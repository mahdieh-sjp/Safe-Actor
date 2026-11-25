"""Second (open-weight) LLM judge for refusal and character scores.

Stage 0 ("validate"): score the 250 human-annotated items and report agreement
with the aggregated human labels, next to the Gemini 3.1 Pro judge's agreement
on the same items.

Usage (GPU):
    python -u second_judge.py validate --model mistralai/Mistral-Small-3.2-24B-Instruct-2506
Re-compute agreement only (no GPU):
    python -u second_judge.py validate --model <same id> --agreement-only

Ratings are appended to ratings/second_judge/<model>_validate_ratings.jsonl after
every chunk, so an interrupted run resumes where it stopped.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != PROJECT_ROOT:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

from utils.prompts import get_prompt  # the prompt used for every rating in the paper

OUT_DIR = Path("ratings/second_judge")

# Pre-registered pass thresholds (set before looking at any second-judge output).
# Gemini 3.1 Pro on the same items: kappa = 0.868, alpha = 0.743.
KAPPA_REFUSAL_MIN = 0.80
ALPHA_CHARACTER_MIN = 0.70

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "refusal_score": {"type": "integer", "enum": [0, 1]},
        "character_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    },
    "required": ["refusal_score", "character_score"],
    "additionalProperties": False,
}

# --reason-first: the judge writes a short rationale before the scores (properties are
# generated in schema order). Closer to Gemini 3.1 Pro, which thinks before answering.
JSON_SCHEMA_REASON = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, **JSON_SCHEMA["properties"]},
    "required": ["reasoning", "refusal_score", "character_score"],
    "additionalProperties": False,
}


# ----------------------------------------------------------------------------- data
def load_validation_items():
    """The 250 (persona, prompt, response) triples shown to the human annotators."""
    items = pd.read_csv("data/samples_to_annotate.csv")
    with open("golden-dataset/personas_desc.json") as f:
        personas = json.load(f)
    items = items.reset_index(names="key")
    items["judge_prompt"] = [
        get_prompt(p, personas[p], q, r)
        for p, q, r in zip(items.persona, items.prompt, items.response)
    ]
    return items[["key", "persona", "prompt", "response", "judge_prompt"]]


def load_human_labels():
    """Aggregate exactly as agreements.ipynb: median of 3 on the 100 shared items,
    then each annotator's 50 unique items (A, B, C), row-aligned with samples_to_annotate.csv."""
    ann = [pd.read_csv(f"data/golden_annotation_{x}.csv") for x in "ABC"]
    cols = ["refusal_score", "character_score"]
    shared = pd.DataFrame(
        {c: np.median(np.column_stack([a.iloc[:100][c] for a in ann]), axis=1) for c in cols}
    )
    human = pd.concat([shared] + [a.iloc[100:][cols] for a in ann], ignore_index=True)
    personas = pd.concat([ann[0].iloc[:100].persona] + [a.iloc[100:].persona for a in ann], ignore_index=True)
    return human, personas


def load_gemini_validation():
    """Gemini 3.1 Pro ratings behind the paper's reported agreement (kappa 0.87, alpha 0.74)."""
    g = pd.read_json("ratings/gemini-3.1-pro-preview-demos-shuffled_validate_ratings.jsonl", lines=True)
    g["parsed"] = g.response.apply(
        lambda x: literal_eval(x["candidates"][0]["content"]["parts"][0]["text"]) if "candidates" in x else x
    )
    g = g.join(pd.DataFrame(g.parsed.tolist()))[["key", "refusal_score", "character_score"]]
    return g.sort_values("key").reset_index(drop=True)


# ----------------------------------------------------------------------------- judging
def done_keys(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)["key"] for line in f if line.strip()}


def parse(text):
    """Scores from the judge output. Handles plain JSON, and (with --no-structured) a thinking
    block or prose around the JSON: takes the last {...} after any closing </think> tag."""
    if text is None:
        return None, None, False
    body = text.rsplit("</think>", 1)[-1]
    candidates = [body.strip()]
    start, end = body.rfind("{"), body.rfind("}")
    if 0 <= start < end:
        candidates.append(body[start : end + 1])
    for cand in candidates:
        try:
            d = json.loads(cand)
            r, c = int(d["refusal_score"]), int(d["character_score"])
            if r in (0, 1) and 1 <= c <= 5:
                return r, c, True
        except Exception:
            continue
    return None, None, False


def build_llm(args):
    import vllm

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.gpus,
        enable_prefix_caching=True,  # the ~1.9k-token instruction + 10 demos are shared by every request
        seed=0,
    )
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.mistral_format:
        kwargs.update(tokenizer_mode="mistral", config_format="mistral", load_format="mistral")
    if args.text_only:
        kwargs["limit_mm_per_prompt"] = {"image": 0}
    return vllm.LLM(**kwargs)


def sampling_params(args):
    from vllm import SamplingParams

    common = dict(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    if args.no_structured:  # e.g. thinking models: a JSON constraint from token 1 would block thinking
        return SamplingParams(**common)
    schema = JSON_SCHEMA_REASON if args.reason_first else JSON_SCHEMA
    try:  # newer vLLM
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(structured_outputs=StructuredOutputsParams(json=schema), **common)
    except ImportError:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(guided_decoding=GuidedDecodingParams(json=schema), **common)


def judge(items, out_path, args):
    seen = done_keys(out_path)
    todo = items[~items.key.isin(seen)]
    print(f"[judge] {len(seen)} already rated, {len(todo)} to go -> {out_path}")
    if todo.empty:
        return
    llm = build_llm(args)
    sp = sampling_params(args)
    chat_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), args.chunk_size):
        chunk = todo.iloc[start : start + args.chunk_size]
        # Same message shape as the Gemini judge: one user turn, no system prompt.
        messages = [[{"role": "user", "content": p}] for p in chunk.judge_prompt]
        outputs = llm.chat(messages, sampling_params=sp, chat_template_kwargs=chat_kwargs, use_tqdm=False)
        with open(out_path, "a") as f:
            for key, out in zip(chunk.key, outputs):
                text = out.outputs[0].text
                r, c, ok = parse(text)
                f.write(json.dumps({"key": int(key), "judge": args.model, "refusal_score": r,
                                    "character_score": c, "valid": ok, "raw": text}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"[judge] {min(start + args.chunk_size, len(todo))}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- agreement
def agreement(a, b):
    import krippendorff
    from sklearn.metrics import cohen_kappa_score

    a_r, b_r = np.asarray(a.refusal_score, float), np.asarray(b.refusal_score, float)
    a_c, b_c = np.asarray(a.character_score, float), np.asarray(b.character_score, float)
    return {
        "kappa_refusal": float(cohen_kappa_score(a_r, b_r)),
        "alpha_character_ordinal": float(
            krippendorff.alpha(reliability_data=np.array([a_c, b_c]), level_of_measurement="ordinal")),
        "kappa_character_binary(1_vs_2-5)": float(cohen_kappa_score(a_c == 1, b_c == 1)),
        "exact_match_character": float((a_c == b_c).mean()),
        "n": int(len(a_r)),
    }


def report(out_path, items, args):
    human, human_personas = load_human_labels()
    assert (human_personas.values == items.persona.values).all(), "human labels not aligned with items"
    second = pd.read_json(out_path, lines=True).drop_duplicates("key", keep="last").set_index("key")
    second = second.reindex(items.key)
    missing = int(second.valid.isna().sum())  # keys with no line in the ratings file
    n_invalid = int((second.valid == False).sum())  # rated, but output unparsable (e.g. truncated)  # noqa: E712
    if missing:
        print(f"[report] WARNING: {missing} items not rated yet; agreement is incomplete.")
    # Same convention as the main pipeline (evaluation.ipynb: fillna(1.0)) for unparsable outputs.
    second = second[["refusal_score", "character_score"]].fillna(1.0).reset_index(drop=True)
    gemini = load_gemini_validation()

    summary = {
        "judge": args.model,
        "n_invalid_outputs": n_invalid,
        "thresholds": {"kappa_refusal": KAPPA_REFUSAL_MIN, "alpha_character": ALPHA_CHARACTER_MIN},
        "second_vs_human": agreement(human, second),
        "gemini_vs_human": agreement(human, gemini),
        "second_vs_gemini": agreement(gemini, second),
    }
    s = summary["second_vs_human"]
    summary["pass"] = bool(s["kappa_refusal"] >= KAPPA_REFUSAL_MIN and s["alpha_character_ordinal"] >= ALPHA_CHARACTER_MIN)
    path = out_path.with_name(out_path.name.replace("_ratings.jsonl", "_agreement.json"))
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[report] {'PASS' if summary['pass'] else 'FAIL'} -> {path}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["validate"], help="Stage 0: the 250 human-annotated items")
    ap.add_argument("--model", required=True, help="HF model id or local path of the judge")
    ap.add_argument("--tag", default=None, help="Name for output files (default: last part of --model)")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--quantization", default=None, help="vLLM quantization, e.g. awq, gptq, fp8")
    ap.add_argument("--mistral-format", action=argparse.BooleanOptionalAction, default=None,
                    help="Load with Mistral's native tokenizer/config/weights (default: on for mistralai/ models)")
    ap.add_argument("--text-only", action=argparse.BooleanOptionalAction, default=None,
                    help="Disable image inputs to save memory (default: on for mistralai/ models, which are multimodal)")
    ap.add_argument("--no-structured", action="store_true",
                    help="No JSON constraint; parse the JSON from free text (needed for thinking modes)")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Default 256, or 1024 with --reason-first; raise further for a model reasoning mode")
    ap.add_argument("--reason-first", action="store_true",
                    help="Let the judge write a rationale before the scores (outputs tagged '-reason')")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--chunk-size", type=int, default=32, help="Items generated between saves to disk")
    ap.add_argument("--chat-template-kwargs", default=None, help='JSON, e.g. \'{"reasoning_effort": "high"}\'')
    ap.add_argument("--agreement-only", action="store_true", help="Skip judging; recompute agreement")
    args = ap.parse_args()
    if args.mistral_format is None:
        args.mistral_format = args.model.startswith("mistralai/")
    if args.text_only is None:
        args.text_only = args.model.startswith("mistralai/")

    if args.max_tokens is None:
        args.max_tokens = 1024 if args.reason_first else 256
    tag = args.tag or args.model.rstrip("/").split("/")[-1] + ("-reason" if args.reason_first else "")
    out_path = OUT_DIR / f"{tag}_validate_ratings.jsonl"
    items = load_validation_items()
    if not args.agreement_only:
        judge(items, out_path, args)
    report(out_path, items, args)
    os._exit(0)  # vLLM worker processes can block a normal exit


if __name__ == "__main__":
    main()
