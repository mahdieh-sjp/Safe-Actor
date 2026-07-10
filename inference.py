import argparse
from pathlib import Path
from utils.seeds import initialize_seeds
from vllm.sampling_params import SamplingParams
from google import genai
import json
import glob
import pandas as pd
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(llm, model, temperature, top_p):
    initialize_seeds()
    model = model.split("/")[-1]
    sampling_params = SamplingParams(max_tokens=8192, temperature=temperature, top_p=top_p)
    DATA_DIR = "./golden-dataset/data/"
    files = glob.glob(f"{DATA_DIR}*_dataset.csv")

    df_list = [pd.read_csv(f) for f in files if "empty" not in f]
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    test_df = merged_df[merged_df["split"] == "test"]
    with open('./golden-dataset/personas_desc.json', 'r') as f:
        personas = json.load(f)
    print(f"\nCharacterful Response Generation for {model}")
        

    for persona_name, persona_desc in personas.items():
        result_path = Path(
            f"./generations/{model}/{persona_name.replace(' ', '_')}.json"
        )
        if result_path.exists():
            print(
                f"Skipping generations from model {model} for role {persona_name}: already computed."
            )
            continue

        persona_instruction = f"You are exactly this character: {persona_name}. {persona_desc}"
        messages = [[
                    {"role": "system", "content": persona_instruction},
                    {"role": "user", "content": q}]
                    for q in test_df[test_df['persona'] == persona_name]["query"]
                ]
        prompts = [llm.get_chat_template(messages, chat_template_kwargs={"enable_thinking": False}) for m in messages]
        print(
            f"Generating responses for role {persona_name} from model {model}"
        )
        responses = []
        print(f"Example prompt: {prompts[0]}")
        llm.generate(prompts[0], sampling_params=sampling_params)
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text.rstrip(" _\n")
            responses.append(generated_text)
        result_path.parent.mkdir(exist_ok=True, parents=True)
        json.dump(responses, open(result_path, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference for model {model}."
    )
    parser.add_argument("model", help="The model model to be prompted.", type=str)
    parser.add_argument("--gpus", help="Number of gpus", type=int, default=1)
    parser.add_argument(
        "--temperature",
        help="Temperature for probabiliy scaling.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--top_p",
        help="Top-p proability of tokens for nucleus sampling",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--dtype", help="dtype to load the model", type=str, default="auto"
    )
    args = parser.parse_args()
    import vllm

    llm = vllm.LLM(
        model=args.model,
        enable_prefix_caching=True,
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.gpus,
        #   download_dir=os.environ["HF_MODELS"],
        gpu_memory_utilization=0.95,
    )
    main(llm, args.model, args.temperature, args.top_p)