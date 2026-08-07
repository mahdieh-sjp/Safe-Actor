import sys
import os

# Force the project root directory to be the #1 priority in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != PROJECT_ROOT:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import gc
from pathlib import Path
from huggingface_hub import snapshot_download
from peft import PeftModel
from utils.seeds import initialize_seeds
from models.openai_client import OpenAIBatchClient
from models.hf_client import HFChatClient
from transformers import AutoModelForCausalLM, AutoProcessor
import json
import glob
import pandas as pd
import torch


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(llm, model, client, temperature):
    initialize_seeds()
    model = model.split("/")[-1]
    DATA_DIR = "./golden-dataset/data/"
    files = glob.glob(f"{DATA_DIR}*_dataset.csv")
    files = sorted(files)

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
        responses = []
        print(
                f"Generating responses for role {persona_name} from model {model}"
            )
        if client is None or client == "hf":
            if client is None:
                sampling_params = SamplingParams(max_tokens=8192, temperature=temperature)
            else:
                sampling_params = {"max_tokens": 8192, "temperature": temperature}
            prompts = [[
                        {"role": "system", "content": persona_instruction},
                        {"role": "user", "content": q}]
                        for q in test_df[test_df['persona'] == persona_name]["prompt"]
                    ]
            print(f"Example prompt: {prompts[0]}")
            llm.chat(prompts[0], sampling_params=sampling_params,
                    chat_template_kwargs={"enable_thinking": False})
            outputs = llm.chat(prompts, sampling_params=sampling_params,
                    chat_template_kwargs={"enable_thinking": False})
            for output in outputs:
                generated_text = output.outputs[0].text.rstrip(" _\n")
                responses.append(generated_text)
        elif client == "openai":
            for q in test_df[test_df['persona'] == persona_name]["prompt"]:
                response = llm.run_prompt(q, persona_instruction).rstrip(" _\n")
                responses.append(response)

        result_path.parent.mkdir(exist_ok=True, parents=True)
        json.dump(responses, open(result_path, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference for model {model}."
    )
    parser.add_argument("model", help="The model model to be prompted.", type=str)
    parser.add_argument("--base_model", help="The base model that was adapted.", type=str, default=None)
    parser.add_argument("--gpus", help="Number of gpus", type=int, default=1)
    parser.add_argument("--client", help="If using an api", type=str, default=None, choices=["openai", "hf"])
    parser.add_argument(
        "--temperature",
        help="Temperature for probabiliy scaling.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dtype", help="dtype to load the model", type=str, default="auto"
    )
    args = parser.parse_args()
    if args.client is None or args.client == "hf":

        if "SFT" in args.model:
            base_model = args.base_model
            model  = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                #trust_remote_code=True,
                dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(
                        model,
                        f"{args.model.replace('+DPO', '')}/final_adapter",
                    )
            model = model.merge_and_unload()
            if "DPO" in args.model:
                model = PeftModel.from_pretrained(
                            model,
                            f"{args.model}/final_adapter",
                        )
                model = model.merge_and_unload()
            model.generation_config.top_p = None

            if args.client == None:
                model.save_pretrained(
                    "/tmp/merged_model",
                    safe_serialization=True,
                )
                if "gemma" in args.model:
                    processor = AutoProcessor.from_pretrained(
                        args.base_model,
                        trust_remote_code=True,
                    )
                    processor.save_pretrained(
                        "/tmp/merged_model"
                    )
                # if "nemotron" in args.model.lower():
                #     config_path = "/tmp/merged_model/config.json"
                #     if os.path.exists(config_path):
                #         with open(config_path, "r") as f:
                #             config_data = json.load(f)
                #         if "layers_block_type" in config_data:
                #             del config_data["layers_block_type"]
                #             with open(config_path, "w") as f:
                #                 json.dump(config_data, f, indent=2)
                #         model = "/tmp/merged_model"

                #     modeling_path = "/tmp/merged_model/modeling_nemotron_h.py"
                #     if os.path.exists(modeling_path):
                #         with open(modeling_path, "r") as f:
                #             code = f.read()
                #         if "or cache_position[-1] >=" in code:
                #             code = code.replace(
                #                 "or cache_position[-1] >=", 
                #                 "or (cache_position is not None and cache_position[-1] >="
                #             )
                #             with open(modeling_path, "w") as f:
                #                 f.write(code)

        else:
            model = args.model
            base_model = model
        if args.client == "hf":
            print(f"Using transformers inference engine for {base_model} (in-memory model, no disk round trip)")
            if isinstance(model, str):
                llm = HFChatClient(model_path=model, tokenizer_path=base_model, dtype=args.dtype, trust_remote_code=True)
            else:
                llm = HFChatClient(model=model, tokenizer_path=base_model, trust_remote_code=True)
        else:
            if ("Qwen" in args.model or "nemotron" in args.model.lower()) and "SFT" in args.model:
                import vllm
                from vllm.sampling_params import SamplingParams

                llm = vllm.LLM(
                    model=model,
                    enable_prefix_caching=True,
                    dtype=args.dtype,
                    #trust_remote_code=True,
                    tensor_parallel_size=args.gpus,
                    model_impl="transformers",
                    #   download_dir=os.environ["HF_MODELS"],
                    gpu_memory_utilization=0.95,
                    tokenizer=base_model
                )
            else:
                import vllm
                from vllm.sampling_params import SamplingParams

                llm = vllm.LLM(
                        model=model,
                        enable_prefix_caching=True,
                        dtype=args.dtype,
                        #trust_remote_code=True,
                        tensor_parallel_size=args.gpus,
                        #   download_dir=os.environ["HF_MODELS"],
                        gpu_memory_utilization=0.95,
                        tokenizer=base_model
                    )
    elif args.client == "openai":
        llm = OpenAIBatchClient(model_name=args.model)
    main(llm, args.model, args.client, args.temperature)

    if "llm" in locals():
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Inference completed. Terminating process.")
    # Use os._exit(0) to bypass blocked vLLM process joins
    os._exit(0)