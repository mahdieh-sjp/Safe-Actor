
import os
import torch
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftModel
import argparse
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
from utils.seeds import initialize_seeds
from utils.prompts import get_system_prompt
import json


os.environ["TOKENIZERS_PARALLELISM"] = "false"

personas = json.load(open("./golden-dataset/personas_desc.json", "r"))

def preprocess_dpo(example):
    persona = example["persona"]
    description = personas[persona]
    return {
        "prompt": [{"role": "system", "content": get_system_prompt(persona, description)},
                   {"role": "user", "content": example["prompt"]}],
        "chosen": [{"role": "assistant", "content": example["preferred_response"]}],
        "rejected": [{"role": "assistant", "content": example["rejected_response"]}],
    }

def preprocess_sft(example):
    persona = example["persona"]
    description = personas[persona]
    return {
        "prompt": [{"role": "system", "content": get_system_prompt(persona, description)},
                   {"role": "user", "content": example["prompt"]}],
        "completion": [{"role": "assistant", "content": example["preferred_response"]}],
    }

def find_resume_checkpoint(output_dir):
    """Return the path to the last checkpoint in output_dir, or None if there isn't one."""
    if not os.path.isdir(output_dir):
        return None
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is not None:
        print(f"Found existing checkpoint at {last_checkpoint}, will resume from there.")
    return last_checkpoint


def main(model, batch_size=16, grad_accumulation_steps=1, dev=False, dev_size=100):
    initialize_seeds()

    model_name = model.split("/")[-1]


    print(f"Training model {model_name} with SFT + DPO")
    
    print("=== LOADING DATASET ===")
    train_data = load_dataset("json", data_files="golden-dataset/train_clean.jsonl")

    print("=== PREPROCESSING DATASET ===")
    sft_dataset = train_data.map(preprocess_sft, remove_columns=["persona", "query_type", "preferred_response", "rejected_response"])["train"]
    dpo_dataset = train_data.map(preprocess_dpo, remove_columns=["persona", "query_type", "preferred_response", "rejected_response"])["train"]
    sft_dataset = sft_dataset.add_column(
        "chat_template_kwargs",
        [{"enable_thinking": False} for _ in range(len(sft_dataset))]
    )
    dpo_dataset = dpo_dataset.add_column(
        "chat_template_kwargs",
        [{"enable_thinking": False} for _ in range(len(dpo_dataset))]
    )


    if dev:
        print(f"=== DEV MODE: USING SUBSAMPLE OF SIZE {dev_size} ===")
        sft_dataset = sft_dataset.shuffle(seed=42).select(range(dev_size))
        dpo_dataset = dpo_dataset.shuffle(seed=42).select(range(dev_size))

    suffix = "-dev" if dev else ""
    sft_output_dir = f"models/{model_name}-SFT{suffix}"
    dpo_output_dir = f"models/{model_name}-SFT+DPO{suffix}"
    print(sft_dataset, dpo_dataset)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules= ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation_steps,
        warmup_steps=.1,
        num_train_epochs=3,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_strategy="epoch",
        lr_scheduler_type="linear",
        seed=42,
        output_dir=sft_output_dir,
        save_strategy="epoch",
    )


    dpo_config = DPOConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation_steps,
        warmup_steps=.1,
        num_train_epochs=3,
        learning_rate=5e-6,
        weight_decay=0.01,
        logging_strategy="epoch",
        lr_scheduler_type="linear",
        seed=42,
        beta=0.1,
        output_dir=dpo_output_dir,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        train_dataset=sft_dataset,
        args=sft_config,
    )

    trainer.model.print_trainable_parameters()

    print("=== RUNNING SFT ===")
    sft_resume = find_resume_checkpoint(sft_output_dir)
    trainer.train(resume_from_checkpoint=sft_resume)

    print("=== SAVING SFT ADAPTER ===")
    sft_adapter_dir = f"{sft_output_dir}/final_adapter"
    trainer.save_model(sft_adapter_dir)

    dpo_model = trainer.model.merge_and_unload()

    # 2. Free SFT trainer state & clear VRAM cache
    del trainer
    torch.cuda.empty_cache()

    trainer =  DPOTrainer(
        model=dpo_model,
        peft_config=peft_config,
        train_dataset=dpo_dataset,
        args=dpo_config,
    )
    trainer.model.print_trainable_parameters()

    print("=== RUNNING DPO ===")
    dpo_resume = find_resume_checkpoint(dpo_output_dir)
    trainer.train(resume_from_checkpoint=dpo_resume)
 
    print("=== SAVING DPO ADAPTER ===")
    dpo_adapter_dir = f"{dpo_output_dir}/final_adapter"
    trainer.save_model(dpo_adapter_dir)
 
    print(f"Done. SFT adapter saved to {sft_adapter_dir}, DPO adapter saved to {dpo_adapter_dir}")



    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference for model {model}."
    )
    parser.add_argument("model", help="The model to be trained", type=str)
    parser.add_argument("--batch_size", help="Batch size for training", type=int, default=16)
    parser.add_argument("--grad_accumulation_steps", help="Gradient accumulation steps for training", type=int, default=1)
    parser.add_argument("--dev", action="store_true", help="Use a small subsample of the dataset for fast dev iteration", default=False)
    parser.add_argument("--dev_size", type=int, default=100, help="Number of samples to use in dev mode")
    args = parser.parse_args()
    main(args.model, batch_size=args.batch_size, grad_accumulation_steps=args.grad_accumulation_steps, dev=args.dev, dev_size=args.dev_size)