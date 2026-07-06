import os
from pathlib import Path

import transformers.utils.hub
if not hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE"):
    transformers.utils.hub.TRANSFORMERS_CACHE = getattr(transformers.utils.hub, "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer


# WRITE THE MODEL NAME HERE
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# WRITE THE TRAINING MODE HERE
MODE = "sft_dpo"  # Options: "sft", "dpo", "sft_dpo"

# WRITE THE DATASET PATH HERE
DATASET_PATH = ""  

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"



def get_model_and_tokenizer(model_name, max_seq_length=1024, dtype=None, load_in_4bit=True):
    """
    Load the base model and tokenizer using FastLanguageModel.
    """
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=None,
    )
    return base_model, tokenizer


def get_peft_model(base_model, r=16, target_modules=None, lora_alpha=16, lora_dropout=0, use_rslora=False):
    """
    Wrap the base model with PEFT (Parameter-Efficient Fine-Tuning) using LoRA.
    """

    if target_modules is None:
        # full list for Qwen and LLaMA/Mistral
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    peft_model = FastLanguageModel.get_peft_model(
        base_model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=use_rslora,
        loftq_config=None,
    )
    return peft_model


def sft_formatting_data(tokenizer, dataset):
    """
    Format the dataset for Supervised Fine-Tuning (SFT) by creating conversation structures and applying the chat template from the tokenizer.
    """
    def formatting_prompts_func(examples):
        convos = []
        for system, prompt, chosen in zip(examples["system"], examples["prompt"], examples["chosen"]):
            convo = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}
            ]
            convos.append(convo)
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    sft_dataset = dataset.map(formatting_prompts_func, batched=True)
    return sft_dataset


def dpo_formatting_data(tokenizer, dataset):
    """
    Format the dataset for Direct Preference Optimization (DPO) by creating conversation structures and applying the chat template from the tokenizer.
    """
    def format_dpo(examples):
        prompts = []
        chosens = []
        rejecteds = []
        for sys_prompt, p, c, r in zip(examples["system"], examples["prompt"], examples["chosen"], examples["rejected"]):
            convo = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": p}]
            prompt_str = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            chosen_str = c + tokenizer.eos_token
            rejected_str = str(r) + tokenizer.eos_token
            prompts.append(prompt_str)
            chosens.append(chosen_str)
            rejecteds.append(rejected_str)

        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    dpo_dataset = dataset.map(format_dpo, batched=True)
    return dpo_dataset



def get_sft_trainer(
        peft_model,
        tokenizer,
        sft_dataset,
        learning_rate=2e-4,
        num_train_epochs=1,
        batch_size=2,
        grad_accum=4,
        output_dir="/temp_SFT"
        ):

    """
    Create and return an SFTTrainer for supervised fine-tuning.
    """

    args = SFTConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=5,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=peft_model,
        processing_class=tokenizer,
        train_dataset=sft_dataset,
        args=args,
    )
    return trainer


def get_dpo_trainer(
        model,
        tokenizer,
        dpo_dataset,
        batch_size=1,
        grad_accum=8,
        learning_rate=5e-5,
        num_train_epochs=1,
        beta=0.1,
        output_dir="/temp_DPO"
        ):

    """
    Create and return a DPOTrainer for direct preference optimization.
    """

    training_args = DPOConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=5,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="none",
        save_strategy="epoch",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        beta=beta,
        train_dataset=dpo_dataset,
        args=training_args,
    )
    return trainer



def get_latest_checkpoint(output_dir):
    checkpoints = [d for d in Path(output_dir).iterdir() if d.is_dir() and "checkpoint" in d.name]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {output_dir}")
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return str(latest_checkpoint)


def get_training_pipeline(mode, dataset_path, model_params, peft_params, sft_params, dpo_params, quantization_method="q4_k_m"):
    """
    Execute the training pipeline based on the specified mode (SFT or DPO).
    """
    print(f"=== TRAINING PIPELINE MODE: {mode.upper()} - MODEL: {model_params.get('model_name', 'Unknown')} ===")
    print("=== LOADING DATASET ===")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    clean_name = model_params.get('model_name', 'Unknown').split('/')[-1]

    if mode == "sft":
        print("\n=== SFT ===")
        base_model, tokenizer = get_model_and_tokenizer(**model_params)
        sft_model = get_peft_model(base_model, **peft_params)
        sft_dataset = sft_formatting_data(tokenizer, dataset)
        sft_trainer = get_sft_trainer(sft_model, tokenizer, sft_dataset, **sft_params)
        sft_trainer.train()
        print("\n=== SAVING SFT MODEL ===")
        sft_trainer.model.save_pretrained_gguf(f"{clean_name}-SFT", tokenizer, quantization_method=quantization_method)
        return sft_trainer.model, tokenizer

    if mode == "dpo":
        print("\n=== DPO ===")
        base_model, tokenizer = get_model_and_tokenizer(**model_params)
        dpo_model = get_peft_model(base_model, **peft_params)
        dpo_dataset = dpo_formatting_data(tokenizer, dataset)
        dpo_trainer = get_dpo_trainer(dpo_model, tokenizer, dpo_dataset, **dpo_params)
        dpo_trainer.train()
        print("\n=== SAVING DPO MODEL ===")
        dpo_trainer.model.save_pretrained_gguf(f"{clean_name}-DPO", tokenizer, quantization_method=quantization_method)
        return dpo_trainer.model, tokenizer

    if mode == "sft_dpo":
        print("\n=== SFT + DPO ===")
        base_model, tokenizer = get_model_and_tokenizer(**model_params)
        sft_model = get_peft_model(base_model, **peft_params)
        sft_dataset = sft_formatting_data(tokenizer, dataset)
        sft_trainer = get_sft_trainer(sft_model, tokenizer, sft_dataset, **sft_params)
        sft_trainer.train()

        # if this mode is gonna be run in two phases, you can save the SFT model here and load it later for DPO training from the last checkpoint

        dpo_dataset = dpo_formatting_data(tokenizer, dataset)
        dpo_trainer = get_dpo_trainer(sft_trainer.model, tokenizer, dpo_dataset, **dpo_params)
        dpo_trainer.train()
        print("\n=== SAVING SFT + DPO MODEL ===")
        dpo_trainer.model.save_pretrained_gguf(f"{clean_name}-SFT-DPO", tokenizer, quantization_method=quantization_method)
        return dpo_trainer.model, tokenizer


def main():

    model_params = {
        "model_name": MODEL_NAME,
        "max_seq_length": 1024,
        "dtype": None,
        "load_in_4bit": True
    }
    peft_params = {
        "r": 16,
        "target_modules": None,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "use_rslora": False
    }
    sft_params = {
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "batch_size": 2,
        "grad_accum": 4,
        "output_dir": "/temp_SFT"
    }

    dpo_params = {
        "batch_size": 1,
        "grad_accum": 8,
        "learning_rate": 5e-5,
        "num_train_epochs": 1,
        "beta": 0.1,
        "output_dir": "/temp_DPO"
    }

    model, tokenizer = get_training_pipeline(
        mode=MODE,
        dataset_path=DATASET_PATH,
        model_params=model_params,
        peft_params=peft_params,
        sft_params=sft_params,
        dpo_params=dpo_params,
        quantization_method="q4_k_m"
    )

if __name__ == "__main__":
    main()