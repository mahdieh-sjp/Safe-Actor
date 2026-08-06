import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

class HFChatClient:
    """
    Minimal drop-in replacement for vllm.LLM's .chat() interface, using plain
    transformers generation. Used for models where vLLM support is currently
    broken/unstable (e.g. Qwen3.5 text-only, see vllm-project/vllm#39316).
    """

    class _Output:
        def __init__(self, text):
            self.outputs = [self._Inner(text)]

        class _Inner:
            def __init__(self, text):
                self.text = text

    def __init__(self, model_path, tokenizer_path=None, dtype="auto", trust_remote_code=True, batch_size=8):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_dtype = torch.bfloat16 if dtype in ("auto", "bfloat16") else getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model_path, trust_remote_code=trust_remote_code
        )
        # left padding required for correct batched causal-LM generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def _build_prompt(self, conversation, chat_template_kwargs):
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )

    @torch.inference_mode()
    def chat(self, prompts, sampling_params=None, chat_template_kwargs=None):
        # accept either a single conversation (list[dict]) or a batch (list[list[dict]]),
        # matching vllm.LLM.chat's calling convention used elsewhere in this script
        if prompts and isinstance(prompts[0], dict):
            conversations = [prompts]
        else:
            conversations = prompts

        max_new_tokens = getattr(sampling_params, "max_tokens", 8192)
        temperature = getattr(sampling_params, "temperature", 0.0)
        top_p = getattr(sampling_params, "top_p", 1.0)
        do_sample = temperature > 0.0

        all_outputs = []
        for i in tqdm(range(0, len(conversations), self.batch_size), desc="Generating"):
            batch = conversations[i:i + self.batch_size]
            texts = [self._build_prompt(c, chat_template_kwargs) for c in batch]

            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(self.device)
            input_len = enc["input_ids"].shape[1]

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            output_ids = self.model.generate(**enc, **gen_kwargs)

            for row in output_ids:
                generated = row[input_len:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                all_outputs.append(self._Output(text))

        return all_outputs