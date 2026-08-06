import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm


class HFChatClient:
    """
    Minimal drop-in replacement for vllm.LLM's .chat() interface, using plain
    transformers generation. Used for models where vLLM support is currently
    broken/unstable (e.g. Qwen3.5 text-only, see vllm-project/vllm#39316).

    Accepts either:
      - a path/Hub ID string (loads fresh, as before), or
      - an already-instantiated model (and optionally tokenizer) object,
        which skips save_pretrained/from_pretrained round-tripping entirely.
        Useful for models whose custom remote code has serialization bugs
        (e.g. Nemotron-H's _tied_weights_keys list/dict mismatch on
        transformers v5+) since the model never needs to be re-saved to disk.
    """

    class _Output:
        def __init__(self, text):
            self.outputs = [self._Inner(text)]

        class _Inner:
            def __init__(self, text):
                self.text = text

    def __init__(
        self,
        model_path=None,
        tokenizer_path=None,
        dtype="auto",
        trust_remote_code=True,
        batch_size=8,
        model=None,
        tokenizer=None,
    ):
        self.batch_size = batch_size

        if model is not None:
            # already-loaded model path: no from_pretrained / save_pretrained round trip
            self.model = model
            self.model.eval()

            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                tok_source = tokenizer_path or model_path
                if tok_source is None:
                    raise ValueError(
                        "When passing a pre-loaded `model`, you must also pass "
                        "either `tokenizer` or `tokenizer_path`/`model_path` to "
                        "load a matching tokenizer from."
                    )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tok_source, trust_remote_code=trust_remote_code
                )
        else:
            if model_path is None:
                raise ValueError("Must pass either `model_path` or a pre-loaded `model`.")

            torch_dtype = torch.bfloat16 if dtype in ("auto", "bfloat16") else getattr(torch, dtype)

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path or model_path, trust_remote_code=trust_remote_code
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
            self.model.eval()

        # Infer the input device from the model itself rather than assuming a
        # single "cuda" device. Matters especially for pre-loaded models that
        # may have been loaded with device_map="auto" and could be sharded
        # across multiple GPUs or partially CPU-offloaded — inputs must go
        # wherever the embedding layer actually lives, not just "cuda:0".
        try:
            self.device = self.model.get_input_embeddings().weight.device
        except AttributeError:
            # fallback for models without a standard get_input_embeddings (rare)
            self.device = next(self.model.parameters()).device

        # left padding required for correct batched causal-LM generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

            output_ids = self.model.generate(**enc, **gen_kwargs)

            for row in output_ids:
                generated = row[input_len:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                all_outputs.append(self._Output(text))

        return all_outputs