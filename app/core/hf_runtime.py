"""Shared Hugging Face causal-LM runtime.

One loaded model serves two jobs: the attention-based document classifier
(a single forward pass with attention weights) and, when `LLM_PROVIDER=local`,
text generation. The model is loaded with `attn_implementation="eager"`
because the fused SDPA kernels never materialise attention weights.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

from app.config import get_settings


def resolve_device(preference: str = "auto") -> str:
    """auto -> mps (Apple Silicon), then cuda, then cpu."""
    import torch

    if preference != "auto":
        return preference
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(preference: str, device: str) -> Any:
    import torch

    explicit = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if preference in explicit:
        return explicit[preference]
    # float16 on Apple MPS overflows inside Qwen's attention and yields NaN
    # weights; bfloat16 has float32's exponent range and stays finite.
    if device == "mps":
        return torch.bfloat16
    return torch.float16 if device == "cuda" else torch.float32


class HFRuntime:
    def __init__(self, model_name: str, device: str, dtype_preference: str = "auto") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.dtype = resolve_dtype(dtype_preference, device)
        self.lock = threading.RLock()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        import transformers

        # The dtype argument was renamed from `torch_dtype` to `dtype` in transformers 4.56.
        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
        dtype_kwarg = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            **{dtype_kwarg: self.dtype},
        )
        self.model.to(device)
        self.model.eval()
        self._torch = torch

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float = 0.0) -> str:
        torch = self._torch
        settings = get_settings()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"]
        limit = settings.local_max_input_tokens
        if input_ids.shape[1] > limit:
            # Keep the head (grounding rules + direct head) and the tail (direct
            # tail + question); drop tokens from the middle, where the
            # indirect-assist material sits.
            keep_head = limit // 2
            keep_tail = limit - keep_head
            input_ids = torch.cat([input_ids[:, :keep_head], input_ids[:, -keep_tail:]], dim=1)
        input_ids = input_ids.to(self.device)

        sampling: dict[str, Any] = (
            {"do_sample": True, "temperature": temperature, "top_p": 0.9}
            if temperature > 0
            else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        )
        with self.lock, torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **sampling,
            )
        generated = output[0, input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


_runtimes: dict[str, HFRuntime] = {}
_runtime_lock = threading.Lock()


def get_runtime(model_name: Optional[str] = None) -> HFRuntime:
    settings = get_settings()
    name = model_name or settings.classifier_model
    with _runtime_lock:
        if name not in _runtimes:
            _runtimes[name] = HFRuntime(name, resolve_device(settings.device), settings.model_dtype)
        return _runtimes[name]


def is_loaded(model_name: Optional[str] = None) -> bool:
    return (model_name or get_settings().classifier_model) in _runtimes
