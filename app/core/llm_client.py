"""Generator LLM clients behind one interface.

    openai / groq / ollama  -> any OpenAI-compatible `/chat/completions` endpoint
    local                   -> the Hugging Face model already loaded for the classifier

Selected by `LLM_PROVIDER` (see `app/config.py`).
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Protocol

import httpx

from app.config import Settings, get_settings

Message = dict[str, str]


class LLMError(RuntimeError):
    """The generator could not produce a completion."""


class LLMClient(Protocol):
    provider: str
    model: str
    parallel: bool  # safe/beneficial to issue several requests at once

    def chat(self, messages: list[Message], max_tokens: int, temperature: Optional[float] = None) -> str: ...


class OpenAICompatibleClient:
    parallel = True

    def __init__(self, provider: str, base_url: str, model: str, api_key: str, timeout: float, temperature: float):
        if not base_url:
            raise LLMError(f"LLM_BASE_URL is not set for provider '{provider}'.")
        if not model:
            raise LLMError(f"LLM_MODEL is not set for provider '{provider}'.")
        if provider in ("openai", "groq") and not api_key:
            raise LLMError(f"LLM_API_KEY is required for provider '{provider}'. Set it in backend/.env.")
        self.provider = provider
        self.model = model
        self._url = f"{base_url}/chat/completions"
        self._temperature = temperature
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def chat(self, messages: list[Message], max_tokens: int, temperature: Optional[float] = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self._temperature if temperature is None else temperature,
        }
        last_error = "no attempt made"
        for attempt in range(4):
            try:
                response = self._client.post(self._url, json=payload)
            except httpx.HTTPError as exc:
                last_error = f"cannot reach {self._url}: {exc}"
                time.sleep(1.5 * (attempt + 1))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                retry_after = response.headers.get("retry-after", "")
                delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2.0 * (attempt + 1)
                time.sleep(min(delay, 30.0))
                continue
            if response.status_code >= 400:
                raise LLMError(f"{self.provider} returned HTTP {response.status_code}: {response.text[:400]}")
            try:
                content = response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise LLMError(f"Unexpected response shape from {self.provider}: {exc}") from exc
            return (content or "").strip()
        raise LLMError(f"{self.provider} request failed after retries ({last_error}).")


class LocalHFClient:
    provider = "local"
    parallel = False

    def __init__(self, model: str, temperature: float):
        self.model = model
        self._temperature = temperature

    def chat(self, messages: list[Message], max_tokens: int, temperature: Optional[float] = None) -> str:
        from app.core.hf_runtime import get_runtime

        try:
            runtime = get_runtime(self.model)
            return runtime.generate(
                messages, max_new_tokens=max_tokens,
                temperature=self._temperature if temperature is None else temperature,
            )
        except Exception as exc:
            raise LLMError(f"Local model '{self.model}' failed: {exc}") from exc


_client: Optional[LLMClient] = None
_client_lock = threading.Lock()


def build_llm_client(settings: Optional[Settings] = None) -> LLMClient:
    settings = settings or get_settings()
    provider = settings.resolved_llm_provider
    if provider == "local":
        return LocalHFClient(settings.resolved_llm_model, settings.llm_temperature)
    return OpenAICompatibleClient(
        provider=provider,
        base_url=settings.resolved_llm_base_url,
        model=settings.resolved_llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        temperature=settings.llm_temperature,
    )


def get_llm_client() -> LLMClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = build_llm_client()
        return _client
