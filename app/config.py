"""Runtime configuration, loaded from environment variables and `backend/.env`.

Every tunable of the pipeline lives here so that the API, the ingestion
scripts and the evaluation suite all read the same values.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Presets for the OpenAI-compatible providers. `LLM_BASE_URL` / `LLM_MODEL`
# override these when set.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:1.5b"},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API ---------------------------------------------------------------
    app_name: str = "Chip-Aware CP-RAG for ESP32"
    # Comma-separated list of origins allowed by CORS. "*" allows any origin.
    frontend_origin: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Knowledge base ----------------------------------------------------
    data_dir: Path = BASE_DIR / "data"
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    collection_name: str = "esp32_docs"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    chunk_words: int = Field(default=180, ge=40, le=600)
    chunk_overlap_words: int = Field(default=30, ge=0, le=200)

    # --- Retrieval ---------------------------------------------------------
    top_k: int = Field(default=8, ge=1, le=20)

    # --- Attention classifier ---------------------------------------------
    classifier_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    classifier_layers: int = Field(default=4, ge=1, le=12)
    classifier_max_chunk_tokens: int = Field(default=256, ge=32, le=1024)
    # auto -> mps, then cuda, then cpu
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    # auto -> bfloat16 on mps, float16 on cuda, float32 on cpu
    model_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    preload_models: bool = False

    # --- Generator LLM -----------------------------------------------------
    # auto -> an OpenAI-compatible API when LLM_API_KEY is set, otherwise the
    # local Hugging Face model (shared with the classifier).
    llm_provider: Literal["auto", "local", "openai", "groq", "ollama"] = "auto"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout: float = 120.0
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    answer_max_tokens: int = Field(default=800, ge=64, le=4096)
    # True: a fuller answer in simple words plus a longer explanation (the website).
    # False: the short exact answer that the benchmark scores.
    detailed_answers: bool = True
    summary_max_tokens: int = Field(default=180, ge=64, le=1024)
    local_max_input_tokens: int = Field(default=6000, ge=512, le=30000)

    # --- Rearranging module ------------------------------------------------
    # Noise documents the rearranging module itself adds to the indirect-assist
    # branch on every query (independent of the robustness noise ratio).
    rearrange_noise_docs: int = Field(default=2, ge=0, le=10)
    # pool -> sample from the built-in distractor pool; llm -> generate them.
    noise_source: Literal["pool", "llm"] = "pool"
    random_seed: int = 42

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.frontend_origin.split(",") if o.strip()]

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        return "openai" if self.llm_api_key else "local"

    @property
    def resolved_llm_base_url(self) -> str:
        preset = PROVIDER_PRESETS.get(self.resolved_llm_provider, {})
        return (self.llm_base_url or preset.get("base_url", "")).rstrip("/")

    @property
    def resolved_llm_model(self) -> str:
        if self.resolved_llm_provider == "local":
            return self.llm_model or self.classifier_model
        preset = PROVIDER_PRESETS.get(self.resolved_llm_provider, {})
        return self.llm_model or preset.get("model", "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
