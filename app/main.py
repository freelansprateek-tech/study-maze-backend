"""FastAPI entrypoint.

    GET  /api/health   service + knowledge-base status
    GET  /api/chips    the chip catalogue for the selector dropdown
    POST /api/query    run the Chip-Aware CP-RAG pipeline

Run from the backend folder:

    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.chips import CHIP_CATALOG, CHIP_IDS, COMMON
from app.config import get_settings
from app.core import hf_runtime
from app.core.classifier import ClassifierError
from app.core.llm_client import LLMError
from app.core.pipeline import get_pipeline
from app.core.retriever import KnowledgeBaseEmptyError
from app.ingest.indexer import kb_counts
from app.schemas import ChipInfo, ChipsResponse, HealthResponse, QueryRequest, QueryResponse

logger = logging.getLogger("cprag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "sentence_transformers", "chromadb"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_COUNTS_TTL_SECONDS = 20.0
_counts_cache: tuple[float, dict[str, int]] = (0.0, {})
_counts_lock = threading.Lock()


def cached_kb_counts(force: bool = False) -> dict[str, int]:
    global _counts_cache
    with _counts_lock:
        stamp, counts = _counts_cache
        if force or not counts or time.monotonic() - stamp > _COUNTS_TTL_SECONDS:
            counts = kb_counts()
            _counts_cache = (time.monotonic(), counts)
        return counts


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    logger.info("LLM provider: %s (%s)", settings.resolved_llm_provider, settings.resolved_llm_model)
    logger.info("CORS origins: %s", ", ".join(settings.cors_origins) or "(none)")
    try:
        counts = cached_kb_counts(force=True)
        total = sum(counts.values())
        if total == 0:
            logger.warning("Knowledge base is EMPTY. Run: python scripts/seed_kb.py")
        else:
            logger.info("Knowledge base: %d chunks %s", total, counts)
    except Exception as exc:  # the API still starts so /api/health can report the problem
        logger.error("Could not open the knowledge base: %s", exc)

    if settings.preload_models:
        logger.info("Preloading the embedding model and %s ...", settings.classifier_model)
        from app.core.embeddings import get_embedder

        get_embedder()
        hf_runtime.get_runtime(settings.classifier_model)
        logger.info("Models loaded on %s", hf_runtime.resolve_device(settings.device))
    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Chip-aware CP-RAG question answering over ESP32-family documentation.",
    lifespan=lifespan,
)

_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in _origins else _origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    cfg = get_settings()
    detail = ""
    try:
        counts = cached_kb_counts()
    except Exception as exc:
        counts, detail = {}, f"Knowledge base unavailable: {exc}"
    total = sum(counts.values())
    if not detail and total == 0:
        detail = "Knowledge base is empty. Run `python scripts/seed_kb.py` in the backend folder."
    return HealthResponse(
        status="ok" if total > 0 else "degraded",
        kb_ready=total > 0,
        kb_chunks=total,
        chunks_by_chip=counts,
        llm_provider=cfg.resolved_llm_provider,
        llm_model=cfg.resolved_llm_model,
        classifier_model=cfg.classifier_model,
        classifier_loaded=hf_runtime.is_loaded(cfg.classifier_model),
        embedding_model=cfg.embedding_model,
        device=hf_runtime.resolve_device(cfg.device),
        version=__version__,
        detail=detail,
    )


@app.get("/api/chips", response_model=ChipsResponse, tags=["catalogue"])
def chips() -> ChipsResponse:
    try:
        counts = cached_kb_counts()
    except Exception:
        counts = {}
    return ChipsResponse(
        chips=[ChipInfo(**CHIP_CATALOG[chip_id], kb_chunks=counts.get(chip_id, 0)) for chip_id in CHIP_IDS],
        common_chunks=counts.get(COMMON, 0),
    )


@app.post("/api/query", response_model=QueryResponse, tags=["query"])
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question for the selected chip, grounded in the indexed documentation."""
    try:
        return get_pipeline().run(request)
    except KnowledgeBaseEmptyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        logger.error("Generator failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"The answer generator failed: {exc}") from exc
    except ClassifierError as exc:
        logger.error("Classifier failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"The attention classifier failed: {exc}") from exc
