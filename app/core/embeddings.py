"""Sentence embeddings with BAAI/bge-small-en-v1.5.

BGE v1.5 is asymmetric: queries get a retrieval instruction prefix, passages
do not. Vectors are L2-normalised, so cosine distance is used in ChromaDB.
"""
from __future__ import annotations

import threading
from typing import Optional, Sequence

from app.config import get_settings

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._lock = threading.Lock()
        self._uses_instruction = "bge" in model_name.lower() and "en" in model_name.lower()

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        with self._lock:
            vectors = self._model.encode(
                list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
            )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        query = BGE_QUERY_INSTRUCTION + text if self._uses_instruction else text
        with self._lock:
            vector = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return vector[0].tolist()


_embedder: Optional[Embedder] = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            settings = get_settings()
            _embedder = Embedder(settings.embedding_model, settings.embedding_device)
        return _embedder
