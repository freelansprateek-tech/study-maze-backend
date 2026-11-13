"""Chip-filtered retrieval over ChromaDB.

The selected chip is a hard metadata filter, not a ranking hint: a chunk
tagged for a different chip can never reach the prompt.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from app.chips import COMMON
from app.config import Settings, get_settings
from app.core.embeddings import get_embedder
from app.ingest.indexer import get_collection
from app.schemas import ChunkTrace


class KnowledgeBaseEmptyError(RuntimeError):
    pass


_STOPWORDS = frozenset(
    "a an the is are was were be been of on in to for with and or by at from as it its this that these those what "
    "which how many much does do can i my you your when where why will would should there than then if have has had "
    "not no chip esp32 happens used use using".split()
)
# Dense retrieval alone favours long explanatory guide pages. Hardware questions
# are usually answered by a short datasheet passage that shares the question's
# exact terms ("RTC memory", "strapping", "VDD_SPI"), so candidates are re-ranked
# with a keyword-overlap term and a small prior per source type.
LEXICAL_WEIGHT = 0.15
SOURCE_PRIOR = {"datasheet": 0.04, "trm": 0.02, "esp-idf-docs": 0.0, "forum": -0.03, "github_issue": -0.04}
CANDIDATE_MULTIPLIER = 5


def content_terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[a-z0-9_]+", text.lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        terms.add(token[:-1] if len(token) > 3 and token.endswith("s") else token)
    return terms


def lexical_overlap(question_terms: set[str], text: str) -> float:
    """Share of the question's content terms that occur in `text` (0..1)."""
    if not question_terms:
        return 0.0
    return len(question_terms & content_terms(text)) / len(question_terms)


class ChipFilteredRetriever:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    @staticmethod
    def chip_filter(selected_chip: str) -> dict[str, Any]:
        """chip ∈ {selected_chip, "common"}"""
        return {"chip": {"$in": [selected_chip, COMMON]}}

    def retrieve(self, question: str, selected_chip: str, k: int, chip_filter: bool = True) -> list[ChunkTrace]:
        collection = get_collection(self.settings)
        total = collection.count()
        if total == 0:
            raise KnowledgeBaseEmptyError(
                "The knowledge base is empty. Run `python scripts/seed_kb.py` in the backend folder first."
            )
        query_vector = get_embedder().embed_query(question)
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(max(k * CANDIDATE_MULTIPLIER, 30), total),
            where=self.chip_filter(selected_chip) if chip_filter else None,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        question_terms = content_terms(question)
        scored = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            similarity = 1.0 - float(distance)
            searchable = f"{metadata.get('title', '')} {metadata.get('section', '')} {text or ''}"
            score = (
                similarity
                + LEXICAL_WEIGHT * lexical_overlap(question_terms, searchable)
                + SOURCE_PRIOR.get(str(metadata.get("source_type", "")), 0.0)
            )
            scored.append((score, chunk_id, text, metadata, distance))
        scored.sort(key=lambda item: item[0], reverse=True)

        chunks: list[ChunkTrace] = []
        for rank, (_score, chunk_id, text, metadata, distance) in enumerate(scored[:k], 1):
            chunks.append(
                ChunkTrace(
                    source_id="",  # assigned once the full context set is known
                    chunk_id=chunk_id,
                    title=str(metadata.get("title", "Untitled")),
                    chip=str(metadata.get("chip", COMMON)),  # type: ignore[arg-type]
                    section=str(metadata.get("section", "")),
                    source_type=str(metadata.get("source_type", "")),
                    url=str(metadata.get("url", "")),
                    text=text or "",
                    retrieval_rank=rank,
                    similarity=round(1.0 - float(distance), 4),
                )
            )
        return chunks
