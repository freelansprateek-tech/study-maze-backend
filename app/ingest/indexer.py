"""ChromaDB persistence for the chip-tagged knowledge base."""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from app.chips import ALL_TAGS
from app.config import Settings, get_settings
from app.core.embeddings import get_embedder
from app.ingest.chunker import chunk_document
import json

from app.ingest.parsers import MANIFEST_NAME, ParsedDocument, discover_files, parse_file
from app.ingest.tagger import chip_from_path, normalize_chip, tag_chunk


@dataclass
class IndexRecord:
    chunk_id: str
    text: str
    title: str
    chip: str
    section: str
    source_type: str
    url: str
    source_path: str

    def metadata(self) -> dict[str, str]:
        return {
            "title": self.title,
            "chip": self.chip,
            "section": self.section,
            "source_type": self.source_type,
            "url": self.url,
            "source_path": self.source_path,
        }


def get_collection(settings: Optional[Settings] = None):
    """Open (or create) the persistent collection. Cosine space for BGE vectors."""
    import chromadb

    settings = settings or get_settings()
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(settings: Optional[Settings] = None) -> None:
    import chromadb

    settings = settings or get_settings()
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    existing = {c.name for c in client.list_collections()}
    if settings.collection_name in existing:
        client.delete_collection(settings.collection_name)


def _chunk_id(document: ParsedDocument, index: int, text: str, base_dir: Optional[Path]) -> str:
    if document.doc_id:
        return document.doc_id if index == 0 else f"{document.doc_id}-{index}"
    source = document.source_path
    if base_dir is not None:
        try:
            head, sep, tail = source.partition("#")
            source = Path(head).resolve().relative_to(base_dir.resolve()).as_posix() + sep + tail
        except ValueError:
            pass
    digest = hashlib.sha1(f"{source}|{index}|{text[:64]}".encode("utf-8")).hexdigest()[:16]
    return f"chunk-{digest}"


def build_records(
    documents: Iterable[ParsedDocument],
    settings: Optional[Settings] = None,
    forced_chip: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> list[IndexRecord]:
    """Chunk parsed documents and tag every chunk with its chip."""
    settings = settings or get_settings()
    forced = normalize_chip(forced_chip)
    records: list[IndexRecord] = []
    for document in documents:
        chunks = chunk_document(document, settings.chunk_words, settings.chunk_overlap_words)
        for chunk in chunks:
            chip = tag_chunk(
                chunk.text,
                source_path=document.source_path,
                explicit_chip=forced or document.chip_hint,
                title=document.title,
            )
            records.append(
                IndexRecord(
                    chunk_id=_chunk_id(document, chunk.index, chunk.text, base_dir),
                    text=chunk.text,
                    title=document.title,
                    chip=chip,
                    section=chunk.section,
                    source_type=document.source_type,
                    url=document.url,
                    source_path=document.source_path,
                )
            )
    return records


def index_records(records: list[IndexRecord], settings: Optional[Settings] = None, batch_size: int = 64) -> int:
    """Embed and upsert. Re-running with the same sources updates in place."""
    if not records:
        return 0
    settings = settings or get_settings()
    collection = get_collection(settings)
    embedder = get_embedder()

    unique: dict[str, IndexRecord] = {}
    for record in records:
        unique[record.chunk_id] = record
    ordered = list(unique.values())

    for start in range(0, len(ordered), batch_size):
        batch = ordered[start:start + batch_size]
        # The title is embedded with the body so that "ESP32-C3 datasheet - ADC"
        # helps a chunk whose own text never repeats the chip name.
        vectors = embedder.embed_documents([f"{r.title} - {r.section}\n{r.text}" for r in batch])
        collection.upsert(
            ids=[r.chunk_id for r in batch],
            embeddings=vectors,
            documents=[r.text for r in batch],
            metadatas=[r.metadata() for r in batch],
        )
    return len(ordered)


def load_manifest(base: Path) -> dict[str, dict[str, str]]:
    """`manifest.json` (written by scripts/fetch_official_docs.py) records, for each
    downloaded file, the URL it came from, so citations can link to the original."""
    path = base / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    files = payload.get("files", payload) if isinstance(payload, dict) else {}
    return {str(k): v for k, v in files.items() if isinstance(v, dict)}


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


def apply_manifest(documents: list[ParsedDocument], entry: dict[str, str]) -> None:
    for document in documents:
        if entry.get("url") and not document.url:
            document.url = entry["url"]
        if entry.get("source_type"):
            document.source_type = entry["source_type"]
        if entry.get("title"):
            document.title = entry["title"]
        if entry.get("chip") and not document.chip_hint:
            document.chip_hint = entry["chip"]


def ingest_path(
    root: Path,
    settings: Optional[Settings] = None,
    forced_chip: Optional[str] = None,
    extensions: Optional[tuple[str, ...]] = None,
) -> tuple[int, Counter, list[str]]:
    """Parse, chunk, tag and index every supported file under `root`.

    Returns (chunks indexed, chunk count per chip, list of files that failed).
    """
    settings = settings or get_settings()
    documents: list[ParsedDocument] = []
    failures: list[str] = []
    base = root if root.is_dir() else root.parent
    manifest = load_manifest(base)
    for path in discover_files(root):
        if extensions and path.suffix.lower() not in extensions:
            continue
        target = normalize_chip(forced_chip) or chip_from_path(str(path))
        try:
            parsed = parse_file(path, target_chip=target)
        except Exception as exc:  # one bad file must not stop a long ingestion run
            failures.append(f"{path}: {exc}")
            continue
        apply_manifest(parsed, manifest.get(_relative(path, base), {}))
        documents.extend(parsed)
    records = build_records(documents, settings, forced_chip=forced_chip, base_dir=base)
    indexed = index_records(records, settings)
    return indexed, Counter(r.chip for r in records), failures


def kb_counts(settings: Optional[Settings] = None) -> dict[str, int]:
    """Number of indexed chunks per chip tag."""
    collection = get_collection(settings)
    counts = {tag: 0 for tag in ALL_TAGS}
    if collection.count() == 0:
        return counts
    metadatas = collection.get(include=["metadatas"])["metadatas"] or []
    for metadata in metadatas:
        chip = str((metadata or {}).get("chip", ""))
        counts[chip] = counts.get(chip, 0) + 1
    return counts
