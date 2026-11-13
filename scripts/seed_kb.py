#!/usr/bin/env python
"""Seed the knowledge base so the backend works immediately.

Loads `data/seed/seed_corpus.json` (ESP32-family documentation chunks plus
noisy community threads, tagged for all five chips and `common`) and runs it
through the real ingestion pipeline: JSON parser -> chunker -> chip tagger ->
BGE embeddings -> ChromaDB.

    python scripts/seed_kb.py            # rebuild the collection from the seed corpus
    python scripts/seed_kb.py --keep     # add/update seed chunks, keep everything else
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.chips import ALL_TAGS  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ingest.indexer import build_records, index_records, kb_counts, reset_collection  # noqa: E402
from app.ingest.parsers import parse_json  # noqa: E402

SEED_FILE = BACKEND_ROOT / "data" / "seed" / "seed_corpus.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-file", type=Path, default=SEED_FILE, help="Seed corpus JSON file.")
    parser.add_argument("--keep", action="store_true", help="Do not wipe the existing collection first.")
    args = parser.parse_args()

    settings = get_settings()
    if not args.seed_file.exists():
        print(f"Seed file not found: {args.seed_file}", file=sys.stderr)
        return 1

    documents = parse_json(args.seed_file)
    records = build_records(documents, settings, base_dir=args.seed_file.parent)
    if not records:
        print("The seed file produced no chunks.", file=sys.stderr)
        return 1

    if not args.keep:
        reset_collection(settings)
        print(f"Reset collection '{settings.collection_name}'.")

    print(f"Embedding {len(records)} chunks with {settings.embedding_model} ...")
    indexed = index_records(records, settings)

    by_chip = Counter(r.chip for r in records)
    by_type = Counter(r.source_type for r in records)
    print(f"\nIndexed {indexed} chunks into {settings.chroma_dir}")
    print("  by chip:   " + ", ".join(f"{tag}={by_chip.get(tag, 0)}" for tag in ALL_TAGS))
    print("  by source: " + ", ".join(f"{name}={count}" for name, count in sorted(by_type.items())))
    totals = kb_counts(settings)
    print(f"  collection now holds {sum(totals.values())} chunks in total")

    missing = [tag for tag in ALL_TAGS if totals.get(tag, 0) == 0]
    if missing:
        print(f"WARNING: no chunks for: {', '.join(missing)}", file=sys.stderr)
        return 1
    print("\nKnowledge base ready. Start the API with: uvicorn app.main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
