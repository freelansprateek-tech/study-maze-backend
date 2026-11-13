#!/usr/bin/env python
"""Ingest real documentation into the knowledge base.

Supported files: .pdf (datasheets, TRMs), .rst / .md (ESP-IDF docs),
.json / .txt (GitHub issues, forum threads).

The chip tag of every chunk comes from, in order: `--chip`, a `chip` field in
a JSON record, the folder or file name, and finally the text itself. The
simplest layout is therefore one folder per chip:

    data/raw/esp32/esp32_datasheet_en.pdf
    data/raw/esp32-s3/esp32-s3_technical_reference_manual_en.pdf
    data/raw/esp32-c6/api-reference/peripherals/gpio.rst
    data/raw/common/forum_threads.json

    python scripts/ingest_docs.py                       # ingest everything under data/raw
    python scripts/ingest_docs.py path/to/file.pdf --chip esp32-c3
    python scripts/ingest_docs.py --dry-run             # parse + tag only, write nothing
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
from app.ingest.indexer import build_records, ingest_path, kb_counts, reset_collection  # noqa: E402
from app.ingest.parsers import discover_files, parse_file  # noqa: E402
from app.ingest.tagger import chip_from_path, normalize_chip  # noqa: E402


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", type=Path, default=settings.data_dir / "raw",
                        help="File or directory to ingest (default: data/raw).")
    parser.add_argument("--chip", choices=ALL_TAGS, help="Force this chip tag for every chunk.")
    parser.add_argument("--reset", action="store_true", help="Wipe the collection before ingesting.")
    parser.add_argument("--ext", nargs="+", metavar="EXT", help="Only ingest files with these extensions, e.g. --ext .pdf")
    parser.add_argument("--dry-run", action="store_true", help="Parse, chunk and tag, but do not index.")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Path not found: {args.path}", file=sys.stderr)
        return 1
    extensions = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext) if args.ext else None
    files = [f for f in discover_files(args.path) if not extensions or f.suffix.lower() in extensions]
    if not files:
        print(f"No supported files (.pdf .rst .md .json .txt) under {args.path}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} file(s) under {args.path}")

    if args.dry_run:
        total: Counter = Counter()
        for path in files:
            target = normalize_chip(args.chip) or chip_from_path(str(path))
            try:
                records = build_records(parse_file(path, target_chip=target), settings, forced_chip=args.chip)
            except Exception as exc:
                print(f"  FAILED {path}: {exc}")
                continue
            tags = Counter(r.chip for r in records)
            total.update(tags)
            print(f"  {path.name}: {len(records)} chunks  {dict(tags)}")
        print(f"\nDry run: {sum(total.values())} chunks  {dict(total)}  (nothing written)")
        return 0

    if args.reset:
        reset_collection(settings)
        print(f"Reset collection '{settings.collection_name}'.")

    indexed, by_chip, failures = ingest_path(args.path, settings, forced_chip=args.chip, extensions=extensions)
    print(f"\nIndexed {indexed} chunks: " + ", ".join(f"{tag}={by_chip.get(tag, 0)}" for tag in ALL_TAGS))
    for failure in failures:
        print(f"  FAILED {failure}", file=sys.stderr)
    totals = kb_counts(settings)
    print(f"Collection now holds {sum(totals.values())} chunks: {totals}")
    return 1 if failures and indexed == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
