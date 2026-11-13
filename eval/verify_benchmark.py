#!/usr/bin/env python
"""Check every benchmark answer against the indexed documentation.

For each question, retrieve the top chunks for its chip and look for the gold
answer in them. This answers two things at once:

  * is the gold answer actually stated in the official documents that were
    ingested (a fact check of the benchmark), and
  * does retrieval surface the chunk that states it (hit@8, hit@20)?

Items whose answer is not found in the top 20 chunks are listed for manual
review: either the gold answer is wrong, the document words it differently, or
the source has not been ingested.

    python eval/verify_benchmark.py
    python eval/verify_benchmark.py --official-only     # ignore GitHub issues / forum chunks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.retriever import ChipFilteredRetriever  # noqa: E402
from eval.metrics import exact_match  # noqa: E402

OFFICIAL_TYPES = {"datasheet", "trm", "esp-idf-docs"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=Path(__file__).resolve().parent / "esp32_qa_benchmark.json")
    parser.add_argument("--official-only", action="store_true", help="Count only datasheet / TRM / ESP-IDF chunks as support.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "benchmark_verification.json")
    args = parser.parse_args()

    items = json.loads(args.benchmark.read_text(encoding="utf-8"))["items"]
    retriever = ChipFilteredRetriever()
    report = []
    hit8 = hit20 = 0
    for item in items:
        chunks = retriever.retrieve(item["question"], item["chip"], 20)
        support = None
        for chunk in chunks:
            if args.official_only and chunk.source_type not in OFFICIAL_TYPES:
                continue
            if exact_match(chunk.text, item["answers"], item.get("all_of")):
                support = chunk
                break
        rank = support.retrieval_rank if support else None
        hit8 += bool(rank and rank <= 8)
        hit20 += bool(rank)
        report.append({
            "id": item["id"], "chip": item["chip"], "question": item["question"], "gold": item["answers"],
            "supported": support is not None, "rank": rank,
            "source": f"{support.title} | {support.section}" if support else None,
            "source_type": support.source_type if support else None,
            "url": support.url if support else None,
        })
        mark = f"rank {rank:>2}" if rank else "NOT FOUND"
        print(f"  {item['id']:<12} {mark:<10} {item['answers'][0]:<28} {(support.title + ' | ' + support.section)[:70] if support else ''}")

    n = len(items)
    print(f"\nGold answer found in top 8: {hit8}/{n}   in top 20: {hit20}/{n}")
    missing = [r["id"] for r in report if not r["supported"]]
    if missing:
        print("Review by hand (answer not found in the top 20 chunks): " + ", ".join(missing))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"hit_at_8": hit8, "hit_at_20": hit20, "n": n, "items": report}, indent=2, ensure_ascii=False))
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
