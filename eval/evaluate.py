#!/usr/bin/env python
"""Benchmark Chip-Aware CP-RAG against CoT-Only and Basic RAG.

For every noise ratio, each system answers every benchmark question; Exact
Match (EM) and F1 are averaged per system.

    python eval/evaluate.py                                   # full run, ratios 0 0.2 0.4 0.6 0.8
    python eval/evaluate.py --limit 10 --noise-ratios 0 0.4   # quick check
    python eval/evaluate.py --systems cp-rag --disable-summarization     # one ablation
    python eval/evaluate.py --ablation-sweep                  # full CP-RAG + every single-component ablation

Systems
    cot-only    no retrieval; the question plus "Let us think step by step."
    basic-rag   top-k retrieval (no chip filter unless --baseline-chip-filter),
                documents concatenated in retrieval order, no step-by-step trigger
    cp-rag      the full pipeline, minus any component switched off with a --disable-* flag

Noise ratio r: round(k*r) of the k context slots hold distractor documents
instead of retrieved chunks. Basic RAG and CP-RAG receive the same distractors.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.chips import CHIP_IDS, chip_label  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.core import assembler  # noqa: E402
from app.core.llm_client import LLMError  # noqa: E402
from app.core.pipeline import CPRagPipeline, make_rng  # noqa: E402
from app.schemas import PipelineOptions, QueryRequest  # noqa: E402
from eval.metrics import exact_match, f1_score  # noqa: E402

DEFAULT_BENCHMARK = Path(__file__).resolve().parent / "esp32_qa_benchmark.json"
DEFAULT_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8]
SYSTEMS = ("cot-only", "basic-rag", "cp-rag")
ABLATIONS = {
    "chip_filter": "w/o chip filter",
    "attention_sort": "w/o attention classification",
    "summarization": "w/o summarization",
    "u_shape_order": "w/o head/tail ordering",
    "noise_shuffle": "w/o noise + shuffle",
}
FINAL_HINT = f'End with one line that starts with "{assembler.FINAL_ANSWER_MARKER}" followed by the short answer.'


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------


def run_cot_only(pipeline: CPRagPipeline, item: dict[str, Any]) -> str:
    label = chip_label(item["chip"])
    messages = [
        {"role": "system", "content": f"You are a technical support assistant for the Espressif {label} microcontroller."},
        {"role": "user", "content": f"Question about the {label}: {item['question']}\n{assembler.COT_TRIGGER}\n{FINAL_HINT}"},
    ]
    return pipeline.llm.chat(messages, max_tokens=pipeline.settings.answer_max_tokens)


def run_basic_rag(pipeline: CPRagPipeline, item: dict[str, Any], ratio: float, k: int, seed: int, chip_filter: bool) -> str:
    label = chip_label(item["chip"])
    question = item["question"]
    rng = make_rng(seed, question, item["chip"])
    # Without the chip filter the chip name in the query is the only chip signal.
    context = pipeline.build_context(f"{label}: {question}" if not chip_filter else question,
                                     item["chip"], k, ratio, rng, chip_filter=chip_filter)
    documents = "\n\n".join(f"[{c.source_id}] {c.title}\n{c.text.strip()}" for c in context)
    messages = [
        {"role": "system", "content": f"You are a technical support assistant for the Espressif {label} microcontroller. "
                                      "Answer the question using the documents provided."},
        {"role": "user", "content": f"Documents:\n\n{documents}\n\nQuestion about the {label}: {question}\n{FINAL_HINT}"},
    ]
    return pipeline.llm.chat(messages, max_tokens=pipeline.settings.answer_max_tokens)


def run_cp_rag(pipeline: CPRagPipeline, item: dict[str, Any], ratio: float, k: int, seed: int, options: PipelineOptions) -> tuple[str, dict[str, Any]]:
    response = pipeline.run(QueryRequest(
        question=item["question"], selected_chip=item["chip"], noise_ratio=ratio, top_k=k, seed=seed, options=options,
    ))
    gold_chunk = item.get("source_chunk_id")
    trace = {
        "grounding_status": response.grounding_status,
        "n_direct": len(response.direct_chunks),
        "n_indirect": len(response.indirect_chunks),
        "gold_retrieved": any(c.chunk_id == gold_chunk for c in response.retrieved_chunks) if gold_chunk else None,
        "gold_direct": any(c.chunk_id == gold_chunk for c in response.direct_chunks) if gold_chunk else None,
        "noise_in_direct": sum(1 for c in response.direct_chunks if c.is_noise),
        "latency_ms": response.timings_ms.get("total"),
    }
    return response.answer, trace


def final_answer(raw: str) -> str:
    """The text after the last 'Final answer:' marker, or the whole output if there is none."""
    answer, _explanation, _reasoning, _cited, _status = assembler.parse_answer(raw, "esp32", set())
    return raw.strip() if answer == assembler.not_covered_statement("esp32") and "cover" not in raw.lower() else answer


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def load_benchmark(path: Path, chips: list[str], limit: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"] if isinstance(payload, dict) else payload
    for item in items:
        missing = [key for key in ("id", "chip", "question", "answers") if key not in item]
        if missing:
            raise ValueError(f"Benchmark item {item.get('id', '?')} is missing: {missing}")
        if item["chip"] not in CHIP_IDS:
            raise ValueError(f"Benchmark item {item['id']} has unknown chip '{item['chip']}'")
    items = [item for item in items if item["chip"] in chips]
    if limit:
        # Take the same number from each chip so a small run still covers all of them.
        per_chip: dict[str, list] = defaultdict(list)
        for item in items:
            per_chip[item["chip"]].append(item)
        picked: list[dict[str, Any]] = []
        index = 0
        while len(picked) < limit and any(index < len(v) for v in per_chip.values()):
            for chip in chips:
                if index < len(per_chip[chip]) and len(picked) < limit:
                    picked.append(per_chip[chip][index])
            index += 1
        items = picked
    return items


def evaluate_variant(
    name: str, system: str, options: PipelineOptions, pipeline: CPRagPipeline, items: list[dict[str, Any]],
    ratios: list[float], k: int, seed: int, baseline_chip_filter: bool, records: list[dict[str, Any]],
) -> dict[float, dict[str, float]]:
    summary: dict[float, dict[str, float]] = {}
    cot_cache: dict[str, str] = {}  # CoT-Only sees no documents, so its output does not depend on the ratio
    for ratio in ratios:
        em_total = f1_total = 0.0
        errors = 0
        started = time.perf_counter()
        for number, item in enumerate(items, start=1):
            trace: dict[str, Any] = {}
            try:
                if system == "cot-only":
                    if item["id"] not in cot_cache:
                        cot_cache[item["id"]] = run_cot_only(pipeline, item)
                    raw = cot_cache[item["id"]]
                    prediction = final_answer(raw)
                elif system == "basic-rag":
                    raw = run_basic_rag(pipeline, item, ratio, k, seed, baseline_chip_filter)
                    prediction = final_answer(raw)
                else:
                    prediction, trace = run_cp_rag(pipeline, item, ratio, k, seed, options)
                    raw = prediction
            except LLMError as exc:
                errors += 1
                raw = prediction = ""
                trace = {"error": str(exc)}
            em = exact_match(prediction, item["answers"], item.get("all_of"))
            f1 = f1_score(prediction, item["answers"])
            em_total += em
            f1_total += f1
            records.append({
                "variant": name, "system": system, "noise_ratio": ratio, "id": item["id"], "chip": item["chip"],
                "category": item.get("category", ""), "question": item["question"], "gold": item["answers"],
                "prediction": prediction, "raw_output": raw, "em": em, "f1": round(f1, 4), **trace,
            })
            print(f"\r  {name:<32} noise={ratio:.1f}  {number}/{len(items)}  EM={em_total / number:.3f}", end="", flush=True)
        n = max(len(items), 1)
        summary[ratio] = {
            "em": round(em_total / n, 4), "f1": round(f1_total / n, 4), "n": len(items), "errors": errors,
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(f"\r  {name:<32} noise={ratio:.1f}  EM={summary[ratio]['em']:.3f}  F1={summary[ratio]['f1']:.3f}"
              f"  ({summary[ratio]['seconds']}s{', %d errors' % errors if errors else ''})        ")
    return summary


def markdown_table(results: dict[str, dict[float, dict[str, float]]], ratios: list[float], metric: str) -> str:
    header = "| System | " + " | ".join(f"noise {r:.1f}" for r in ratios) + " | mean |"
    rule = "|---|" + "---:|" * (len(ratios) + 1)
    lines = [header, rule]
    for name, per_ratio in results.items():
        values = [per_ratio[r][metric] for r in ratios]
        cells = " | ".join(f"{v * 100:.1f}" for v in values)
        lines.append(f"| {name} | {cells} | {sum(values) / len(values) * 100:.1f} |")
    return "\n".join(lines)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--noise-ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--chips", nargs="+", choices=CHIP_IDS, default=list(CHIP_IDS))
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only this many questions (spread over the chips).")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--baseline-chip-filter", action="store_true",
                        help="Give Basic RAG the chip metadata filter too (default: unfiltered retrieval).")
    ablation = parser.add_argument_group("CP-RAG ablations (each flag removes one component)")
    ablation.add_argument("--disable-chip-filter", action="store_true")
    ablation.add_argument("--disable-attention-sort", action="store_true")
    ablation.add_argument("--disable-summarization", action="store_true")
    ablation.add_argument("--disable-u-shape-order", action="store_true")
    ablation.add_argument("--disable-noise-shuffle", action="store_true")
    ablation.add_argument("--ablation-sweep", action="store_true",
                          help="Run full CP-RAG and then each single-component ablation.")
    args = parser.parse_args()

    for ratio in args.noise_ratios:
        if not 0.0 <= ratio <= 0.8:
            parser.error("noise ratios must be between 0.0 and 0.8")

    items = load_benchmark(args.benchmark, args.chips, args.limit)
    if not items:
        print("No benchmark items selected.", file=sys.stderr)
        return 1

    options = PipelineOptions(
        chip_filter=not args.disable_chip_filter,
        attention_sort=not args.disable_attention_sort,
        summarization=not args.disable_summarization,
        u_shape_order=not args.disable_u_shape_order,
        noise_shuffle=not args.disable_noise_shuffle,
    )
    disabled = [label for field, label in ABLATIONS.items() if not getattr(options, field)]

    variants: list[tuple[str, str, PipelineOptions]] = []
    for system in args.systems:
        if system != "cp-rag":
            variants.append(({"cot-only": "CoT-Only", "basic-rag": "Basic RAG"}[system], system, PipelineOptions()))
        elif args.ablation_sweep:
            variants.append(("Chip-Aware CP-RAG (full)", system, PipelineOptions()))
            for field, label in ABLATIONS.items():
                variants.append((f"CP-RAG {label}", system, PipelineOptions(**{field: False})))
        else:
            name = "Chip-Aware CP-RAG" + (f" ({', '.join(disabled)})" if disabled else " (full)")
            variants.append((name, system, options))

    # The benchmark scores the short exact answer, not the long student-friendly one.
    settings = settings.model_copy(update={"detailed_answers": False})
    pipeline = CPRagPipeline(settings)
    print(f"Benchmark: {args.benchmark.name}  |  {len(items)} questions  |  k={args.top_k}  |  seed={args.seed}")
    print(f"Generator: {pipeline.llm.provider} / {pipeline.llm.model}   Classifier: {settings.classifier_model}\n")

    records: list[dict[str, Any]] = []
    results: dict[str, dict[float, dict[str, float]]] = {}
    for name, system, variant_options in variants:
        results[name] = evaluate_variant(
            name, system, variant_options, pipeline, items, args.noise_ratios, args.top_k, args.seed,
            args.baseline_chip_filter, records,
        )

    em_table = markdown_table(results, args.noise_ratios, "em")
    f1_table = markdown_table(results, args.noise_ratios, "f1")
    print("\nExact Match (%)\n" + em_table)
    print("\nF1 (%)\n" + f1_table)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "benchmark": str(args.benchmark), "n_questions": len(items), "top_k": args.top_k, "seed": args.seed,
        "noise_ratios": args.noise_ratios, "llm_provider": pipeline.llm.provider, "llm_model": pipeline.llm.model,
        "classifier_model": settings.classifier_model, "embedding_model": settings.embedding_model,
        "baseline_chip_filter": args.baseline_chip_filter,
        "summary": {name: {str(r): v for r, v in per_ratio.items()} for name, per_ratio in results.items()},
        "records": records,
    }
    json_path = args.out / f"eval-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / f"eval-{stamp}.md").write_text(
        f"# Evaluation {stamp}\n\n{len(items)} questions, k={args.top_k}, generator `{pipeline.llm.model}`.\n\n"
        f"## Exact Match (%)\n\n{em_table}\n\n## F1 (%)\n\n{f1_table}\n", encoding="utf-8")
    with (args.out / f"eval-{stamp}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "noise_ratio", "em", "f1", "n", "errors", "seconds"])
        for name, per_ratio in results.items():
            for ratio, values in per_ratio.items():
                writer.writerow([name, ratio, values["em"], values["f1"], values["n"], values["errors"], values["seconds"]])
    print(f"\nSaved: {json_path}  (+ .md and .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
