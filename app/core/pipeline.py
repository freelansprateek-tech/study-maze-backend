"""The online CP-RAG query pipeline.

    1. chip-filtered retrieval           retriever.py
    2. attention-based classification    classifier.py
    3. dual-branch tailored processing   processors.py
    4. position-aware prompt assembly    assembler.py
    5. grounded generation               llm_client.py

`CPRagPipeline.run` returns the answer together with a full trace of every
stage, which is what the frontend's X-Ray inspector renders.
"""
from __future__ import annotations

import math
import random
import threading
import time
from typing import Optional

from app.chips import chip_label
from app.config import Settings, get_settings
from app.core import assembler
from app.core.classifier import AttentionDocumentClassifier, ClassifierDocument
from app.core.hf_runtime import resolve_device
from app.core.llm_client import LLMClient, get_llm_client
from app.core.noise import NoiseProvider
from app.core.processors import DirectAssistProcessor, RearrangingModule, render_direct_item
from app.core.retriever import ChipFilteredRetriever
from app.schemas import (
    AttentionScore,
    ChunkTrace,
    Citation,
    ClassifierInfo,
    DirectSummary,
    ModelInfo,
    PipelineOptions,
    PromptItem,
    QueryRequest,
    QueryResponse,
)


def make_rng(seed: int, question: str, chip: str) -> random.Random:
    """Deterministic per (seed, question, chip): a rerun reproduces the same noise and order."""
    return random.Random(f"{seed}|{chip}|{question.strip().lower()}")


def noise_slots(k: int, noise_ratio: float) -> tuple[int, int]:
    """Split k context slots into (retrieved, distractor) counts for a noise ratio.

    At least one retrieved chunk is always kept.
    """
    n_noise = min(int(round(k * noise_ratio)), k - 1)
    return k - n_noise, n_noise


def citation_label(chunk: ChunkTrace) -> str:
    return f"[{chunk.title} | {chip_label(chunk.chip) if not chunk.is_noise else 'Noise'} | {chunk.section or '-'}]"


class CPRagPipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[LLMClient] = None,
        retriever: Optional[ChipFilteredRetriever] = None,
        classifier: Optional[AttentionDocumentClassifier] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or get_llm_client()
        self.retriever = retriever or ChipFilteredRetriever(self.settings)
        self.classifier = classifier or AttentionDocumentClassifier(settings=self.settings)
        self.noise_provider = NoiseProvider(self.llm, self.settings.noise_source)
        self.direct_processor = DirectAssistProcessor(self.llm, self.settings.summary_max_tokens)
        self.rearranger = RearrangingModule(self.noise_provider)

    # -- stage 1 --------------------------------------------------------------

    def build_context(
        self,
        question: str,
        chip: str,
        k: int,
        noise_ratio: float,
        rng: random.Random,
        chip_filter: bool = True,
    ) -> list[ChunkTrace]:
        """Retrieve, then replace a `noise_ratio` share of the k slots with distractors.

        The distractors are inserted at random positions and are *not* marked
        for the later stages: the classifier has to recognise them on its own.
        The baselines in `eval/` are fed this same list.
        """
        n_real, n_noise = noise_slots(k, noise_ratio)
        retrieved = self.retriever.retrieve(question, chip, n_real, chip_filter=chip_filter)
        noise = self.noise_provider.sample(n_noise, rng, question, "environment")

        context: list[ChunkTrace] = list(retrieved)
        for doc in noise:
            context.insert(rng.randint(0, len(context)), doc)
        for number, chunk in enumerate(context, start=1):
            chunk.source_id = f"S{number}"
        return context

    # -- stage 2 --------------------------------------------------------------

    def classify(
        self, question: str, chip: str, context: list[ChunkTrace], use_attention: bool
    ) -> tuple[list[ChunkTrace], list[ChunkTrace], list[AttentionScore], ClassifierInfo]:
        by_id = {c.source_id: c for c in context}
        if not use_attention:
            # Ablation: no attention signal. The top half of the context order is
            # treated as direct-assist.
            cut = math.ceil(len(context) / 2)
            return list(context[:cut]), list(context[cut:]), [], ClassifierInfo(method="retrieval-rank")

        result = self.classifier.classify(
            question,
            chip_label(chip),
            [ClassifierDocument(c.source_id, c.title, c.text) for c in context],
        )
        scores: list[AttentionScore] = []
        for score in result.scores:
            chunk = by_id[score.source_id]
            chunk.attention_score = score.raw_score
            scores.append(
                AttentionScore(
                    source_id=score.source_id,
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    raw_score=score.raw_score,
                    normalized_score=score.normalized_score,
                    per_layer=score.per_layer,
                    token_span=score.token_span,
                    n_tokens=score.token_span[1] - score.token_span[0],
                    cluster=score.cluster,  # type: ignore[arg-type]
                    is_noise=chunk.is_noise,
                )
            )
        scores.sort(key=lambda s: s.raw_score, reverse=True)
        direct = [by_id[s.source_id] for s in scores if s.cluster == "direct"]  # strongest first
        direct_ids = {c.source_id for c in direct}
        indirect = [c for c in context if c.source_id not in direct_ids]  # context order
        info = ClassifierInfo(
            method="attention-kmeans",
            model=result.model,
            device=result.device,
            layers_used=result.layers_used,
            question_span=result.question_span,
            total_tokens=result.total_tokens,
            direct_centroid=result.direct_centroid,
            indirect_centroid=result.indirect_centroid,
        )
        return direct, indirect, scores, info

    # -- full run -------------------------------------------------------------

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            llm_provider=self.llm.provider,
            llm_model=self.llm.model,
            classifier_model=self.settings.classifier_model,
            embedding_model=self.settings.embedding_model,
            device=resolve_device(self.settings.device),
        )

    def run(self, request: QueryRequest) -> QueryResponse:
        settings = self.settings
        options: PipelineOptions = request.options
        chip = request.selected_chip
        question = request.question
        k = request.top_k or settings.top_k
        seed = settings.random_seed if request.seed is None else request.seed
        rng = make_rng(seed, question, chip)
        timings: dict[str, float] = {}
        started = time.perf_counter()

        def lap(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = round((now - since) * 1000, 1)
            return now

        # 1. Chip-filtered retrieval (+ robustness noise)
        t = time.perf_counter()
        context = self.build_context(question, chip, k, request.noise_ratio, rng, options.chip_filter)
        t = lap("retrieval", t)
        retrieved = [c for c in context if not c.is_noise]

        # 2. Attention-based document classification
        direct, indirect, attention_scores, classifier_info = self.classify(
            question, chip, context, options.attention_sort
        )
        t = lap("classification", t)

        # 3a. Direct-Assist: multilevel semantic optimization
        if options.summarization:
            summaries = self.direct_processor.process(question, chip_label(chip), direct)
        else:
            summaries = DirectAssistProcessor.passthrough(direct)
        summary_by_id: dict[str, DirectSummary] = {s.source_id: s for s in summaries}
        # A chunk whose summary found nothing relevant is demoted to the middle.
        demoted = [c for c in direct if not summary_by_id[c.source_id].relevant]
        for chunk in demoted:
            chunk.demoted = True
        direct = [c for c in direct if summary_by_id[c.source_id].relevant]
        indirect = indirect + demoted
        t = lap("direct_processing", t)

        # 3b. Indirect-Assist: rearranging module
        injected: list[ChunkTrace] = []
        if options.noise_shuffle:
            middle, injected = self.rearranger.process(
                indirect,
                question,
                settings.rearrange_noise_docs,
                rng,
                exclude_titles={c.title for c in context if c.is_noise},
            )
            for number, chunk in enumerate(injected, start=len(context) + 1):
                chunk.source_id = f"S{number}"
        else:
            middle = list(indirect)
        t = lap("indirect_processing", t)

        # 4. Position-aware prompt assembly
        chunks_by_source = {c.source_id: c for c in context + injected}
        direct_items = [
            PromptItem(source_id=c.source_id, kind="direct", title=c.title,
                       text=render_direct_item(summary_by_id[c.source_id], c.text))
            for c in direct
        ]
        middle_items = [
            PromptItem(source_id=c.source_id, kind="noise" if c.is_noise else "indirect", title=c.title, text=c.text)
            for c in middle
        ]
        messages, layout = assembler.assemble_prompt(
            chip, question, direct_items, middle_items, chunks_by_source,
            u_shape=options.u_shape_order,
            flat_order=[c.source_id for c in context + injected],
            detailed=settings.detailed_answers,
        )
        final_prompt = assembler.render_messages(messages)
        t = lap("assembly", t)

        # 5. Grounded generation. With no retrieved documentation there is
        #    nothing to ground an answer in, so the LLM is not consulted.
        valid_ids = set(chunks_by_source)
        grounding_note = ""
        if not retrieved:
            answer, explanation, reasoning, cited, status = assembler.not_covered_statement(chip), "", "", [], "not_covered"
        else:
            raw = self.llm.chat(messages, max_tokens=settings.answer_max_tokens)
            answer, explanation, reasoning, cited, status = assembler.parse_answer(raw, chip, valid_ids)
            # Every value in the answer must exist in a document that was in the
            # prompt. An answer that fails is withheld, not shown with a warning.
            if status != "not_covered":
                source_texts = [f"{c.title} {c.text}" for c in chunks_by_source.values()]
                source_texts += [question, chip_label(chip)]
                missing = assembler.unsupported_values(answer, source_texts)
                if missing:
                    grounding_note = (
                        "The generated answer was withheld because it contained values that appear in none of "
                        f"the retrieved documents: {', '.join(missing[:6])}. Generated text: {answer[:400]}"
                    )
                    answer, explanation, cited, status = assembler.not_covered_statement(chip), "", [], "withheld"
                else:
                    explanation = assembler.keep_supported_sentences(explanation, source_texts)
        lap("generation", t)
        timings["total"] = round((time.perf_counter() - started) * 1000, 1)

        direct_ids = {c.source_id for c in direct}
        citations = [
            Citation(
                source_id=c.source_id,
                label=citation_label(c),
                title=c.title,
                chip=c.chip,
                section=c.section,
                source_type=c.source_type,
                url=c.url,
                role="noise" if c.is_noise else ("direct" if c.source_id in direct_ids else "indirect"),
                cited=c.source_id in cited,
                excerpt=c.text,
            )
            for c in sorted(chunks_by_source.values(), key=lambda c: int(c.source_id[1:]))
        ]

        return QueryResponse(
            answer=answer,
            explanation=explanation,
            reasoning=reasoning,
            grounding_status=status,
            grounding_note=grounding_note,
            selected_chip=chip,
            question=question,
            noise_ratio=request.noise_ratio,
            top_k=k,
            options=options,
            retrieved_chunks=retrieved,
            direct_chunks=direct,
            indirect_chunks=[c for c in middle if not c.is_noise],
            noise_chunks=[c for c in context + injected if c.is_noise],
            attention_scores=attention_scores,
            classifier=classifier_info,
            summaries=[s for s in summaries],
            prompt_layout=layout,
            final_prompt=final_prompt,
            citations=citations,
            timings_ms=timings,
            models=self.model_info(),
        )


_pipeline: Optional[CPRagPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> CPRagPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = CPRagPipeline()
        return _pipeline
