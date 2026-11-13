"""Pydantic v2 schemas: the public API contract and the internal pipeline trace.

The frontend mirrors these types by hand in `src/api-types.ts`; keep the two in
step when a field changes.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ChipId = Literal["esp32", "esp32-s3", "esp32-c3", "esp32-c6", "esp32-s2"]
ChipTag = Literal["esp32", "esp32-s3", "esp32-c3", "esp32-c6", "esp32-s2", "common", "noise"]
AssistRole = Literal["direct", "indirect"]
NoiseOrigin = Literal["environment", "rearrange"]
GroundingStatus = Literal["grounded", "uncited", "not_covered", "withheld"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class PipelineOptions(StrictModel):
    """Component switches. All on = full CP-RAG. Each one off = one ablation."""

    chip_filter: bool = Field(True, description="Restrict retrieval to the selected chip + common.")
    attention_sort: bool = Field(True, description="Classify with attention + K-Means (else retrieval rank).")
    summarization: bool = Field(True, description="Multilevel semantic optimization of direct-assist chunks.")
    u_shape_order: bool = Field(True, description="Head/tail placement of direct-assist content.")
    noise_shuffle: bool = Field(True, description="Rearranging module: inject noise and shuffle indirect chunks.")


class QueryRequest(StrictModel):
    question: str = Field(..., min_length=3, max_length=1000)
    selected_chip: ChipId
    noise_ratio: float = Field(0.0, ge=0.0, le=0.8, description="Share of the k context slots filled with distractors.")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    seed: Optional[int] = Field(None, description="Seed for noise sampling and shuffling.")
    options: PipelineOptions = Field(default_factory=PipelineOptions)

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return value


# ---------------------------------------------------------------------------
# Pipeline trace
# ---------------------------------------------------------------------------


class ChunkTrace(StrictModel):
    """One context document as it moves through the pipeline."""

    source_id: str = Field(..., description="Tag used in the prompt and in citations, e.g. S3.")
    chunk_id: str
    title: str
    chip: ChipTag
    section: str = ""
    source_type: str = ""
    url: str = ""
    text: str
    retrieval_rank: Optional[int] = None
    similarity: Optional[float] = None
    is_noise: bool = False
    noise_origin: Optional[NoiseOrigin] = None
    attention_score: Optional[float] = None
    demoted: bool = Field(False, description="Clustered direct, but its summary found nothing relevant.")


class AttentionScore(StrictModel):
    source_id: str
    chunk_id: str
    title: str
    raw_score: float = Field(..., description="Length-normalised question->chunk attention, mean of the layers used.")
    normalized_score: float = Field(..., description="raw_score as a share of the total over all chunks.")
    per_layer: list[float] = Field(default_factory=list)
    token_span: tuple[int, int]
    n_tokens: int
    cluster: AssistRole
    is_noise: bool = False


class ClassifierInfo(StrictModel):
    method: Literal["attention-kmeans", "retrieval-rank"]
    model: str = ""
    device: str = ""
    layers_used: list[int] = Field(default_factory=list)
    question_span: Optional[tuple[int, int]] = None
    total_tokens: int = 0
    direct_centroid: Optional[float] = None
    indirect_centroid: Optional[float] = None


class Entity(StrictModel):
    text: str
    type: Literal["pin", "register", "electrical", "api", "config", "peripheral", "other"]


class Relation(StrictModel):
    subject: str
    predicate: str
    object: str


class DirectSummary(StrictModel):
    source_id: str
    chunk_id: str
    title: str
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    summary: str
    evidence: str = Field("", description="Sentences copied verbatim from the chunk, sent alongside an LLM summary.")
    method: Literal["llm", "extractive", "raw"]
    relevant: bool = True


class PromptItem(StrictModel):
    source_id: str
    kind: Literal["direct", "indirect", "noise"]
    title: str
    text: str


class PromptSegment(StrictModel):
    position: Literal["system", "head", "middle", "tail", "flat", "question"]
    label: str
    text: str = ""
    items: list[PromptItem] = Field(default_factory=list)
    approx_tokens: int = 0


class Citation(StrictModel):
    source_id: str
    label: str = Field(..., description="[Doc Title | Chip | Section]")
    title: str
    chip: ChipTag
    section: str = ""
    source_type: str = ""
    url: str = ""
    role: Literal["direct", "indirect", "noise"]
    cited: bool = False
    excerpt: str = ""


class ModelInfo(StrictModel):
    llm_provider: str
    llm_model: str
    classifier_model: str
    embedding_model: str
    device: str


class QueryResponse(StrictModel):
    answer: str
    explanation: str = Field("", description="Longer plain-language explanation of the answer.")
    reasoning: str = ""
    grounding_status: GroundingStatus
    grounding_note: str = Field("", description="Why an answer was withheld, when it was.")
    selected_chip: ChipId
    question: str
    noise_ratio: float
    top_k: int
    options: PipelineOptions

    retrieved_chunks: list[ChunkTrace]
    direct_chunks: list[ChunkTrace]
    indirect_chunks: list[ChunkTrace]
    noise_chunks: list[ChunkTrace]

    attention_scores: list[AttentionScore]
    classifier: ClassifierInfo
    summaries: list[DirectSummary]
    prompt_layout: list[PromptSegment]
    final_prompt: str
    citations: list[Citation]

    timings_ms: dict[str, float]
    models: ModelInfo


# ---------------------------------------------------------------------------
# Catalogue + health
# ---------------------------------------------------------------------------


class ChipInfo(StrictModel):
    id: ChipId
    label: str
    architecture: str
    cores: str
    max_frequency_mhz: int
    wifi: str
    bluetooth: str
    usb: str
    extra_radio: Optional[str] = None
    gpio_count: int
    sample_questions: list[str]
    kb_chunks: int = 0


class ChipsResponse(StrictModel):
    chips: list[ChipInfo]
    common_chunks: int = 0


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    kb_ready: bool
    kb_chunks: int
    chunks_by_chip: dict[str, int]
    llm_provider: str
    llm_model: str
    classifier_model: str
    classifier_loaded: bool
    embedding_model: str
    device: str
    version: str
    detail: str = ""
