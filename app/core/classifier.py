"""Attention-based document classifier (CP-RAG).

A small open LLM reads all context documents followed by the question in one
forward pass. For every document i with token span [s_i, e_i) and the question
span [q_s, q_e), the contribution score is

    score_i = mean over the last L layers of
              ( 1 / (e_i - s_i) ) * sum_{t in [s_i, e_i)}
                  mean over heads h, question tokens q in [q_s, q_e) of A[l, h, q, t]

i.e. the average attention a question token pays to a token of that document.
Dividing by the span length removes the advantage long documents would
otherwise have. K-Means with K = 2 then splits the scores: the cluster with
the higher centroid is Direct-Assist, the other Indirect-Assist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.core.hf_runtime import HFRuntime, get_runtime

CLASSIFIER_SYSTEM = (
    "You are a hardware documentation assistant. Read the documents and answer the question using them."
)


class ClassifierError(RuntimeError):
    pass


@dataclass
class ClassifierDocument:
    source_id: str
    title: str
    text: str


@dataclass
class TokenizedPrompt:
    input_ids: list[int]
    doc_spans: list[tuple[int, int]]
    question_span: tuple[int, int]


@dataclass
class DocumentScore:
    source_id: str
    raw_score: float
    normalized_score: float
    per_layer: list[float]
    token_span: tuple[int, int]
    cluster: str  # "direct" | "indirect"


@dataclass
class ClassificationResult:
    scores: list[DocumentScore]
    layers_used: list[int] = field(default_factory=list)
    question_span: Optional[tuple[int, int]] = None
    total_tokens: int = 0
    direct_centroid: Optional[float] = None
    indirect_centroid: Optional[float] = None
    model: str = ""
    device: str = ""


def kmeans_split(scores: Sequence[float]) -> tuple[list[str], Optional[float], Optional[float]]:
    """Cluster 1-D scores with KMeans(n_clusters=2, random_state=42).

    Returns (label per score, direct centroid, indirect centroid). The cluster
    with the higher centroid is "direct". With fewer than two distinct scores
    there is nothing to separate, so everything is treated as direct.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    if len(values) == 0:
        return [], None, None
    if len(values) < 2 or float(np.ptp(values)) < 1e-12:
        return ["direct"] * len(values), float(values.mean()), None

    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=2, random_state=42, n_init=10).fit(values)
    centroids = model.cluster_centers_.ravel()
    direct_cluster = int(np.argmax(centroids))
    labels = ["direct" if int(label) == direct_cluster else "indirect" for label in model.labels_]
    return labels, float(centroids[direct_cluster]), float(centroids[1 - direct_cluster])


class AttentionDocumentClassifier:
    def __init__(self, runtime: Optional[HFRuntime] = None, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._runtime = runtime

    @property
    def runtime(self) -> HFRuntime:
        if self._runtime is None:
            self._runtime = get_runtime(self.settings.classifier_model)
        return self._runtime

    # -- prompt construction with exact span tracking ------------------------

    def tokenize_with_spans(self, question: str, chip_label: str, documents: Sequence[ClassifierDocument]) -> TokenizedPrompt:
        """Tokenise piece by piece so every span boundary is known exactly.

        Each piece is encoded on its own and the id lists are concatenated; the
        model sees exactly these ids, so [s_i, e_i) and [q_s, q_e) index the
        attention matrices without any offset arithmetic.
        """
        tokenizer = self.runtime.tokenizer
        max_tokens = self.settings.classifier_max_chunk_tokens

        def encode(text: str) -> list[int]:
            return tokenizer(text, add_special_tokens=False)["input_ids"]

        ids: list[int] = encode(
            f"<|im_start|>system\n{CLASSIFIER_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\nDocuments about the {chip_label}:\n\n"
        )
        doc_spans: list[tuple[int, int]] = []
        for number, document in enumerate(documents, start=1):
            ids.extend(encode(f"[Document {number}]\n"))
            body = encode(f"{document.title}\n{document.text.strip()}")[:max_tokens]
            start = len(ids)
            ids.extend(body)
            doc_spans.append((start, len(ids)))
            ids.extend(encode("\n\n"))

        ids.extend(encode("Question: "))
        q_start = len(ids)
        ids.extend(encode(question.strip()))
        question_span = (q_start, len(ids))
        ids.extend(encode("<|im_end|>\n<|im_start|>assistant\n"))
        return TokenizedPrompt(ids, doc_spans, question_span)

    # -- forward pass ---------------------------------------------------------

    def question_to_token_attention(self, prompt: TokenizedPrompt) -> tuple[np.ndarray, list[int]]:
        """Return an array [L, T]: for each of the last L layers, the attention
        each prompt token receives, averaged over heads and question tokens."""
        import torch

        runtime = self.runtime
        model = runtime.model
        attention_modules = [m for name, m in model.named_modules() if name.endswith("self_attn")]
        n_layers = len(attention_modules) or int(getattr(model.config, "num_hidden_layers", 0))
        if n_layers == 0:
            raise ClassifierError("Could not locate the model's attention layers.")
        wanted = min(self.settings.classifier_layers, n_layers)
        target_layers = list(range(n_layers - wanted, n_layers))
        q_start, q_end = prompt.question_span
        if q_end <= q_start:
            raise ClassifierError("The question produced no tokens.")

        captured: dict[int, "torch.Tensor"] = {}

        def make_hook(layer_index: int):
            # A full [heads, T, T] tensor per layer is large. The hook reduces the
            # weights of the layers we need to a [T] vector and returns None in
            # their place, so no layer's attention tensor outlives its own step.
            def hook(_module, _inputs, output):
                if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                    return output
                weights = output[1]  # [batch, heads, T, T]
                if layer_index in target_layers:
                    captured[layer_index] = weights[0, :, q_start:q_end, :].float().mean(dim=(0, 1)).cpu()
                return (output[0], None) + tuple(output[2:])
            return hook

        handles = [module.register_forward_hook(make_hook(i)) for i, module in enumerate(attention_modules)]
        input_ids = torch.tensor([prompt.input_ids], dtype=torch.long, device=runtime.device)
        try:
            with runtime.lock, torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        if len(captured) < len(target_layers):
            # Hooks saw no weights (unusual module layout): fall back to the
            # attentions returned by the model itself.
            attentions = getattr(outputs, "attentions", None)
            if not attentions or attentions[-1] is None:
                raise ClassifierError(
                    "The model returned no attention weights. It must be loaded with attn_implementation='eager'."
                )
            for layer_index in target_layers:
                weights = attentions[layer_index]
                captured[layer_index] = weights[0, :, q_start:q_end, :].float().mean(dim=(0, 1)).cpu()

        matrix = torch.stack([captured[i] for i in target_layers]).numpy().astype(np.float64)
        # Release the forward pass's buffers. Without this the MPS allocator keeps
        # every query's activations cached and each classification gets slower.
        del outputs, input_ids, captured
        import gc

        gc.collect()
        if runtime.device == "mps":
            torch.mps.empty_cache()
        elif runtime.device == "cuda":
            torch.cuda.empty_cache()
        if not np.isfinite(matrix).all():
            raise ClassifierError(
                "Attention weights contain NaN/Inf. Set MODEL_DTYPE=bfloat16 (or float32) in backend/.env and restart."
            )
        return matrix, target_layers

    # -- public API -----------------------------------------------------------

    def classify(self, question: str, chip_label: str, documents: Sequence[ClassifierDocument]) -> ClassificationResult:
        runtime = self.runtime
        if not documents:
            return ClassificationResult(scores=[], model=runtime.model_name, device=runtime.device)

        prompt = self.tokenize_with_spans(question, chip_label, documents)
        matrix, layers = self.question_to_token_attention(prompt)

        per_layer_scores: list[list[float]] = []
        raw_scores: list[float] = []
        for start, end in prompt.doc_spans:
            length = max(end - start, 1)
            layer_values = (matrix[:, start:end].sum(axis=1) / length).tolist()  # length-normalised
            per_layer_scores.append(layer_values)
            raw_scores.append(float(np.mean(layer_values)))

        labels, direct_centroid, indirect_centroid = kmeans_split(raw_scores)
        total = sum(raw_scores) or 1.0
        scores = [
            DocumentScore(
                source_id=document.source_id,
                raw_score=raw_scores[i],
                normalized_score=raw_scores[i] / total,
                per_layer=per_layer_scores[i],
                token_span=prompt.doc_spans[i],
                cluster=labels[i],
            )
            for i, document in enumerate(documents)
        ]
        return ClassificationResult(
            scores=scores,
            layers_used=layers,
            question_span=prompt.question_span,
            total_tokens=len(prompt.input_ids),
            direct_centroid=direct_centroid,
            indirect_centroid=indirect_centroid,
            model=runtime.model_name,
            device=runtime.device,
        )
