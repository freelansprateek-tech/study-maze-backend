"""Dual-branch tailored processing.

Direct-Assist  -> Multilevel Semantic Optimization
    level 1  hardware entities (pins, registers, electrical limits, APIs)
    level 2  relations between those entities
    level 3  a concise, question-focused summary built on levels 1 and 2

Indirect-Assist -> Rearranging Module
    add noise documents, then shuffle, so that no indirect document can gain
    influence from a privileged position in the prompt.
"""
from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from app.core.llm_client import LLMClient, LLMError
from app.core.noise import NoiseProvider
from app.schemas import ChunkTrace, DirectSummary, Entity, Relation

# ---------------------------------------------------------------------------
# Level 1: hardware entity extraction (deterministic patterns)
# ---------------------------------------------------------------------------

_UNIT = r"(?:µA|μA|uA|mA|nA|A|mV|V|MHz|kHz|GHz|Hz|KB|kB|MB|GB|Mbps|kbps|bps|dBm|dB|ms|µs|μs|us|kΩ|Ω|ohm|°C|bits?|-bit|baud)"
_ENTITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("pin", re.compile(r"\b(?:GPIO|IO)\s?\d{1,2}(?:\s?(?:-|–|to)\s?(?:GPIO|IO)?\s?\d{1,2})?\b")),
    ("pin", re.compile(r"\b(?:MTDI|MTDO|MTMS|MTCK|U\d(?:TXD|RXD)|CHIP_PU|EN|VDD_SPI|VDD_SDIO|VDD3P3(?:_RTC|_CPU)?)\b")),
    ("register", re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:_[A-Z0-9]+)+_REG\b")),
    ("config", re.compile(r"\bCONFIG_[A-Z0-9_]+\b")),
    ("api", re.compile(r"\b(?:esp|gpio|adc|dac|ledc|i2c|spi|uart|nvs|rmt|twai|i2s|touch|rtc|wifi|vTask|xTask|pdMS)[A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?:\(\))?")),
    ("api", re.compile(r"\bidf\.py(?:\s+-p\s+\S+)?(?:\s+[a-z][a-z\-]+){1,2}")),
    ("api", re.compile(r"\b(?:ADC|WIFI|GPIO|LEDC|ESP|I2C|SPI|UART)_[A-Z0-9]+(?:_[A-Z0-9]+)+\b")),
    ("electrical", re.compile(rf"(?<![\w.])\d+(?:\.\d+)?\s?(?:(?:-|–|to)\s?\d+(?:\.\d+)?\s?)?{_UNIT}(?![A-Za-z])")),
    ("electrical", re.compile(r"\b0x[0-9A-Fa-f]{1,8}\b")),
    ("peripheral", re.compile(r"\b(?:ADC[12]?|DAC[12]?|UART\d?|I2C\d?|I2S\d?|SPI\d?|TWAI|RMT|LEDC|PWM|USB(?: OTG| Serial/JTAG)?|JTAG|SAR ADC|PSRAM|ULP|Wi-Fi 6|Wi-Fi|Bluetooth(?: LE)?|Zigbee|Thread|IEEE 802\.15\.4|802\.11(?:ax|b/g/n|[abgn]+)?)\b")),
]


def extract_hardware_entities(text: str, limit: int = 24) -> list[Entity]:
    """Pattern-based extraction of the entity classes that matter in hardware docs."""
    found: list[tuple[int, Entity]] = []
    seen: set[str] = set()
    covered: list[tuple[int, int]] = []
    for entity_type, pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            value = " ".join(match.group(0).split())
            key = value.lower()
            span = match.span()
            # A value already captured inside a longer entity is not repeated
            # (e.g. "ADC" inside "ADC_ATTEN_DB_12").
            if key in seen or any(s <= span[0] and span[1] <= e for s, e in covered):
                continue
            seen.add(key)
            covered.append(span)
            found.append((span[0], Entity(text=value, type=entity_type)))  # type: ignore[arg-type]
    found.sort(key=lambda item: item[0])
    return [entity for _, entity in found[:limit]]


# ---------------------------------------------------------------------------
# Extractive fallback summary
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were be been of on in to for with and or by at from as it its this that these those "
    "what which how many much does do can i my you your when where why will would should there than then if "
    "have has had not no yes use used using chip esp32 pin pins".split()
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`(\[])")
_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9_\-/.]*", text.lower()) if t not in _STOPWORDS and len(t) > 1}


def extractive_summary(question: str, text: str, max_sentences: int = 3) -> str:
    """Pick the sentences that share the most terms with the question, in original order."""
    sentences = [s.strip() for s in _SENTENCES.split(" ".join(text.split())) if s.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    q_terms = _terms(question)
    scored = []
    for index, sentence in enumerate(sentences):
        overlap = len(q_terms & _terms(sentence))
        has_fact = 0.5 if _NUMBER.search(sentence) else 0.0
        scored.append((overlap + has_fact, -index, index))
    keep = sorted(index for _, _, index in sorted(scored, reverse=True)[:max_sentences])
    return " ".join(sentences[i] for i in keep)


def numbers_are_grounded(summary: str, source: str, question: str) -> bool:
    """Every number in the summary must already exist in the source chunk or the question.

    A summary is the only place the pipeline rewrites documentation, so this is
    where an invented value could enter. A summary that fails falls back to
    sentences copied verbatim from the chunk.
    """
    allowed = set(_NUMBER.findall(source)) | set(_NUMBER.findall(question))
    return all(number in allowed for number in _NUMBER.findall(summary))


_NEGATION = re.compile(
    r"\b(?:no|not|never|none|cannot|without|neither|nor|only|unsupported|unavailable|unable|lacks?|"
    r"except|instead)\b|n't\b",
    re.I,
)


def has_negation(text: str) -> bool:
    return _NEGATION.search(text) is not None


def polarity_preserved(summary: str, evidence: str) -> bool:
    """Reject a summary that drops a negation or restriction present in its evidence.

    "The 5 GHz band is not supported" summarised as "supports 5 GHz" is the
    most damaging error a hardware summary can make, and a small model makes
    it. When the sentences a summary rests on contain a negation or an "only"
    and the summary contains none, the summary is not trusted.
    """
    return has_negation(summary) or not has_negation(evidence)


def vocabulary_supported(summary: str, source: str, question: str, threshold: float = 0.8) -> bool:
    """A summary may only recombine words its source (or the question) already uses.

    A model asked to summarise a chunk "for the question" will sometimes answer
    the question from a chunk that does not contain the answer, e.g. attribute
    "selects the flash voltage" to a pin table that never mentions voltage.
    Such a summary introduces content words that are absent from the chunk.
    """
    summary_terms = _terms(summary)
    if not summary_terms:
        return False
    allowed = _terms(source) | _terms(question)
    stems = {t.rstrip("s") for t in allowed}
    supported = sum(1 for t in summary_terms if t in allowed or t.rstrip("s") in stems)
    return supported / len(summary_terms) >= threshold


def validate_relations(relations: Sequence[Relation], text: str) -> list[Relation]:
    """Keep a relation only if a single sentence of the chunk mentions both of its
    ends, and the relation does not turn a negated statement into a positive one."""
    sentences = [s.lower() for s in _SENTENCES.split(" ".join(text.split()))]
    kept: list[Relation] = []
    for relation in relations:
        subject, obj = relation.subject.lower(), relation.object.lower()
        support = [s for s in sentences if subject in s and obj in s]
        if not support:
            continue
        triple = f"{relation.subject} {relation.predicate} {relation.object}"
        if not has_negation(triple) and all(has_negation(s) for s in support):
            continue
        kept.append(relation)
    return kept


# ---------------------------------------------------------------------------
# Direct-Assist branch
# ---------------------------------------------------------------------------

_MSO_SYSTEM = (
    "You are a hardware documentation analyst. You work only from the document you are given and never add "
    "facts from memory. You copy numbers, pin names, register names and API names exactly as written."
)

_MSO_PROMPT = """Chip: {chip}
Question: {question}

Document "{title}":
\"\"\"
{text}
\"\"\"

Hardware terms already detected in the document: {hints}

Tables in the document appear as rows of cells separated by " | ", with the header row first; read each value
under its own column header.

Do three things, using ONLY the document above:
1. ENTITIES - list the hardware entities that matter for the question (pins, registers, electrical limits, APIs, peripherals).
2. RELATIONS - state how those entities relate, as triples.
3. SUMMARY - write 1 to 3 sentences that restate what the document says about the subject of the question, with
exact values. Report only what the document states. Do not answer the question yourself, and do not connect a
pin, value or function to the question unless the document makes that connection in its own words.
If the document contains nothing useful for the question, write IRRELEVANT as the summary.

Answer in exactly this format and nothing else:
ENTITIES: <entity>; <entity>; <entity>
RELATIONS: <entity> | <verb phrase> | <entity or value>; <entity> | <verb phrase> | <entity or value>
SUMMARY: <your summary>

Example of the format (about a different document):
ENTITIES: GPIO12; VDD_SDIO; 1.8 V
RELATIONS: GPIO12 | selects the voltage of | VDD_SDIO; VDD_SDIO | is set to | 1.8 V when GPIO12 is high
SUMMARY: GPIO12 selects the VDD_SDIO flash voltage; pulling it high sets 1.8 V."""

_FIELD = re.compile(r"^\s*\**\s*(ENTITIES|RELATIONS|SUMMARY)\s*\**\s*[:\-]\s*", re.I | re.M)
_ENTITY_TYPES = ("pin", "register", "electrical", "api", "config", "peripheral")


def parse_mso_output(raw: str) -> tuple[list[str], list[Relation], str]:
    """Parse the ENTITIES / RELATIONS / SUMMARY block. Tolerant of small format drift."""
    fields: dict[str, str] = {}
    matches = list(_FIELD.finditer(raw))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        fields.setdefault(match.group(1).upper(), raw[match.end():end].strip())

    entities = [
        re.sub(r"\s*\((?:%s|other|value)\)\s*$" % "|".join(_ENTITY_TYPES), "", item.strip(" -*\n"), flags=re.I)
        for item in re.split(r"[;\n]", fields.get("ENTITIES", ""))
    ]
    entities = [e for e in entities if 1 < len(e) <= 80 and e.lower() not in ("none", "n/a")]

    relations: list[Relation] = []
    for item in re.split(r"[;\n]", fields.get("RELATIONS", "")):
        parts = [p.strip(" -*") for p in item.split("|")]
        placeholder = any(p.lower() in ("relation", "subject", "object") or p.startswith("<") for p in parts)
        if len(parts) == 3 and all(parts) and all(len(p) <= 120 for p in parts) and not placeholder:
            relations.append(Relation(subject=parts[0], predicate=parts[1], object=parts[2]))

    summary = " ".join(fields.get("SUMMARY", "").split())
    if not matches:  # the model ignored the format: treat the whole reply as the summary
        summary = " ".join(raw.split())
    return entities, relations[:8], summary


def _classify_entity(text: str) -> str:
    for entity_type, pattern in _ENTITY_PATTERNS:
        if pattern.fullmatch(text) or pattern.search(text):
            return entity_type
    return "other"


class DirectAssistProcessor:
    """Multilevel Semantic Optimization of direct-assist chunks."""

    def __init__(self, llm: LLMClient, max_tokens: int = 220) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def _process_one(self, question: str, chip_label: str, chunk: ChunkTrace) -> DirectSummary:
        pattern_entities = extract_hardware_entities(chunk.text)
        hints = ", ".join(e.text for e in pattern_entities[:16]) or "none"
        prompt = _MSO_PROMPT.format(
            chip=chip_label, question=question, title=chunk.title, text=chunk.text.strip(), hints=hints
        )
        raw = self.llm.chat(
            [{"role": "system", "content": _MSO_SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )
        llm_entities, relations, summary = parse_mso_output(raw)

        # Keep only LLM entities that literally occur in the chunk, then add the
        # pattern-detected ones the LLM named or that the question asks about.
        chunk_lower = chunk.text.lower()
        merged: dict[str, Entity] = {}
        for name in llm_entities:
            if name.lower() in chunk_lower:
                merged[name.lower()] = Entity(text=name, type=_classify_entity(name))  # type: ignore[arg-type]
        for entity in pattern_entities:
            merged.setdefault(entity.text.lower(), entity)
        entities = list(merged.values())[:16]
        relations = validate_relations(relations, chunk.text)
        evidence = extractive_summary(question, chunk.text, max_sentences=2)

        irrelevant = summary.strip(" .\"'").upper() == "IRRELEVANT" or summary.upper().startswith("IRRELEVANT")
        # A small summariser sometimes dismisses the very chunk that holds the
        # answer. Its verdict is only accepted when the chunk also shares almost
        # no vocabulary with the question; otherwise the chunk is kept and
        # condensed by sentence selection.
        shared_terms = len(_terms(question) & _terms(f"{chunk.title} {chunk.text}"))
        if irrelevant and shared_terms >= 2:
            return DirectSummary(
                source_id=chunk.source_id, chunk_id=chunk.chunk_id, title=chunk.title,
                entities=entities, relations=relations,
                summary=extractive_summary(question, chunk.text), method="extractive",
            )
        if irrelevant:
            return DirectSummary(
                source_id=chunk.source_id, chunk_id=chunk.chunk_id, title=chunk.title,
                entities=entities, relations=[], summary="No content relevant to the question.",
                method="llm", relevant=False,
            )
        method = "llm"
        faithful = (
            len(summary.split()) >= 4
            and numbers_are_grounded(summary, chunk.text, question)
            and polarity_preserved(summary, evidence)
            and vocabulary_supported(summary, f"{chunk.title} {chunk.text}", question)
        )
        if not faithful:
            summary, method = extractive_summary(question, chunk.text), "extractive"
        # An LLM summary never travels alone: the sentences it should rest on are
        # sent with it, verbatim, so a paraphrase that drifts from the source can
        # be caught by the generator (and by the reader, in the X-Ray view).
        return DirectSummary(
            source_id=chunk.source_id, chunk_id=chunk.chunk_id, title=chunk.title,
            entities=entities, relations=relations, summary=summary,
            evidence=evidence if method == "llm" else "", method=method,  # type: ignore[arg-type]
        )

    def _safe(self, question: str, chip_label: str, chunk: ChunkTrace) -> DirectSummary:
        try:
            return self._process_one(question, chip_label, chunk)
        except LLMError:
            # The summariser is an optimisation; if it is unavailable the chunk
            # is still used, condensed by sentence selection.
            return DirectSummary(
                source_id=chunk.source_id, chunk_id=chunk.chunk_id, title=chunk.title,
                entities=extract_hardware_entities(chunk.text)[:16], relations=[],
                summary=extractive_summary(question, chunk.text), method="extractive",
            )

    def process(self, question: str, chip_label: str, chunks: Sequence[ChunkTrace]) -> list[DirectSummary]:
        if not chunks:
            return []
        if getattr(self.llm, "parallel", False) and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
                return list(pool.map(lambda c: self._safe(question, chip_label, c), chunks))
        return [self._safe(question, chip_label, chunk) for chunk in chunks]

    @staticmethod
    def passthrough(chunks: Sequence[ChunkTrace]) -> list[DirectSummary]:
        """Ablation: no summarisation, the raw chunk text is used as-is."""
        return [
            DirectSummary(
                source_id=c.source_id, chunk_id=c.chunk_id, title=c.title,
                entities=extract_hardware_entities(c.text)[:16], relations=[],
                summary=c.text.strip(), method="raw",
            )
            for c in chunks
        ]


def render_direct_item(summary: DirectSummary, source_text: str = "") -> str:
    """Text placed in the prompt for one direct-assist document.

    The summary leads, so the question-relevant point is the first thing the
    generator reads, and the original chunk follows in full. Summaries of pin
    tables and electrical tables are exactly where a paraphrase goes wrong, so
    the generator always has the source wording to check against.
    """
    if summary.method == "raw":
        return summary.summary
    source_text = source_text.strip()
    if not source_text or source_text == summary.summary.strip():
        return summary.summary
    # Entities and relations are shown in the X-Ray view but are not sent to the
    # generator: a wrong triple is harder to spot than a wrong sentence.
    return f"Summary: {summary.summary}\nSource text: {source_text}"


# ---------------------------------------------------------------------------
# Indirect-Assist branch
# ---------------------------------------------------------------------------


class RearrangingModule:
    """Mix indirect-assist chunks with noise documents and shuffle the result."""

    def __init__(self, noise_provider: NoiseProvider) -> None:
        self.noise_provider = noise_provider

    def process(
        self,
        indirect_chunks: Sequence[ChunkTrace],
        question: str,
        n_noise: int,
        rng: random.Random,
        exclude_titles: Optional[set[str]] = None,
    ) -> tuple[list[ChunkTrace], list[ChunkTrace]]:
        """Return (shuffled middle documents, the noise documents that were added)."""
        injected = self.noise_provider.sample(n_noise, rng, question, "rearrange", exclude_titles)
        middle = list(indirect_chunks) + injected
        rng.shuffle(middle)
        return middle, injected
