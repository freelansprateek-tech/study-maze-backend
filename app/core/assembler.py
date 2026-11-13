"""Position-aware prompt assembly and answer parsing.

LLMs use the beginning and the end of a long context far better than its
middle ("Lost in the Middle", Liu et al. 2024). The prompt is therefore laid
out as

    System grounding rules
    Direct_Head      first half of the direct-assist summaries, strongest first
    Middle           shuffled indirect-assist chunks + noise documents
    Direct_Tail      second half of the direct-assist summaries, strongest last
    Question + "Let us think step by step."

so the highest-contribution content sits at both edges, next to the rules and
next to the question.
"""
from __future__ import annotations

import re
from typing import Sequence

from app.chips import chip_label
from app.schemas import ChunkTrace, GroundingStatus, PromptItem, PromptSegment

COT_TRIGGER = "Let us think step by step."
FINAL_ANSWER_MARKER = "Final answer:"
EXPLANATION_MARKER = "Explanation:"


def not_covered_statement(chip: str) -> str:
    return f"The documentation for the {chip_label(chip)} does not cover this detail."


def system_rules(chip: str, detailed: bool = False) -> str:
    label = chip_label(chip)
    return (
        f"You are a technical support assistant for the Espressif {label} microcontroller.\n"
        "Grounding rules:\n"
        f"1. Answer ONLY with facts stated in the numbered source documents in the user message. "
        f"Never use your own knowledge of the {label} or of any other chip.\n"
        "2. The sources differ in quality. Some are unrelated or describe other hardware; ignore those. "
        "Values for a different chip are never an answer.\n"
        "3. Where a source gives a summary followed by its source text, the source text is authoritative: "
        "if they disagree, follow the source text. Tables appear as rows of cells separated by ' | ' with the "
        "header row first.\n"
        "4. After every fact, cite the source it came from with its tag in square brackets, for example [S2].\n"
        f"5. If the sources do not contain the answer, reply exactly: \"{not_covered_statement(chip)}\" "
        "Do not guess and do not give a partial answer from memory. If the sources do not state the answer in so "
        "many words but state a fact that settles it (for example the only frequency band that is supported), "
        "answer with that fact and say that this is what the documentation states.\n"
        f"6. Reason briefly step by step, then end with one line that starts with \"{FINAL_ANSWER_MARKER}\" "
        "followed by the short, exact answer and its citation tags."
        + (
            "\n7. Your readers are students who are new to microcontrollers. In the final answer and the "
            "explanation use short sentences and simple everyday words, and say what a technical term means in a "
            "few words the first time you use it. Simple wording never changes the facts: every pin name, number "
            "and unit must be copied exactly from the sources."
            if detailed else ""
        )
    )


def split_head_tail(direct_items: Sequence[PromptItem]) -> tuple[list[PromptItem], list[PromptItem]]:
    """`direct_items` must be sorted strongest first.

    Head takes the first half in that order. Tail takes the second half
    reversed, so within the tail the stronger document is the one closest to
    the question.
    """
    items = list(direct_items)
    half = (len(items) + 1) // 2
    return items[:half], list(reversed(items[half:]))


def _render_item(item: PromptItem, meta: dict[str, ChunkTrace]) -> str:
    chunk = meta.get(item.source_id)
    header = f"[{item.source_id}] {item.title}"
    if chunk is not None and not chunk.is_noise:
        details = " | ".join(p for p in (chip_label(chunk.chip), chunk.section) if p)
        if details:
            header += f" ({details})"
    return f"{header}\n{item.text.strip()}"


def _approx_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def assemble_prompt(
    chip: str,
    question: str,
    direct_items: Sequence[PromptItem],
    middle_items: Sequence[PromptItem],
    chunks_by_source: dict[str, ChunkTrace],
    u_shape: bool = True,
    flat_order: Sequence[str] = (),
    detailed: bool = False,
) -> tuple[list[dict[str, str]], list[PromptSegment]]:
    """Build the chat messages and the layout trace shown by the X-Ray inspector.

    With `u_shape=False` (ablation) all documents are listed once in
    `flat_order` (the original retrieval order), with no head/tail placement.

    With `detailed=True` the model writes a fuller answer in simple words and
    a longer explanation after it, for readers who are new to the subject.
    With `detailed=False` it writes the short exact answer the benchmark scores.
    """
    system_text = system_rules(chip, detailed)
    # The format reminder sits directly above the question: small models follow
    # an instruction near the end of the prompt far more reliably than one in
    # the system message alone.
    format_text = (
        "### How to answer\n"
        "Use only the source documents above. Write 2 to 4 short reasoning steps and put the source tag, "
        "such as [S1], after every fact you use. Then write one last line in exactly this form:\n"
        f"{FINAL_ANSWER_MARKER} <short exact answer> [S#]\n"
        f"If the sources do not contain the answer, the last line must be:\n"
        f"{FINAL_ANSWER_MARKER} {not_covered_statement(chip)}"
    )
    if detailed:
        format_text = (
            "### How to answer\n"
            "Use only the source documents above. Write for a student who is new to microcontrollers: short "
            "sentences, simple everyday words, and a few words of explanation for any technical term.\n"
            "First write 2 to 4 short reasoning steps and put the source tag, such as [S1], after every fact you "
            "use. Then write these two parts, each starting with its label exactly as shown:\n"
            f"{FINAL_ANSWER_MARKER} <2 to 3 simple sentences. Give the exact answer first, then say in plain "
            "words what it means. Put a source tag after every fact.>\n"
            f"{EXPLANATION_MARKER} <5 to 8 simple sentences that explain the answer in more depth: why it is "
            "so, what it means in practice when you build a project, and what to be careful about. Use only "
            "facts from the sources and put a source tag after every fact. Do not add any pin name, number or "
            "value that is not in the sources.>\n"
            "If the sources do not contain the answer, write only this line and no explanation:\n"
            f"{FINAL_ANSWER_MARKER} {not_covered_statement(chip)}"
        )
    question_text = f"Question about the {chip_label(chip)}: {question.strip()}\n{COT_TRIGGER}"
    segments: list[PromptSegment] = [
        PromptSegment(position="system", label="System grounding rules", text=system_text,
                      approx_tokens=_approx_tokens(system_text)),
    ]
    blocks: list[str] = []

    def add(position: str, label: str, heading: str, items: Sequence[PromptItem]) -> None:
        if not items:
            return
        body = "\n\n".join(_render_item(item, chunks_by_source) for item in items)
        text = f"### {heading}\n{body}"
        blocks.append(text)
        segments.append(PromptSegment(position=position, label=label, text=text, items=list(items),  # type: ignore[arg-type]
                                      approx_tokens=_approx_tokens(text)))

    if u_shape:
        head, tail = split_head_tail(direct_items)
        add("head", "Head — Direct-Assist (first half)", "Key source documents, part 1", head)
        add("middle", "Middle — Indirect-Assist + shuffled noise", "Additional source documents", middle_items)
        add("tail", "Tail — Direct-Assist (second half)", "Key source documents, part 2", tail)
    else:
        by_id = {item.source_id: item for item in list(direct_items) + list(middle_items)}
        ordered = [by_id[sid] for sid in flat_order if sid in by_id]
        ordered += [item for sid, item in by_id.items() if sid not in set(flat_order)]
        add("flat", "Flat — retrieval order (no head/tail placement)", "Source documents", ordered)

    if not blocks:
        blocks.append("### Source documents\n(no documents were retrieved)")
    segments.append(PromptSegment(position="question", label="User question + step-by-step trigger",
                                  text=f"{format_text}\n\n{question_text}",
                                  approx_tokens=_approx_tokens(format_text + question_text)))

    user_text = "\n\n".join(blocks + [format_text, question_text])
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
    return messages, segments


def render_messages(messages: Sequence[dict[str, str]]) -> str:
    return "\n\n".join(f"<<{m['role'].upper()}>>\n{m['content']}" for m in messages)


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

_CITATION = re.compile(r"\[\s*((?:S\d+\s*(?:,|;|and|&)?\s*)+)\]", re.I)
_FINAL = re.compile(r"\**\s*final\s+answer\s*\**\s*[:\-]\s*\**", re.I)
_EXPLANATION = re.compile(r"(?:^|\n)\s*(?:#+\s*)?\**\s*explanation\s*\**\s*[:\-]\s*\**", re.I)
_NOT_COVERED = re.compile(r"do(?:es)?\s+not\s+cover\s+this\s+detail", re.I)


def extract_citations(text: str, valid_ids: set[str]) -> list[str]:
    cited: list[str] = []
    for group in _CITATION.findall(text):
        for source_id in re.findall(r"S\d+", group, flags=re.I):
            source_id = source_id.upper()
            if source_id in valid_ids and source_id not in cited:
                cited.append(source_id)
    return cited


def parse_answer(raw: str, chip: str, valid_ids: set[str]) -> tuple[str, str, str, list[str], GroundingStatus]:
    """Split the generation into (answer, explanation, reasoning, cited source ids, grounding status).

    `explanation` is the optional longer part after the "Explanation:" label; it
    is empty when the model was not asked for one or did not write one.
    """
    text = raw.strip()
    matches = list(_FINAL.finditer(text))
    if matches:
        last = matches[-1]
        reasoning, answer = text[: last.start()].strip(), text[last.end():].strip()
    else:
        reasoning, answer = "", text
    explanation = ""
    split = _EXPLANATION.search(answer)
    if split:
        answer, explanation = answer[: split.start()], answer[split.end():].strip().strip("*").strip()
    answer = answer.strip().strip("*").strip()

    if not answer or _NOT_COVERED.search(answer) or (not matches and _NOT_COVERED.search(text)):
        return not_covered_statement(chip), "", reasoning, [], "not_covered"

    all_citations = extract_citations(answer, valid_ids)
    for part in (explanation, reasoning):
        all_citations += [c for c in extract_citations(part, valid_ids) if c not in all_citations]
    status: GroundingStatus = "grounded" if all_citations else "uncited"
    return answer, explanation, reasoning, all_citations, status


# ---------------------------------------------------------------------------
# Post-generation value check
# ---------------------------------------------------------------------------

_VALUE_TOKEN = re.compile(r"[A-Za-z_]*\d[\w.]*(?:-[A-Za-z]\d\w*)?(?:\s?(?:µA|μA|uA|mA|mV|V|MHz|kHz|GHz|Hz|KB|kB|MB|Mbps|kbps|dBm|ms|bits?)\b)?")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", re.M)


def _canonical(text: str) -> str:
    text = _CITATION.sub(" ", text).lower().replace("µ", "u").replace("μ", "u")
    # "10 uA" / "12-bit" -> "10ua" / "12bit" (known units only)
    text = re.sub(r"(?<=\d)[\s\-]+(?=(?:ua|ma|mv|v|mhz|khz|ghz|hz|kb|mb|mbps|kbps|dbm|ms|bits?)\b)", "", text)
    return re.sub(r"(?<=\d)(bit)s\b", r"\1", text)


def unsupported_values(answer: str, sources: Sequence[str]) -> list[str]:
    """Values in the answer (numbers, quantities with units, identifiers such as
    GPIO9 or 0x1000) that appear in none of the given source texts.

    This is the last line of defence for the grounding rule: a pin number or an
    electrical limit that no retrieved document contains must not reach the user.
    """
    haystack = _canonical(" \n ".join(sources))
    missing: list[str] = []
    cleaned = _LIST_MARKER.sub("", _CITATION.sub(" ", answer))
    for match in _VALUE_TOKEN.finditer(cleaned):
        raw = match.group(0).strip().rstrip(".")
        token = _canonical(raw).strip().rstrip(".")
        if not token or re.fullmatch(r"s\d+", token):
            continue
        if re.search(r"(?<![\w.])%s(?![\w])" % re.escape(token), haystack) is None and raw not in missing:
            missing.append(raw)
    return missing


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z0-9\-•*])|\n+")


def keep_supported_sentences(text: str, sources: Sequence[str]) -> str:
    """Drop every sentence of `text` that contains a value found in no source.

    Used for the long explanation: one unsupported number should remove the
    sentence that carries it, not the whole explanation.
    """
    kept = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip() and not unsupported_values(s, sources)]
    return "\n".join(kept) if "\n" in text.strip() else " ".join(kept)
