"""Section-aware chunking.

Sections are never merged across a heading. Inside a section, paragraphs are
packed up to `chunk_words`; consecutive chunks of the same section overlap by
`overlap_words` so a fact split across a boundary is still retrievable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingest.parsers import ParsedDocument

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\[(])")


@dataclass
class Chunk:
    text: str
    section: str
    index: int


def _split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    words = paragraph.split()
    if len(words) <= limit:
        return [paragraph]
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _SENTENCE_END.split(paragraph):
        n = len(sentence.split())
        if n > limit:  # a single run-on "sentence" (tables, logs): hard split
            if current:
                pieces.append(" ".join(current))
                current, count = [], 0
            tokens = sentence.split()
            pieces.extend(" ".join(tokens[i:i + limit]) for i in range(0, len(tokens), limit))
            continue
        if count + n > limit and current:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_section(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if paragraph:
            paragraphs.extend(_split_long_paragraph(paragraph, chunk_words))

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for paragraph in paragraphs:
        n = len(paragraph.split())
        if current and count + n > chunk_words:
            chunks.append("\n\n".join(current))
            tail = " ".join(" ".join(current).split()[-overlap_words:]) if overlap_words else ""
            current = [tail] if tail else []
            count = len(tail.split())
        current.append(paragraph)
        count += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(
    document: ParsedDocument,
    chunk_words: int = 180,
    overlap_words: int = 30,
    min_words: int = 12,
    merge_below: int = 40,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    carry = ""  # a section too short to stand alone (often one intro sentence) joins the next one
    carry_heading = ""
    for position, section in enumerate(document.sections):
        text = f"{carry}\n\n{section.text}" if carry else section.text
        heading = carry_heading or section.heading
        is_last = position == len(document.sections) - 1
        if len(text.split()) < merge_below and not is_last:
            carry, carry_heading = text, heading
            continue
        carry, carry_heading = "", ""
        for piece in chunk_section(text, chunk_words, overlap_words):
            if len(piece.split()) < min_words:
                continue
            chunks.append(Chunk(text=piece, section=heading, index=len(chunks)))
    return chunks
