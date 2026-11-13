"""Chip metadata classifier.

Assigns every chunk one tag out of
    esp32, esp32-s3, esp32-c3, esp32-c6, esp32-s2, common

Priority, highest first:
  1. an explicit chip supplied by the source (JSON `chip` field, CLI `--chip`);
  2. the file path (`data/raw/esp32-s3/...`, `esp32c3_datasheet_en.pdf`);
  3. the chip names mentioned in the text itself;
  4. `common` when nothing identifies a single chip.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from app.chips import ALL_TAGS, CHIP_IDS, COMMON

# Variant chips first so "ESP32-S3" is never read as plain "ESP32".
_VARIANT_PATTERNS: dict[str, re.Pattern[str]] = {
    "esp32-s3": re.compile(r"esp32[\s_\-]?s3", re.I),
    "esp32-s2": re.compile(r"esp32[\s_\-]?s2", re.I),
    "esp32-c3": re.compile(r"esp32[\s_\-]?c3|esp8685", re.I),
    "esp32-c6": re.compile(r"esp32[\s_\-]?c6", re.I),
}
# Plain ESP32: not followed by a series suffix such as -S3, C6, -H2, -P4.
_CLASSIC_PATTERN = re.compile(r"esp32(?![\s_\-]?[scph]\d)(?:[\s_\-]?(?:wroom|wrover|d0wd|pico|u4wdh|devkitc))?", re.I)

_ALIASES = {
    "esp32s3": "esp32-s3", "esp32_s3": "esp32-s3",
    "esp32s2": "esp32-s2", "esp32_s2": "esp32-s2",
    "esp32c3": "esp32-c3", "esp32_c3": "esp32-c3",
    "esp32c6": "esp32-c6", "esp32_c6": "esp32-c6",
    "all": COMMON, "generic": COMMON, "any": COMMON, "": COMMON,
}


def normalize_chip(value: Optional[str]) -> Optional[str]:
    """Map a free-form chip name to a canonical tag, or None if unrecognised."""
    if value is None:
        return None
    key = value.strip().lower()
    key = _ALIASES.get(key, key)
    return key if key in ALL_TAGS else None


def count_chip_mentions(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for chip, pattern in _VARIANT_PATTERNS.items():
        counts[chip] = len(pattern.findall(text))
    counts["esp32"] = len(_CLASSIC_PATTERN.findall(text))
    return Counter({chip: n for chip, n in counts.items() if n > 0})


def chip_from_path(path: str) -> Optional[str]:
    """Chip named by a directory or file name. Directories closest to the file win."""
    parts = [p for p in re.split(r"[\\/#]", path.lower()) if p]
    for part in reversed(parts):
        if part == COMMON or part.startswith("common"):
            return COMMON
        mentions = count_chip_mentions(part)
        if len(mentions) == 1:
            return next(iter(mentions))
    return None


def chip_from_text(text: str, dominance: float = 0.7) -> str:
    """Tag by content: one clearly dominant chip, otherwise `common`."""
    mentions = count_chip_mentions(text)
    if not mentions:
        return COMMON
    if len(mentions) == 1:
        return next(iter(mentions))
    (top_chip, top_count), (_, second_count) = mentions.most_common(2)
    total = sum(mentions.values())
    if top_count / total >= dominance and top_count > second_count:
        return top_chip
    return COMMON


def tag_chunk(
    text: str,
    source_path: str = "",
    explicit_chip: Optional[str] = None,
    title: str = "",
) -> str:
    explicit = normalize_chip(explicit_chip)
    if explicit:
        return explicit
    from_path = chip_from_path(source_path) if source_path else None
    if from_path:
        return from_path
    # The title names the device far more reliably than body text does.
    from_title = count_chip_mentions(title)
    if len(from_title) == 1:
        return next(iter(from_title))
    return chip_from_text(f"{title}\n{text}")


__all__ = [
    "CHIP_IDS",
    "chip_from_path",
    "chip_from_text",
    "count_chip_mentions",
    "normalize_chip",
    "tag_chunk",
]
