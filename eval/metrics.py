"""Answer-matching metrics: Exact Match (EM) and token-level F1.

EM follows the convention of RAG robustness benchmarks (RGB, CP-RAG): a
prediction is correct when it *contains* a gold answer, after normalisation.
Matching is boundary-aware, so the gold answer "6" does not match "16" or
"GPIO6", and "0x0" does not match "0x1000".
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Sequence

_CITATION = re.compile(r"\[\s*(?:S\d+[\s,;&]*(?:and\s*)?)+\]", re.I)
_UNITS = r"(?:ua|ma|na|mv|v|mhz|khz|ghz|hz|kb|mb|gb|kbps|mbps|dbm|ms|us|bit)"
_ARTICLES = re.compile(r"\b(?:a|an|the)\b")
_NUMBER_WORDS = {
    word: str(value) for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty".split()
    )
}
_NUMBER_WORD = re.compile(r"\b(%s)\b" % "|".join(_NUMBER_WORDS))


def normalize(text: str) -> str:
    text = _CITATION.sub(" ", text).lower()
    text = text.replace("µ", "u").replace("μ", "u").replace("–", "-").replace("—", "-")
    text = _NUMBER_WORD.sub(lambda match: _NUMBER_WORDS[match.group(1)], text)  # "seven channels" -> "7 channels"
    text = re.sub(r"\b(gpio|io|adc|dac|uart|i2c|spi)s?\s+(?=\d)", r"\1", text)  # "GPIO 25" -> "gpio25"
    text = re.sub(r"\bbits\b", "bit", text)
    text = re.sub(r"(?<=\d)[\s\-]+(?=%s\b)" % _UNITS, "", text)  # "10 uA" / "13-bit" -> "10ua" / "13bit"
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 2,500 -> 2500
    text = re.sub(r"[^a-z0-9.\s_/]", " ", text)
    text = re.sub(r"\.(?!\d)", " ", text)  # keep decimal points only
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def contains_answer(prediction: str, answer: str) -> bool:
    pred, gold = normalize(prediction), normalize(answer)
    if not gold:
        return False
    return re.search(r"(?<![a-z0-9.])%s(?![a-z0-9])" % re.escape(gold), pred) is not None


def exact_match(prediction: str, answers: Sequence[str], all_of: Optional[Sequence[str]] = None) -> float:
    if all_of:
        return float(all(contains_answer(prediction, part) for part in all_of))
    return float(any(contains_answer(prediction, answer) for answer in answers))


def _f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, answers: Sequence[str]) -> float:
    """Best token-level F1 of the prediction against any accepted answer (SQuAD style)."""
    pred_tokens = normalize(prediction).split()
    return max((_f1(pred_tokens, normalize(answer).split()) for answer in answers), default=0.0)
