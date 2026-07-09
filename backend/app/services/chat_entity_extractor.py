from __future__ import annotations

import re
import unicodedata
from typing import Iterable


GENERIC_TERMS = {
    "adr",
    "bệnh nhân",
    "cách dùng",
    "chống chỉ định",
    "dịch",
    "dùng",
    "gan",
    "gì",
    "không",
    "không dùng khi nào",
    "liều",
    "lưu ý",
    "mang thai",
    "nhau",
    "nào",
    "suy gan",
    "suy thận",
    "tác dụng không mong muốn",
    "tác dụng phụ",
    "thai",
    "thận",
    "thận trọng",
    "thuốc",
    "thuốc nào",
    "uống",
}

CONNECTOR_PATTERNS = [
    r"\bcó\s+tương\s+tác\s+với\b",
    r"\bdùng\s+cùng\b",
    r"\buống\s+cùng\b",
    r"\bdùng\s+chung\b",
    r"\buống\s+chung\b",
    r"\bphối\s+hợp\s+với\b",
    r"\bkết\s+hợp\s+với\b",
    r"\bvới\b",
    r"\bvà\b",
]

TRAILING_PATTERNS = [
    r"\bcó\s+những\s+tương\s+tác\s+thuốc\s+nào\b.*$",
    r"\btương\s+tác\s+với\s+thuốc\s+nào\b.*$",
    r"\bcó\s+tương\s+tác\b.*$",
    r"\btương\s+tác\b.*$",
    r"\bcó\s+sao\b.*$",
    r"\bcó\s+ảnh\s+hưởng\b.*$",
    r"\bcó\s+tác\s+dụng\s+phụ\b.*$",
    r"\btác\s+dụng\s+phụ\b.*$",
    r"\btác\s+dụng\s+không\s+mong\s+muốn\b.*$",
    r"\bdùng\s+cần\s+lưu\s+ý\b.*$",
    r"\bcần\s+thận\s+trọng\b.*$",
    r"\bthận\s+trọng\b.*$",
    r"\bchống\s+chỉ\s+định\b.*$",
    r"\bkhông\s+dùng\s+khi\s+nào\b.*$",
    r"\bliều\b.*$",
    r"\bcách\s+dùng\b.*$",
    r"\buống\s+thế\s+nào\b.*$",
    r"\bcần\s+lưu\s+ý\b.*$",
    r"\blưu\s+ý\b.*$",
    r"\bkhông\b$",
    r"\bgì\b$",
]

LEADING_NOISE_RE = re.compile(
    r"^(cho\s+tôi\s+hỏi|cho\s+tôi\s+thông\s+tin\s+về|"
    r"xin\s+hỏi|hỏi|thông\s+tin\s+về|thuốc|dịch\s+truyền)\s+",
    flags=re.IGNORECASE,
)
PAIR_SEPARATOR_RE = re.compile(r"\b(?:và|với)\b", flags=re.IGNORECASE)
DRUG_TOKEN_RE = re.compile(
    r"[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9.+/-]*(?:\s+[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9.+/-]*){0,3}"
)


def _fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    folded = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return folded.replace("đ", "d")


def _clean_candidate(value: str) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = re.sub(r"[?!.:,;()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -/")
    text = LEADING_NOISE_RE.sub("", text).strip()
    for pattern in TRAILING_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", text).strip(" -/")


def _is_generic(value: str) -> bool:
    folded = _fold_text(value)
    if len(folded) < 3:
        return True
    generic_terms = {_fold_text(term) for term in GENERIC_TERMS}
    return folded in generic_terms


def _deduplicate(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _fold_text(value)
        if not value or key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _candidate_parts(value: str) -> list[str]:
    candidate = _clean_candidate(value)
    if not candidate:
        return []
    return [
        _clean_candidate(part)
        for part in PAIR_SEPARATOR_RE.split(candidate)
        if _clean_candidate(part)
    ]


def _mention_from_candidate(value: str) -> str | None:
    candidate = _clean_candidate(value)
    if not candidate or _is_generic(candidate):
        return None

    token_match = DRUG_TOKEN_RE.search(candidate)
    if not token_match:
        return None

    mention = _clean_candidate(token_match.group(0))
    if not mention or _is_generic(mention):
        return None
    return mention


class ChatEntityExtractor:
    """Deterministic drug mention extractor for short chat questions."""

    def extract(self, message: str, max_mentions: int = 4) -> list[str]:
        text = unicodedata.normalize("NFC", str(message or ""))
        candidates: list[str] = []

        for pattern in CONNECTOR_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            candidates.extend(_candidate_parts(text[: match.start()]))
            candidates.extend(_candidate_parts(text[match.end() :]))
            break

        if not candidates:
            candidates.extend(_candidate_parts(text))

        mentions = [
            mention
            for candidate in candidates
            if (mention := _mention_from_candidate(candidate)) is not None
        ]
        return _deduplicate(mentions)[:max_mentions]
