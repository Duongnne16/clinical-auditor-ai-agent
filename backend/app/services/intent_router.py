from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

from backend.app.services.chat_entity_extractor import ChatEntityExtractor


DRUG_INTERACTION_QUERY = "drug_interaction_query"
SINGLE_DRUG_QUERY = "single_drug_query"
OUT_OF_SCOPE = "out_of_scope"

INTERACTION_KEYWORDS = [
    "tương tác",
    "dùng cùng",
    "uống cùng",
    "dùng chung",
    "uống chung",
    "phối hợp",
    "kết hợp",
    "có sao không",
    "có ảnh hưởng không",
]

SINGLE_DRUG_INTERACTION_LOOKUP_PATTERNS = [
    r"\bco\s+nhung\s+tuong\s+tac\s+thuoc\s+nao\b",
    r"\btuong\s+tac\s+voi\s+thuoc\s+nao\b",
    r"\bco\s+tuong\s+tac\s+gi\b",
]

TOPIC_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "adverse_effect",
        ["tác dụng phụ", "tác dụng không mong muốn", "adr"],
    ),
    ("contraindication", ["chống chỉ định", "không dùng khi nào"]),
    ("caution", ["thận trọng", "lưu ý", "cần chú ý"]),
    ("dosage", ["liều", "cách dùng", "uống thế nào"]),
    ("pregnancy_lactation", ["thai", "mang thai", "cho con bú"]),
    ("renal", ["suy thận", "thận"]),
    ("hepatic", ["suy gan", "gan"]),
]

MEDICAL_KEYWORDS = [
    "adr",
    "bệnh nhân",
    "chống chỉ định",
    "cách dùng",
    "dược",
    "hoạt chất",
    "kê đơn",
    "liều",
    "lưu ý",
    "mang thai",
    "suy gan",
    "suy thận",
    "tác dụng phụ",
    "thận trọng",
    "thuốc",
    "tương tác",
    "uống",
]

CLEAR_TRANSLATION_PATTERNS = [
    r"\bdich\s+cau\s+nay\b",
    r"\bdich\s+sang\s+tieng\b",
    r"\bdich\s+.*\s+sang\s+tieng\b",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\bbai\s+van\b",
    r"\blam\s+tho\b",
    r"\bthoi\s+tiet\b",
    r"\bdau\s+tu\b",
    r"\bchung\s+khoan\b",
    r"\bcode\b",
    r"\blap\s+trinh\b",
]


@dataclass
class IntentClassification:
    intent: str
    confidence: float
    drug_mentions: list[str] = field(default_factory=list)
    topic: str = "general"
    warnings: list[str] = field(default_factory=list)


def _fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).replace(
        "đ", "d"
    )


def _contains_any(text: str, phrases: list[str]) -> bool:
    folded = _fold_text(text)
    return any(_fold_text(phrase) in folded for phrase in phrases)


def _matches_any(text: str, patterns: list[str]) -> bool:
    folded = _fold_text(text)
    return any(re.search(pattern, folded) for pattern in patterns)


class IntentRouter:
    """Deterministic lightweight router for chat MVP intents."""

    def __init__(self, extractor: ChatEntityExtractor | None = None) -> None:
        self.extractor = extractor or ChatEntityExtractor()

    @staticmethod
    def _topic(message: str) -> str:
        for topic, keywords in TOPIC_KEYWORDS:
            if _contains_any(message, keywords):
                return topic
        return "general"

    @staticmethod
    def _is_clear_out_of_scope(message: str) -> bool:
        if _matches_any(message, CLEAR_TRANSLATION_PATTERNS):
            return True
        if _contains_any(message, MEDICAL_KEYWORDS):
            return False
        return _matches_any(message, OUT_OF_SCOPE_PATTERNS)

    @staticmethod
    def _is_single_drug_interaction_lookup(message: str) -> bool:
        return _matches_any(message, SINGLE_DRUG_INTERACTION_LOOKUP_PATTERNS)

    def classify(self, message: str) -> IntentClassification:
        if self._is_clear_out_of_scope(message):
            return IntentClassification(
                intent=OUT_OF_SCOPE,
                confidence=0.95,
                drug_mentions=[],
                topic="general",
            )

        drug_mentions = self.extractor.extract(message)

        if _contains_any(message, INTERACTION_KEYWORDS):
            if (
                len(drug_mentions) == 1
                and self._is_single_drug_interaction_lookup(message)
            ):
                return IntentClassification(
                    intent=SINGLE_DRUG_QUERY,
                    confidence=0.85,
                    drug_mentions=drug_mentions,
                    topic="interaction",
                )
            warnings = []
            if len(drug_mentions) < 2:
                warnings.append("interaction_query_requires_two_drugs")
            return IntentClassification(
                intent=DRUG_INTERACTION_QUERY,
                confidence=0.9 if len(drug_mentions) >= 2 else 0.65,
                drug_mentions=drug_mentions,
                topic="interaction",
                warnings=warnings,
            )

        topic = self._topic(message)
        if topic != "general" or drug_mentions or _contains_any(message, MEDICAL_KEYWORDS):
            warnings = []
            if not drug_mentions:
                warnings.append("single_drug_query_requires_drug_name")
            return IntentClassification(
                intent=SINGLE_DRUG_QUERY,
                confidence=0.85 if drug_mentions else 0.55,
                drug_mentions=drug_mentions,
                topic=topic,
                warnings=warnings,
            )

        return IntentClassification(
            intent=OUT_OF_SCOPE,
            confidence=0.75,
            drug_mentions=drug_mentions,
            topic="general",
        )
