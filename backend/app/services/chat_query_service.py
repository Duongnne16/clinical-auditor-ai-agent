from __future__ import annotations

import re
import unicodedata
from typing import Any

from backend.app.schemas.chat import ChatRequest
from backend.app.services.doctor_report_text_safety import (
    sanitize_doctor_report_text,
)
from backend.app.services.intent_router import (
    DRUG_INTERACTION_QUERY,
    OUT_OF_SCOPE,
    SINGLE_DRUG_QUERY,
    IntentClassification,
    IntentRouter,
)
from backend.app.services.ingredient_evidence_resolver import IngredientEvidenceResolver
from backend.app.services.normalize_drugs_service import NormalizeDrugsService
from backend.app.services.qdrant_retriever_service import QdrantRetrieverService


OUT_OF_SCOPE_REFUSAL = (
    "Hệ thống này chỉ hỗ trợ tra cứu và rà soát thông tin liên quan đến thuốc, "
    "đơn thuốc, tương tác thuốc và một số lưu ý sử dụng thuốc. Vui lòng nhập "
    "câu hỏi trong phạm vi này."
)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "Hiện hệ thống chưa tìm thấy đủ bằng chứng phù hợp trong cơ sở dữ liệu để "
    "trả lời chắc chắn câu hỏi này. Bác sĩ/dược sĩ nên kiểm tra lại tên "
    "thuốc/hoạt chất và đối chiếu thêm với nguồn chuyên môn."
)

INTERACTION_MISSING_DRUGS_ANSWER = (
    "Vui lòng nhập rõ tên cả hai thuốc hoặc hoạt chất cần rà soát tương tác."
)

SINGLE_DRUG_MISSING_ANSWER = (
    "Vui lòng nhập rõ tên thuốc hoặc hoạt chất cần tra cứu."
)

TOPIC_QUERY_TYPES = {
    "adverse_effect": "adverse_effect",
    "contraindication": "contraindication",
    "caution": "precaution",
    "dosage": "dose",
    "pregnancy_lactation": "pregnancy_lactation",
    "renal": "renal_hepatic",
    "hepatic": "renal_hepatic",
    "general": "general",
}

TOPIC_LABELS = {
    "adverse_effect": "tác dụng không mong muốn",
    "contraindication": "chống chỉ định",
    "caution": "thận trọng và lưu ý",
    "dosage": "liều lượng và cách dùng",
    "pregnancy_lactation": "thai kỳ/cho con bú",
    "renal": "chức năng thận",
    "hepatic": "chức năng gan",
    "general": "thông tin thuốc",
}

SINGLE_QUERY_TEXT = {
    "adverse_effect": "{drug} tác dụng không mong muốn tác dụng phụ",
    "contraindication": "{drug} chống chỉ định",
    "caution": "{drug} thận trọng lưu ý",
    "dosage": "{drug} liều lượng cách dùng",
    "pregnancy_lactation": "{drug} thai kỳ cho con bú",
    "renal": "{drug} suy thận chức năng thận",
    "hepatic": "{drug} suy gan chức năng gan",
    "general": "{drug} thông tin thuốc",
}

FORBIDDEN_REPLACEMENTS = [
    (r"không\s+an\s+toàn", "cần được rà soát"),
    (r"an\s+toàn", "cần được rà soát"),
    (r"không\s+nên\s+dùng", "cần được bác sĩ/dược sĩ rà soát"),
    (r"không\s+được\s+dùng", "cần được bác sĩ/dược sĩ rà soát"),
    (r"không\s+được\s+trộn", "cần được bác sĩ/dược sĩ rà soát"),
    (r"không\s+dùng\s+được", "cần được bác sĩ/dược sĩ rà soát"),
    (r"dùng\s+được", "cần được đánh giá theo bối cảnh lâm sàng"),
    (r"ngừng\s+dùng\s+thuốc", "rà soát thuốc"),
    (r"phải\s+ngừng", "cần được nhân viên y tế đánh giá và xử trí phù hợp"),
    (r"cần\s+ngừng", "cần được nhân viên y tế đánh giá và xử trí phù hợp"),
    (r"ngừng\s+thuốc", "rà soát thuốc"),
    (r"dừng\s+thuốc", "rà soát thuốc"),
    (r"thăm\s+khám\s+thầy\s+thuốc", "được nhân viên y tế đánh giá phù hợp"),
    (r"phải\s+thăm\s+khám", "cần được nhân viên y tế đánh giá phù hợp"),
    (r"cần\s+thăm\s+khám", "cần được nhân viên y tế đánh giá phù hợp"),
    (r"đổi\s+thuốc", "cân nhắc phương án xử trí phù hợp"),
    (r"thay\s+thuốc", "cân nhắc phương án xử trí phù hợp"),
    (r"tăng\s+liều", "rà soát liều"),
    (r"giảm\s+liều", "rà soát liều"),
    (r"kê\s+thêm", "cân nhắc phương án điều trị phù hợp"),
]

UNSAFE_EVIDENCE_ACTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ngừng\s+dùng\s+thuốc",
        r"ngừng\s+thuốc",
        r"dừng\s+thuốc",
        r"phải\s+ngừng",
        r"cần\s+ngừng",
        r"không\s+nên\s+dùng",
        r"không\s+được\s+dùng",
        r"không\s+được\s+trộn",
        r"thăm\s+khám\s+thầy\s+thuốc",
        r"phải\s+thăm\s+khám",
        r"cần\s+thăm\s+khám",
        r"tăng\s+liều",
        r"giảm\s+liều",
        r"đổi\s+thuốc",
        r"thay\s+thuốc",
        r"kê\s+thêm",
    )
)

FALLBACK_TOPIC_SUMMARIES = {
    "adverse_effect": (
        "Nguồn tham khảo ghi nhận một số tác dụng không mong muốn cần được "
        "bác sĩ/dược sĩ lưu ý khi rà soát."
    ),
    "interaction": (
        "Nguồn tham khảo có thông tin liên quan đến mục tương tác thuốc; "
        "bác sĩ/dược sĩ nên đối chiếu thêm trước khi quyết định lâm sàng."
    ),
}

PARACETAMOL_SKIN_REACTION_SUMMARY = (
    "Nguồn tham khảo ghi nhận một số phản ứng da nghiêm trọng hiếm gặp liên quan "
    "đến Paracetamol. Bác sĩ/dược sĩ nên đối chiếu với biểu hiện lâm sàng và "
    "tiền sử người bệnh khi rà soát."
)


def _deduplicate(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _clean_text(value: Any, max_length: int = 300) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = _normalize_snippet_text(text)
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0].strip()


def _normalize_snippet_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    return text


def _public_sources(chunks: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for chunk in chunks:
        key = (
            str(chunk.get("source") or ""),
            str(chunk.get("slug") or ""),
            str(chunk.get("section") or ""),
            str(chunk.get("url") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "rank": len(sources) + 1,
                "source": chunk.get("source"),
                "title": chunk.get("title"),
                "entity_name": chunk.get("entity_name"),
                "section": chunk.get("section"),
                "section_title": chunk.get("section_title"),
                "url": chunk.get("url"),
                "slug": chunk.get("slug"),
            }
        )
        if len(sources) >= limit:
            break
    return sources


class ChatQueryService:
    """Thin chat orchestration over existing normalization and retrieval services."""

    def __init__(
        self,
        intent_router: IntentRouter | None = None,
        normalizer: NormalizeDrugsService | None = None,
        retriever: QdrantRetrieverService | None = None,
        ingredient_resolver: IngredientEvidenceResolver | None = None,
        top_k: int = 5,
    ) -> None:
        self.intent_router = intent_router or IntentRouter()
        self.normalizer = normalizer or NormalizeDrugsService()
        self.retriever = retriever or QdrantRetrieverService()
        self.ingredient_resolver = ingredient_resolver or IngredientEvidenceResolver()
        self.top_k = top_k

    @staticmethod
    def _sanitize_answer(answer: str) -> str:
        safe = sanitize_doctor_report_text(answer)
        for pattern, replacement in FORBIDDEN_REPLACEMENTS:
            safe = re.sub(pattern, replacement, safe, flags=re.IGNORECASE)
        return _normalize_snippet_text(safe)

    @staticmethod
    def _is_unsafe_evidence_sentence(sentence: str) -> bool:
        return any(
            pattern.search(sentence) for pattern in UNSAFE_EVIDENCE_ACTION_PATTERNS
        )

    @staticmethod
    def _is_broken_leading_fragment(sentence: str) -> bool:
        stripped = sentence.strip()
        if not stripped:
            return False
        first_char = stripped[0]
        first_word = stripped.split(" ", 1)[0]
        return first_char.islower() and len(first_word) <= 5

    @staticmethod
    def _limit_complete_sentences(sentences: list[str], max_length: int) -> str:
        selected: list[str] = []
        current_length = 0
        for sentence in sentences:
            sentence = _normalize_snippet_text(sentence)
            if not sentence:
                continue
            next_length = current_length + len(sentence) + (1 if selected else 0)
            if next_length > max_length:
                break
            selected.append(sentence)
            current_length = next_length
            if len(selected) == 2:
                break

        if selected:
            return _normalize_snippet_text(" ".join(selected))

        if not sentences:
            return ""
        first = _normalize_snippet_text(sentences[0])
        if len(first) <= max_length:
            return first
        return _clean_text(first, max_length=max_length)

    @staticmethod
    def _clean_evidence_snippet(text: Any, max_length: int = 300) -> str:
        value = unicodedata.normalize("NFC", str(text or ""))
        if "Nội dung:" in value:
            value = value.rsplit("Nội dung:", 1)[-1]
        metadata_labels = (
            "Hoạt chất:",
            "Slug:",
            "Mục:",
            "Nguồn:",
            "URL:",
            "Nội dung:",
        )
        lines = []
        for line in value.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(label) for label in metadata_labels):
                continue
            lines.append(stripped)
        cleaned = " ".join(lines)
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        cleaned = _normalize_snippet_text(cleaned)
        if not cleaned:
            return ""

        raw_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if sentence.strip()
        ]
        safe_sentences: list[str] = []
        for index, sentence in enumerate(raw_sentences):
            if index == 0 and ChatQueryService._is_broken_leading_fragment(sentence):
                continue
            if ChatQueryService._is_unsafe_evidence_sentence(sentence):
                continue
            safe_sentences.append(sentence)

        return ChatQueryService._limit_complete_sentences(
            safe_sentences,
            max_length=max_length,
        )

    @classmethod
    def _summary_from_chunks(
        cls,
        chunks: list[dict[str, Any]],
        topic: str | None,
        query_type: str,
    ) -> str:
        snippets: list[str] = []
        removed_unsafe_text = ""
        for chunk in chunks:
            raw_text = str(chunk.get("text") or "")
            if any(
                pattern.search(raw_text)
                for pattern in UNSAFE_EVIDENCE_ACTION_PATTERNS
            ):
                removed_unsafe_text = f"{removed_unsafe_text} {raw_text}".strip()
            text = cls._clean_evidence_snippet(chunk.get("text"))
            if text:
                snippets.append(text)
            if len(snippets) == 2:
                break
        if not snippets:
            if topic == "adverse_effect" and cls._mentions_paracetamol_skin_reaction(
                chunks,
                removed_unsafe_text,
            ):
                return PARACETAMOL_SKIN_REACTION_SUMMARY
            return FALLBACK_TOPIC_SUMMARIES.get(
                topic or query_type,
                "Nguồn tham khảo có thông tin liên quan; bác sĩ/dược sĩ nên đối chiếu với tình trạng người bệnh và nguồn tham khảo.",
            )
        return " ".join(snippets)

    @staticmethod
    def _mentions_paracetamol_skin_reaction(
        chunks: list[dict[str, Any]],
        unsafe_text: str,
    ) -> bool:
        haystack_parts = [unsafe_text]
        for chunk in chunks:
            haystack_parts.extend(
                str(chunk.get(key) or "")
                for key in ("slug", "entity_name", "title", "section_title", "text")
            )
        haystack = " ".join(haystack_parts).casefold()
        has_paracetamol = "paracetamol" in haystack
        has_skin_reaction = any(
            term in haystack
            for term in (
                "ban",
                "da",
                "stevens-johnson",
                "hoại tử biểu bì",
                "phản ứng da",
            )
        )
        return has_paracetamol and has_skin_reaction

    @staticmethod
    def _direct_ingredient_output(
        mention: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        best_match = evidence.get("best_match")
        best_match = best_match if isinstance(best_match, dict) else {}
        return {
            "name": mention,
            "normalized_name": best_match.get("evidence_slug") or mention.lower(),
            "strength_raw": None,
            "strength_value": None,
            "strength_unit": None,
            "evidence_status": evidence.get("status", "unresolved"),
            "evidence_name": best_match.get("evidence_name"),
            "evidence_slug": best_match.get("evidence_slug"),
            "evidence_match_type": best_match.get("match_type"),
            "evidence_score": best_match.get("score"),
            "evidence_requires_review": bool(evidence.get("requires_review")),
            "evidence_warnings": list(evidence.get("warnings") or []),
        }

    @staticmethod
    def _medication_has_resolved_evidence(medication: dict[str, Any]) -> bool:
        ingredients = medication.get("active_ingredients")
        if not isinstance(ingredients, list):
            return False
        return any(
            isinstance(ingredient, dict)
            and ingredient.get("evidence_status") == "resolved"
            and ingredient.get("evidence_slug")
            for ingredient in ingredients
        )

    @staticmethod
    def _recompute_evidence_slugs(normalized: dict[str, Any]) -> None:
        slugs: list[str] = []
        for medication in normalized.get("medications") or []:
            if not isinstance(medication, dict):
                continue
            for ingredient in medication.get("active_ingredients") or []:
                if not isinstance(ingredient, dict):
                    continue
                if ingredient.get("evidence_status") != "resolved":
                    continue
                slug = ingredient.get("evidence_slug")
                if slug:
                    slugs.append(str(slug))
        normalized["resolved_evidence_slugs"] = slugs
        normalized["unique_evidence_slugs"] = sorted(set(slugs))

    def _prefer_direct_ingredient_mentions(
        self,
        normalized: dict[str, Any],
        mentions: list[str],
    ) -> dict[str, Any]:
        medications = normalized.get("medications")
        if not isinstance(medications, list):
            return normalized

        for index, mention in enumerate(mentions):
            if index >= len(medications) or not isinstance(medications[index], dict):
                continue
            medication = medications[index]
            if self._medication_has_resolved_evidence(medication):
                continue

            evidence = self.ingredient_resolver.resolve(mention, top_k=5)
            if evidence.get("status") != "resolved":
                continue

            medication.update(
                {
                    "mapping_status": (
                        medication.get("mapping_status") or "ingredient_only"
                    ),
                    "mapping_match_type": (
                        medication.get("mapping_match_type")
                        or "chat_direct_ingredient"
                    ),
                    "active_ingredients": [
                        self._direct_ingredient_output(mention, evidence)
                    ],
                    "requires_review": bool(evidence.get("requires_review")),
                    "warnings": _deduplicate(
                        [
                            *list(medication.get("warnings") or []),
                            "chat_direct_ingredient_preferred",
                        ]
                    ),
                }
            )

        self._recompute_evidence_slugs(normalized)
        return normalized

    @staticmethod
    def _response(
        classification: IntentClassification,
        answer: str,
        normalized_result: dict[str, Any] | None = None,
        sources: list[dict[str, Any]] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "message": answer,
            "answer": answer,
            "intent": classification.intent,
            "normalized_drugs": (
                list((normalized_result or {}).get("medications") or [])
            ),
            "sources": sources or [],
            "warnings": _deduplicate([*classification.warnings, *(warnings or [])]),
        }

    def answer(self, request: ChatRequest) -> dict[str, Any]:
        classification = self.intent_router.classify(request.message)

        if classification.intent == OUT_OF_SCOPE:
            return self._response(
                classification,
                self._sanitize_answer(OUT_OF_SCOPE_REFUSAL),
            )

        if classification.intent == DRUG_INTERACTION_QUERY:
            return self._answer_interaction(request.message, classification)

        if classification.intent == SINGLE_DRUG_QUERY:
            return self._answer_single_drug(classification)

        unknown_answer = self._sanitize_answer(OUT_OF_SCOPE_REFUSAL)
        return self._response(classification, unknown_answer)

    def _answer_interaction(
        self, message: str, classification: IntentClassification
    ) -> dict[str, Any]:
        mentions = classification.drug_mentions[:2]
        if len(mentions) < 2:
            return self._response(
                classification,
                self._sanitize_answer(INTERACTION_MISSING_DRUGS_ANSWER),
                warnings=["missing_interaction_drug_mentions"],
            )

        normalized = self._prefer_direct_ingredient_mentions(
            self.normalizer.normalize_many(mentions, top_k=5),
            mentions,
        )
        query_text = f"{mentions[0]} {mentions[1]} tương tác thuốc"
        evidence = self.retriever.retrieve_for_normalized_result(
            normalized,
            query_type="interaction",
            query_text=query_text,
            top_k=self.top_k,
        )
        return self._compose_evidence_answer(
            classification=classification,
            normalized=normalized,
            evidence=evidence,
            fallback_drugs=mentions,
            query_text=message,
        )

    def _answer_single_drug(
        self, classification: IntentClassification
    ) -> dict[str, Any]:
        mentions = classification.drug_mentions[:1]
        if not mentions:
            return self._response(
                classification,
                self._sanitize_answer(SINGLE_DRUG_MISSING_ANSWER),
                warnings=["missing_single_drug_mention"],
            )

        normalized = self._prefer_direct_ingredient_mentions(
            self.normalizer.normalize_many(mentions, top_k=5),
            mentions,
        )
        topic = classification.topic or "general"
        query_type = TOPIC_QUERY_TYPES.get(topic, "general")
        query_text = SINGLE_QUERY_TEXT.get(topic, SINGLE_QUERY_TEXT["general"]).format(
            drug=mentions[0]
        )
        evidence = self.retriever.retrieve_for_normalized_result(
            normalized,
            query_type=query_type,
            query_text=query_text,
            top_k=self.top_k,
        )
        return self._compose_evidence_answer(
            classification=classification,
            normalized=normalized,
            evidence=evidence,
            fallback_drugs=mentions,
            query_text=query_text,
        )

    def _compose_evidence_answer(
        self,
        classification: IntentClassification,
        normalized: dict[str, Any],
        evidence: dict[str, Any],
        fallback_drugs: list[str],
        query_text: str,
    ) -> dict[str, Any]:
        chunks = [
            chunk
            for chunk in evidence.get("chunks", [])
            if isinstance(chunk, dict) and chunk.get("text")
        ]
        warnings = list(evidence.get("warnings") or [])
        sources = _public_sources(chunks)

        if not chunks:
            return self._response(
                classification,
                self._sanitize_answer(INSUFFICIENT_EVIDENCE_ANSWER),
                normalized_result=normalized,
                sources=sources,
                warnings=[*warnings, "insufficient_evidence"],
            )

        query_type = "interaction" if classification.intent == DRUG_INTERACTION_QUERY else (
            TOPIC_QUERY_TYPES.get(classification.topic or "general", "general")
        )
        summary = self._summary_from_chunks(
            chunks,
            topic=classification.topic,
            query_type=query_type,
        )
        if classification.intent == DRUG_INTERACTION_QUERY:
            drug_a, drug_b = fallback_drugs[:2]
            answer = (
                "Với dữ liệu hiện có, hệ thống ghi nhận cần rà soát khả năng "
                f"tương tác giữa {drug_a} và {drug_b}. {summary}. "
                "Bác sĩ/dược sĩ nên đối chiếu với tình trạng người bệnh và "
                "nguồn tham khảo trước khi quyết định lâm sàng."
            )
        else:
            drug = fallback_drugs[0]
            topic = TOPIC_LABELS.get(classification.topic, TOPIC_LABELS["general"])
            answer = (
                "Với dữ liệu hiện có, hệ thống ghi nhận một số thông tin cần "
                f"lưu ý về {drug} liên quan đến {topic}. {summary}. "
                "Bác sĩ/dược sĩ nên đối chiếu với tình trạng người bệnh và "
                "nguồn tham khảo trước khi quyết định lâm sàng."
            )

        return self._response(
            classification,
            self._sanitize_answer(answer),
            normalized_result=normalized,
            sources=sources,
            warnings=warnings,
        )
