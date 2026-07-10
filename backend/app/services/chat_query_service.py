from __future__ import annotations

import re
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
from backend.app.services.gemini_chat_answer_client import GeminiChatAnswerClient
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

GEMINI_CHAT_ANSWER_FAILED = "gemini_chat_answer_failed"

GEMINI_CHAT_FAILURE_ANSWER = (
    "Hiện hệ thống chưa thể sinh câu trả lời AI từ bằng chứng đã truy xuất. "
    "Bác sĩ/dược sĩ nên kiểm tra lại nguồn tham khảo."
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
    "interaction": "interaction",
    "general": "general",
}

SINGLE_QUERY_TEXT = {
    "adverse_effect": "{drug} tác dụng không mong muốn tác dụng phụ",
    "contraindication": "{drug} chống chỉ định",
    "caution": "{drug} thận trọng lưu ý",
    "dosage": "{drug} liều lượng cách dùng",
    "pregnancy_lactation": "{drug} thai kỳ cho con bú",
    "renal": "{drug} suy thận chức năng thận",
    "hepatic": "{drug} suy gan chức năng gan",
    "interaction": "{drug} tương tác thuốc",
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

def _deduplicate(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


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


GEMINI_EVIDENCE_CHUNK_FIELDS = (
    "rank",
    "slug",
    "entity_name",
    "section",
    "section_title",
    "source",
    "title",
    "url",
    "text",
)


def _gemini_evidence_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        sanitized_chunks.append(
            {
                key: chunk.get(key)
                for key in GEMINI_EVIDENCE_CHUNK_FIELDS
                if chunk.get(key) is not None
            }
        )
    return sanitized_chunks


class ChatQueryService:
    """Thin chat orchestration over existing normalization and retrieval services."""

    def __init__(
        self,
        intent_router: IntentRouter | None = None,
        normalizer: NormalizeDrugsService | None = None,
        retriever: QdrantRetrieverService | None = None,
        ingredient_resolver: IngredientEvidenceResolver | None = None,
        answer_client: Any | None = None,
        top_k: int = 5,
    ) -> None:
        self.intent_router = intent_router or IntentRouter()
        self.normalizer = normalizer or NormalizeDrugsService()
        self.retriever = retriever or QdrantRetrieverService()
        self.ingredient_resolver = ingredient_resolver or IngredientEvidenceResolver()
        self.answer_client = answer_client or GeminiChatAnswerClient()
        self.top_k = top_k

    @staticmethod
    def _sanitize_answer(answer: str) -> str:
        safe = sanitize_doctor_report_text(answer)
        for pattern, replacement in FORBIDDEN_REPLACEMENTS:
            safe = re.sub(pattern, replacement, safe, flags=re.IGNORECASE)
        return _normalize_snippet_text(safe)

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

    @staticmethod
    def _build_answer_payload(
        classification: IntentClassification,
        normalized: dict[str, Any],
        chunks: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        fallback_drugs: list[str],
        question: str,
        query_type: str,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "intent": classification.intent,
            "topic": classification.topic,
            "query_type": query_type,
            "drug_mentions": list(fallback_drugs),
            "normalized_drugs": list(normalized.get("medications") or []),
            "evidence_chunks": _gemini_evidence_chunks(chunks),
            "sources": sources,
            "safety_rules": {
                "no_diagnosis": True,
                "no_prescribing": True,
                "use_only_evidence": True,
            },
        }

    @staticmethod
    def _validate_answer_client_result(result: Any) -> tuple[str, list[str]]:
        if not isinstance(result, dict):
            raise ValueError("chat_answer_client_result_invalid")
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("chat_answer_client_answer_invalid")
        warnings = result.get("warnings", [])
        if warnings is None:
            warnings = []
        if not isinstance(warnings, list) or not all(
            isinstance(warning, str) for warning in warnings
        ):
            raise ValueError("chat_answer_client_warnings_invalid")
        return answer.strip(), warnings

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
            return self._answer_single_drug(request.message, classification)

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
        self, message: str, classification: IntentClassification
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
            query_text=message,
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
        payload = self._build_answer_payload(
            classification=classification,
            normalized=normalized,
            chunks=chunks,
            sources=sources,
            fallback_drugs=fallback_drugs,
            question=query_text,
            query_type=query_type,
        )
        try:
            gemini_result = self.answer_client.answer(payload)
            answer, answer_warnings = self._validate_answer_client_result(
                gemini_result
            )
        except Exception:
            answer = GEMINI_CHAT_FAILURE_ANSWER
            answer_warnings = [GEMINI_CHAT_ANSWER_FAILED]

        return self._response(
            classification,
            self._sanitize_answer(answer),
            normalized_result=normalized,
            sources=sources,
            warnings=[*warnings, *answer_warnings],
        )
