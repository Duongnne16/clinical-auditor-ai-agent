from __future__ import annotations

import itertools
import re
from typing import Any, Iterable


VALID_RISK_TYPES = {
    "interaction",
    "contraindication",
    "precaution",
    "pregnancy_lactation",
    "renal_hepatic",
    "adverse_effect",
    "overdose",
    "general",
}

VALID_RISK_LEVELS = {"low", "moderate", "high", "unknown"}

PATIENT_CONTEXT_FIELDS = [
    "age",
    "sex",
    "allergies",
    "pregnancy_status",
    "renal_function",
    "hepatic_function",
    "diagnoses",
    "current_medications",
]


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _clean_snippet(text: Any, max_length: int = 700) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned[:max_length]


class RiskAnalyzerService:
    """Evidence-grounded risk analysis context builder and validator.

    TODO SafetyLayer:
    - block unsafe recommendation wording such as self-stopping medication.
    - block self-increasing dose instructions.
    - block self-changing/replacing medication instructions.
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        max_chunks_per_query_type: int = 5,
    ) -> None:
        if max_chunks_per_query_type <= 0:
            raise ValueError("max_chunks_per_query_type must be greater than 0")
        self.llm_client = llm_client
        self.max_chunks_per_query_type = max_chunks_per_query_type

    @staticmethod
    def _missing_information(patient_context: dict[str, Any] | None) -> list[str]:
        context = patient_context or {}
        return [
            field
            for field in PATIENT_CONTEXT_FIELDS
            if context.get(field) in (None, "", [], {})
        ]

    @staticmethod
    def _medication_context(
        normalized_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        medications = normalized_result.get("medications")
        if not isinstance(medications, list):
            return []

        output: list[dict[str, Any]] = []
        for medication in medications:
            if not isinstance(medication, dict):
                continue
            ingredients = medication.get("active_ingredients")
            ingredient_context: list[dict[str, Any]] = []
            if isinstance(ingredients, list):
                for ingredient in ingredients:
                    if not isinstance(ingredient, dict):
                        continue
                    ingredient_context.append(
                        {
                            "name": ingredient.get("name"),
                            "evidence_slug": ingredient.get("evidence_slug"),
                            "strength_raw": ingredient.get("strength_raw"),
                            "strength_value": ingredient.get("strength_value"),
                            "strength_unit": ingredient.get("strength_unit"),
                        }
                    )

            output.append(
                {
                    "raw_name": medication.get("raw_name"),
                    "raw_line": medication.get("raw_line"),
                    "generic_text": medication.get("generic_text"),
                    "brand_text": medication.get("brand_text"),
                    "mapping_status": medication.get("mapping_status"),
                    "requires_review": medication.get("requires_review"),
                    "warnings": list(medication.get("warnings") or []),
                    "active_ingredients": ingredient_context,
                }
            )
        return output

    @staticmethod
    def _evidence_slugs_from_medications(
        medications: list[dict[str, Any]],
    ) -> list[str]:
        slugs: list[str] = []
        for medication in medications:
            for ingredient in medication.get("active_ingredients", []):
                slug = ingredient.get("evidence_slug")
                if slug:
                    slugs.append(str(slug))
        return _deduplicate(slugs)

    @staticmethod
    def _interaction_candidates(slugs: list[str]) -> list[dict[str, str]]:
        return [
            {"slug_a": slug_a, "slug_b": slug_b}
            for slug_a, slug_b in itertools.combinations(slugs, 2)
        ]

    @staticmethod
    def _chunk_context(chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": chunk.get("chunk_id"),
            "slug": chunk.get("slug"),
            "section": chunk.get("section"),
            "source": chunk.get("source"),
            "url": chunk.get("url"),
            "snippet": _clean_snippet(chunk.get("text")),
        }

    def build_evidence_context(
        self,
        normalized_result: dict[str, Any],
        evidence_bundle: dict[str, Any],
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        medications = self._medication_context(normalized_result)
        slugs = self._evidence_slugs_from_medications(medications)
        evidence_by_query_type: dict[str, list[dict[str, Any]]] = {}
        valid_refs: list[str] = []

        query_results = evidence_bundle.get("query_results") or {}
        if isinstance(query_results, dict):
            for query_type, result in query_results.items():
                chunks = []
                if isinstance(result, dict):
                    chunks = result.get("chunks") or []
                if not isinstance(chunks, list):
                    chunks = []

                contexts: list[dict[str, Any]] = []
                for chunk in chunks[: self.max_chunks_per_query_type]:
                    if not isinstance(chunk, dict):
                        continue
                    context = self._chunk_context(chunk)
                    contexts.append(context)
                    chunk_id = context.get("chunk_id")
                    if chunk_id:
                        valid_refs.append(str(chunk_id))
                evidence_by_query_type[str(query_type)] = contexts

        return {
            "patient_context": patient_context or {},
            "medications": medications,
            "interaction_candidates": self._interaction_candidates(slugs),
            "evidence_by_query_type": evidence_by_query_type,
            "valid_evidence_refs": _deduplicate(valid_refs),
            "missing_information": self._missing_information(patient_context),
        }

    @staticmethod
    def _valid_ref_set(valid_evidence_refs: set[str] | list[str]) -> set[str]:
        return {str(ref) for ref in valid_evidence_refs if ref}

    def validate_llm_analysis(
        self,
        llm_result: dict[str, Any],
        valid_evidence_refs: set[str] | list[str],
    ) -> dict[str, Any]:
        valid_refs = self._valid_ref_set(valid_evidence_refs)
        warnings: list[str] = []
        overall_risk_level = str(
            llm_result.get("overall_risk_level", "unknown")
        )
        if overall_risk_level not in VALID_RISK_LEVELS:
            overall_risk_level = "unknown"
            warnings.append("invalid_overall_risk_level_normalized")

        sanitized_items: list[dict[str, Any]] = []
        risk_items = llm_result.get("risk_items")
        if not isinstance(risk_items, list):
            risk_items = []

        for item in risk_items:
            if not isinstance(item, dict):
                continue

            risk_type = str(item.get("risk_type", "general"))
            if risk_type not in VALID_RISK_TYPES:
                risk_type = "general"
                warnings.append("invalid_risk_type_normalized")

            severity = str(item.get("severity", "unknown"))
            if severity not in VALID_RISK_LEVELS:
                severity = "unknown"
                warnings.append("invalid_severity_normalized")

            refs = item.get("evidence_refs")
            if not isinstance(refs, list):
                refs = []
            refs_as_text = [str(ref) for ref in refs if ref]
            filtered_refs = [ref for ref in refs_as_text if ref in valid_refs]
            if len(filtered_refs) != len(refs_as_text):
                warnings.append("unsupported_evidence_ref_removed")
            filtered_refs = _deduplicate(filtered_refs)
            if not filtered_refs:
                warnings.append("risk_item_removed_due_to_missing_evidence")
                continue

            affected_slugs = item.get("affected_slugs")
            if not isinstance(affected_slugs, list):
                affected_slugs = []

            sanitized_items.append(
                {
                    "risk_type": risk_type,
                    "severity": severity,
                    "title": item.get("title"),
                    "explanation": item.get("explanation"),
                    "affected_slugs": [
                        str(slug) for slug in affected_slugs if slug
                    ],
                    "evidence_refs": filtered_refs,
                    "recommendation": item.get("recommendation"),
                }
            )

        missing_information = llm_result.get("missing_information")
        if not isinstance(missing_information, list):
            missing_information = []

        original_overall_risk_level = llm_result.get(
            "overall_risk_level", "unknown"
        )
        if not sanitized_items and (
            risk_items or original_overall_risk_level != "unknown"
        ):
            overall_risk_level = "unknown"
            warnings.append(
                "overall_risk_level_reset_due_to_no_supported_risk_items"
            )

        return {
            "overall_risk_level": overall_risk_level,
            "risk_items": sanitized_items,
            "missing_information": [
                str(item) for item in missing_information if item
            ],
            "warnings": _deduplicate(warnings),
            "errors": [],
        }

    @staticmethod
    def _has_evidence(evidence_bundle: dict[str, Any] | None) -> bool:
        if not isinstance(evidence_bundle, dict):
            return False
        unique_chunks = evidence_bundle.get("unique_chunks")
        return isinstance(unique_chunks, list) and bool(unique_chunks)

    def _call_llm(self, evidence_context: dict[str, Any]) -> dict[str, Any]:
        if hasattr(self.llm_client, "analyze_risks"):
            result = self.llm_client.analyze_risks(evidence_context)
        else:
            result = self.llm_client(evidence_context)
        if not isinstance(result, dict):
            return {}
        return result

    def analyze(
        self,
        normalized_result: dict[str, Any],
        evidence_bundle: dict[str, Any] | None,
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._has_evidence(evidence_bundle):
            return {
                "status": "insufficient_evidence",
                "overall_risk_level": "unknown",
                "risk_items": [],
                "evidence_context": None,
                "missing_information": self._missing_information(patient_context),
                "warnings": ["no_evidence_available_for_analysis"],
                "errors": [],
            }

        evidence_context = self.build_evidence_context(
            normalized_result=normalized_result,
            evidence_bundle=evidence_bundle or {},
            patient_context=patient_context,
        )

        if self.llm_client is None:
            return {
                "status": "analysis_context_ready",
                "overall_risk_level": "unknown",
                "risk_items": [],
                "evidence_context": evidence_context,
                "missing_information": evidence_context["missing_information"],
                "warnings": ["analysis_not_run_without_llm"],
                "errors": [],
            }

        try:
            llm_result = self._call_llm(evidence_context)
            validation = self.validate_llm_analysis(
                llm_result,
                set(evidence_context["valid_evidence_refs"]),
            )
        except Exception:
            return {
                "status": "analysis_failed",
                "overall_risk_level": "unknown",
                "risk_items": [],
                "evidence_context": evidence_context,
                "missing_information": evidence_context["missing_information"],
                "warnings": [],
                "errors": ["risk_analysis_failed"],
            }

        return {
            "status": "analysis_ready",
            "overall_risk_level": validation["overall_risk_level"],
            "risk_items": validation["risk_items"],
            "evidence_context": evidence_context,
            "missing_information": _deduplicate(
                [
                    *evidence_context["missing_information"],
                    *validation["missing_information"],
                ]
            ),
            "warnings": validation["warnings"],
            "errors": validation["errors"],
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "service": "RiskAnalyzerService",
            "llm_enabled": self.llm_client is not None,
            "max_chunks_per_query_type": self.max_chunks_per_query_type,
            "valid_risk_types": sorted(VALID_RISK_TYPES),
            "valid_risk_levels": sorted(VALID_RISK_LEVELS),
        }
