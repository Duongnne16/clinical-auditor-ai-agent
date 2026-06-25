from __future__ import annotations

from copy import deepcopy
import re
import unicodedata
from typing import Any, Iterable


ERROR_STATUSES = {"analysis_failed", "error", "failed"}
RISK_LEVEL_PRIORITY = {"unknown": 0, "low": 1, "moderate": 2, "high": 3}

GENERIC_SAFE_RECOMMENDATION = (
    "Bác sĩ/dược sĩ cần rà soát lại chỉ định, liều dùng hoặc phối hợp thuốc "
    "dựa trên tình trạng lâm sàng của bệnh nhân và cân nhắc phương án xử trí phù hợp."
)
INTERACTION_SAFE_RECOMMENDATION = (
    "Bác sĩ/dược sĩ cần rà soát lại phối hợp thuốc này, đối chiếu với tình trạng "
    "lâm sàng của bệnh nhân và cân nhắc phương án xử trí phù hợp."
)
CONTRAINDICATION_SAFE_RECOMMENDATION = (
    "Bác sĩ/dược sĩ cần rà soát lại chống chỉ định được nêu trong bằng chứng và "
    "cân nhắc phương án điều trị phù hợp dựa trên tình trạng lâm sàng của bệnh nhân."
)
RENAL_UNCERTAINTY_TEXT = (
    "Thông tin chức năng thận hiện có cần được đối chiếu thêm với bối cảnh lâm sàng "
    "và tiêu chuẩn chuyên môn trước khi kết luận bệnh lý thận."
)
HEPATIC_UNCERTAINTY_TEXT = (
    "Thông tin chức năng gan hiện có cần được đối chiếu thêm với bối cảnh lâm sàng "
    "và tiêu chuẩn chuyên môn trước khi kết luận bệnh lý gan."
)

UNSAFE_RECOMMENDATION_PHRASES = [
    "ngừng thuốc",
    "dừng thuốc",
    "ngưng thuốc",
    "tự ngừng",
    "tự dừng",
    "tự ý ngừng",
    "thay thế bằng",
    "đổi sang",
    "tăng liều",
    "giảm liều",
    "kê thêm",
    "bắt đầu dùng",
    "stop taking",
    "discontinue",
    "switch to",
    "replace with",
    "increase the dose",
    "decrease the dose",
    "start taking",
]
UNSAFE_RECOMMENDATION_REGEXES = [
    r"\b(tam\s+)?(ngung|dung)\b",
    r"\b(thay\s+the|doi\s+sang)\b",
    r"\b(tang\s+lieu|giam\s+lieu|ke\s+them|bat\s+dau\s+dung)\b",
    r"\b(stop\s+taking|discontinue|switch\s+to)\b",
    r"\breplace\b.+\bwith\b",
    r"\b(increase\s+the\s+dose|decrease\s+the\s+dose|start\s+taking)\b",
]

RENAL_DIAGNOSIS_TERMS = [
    "kidney disease",
    "chronic kidney disease",
    "ckd",
    "renal impairment",
    "renal disease",
    "suy thận",
    "bệnh thận",
]
HEPATIC_DIAGNOSIS_TERMS = [
    "liver disease",
    "hepatic impairment",
    "hepatic disease",
    "suy gan",
    "bệnh gan",
]

RENAL_UNSUPPORTED_PHRASES = [
    "được coi là suy thận nhẹ",
    "bệnh nhân bị suy thận",
    "bệnh nhân suy thận",
    "bệnh nhân mắc bệnh thận",
    "is considered mild renal impairment",
    "has renal impairment",
    "has kidney disease",
]
HEPATIC_UNSUPPORTED_PHRASES = [
    "bệnh nhân bị suy gan",
    "bệnh nhân suy gan",
    "bệnh nhân mắc bệnh gan",
    "has hepatic impairment",
    "has liver disease",
]
MILD_RENAL_INFERENCE_TERMS = [
    "suy thận nhẹ",
    "mild renal impairment",
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


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value or "")


def _fold_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", _text(value).casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _contains_any(text: Any, phrases: Iterable[str]) -> bool:
    folded = _fold_text(text)
    return any(_fold_text(phrase) in folded for phrase in phrases)


def _matches_any_regex(text: Any, patterns: Iterable[str]) -> bool:
    folded = _fold_text(text)
    return any(re.search(pattern, folded, flags=re.IGNORECASE) for pattern in patterns)


def _has_unsafe_recommendation(text: Any) -> bool:
    return _contains_any(text, UNSAFE_RECOMMENDATION_PHRASES) or _matches_any_regex(
        text, UNSAFE_RECOMMENDATION_REGEXES
    )


def _risk_items(value: Any) -> list[dict[str, Any]]:
    return [item for item in _as_list(value) if isinstance(item, dict)]


class SafetyLayerService:
    """Deterministic safety sanitizer for structured risk analysis."""

    @staticmethod
    def _valid_evidence_refs(
        risk_analysis: dict[str, Any],
        evidence_bundle: dict[str, Any] | None,
    ) -> set[str]:
        refs: list[str] = []
        context = risk_analysis.get("evidence_context")
        if isinstance(context, dict):
            refs.extend(
                str(ref)
                for ref in _as_list(context.get("valid_evidence_refs"))
                if ref
            )

        if not refs and isinstance(evidence_bundle, dict):
            for key in ("unique_chunks", "all_chunks"):
                for chunk in _as_list(evidence_bundle.get(key)):
                    if isinstance(chunk, dict) and chunk.get("chunk_id"):
                        refs.append(str(chunk["chunk_id"]))
                if refs:
                    break
        return set(refs)

    @staticmethod
    def _diagnoses_text(patient_context: dict[str, Any] | None) -> str:
        if not isinstance(patient_context, dict):
            return ""
        diagnoses = patient_context.get("diagnoses")
        if isinstance(diagnoses, list):
            return " ".join(str(item) for item in diagnoses if item)
        return _text(diagnoses)

    @classmethod
    def _has_supported_renal_diagnosis(
        cls, patient_context: dict[str, Any] | None
    ) -> bool:
        return _contains_any(cls._diagnoses_text(patient_context), RENAL_DIAGNOSIS_TERMS)

    @classmethod
    def _has_supported_hepatic_diagnosis(
        cls, patient_context: dict[str, Any] | None
    ) -> bool:
        return _contains_any(cls._diagnoses_text(patient_context), HEPATIC_DIAGNOSIS_TERMS)

    @staticmethod
    def _egfr_value(patient_context: dict[str, Any] | None) -> float | None:
        if not isinstance(patient_context, dict):
            return None
        renal_function = _text(patient_context.get("renal_function"))
        match = re.search(
            r"egfr\s*(?:[<≤]\s*)?(\d+(?:[.,]\d+)?)",
            renal_function,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return None

    @classmethod
    def _has_egfr_below_30(cls, patient_context: dict[str, Any] | None) -> bool:
        if not isinstance(patient_context, dict):
            return False
        renal_text = _fold_text(patient_context.get("renal_function"))
        if re.search(r"egfr\s*[<≤]\s*30", renal_text):
            return True
        value = cls._egfr_value(patient_context)
        return value is not None and value < 30

    @staticmethod
    def _risk_text(item: dict[str, Any]) -> str:
        return " ".join(
            _text(item.get(field))
            for field in ("title", "explanation", "recommendation")
        )

    @classmethod
    def _is_metformin_egfr_contraindication(
        cls,
        item: dict[str, Any],
        patient_context: dict[str, Any] | None,
    ) -> bool:
        if str(item.get("severity", "")).casefold() != "high":
            return False
        if not cls._has_egfr_below_30(patient_context):
            return False

        text = _fold_text(cls._risk_text(item))
        affected_slugs_text = _fold_text(" ".join(str(slug) for slug in _as_list(item.get("affected_slugs"))))
        evidence_refs_text = _fold_text(" ".join(str(ref) for ref in _as_list(item.get("evidence_refs"))))
        risk_type = str(item.get("risk_type", "")).casefold()

        has_metformin = (
            "metformin" in text
            or "metformin" in affected_slugs_text
            or "metformin" in evidence_refs_text
        )
        has_contraindication = (
            risk_type == "contraindication"
            or "contraindication" in text
            or "chong chi dinh" in text
            or "chong_chi_dinh" in evidence_refs_text
        )
        return has_metformin and has_contraindication

    @staticmethod
    def _safe_recommendation(risk_type: Any) -> str:
        if risk_type == "interaction":
            return INTERACTION_SAFE_RECOMMENDATION
        if risk_type == "contraindication":
            return CONTRAINDICATION_SAFE_RECOMMENDATION
        return GENERIC_SAFE_RECOMMENDATION

    @classmethod
    def _rewrite_unsafe_recommendation(
        cls, item: dict[str, Any], warnings: list[str]
    ) -> None:
        recommendation = item.get("recommendation")
        if not recommendation:
            return
        if not _has_unsafe_recommendation(recommendation):
            return
        item["recommendation"] = cls._safe_recommendation(item.get("risk_type"))
        warnings.append("safety_rewrote_unsafe_recommendation")

    @classmethod
    def _handle_unsupported_diagnosis(
        cls,
        item: dict[str, Any],
        patient_context: dict[str, Any] | None,
        warnings: list[str],
    ) -> dict[str, Any] | None:
        combined_text = cls._risk_text(item)
        renal_claim = _contains_any(combined_text, RENAL_UNSUPPORTED_PHRASES)
        hepatic_claim = _contains_any(combined_text, HEPATIC_UNSUPPORTED_PHRASES)
        if not renal_claim and not hepatic_claim:
            return item

        if renal_claim:
            if cls._has_supported_renal_diagnosis(
                patient_context
            ) or cls._is_metformin_egfr_contraindication(item, patient_context):
                return item
            if (
                str(item.get("severity", "")).casefold() == "low"
                and _contains_any(combined_text, MILD_RENAL_INFERENCE_TERMS)
            ):
                warnings.append("safety_removed_unsupported_diagnosis_inference")
                return None
            cls._rewrite_diagnosis_wording(item, RENAL_UNSUPPORTED_PHRASES, RENAL_UNCERTAINTY_TEXT)
            warnings.append("safety_rewrote_unsupported_diagnosis_wording")

        if hepatic_claim:
            if cls._has_supported_hepatic_diagnosis(patient_context):
                return item
            cls._rewrite_diagnosis_wording(
                item, HEPATIC_UNSUPPORTED_PHRASES, HEPATIC_UNCERTAINTY_TEXT
            )
            warnings.append("safety_rewrote_unsupported_diagnosis_wording")
        return item

    @staticmethod
    def _rewrite_diagnosis_wording(
        item: dict[str, Any],
        phrases: list[str],
        replacement: str,
    ) -> None:
        for field in ("title", "explanation", "recommendation"):
            value = item.get(field)
            if not value or not _contains_any(value, phrases):
                continue
            item[field] = replacement

    @staticmethod
    def _normalized_has_unresolved(normalized_result: dict[str, Any]) -> bool:
        for key in ("unresolved_ingredients", "unmapped_medications"):
            value = normalized_result.get(key)
            if isinstance(value, list) and value:
                return True
            if isinstance(value, int) and value > 0:
                return True

        summary = normalized_result.get("summary")
        if isinstance(summary, dict):
            for key in ("unmatched_medications", "unresolved_ingredients"):
                value = summary.get(key)
                if isinstance(value, list) and value:
                    return True
                if isinstance(value, int) and value > 0:
                    return True

        for medication in _as_list(normalized_result.get("medications")):
            if not isinstance(medication, dict):
                continue
            if medication.get("mapping_status") == "unmatched":
                return True
        return False

    @staticmethod
    def _normalized_requires_review(normalized_result: dict[str, Any]) -> bool:
        if normalized_result.get("requires_review") is True:
            return True
        summary = normalized_result.get("summary")
        if isinstance(summary, dict) and summary.get("requires_review") is True:
            return True
        for medication in _as_list(normalized_result.get("medications")):
            if isinstance(medication, dict) and medication.get("requires_review") is True:
                return True
        return False

    @staticmethod
    def _recomputed_level(risk_items: list[dict[str, Any]]) -> str:
        highest = "unknown"
        for item in risk_items:
            severity = str(item.get("severity", "unknown")).casefold()
            if RISK_LEVEL_PRIORITY.get(severity, 0) > RISK_LEVEL_PRIORITY[highest]:
                highest = severity
        return highest

    def apply(
        self,
        risk_analysis: dict[str, Any],
        normalized_result: dict[str, Any],
        evidence_bundle: dict[str, Any] | None,
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = deepcopy(risk_analysis) if isinstance(risk_analysis, dict) else {}
        normalized = deepcopy(normalized_result) if isinstance(normalized_result, dict) else {}
        warnings = _deduplicate(
            [str(warning) for warning in _as_list(result.get("warnings")) if warning]
        )
        errors = _deduplicate(
            [str(error) for error in _as_list(result.get("errors")) if error]
        )
        status = str(result.get("status", "unknown"))
        original_level = str(result.get("overall_risk_level", "unknown"))
        valid_refs = self._valid_evidence_refs(result, evidence_bundle)

        if self._normalized_has_unresolved(normalized):
            warnings.append("safety_unresolved_medications")
        if self._normalized_requires_review(normalized):
            warnings.append("safety_mapping_requires_review")

        if not valid_refs:
            result["status"] = status if status in ERROR_STATUSES else "insufficient_evidence"
            result["overall_risk_level"] = "unknown"
            result["risk_items"] = []
            result["missing_information"] = [
                str(item) for item in _as_list(result.get("missing_information")) if item
            ]
            warnings.append("safety_no_valid_evidence")
            result["warnings"] = _deduplicate(warnings)
            result["errors"] = errors
            return result

        kept_items: list[dict[str, Any]] = []
        for item in _risk_items(result.get("risk_items")):
            refs = [str(ref) for ref in _as_list(item.get("evidence_refs")) if ref]
            filtered_refs = _deduplicate([ref for ref in refs if ref in valid_refs])
            if not filtered_refs:
                warnings.append("safety_removed_risk_item_without_evidence")
                continue
            item["evidence_refs"] = filtered_refs
            self._rewrite_unsafe_recommendation(item, warnings)
            item = self._handle_unsupported_diagnosis(item, patient_context, warnings)
            if item is None:
                continue
            kept_items.append(item)

        result["risk_items"] = kept_items
        recomputed = self._recomputed_level(kept_items)
        result["overall_risk_level"] = recomputed
        if recomputed != original_level:
            warnings.append("safety_overall_risk_recomputed")
        result["missing_information"] = [
            str(item) for item in _as_list(result.get("missing_information")) if item
        ]
        result["warnings"] = _deduplicate(warnings)
        result["errors"] = errors
        return result

    def get_stats(self) -> dict[str, Any]:
        return {
            "service": "SafetyLayerService",
            "deterministic": True,
        }
