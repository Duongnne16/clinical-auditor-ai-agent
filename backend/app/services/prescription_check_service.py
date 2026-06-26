from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from backend.app.services.normalize_drugs_service import NormalizeDrugsService
from backend.app.services.qdrant_retriever_service import QdrantRetrieverService


CLEAR_HEADER_PREFIXES = (
    "bệnh nhân:",
    "benh nhan:",
    "chẩn đoán:",
    "chan doan:",
    "ngày kê đơn:",
    "ngay ke don:",
    "lời dặn:",
    "loi dan:",
)
NUMBERED_MEDICATION_RE = re.compile(r"^\s*\d+\s*(?:[.)-])\s+\S+")


def _fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", str(value or "").casefold())
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return without_marks.replace("đ", "d")


def _is_footer_or_signature_line(line: str) -> bool:
    folded = _fold_text(line).strip(" .")
    if not folded:
        return False
    compact = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    if re.fullmatch(r"ngay(?: \d+)? thang(?: \d+)? nam(?: \d+)?", compact):
        return True
    if compact.startswith("bac si"):
        return True
    return compact in {"ky ten", "chu ky"}


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


class PrescriptionCheckService:
    """Orchestrate normalization and evidence retrieval for prescriptions.

    TODO production:
    - cache SentenceTransformer / embedding model singleton.
    - reuse Qdrant client.
    - avoid loading embedding model per request.
    """

    def __init__(
        self,
        normalizer: Any | None = None,
        retriever: Any | None = None,
    ) -> None:
        self.normalizer = normalizer or NormalizeDrugsService()
        self.retriever = retriever or QdrantRetrieverService()

    @staticmethod
    def _base_output(
        *,
        status: str,
        doctor_id: str | None,
        patient_context: dict[str, Any] | None,
        raw_text: str | None,
        medication_lines: list[str],
        normalized_result: dict[str, Any] | None = None,
        evidence_bundle: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "doctor_id": doctor_id,
            "patient_context": patient_context or {},
            "input": {
                "raw_text": raw_text,
                "line_count": len(medication_lines),
                "medication_lines": medication_lines,
            },
            "normalized_result": normalized_result,
            "evidence_bundle": evidence_bundle,
            "warnings": _deduplicate(warnings or []),
            "errors": _deduplicate(errors or []),
        }

    @staticmethod
    def _is_clear_header(line: str) -> bool:
        normalized = line.casefold()
        return any(
            normalized.startswith(prefix) for prefix in CLEAR_HEADER_PREFIXES
        )

    def extract_medication_lines(self, prescription_text: str) -> list[str]:
        if not isinstance(prescription_text, str):
            return []
        medication_lines: list[str] = []
        for raw_line in prescription_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if self._is_clear_header(line):
                continue
            if _is_footer_or_signature_line(line):
                continue
            medication_lines.append(line)
        return medication_lines

    @staticmethod
    def group_medication_blocks(medication_lines: list[str]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for line in medication_lines:
            if _is_footer_or_signature_line(line):
                continue
            if NUMBERED_MEDICATION_RE.match(line):
                blocks.append({"raw_line": line})
                continue
            if not blocks:
                blocks.append({"raw_line": line})
                continue
            instruction = str(blocks[-1].get("instruction") or "")
            blocks[-1]["instruction"] = (
                f"{instruction}\n{line}".strip() if instruction else line
            )
        return blocks

    @staticmethod
    def _clean_lines(medication_lines: list[str]) -> list[str]:
        return [
            line.strip()
            for line in medication_lines
            if isinstance(line, str) and line.strip()
        ]

    @staticmethod
    def _extract_evidence_slugs(
        normalized_result: dict[str, Any] | None,
    ) -> list[str]:
        if not isinstance(normalized_result, dict):
            return []

        explicit_slugs = normalized_result.get("unique_evidence_slugs")
        if isinstance(explicit_slugs, list):
            return _deduplicate(
                [str(slug) for slug in explicit_slugs if slug]
            )

        slugs: list[str] = []
        medications = normalized_result.get("medications")
        if not isinstance(medications, list):
            return []
        for medication in medications:
            if not isinstance(medication, dict):
                continue
            ingredients = medication.get("active_ingredients")
            if not isinstance(ingredients, list):
                continue
            for ingredient in ingredients:
                if not isinstance(ingredient, dict):
                    continue
                slug = ingredient.get("evidence_slug")
                if slug:
                    slugs.append(str(slug))
        return _deduplicate(slugs)

    @staticmethod
    def _normalizer_warnings(normalized_result: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        top_level = normalized_result.get("warnings")
        if isinstance(top_level, list):
            warnings.extend(str(warning) for warning in top_level if warning)
        medications = normalized_result.get("medications")
        if isinstance(medications, list):
            for medication in medications:
                if not isinstance(medication, dict):
                    continue
                medication_warnings = medication.get("warnings")
                if isinstance(medication_warnings, list):
                    warnings.extend(
                        str(warning)
                        for warning in medication_warnings
                        if warning
                    )
        return warnings

    @staticmethod
    def _requires_review(normalized_result: dict[str, Any]) -> bool:
        if normalized_result.get("requires_review") is True:
            return True
        summary = normalized_result.get("summary")
        if isinstance(summary, dict) and summary.get("requires_review") is True:
            return True
        medications = normalized_result.get("medications")
        if isinstance(medications, list):
            return any(
                isinstance(medication, dict)
                and medication.get("requires_review") is True
                for medication in medications
            )
        return False

    def check_text(
        self,
        prescription_text: str,
        doctor_id: str | None = None,
        patient_context: dict[str, Any] | None = None,
        query_types: list[str] | None = None,
        top_k_per_type: int = 8,
    ) -> dict[str, Any]:
        if top_k_per_type <= 0:
            raise ValueError("top_k_per_type must be greater than 0")
        if not isinstance(prescription_text, str) or not prescription_text.strip():
            return self._base_output(
                status="invalid_input",
                doctor_id=doctor_id,
                patient_context=patient_context,
                raw_text=prescription_text if isinstance(prescription_text, str) else None,
                medication_lines=[],
                warnings=["empty_prescription_text"],
            )

        medication_lines = self.extract_medication_lines(prescription_text)
        medication_blocks = self.group_medication_blocks(medication_lines)
        return self._check_prepared_lines(
            medications=medication_blocks,
            raw_text=prescription_text,
            doctor_id=doctor_id,
            patient_context=patient_context,
            query_types=query_types,
            top_k_per_type=top_k_per_type,
        )

    def check_lines(
        self,
        medication_lines: list[str],
        doctor_id: str | None = None,
        patient_context: dict[str, Any] | None = None,
        query_types: list[str] | None = None,
        top_k_per_type: int = 8,
    ) -> dict[str, Any]:
        if top_k_per_type <= 0:
            raise ValueError("top_k_per_type must be greater than 0")
        return self._check_prepared_lines(
            medications=self._clean_lines(medication_lines),
            raw_text=None,
            doctor_id=doctor_id,
            patient_context=patient_context,
            query_types=query_types,
            top_k_per_type=top_k_per_type,
        )

    def _check_prepared_lines(
        self,
        *,
        medications: list[dict[str, Any] | str],
        raw_text: str | None,
        doctor_id: str | None,
        patient_context: dict[str, Any] | None,
        query_types: list[str] | None,
        top_k_per_type: int,
    ) -> dict[str, Any]:
        medication_lines = [
            str(medication.get("raw_line") or "").strip()
            if isinstance(medication, dict)
            else str(medication).strip()
            for medication in medications
        ]
        medication_lines = [line for line in medication_lines if line]
        if not medication_lines:
            return self._base_output(
                status="invalid_input",
                doctor_id=doctor_id,
                patient_context=patient_context,
                raw_text=raw_text,
                medication_lines=[],
                warnings=["no_medication_lines_found"],
            )

        try:
            normalized_inputs = [
                medication
                for medication in medications
                if (
                    isinstance(medication, dict)
                    and str(medication.get("raw_line") or "").strip()
                )
                or (isinstance(medication, str) and medication.strip())
            ]
            normalized_inputs = [
                medication
                if isinstance(medication, dict)
                else {"raw_line": str(medication).strip()}
                for medication in normalized_inputs
            ]
            normalized_result = self.normalizer.normalize_many(normalized_inputs)
        except Exception:
            return self._base_output(
                status="error",
                doctor_id=doctor_id,
                patient_context=patient_context,
                raw_text=raw_text,
                medication_lines=medication_lines,
                warnings=[],
                errors=["normalization_failed"],
            )

        warnings = self._normalizer_warnings(normalized_result)
        if self._requires_review(normalized_result):
            warnings.append("some_medications_require_review")

        evidence_slugs = self._extract_evidence_slugs(normalized_result)
        if not evidence_slugs:
            warnings.append("no_evidence_slugs_available")
            return self._base_output(
                status="insufficient_information",
                doctor_id=doctor_id,
                patient_context=patient_context,
                raw_text=raw_text,
                medication_lines=medication_lines,
                normalized_result=normalized_result,
                evidence_bundle=None,
                warnings=warnings,
            )

        try:
            evidence_bundle = self.retriever.build_prescription_evidence_bundle(
                normalized_result=normalized_result,
                query_types=query_types,
                top_k_per_type=top_k_per_type,
            )
        except Exception:
            return self._base_output(
                status="evidence_retrieval_failed",
                doctor_id=doctor_id,
                patient_context=patient_context,
                raw_text=raw_text,
                medication_lines=medication_lines,
                normalized_result=normalized_result,
                evidence_bundle=None,
                warnings=warnings,
                errors=["evidence_retrieval_failed"],
            )

        bundle_warnings = evidence_bundle.get("warnings")
        if isinstance(bundle_warnings, list):
            warnings.extend(str(warning) for warning in bundle_warnings if warning)

        unique_chunks = evidence_bundle.get("unique_chunks")
        if not unique_chunks:
            warnings.append("no_evidence_chunks_retrieved")
            status = "evidence_unavailable"
        else:
            status = "evidence_ready"

        return self._base_output(
            status=status,
            doctor_id=doctor_id,
            patient_context=patient_context,
            raw_text=raw_text,
            medication_lines=medication_lines,
            normalized_result=normalized_result,
            evidence_bundle=evidence_bundle,
            warnings=warnings,
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "service": "PrescriptionCheckService",
            "normalizer": self.normalizer.get_stats(),
            "retriever": self.retriever.get_stats(),
        }
