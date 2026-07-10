from __future__ import annotations

import re
from typing import Any, Iterable


SAFETY_DISCLAIMER = (
    "Báo cáo này chỉ nhằm hỗ trợ bác sĩ/dược sĩ rà soát đơn thuốc dựa trên "
    "dữ liệu hiện có và không thay thế quyết định chuyên môn. Bác sĩ/dược sĩ "
    "là người chịu trách nhiệm đánh giá cuối cùng dựa trên tình trạng lâm sàng "
    "thực tế của bệnh nhân."
)

CONTEXT_READY_SUMMARY = (
    "Hệ thống đã chuẩn bị bằng chứng và ngữ cảnh để bác sĩ/dược sĩ rà soát."
)

NO_SUPPORTED_RISK_SUMMARY = (
    "Trong phạm vi dữ liệu đã truy xuất, hệ thống chưa ghi nhận điểm cần rà soát có bằng chứng phù hợp."
)

DOCTOR_FACING_MAPPING_WARNING = (
    "Một số dòng thuốc chưa được hệ thống nhận diện chắc chắn, cần rà soát lại."
)
DECISIVE_REPLACEMENTS = (
    (
        r"(?i)đơn thuốc\s+(?:không\s+)?an toàn",
        "đơn thuốc cần được bác sĩ/dược sĩ rà soát",
    ),
    (r"(?i)không\s+dùng\s+được", "cần rà soát trước khi sử dụng"),
    (r"(?i)dùng\s+được", "cần được đánh giá theo bối cảnh lâm sàng"),
    (r"(?i)ngừng\s+thuốc", "rà soát thuốc"),
    (r"(?i)đổi\s+thuốc", "cân nhắc phương án xử trí phù hợp"),
    (r"(?i)tăng\s+liều", "rà soát liều"),
    (r"(?i)giảm\s+liều", "rà soát liều"),
)
TECHNICAL_MAPPING_WARNINGS = {
    "no_mapping_found",
    "drug_mapping_not_found",
    "drug_or_ingredient_not_found",
    "ingredient_evidence_requires_review",
    "safety_mapping_requires_review",
    "safety_unresolved_medications",
    "some_medications_require_review",
}


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _normalize_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    if max_length <= 1:
        return text[:max_length]
    return text[: max_length - 1].rstrip() + "…"


def _doctor_safe_text(value: Any) -> str:
    text = str(value or "")
    for pattern, replacement in DECISIVE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text


def _doctor_facing_warnings(warnings: list[str]) -> list[str]:
    messages: list[str] = []
    if any(warning in TECHNICAL_MAPPING_WARNINGS for warning in warnings):
        messages.append(DOCTOR_FACING_MAPPING_WARNING)
    return _deduplicate(messages)


def _checked_query_types(evidence_bundle: dict[str, Any] | None) -> list[str]:
    if not isinstance(evidence_bundle, dict):
        return []
    query_results = evidence_bundle.get("query_results")
    if not isinstance(query_results, dict):
        return []
    return _deduplicate(str(query_type) for query_type in query_results if query_type)


class ReportGeneratorService:
    """Format validated prescription-risk analysis into an audit report.

    This service does not infer clinical risk, create new risk items, call an
    LLM, or retrieve evidence. It only formats already-prepared data.
    """

    def __init__(self, max_evidence_snippet_length: int = 400) -> None:
        if max_evidence_snippet_length <= 0:
            raise ValueError("max_evidence_snippet_length must be greater than 0")
        self.max_evidence_snippet_length = max_evidence_snippet_length

    @staticmethod
    def _ingredient_summary(ingredient: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": ingredient.get("name"),
            "evidence_slug": ingredient.get("evidence_slug"),
            "strength_raw": ingredient.get("strength_raw"),
            "strength_value": ingredient.get("strength_value"),
            "strength_unit": ingredient.get("strength_unit"),
        }

    def build_medication_summary(
        self, normalized_result: dict[str, Any]
    ) -> list[dict[str, Any]]:
        medications = normalized_result.get("medications")
        if not isinstance(medications, list):
            return []

        output: list[dict[str, Any]] = []
        for medication in medications:
            if not isinstance(medication, dict):
                continue

            ingredients = medication.get("active_ingredients")
            ingredient_summary: list[dict[str, Any]] = []
            if isinstance(ingredients, list):
                ingredient_summary = [
                    self._ingredient_summary(ingredient)
                    for ingredient in ingredients
                    if isinstance(ingredient, dict)
                ]

            warnings = medication.get("warnings")
            if not isinstance(warnings, list):
                warnings = []

            output.append(
                {
                    "raw_name": medication.get("raw_name"),
                    "raw_line": medication.get("raw_line"),
                    "instruction": medication.get("instruction"),
                    "generic_text": medication.get("generic_text"),
                    "brand_text": medication.get("brand_text"),
                    "mapping_status": medication.get("mapping_status"),
                    "requires_review": medication.get("requires_review") is True,
                    "warnings": [str(warning) for warning in warnings if warning],
                    "active_ingredients": ingredient_summary,
                }
            )
        return output

    def build_evidence_index(
        self, evidence_bundle: dict[str, Any] | None
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(evidence_bundle, dict):
            return {}
        chunks = evidence_bundle.get("unique_chunks")
        if not isinstance(chunks, list):
            return {}

        evidence_index: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            chunk_id_text = str(chunk_id)
            if chunk_id_text in evidence_index:
                continue
            text = chunk.get("text") or chunk.get("content") or ""
            evidence_index[chunk_id_text] = {
                "chunk_id": chunk_id_text,
                "slug": chunk.get("slug"),
                "section": chunk.get("section"),
                "section_title": chunk.get("section_title"),
                "source": chunk.get("source"),
                "source_type": chunk.get("source_type"),
                "url": chunk.get("url"),
                "snippet": _truncate(
                    _normalize_whitespace(text),
                    self.max_evidence_snippet_length,
                ),
            }
        return evidence_index

    def attach_evidence_to_risk_items(
        self,
        risk_items: list[dict[str, Any]],
        evidence_index: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in risk_items:
            if not isinstance(item, dict):
                continue
            refs = item.get("evidence_refs")
            if not isinstance(refs, list):
                refs = []
            evidence = [
                evidence_index[str(ref)]
                for ref in refs
                if ref and str(ref) in evidence_index
            ]
            copied = {
                "risk_type": item.get("risk_type"),
                "severity": item.get("severity"),
                "title": item.get("title"),
                "explanation": item.get("explanation"),
                "affected_slugs": item.get("affected_slugs")
                if isinstance(item.get("affected_slugs"), list)
                else [],
                "evidence_refs": [str(ref) for ref in refs if ref],
                "recommendation": item.get("recommendation"),
                "evidence": evidence,
            }
            output.append(copied)
        return output

    @staticmethod
    def _missing_evidence_ref_warning(
        risk_items: list[dict[str, Any]],
        evidence_index: dict[str, dict[str, Any]],
    ) -> list[str]:
        for item in risk_items:
            if not isinstance(item, dict):
                continue
            refs = item.get("evidence_refs")
            if not isinstance(refs, list):
                continue
            if any(ref and str(ref) not in evidence_index for ref in refs):
                return ["risk_item_evidence_ref_missing_in_bundle"]
        return []

    @staticmethod
    def _medications_requiring_review(
        medication_summary: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            medication
            for medication in medication_summary
            if medication.get("requires_review") is True
            or medication.get("mapping_status") == "unmatched"
        ]

    @staticmethod
    def _base_summary(
        status: str,
        risk_items: list[dict[str, Any]],
        overall_risk_level: str,
    ) -> str:
        if status == "analysis_context_ready":
            return CONTEXT_READY_SUMMARY
        if status == "analysis_ready":
            if risk_items:
                return (
                    "Đã ghi nhận cảnh báo nguy cơ có bằng chứng hợp lệ. "
                    f"Mức nguy cơ tổng quan: {overall_risk_level}."
                )
            return NO_SUPPORTED_RISK_SUMMARY
        if status == "insufficient_evidence":
            return "Chưa đủ bằng chứng để tạo báo cáo phân tích nguy cơ."
        if status == "analysis_failed":
            return "Quá trình phân tích nguy cơ không thành công."
        return "Không thể tạo báo cáo từ kết quả phân tích nguy cơ hiện tại."

    @staticmethod
    def _append_missing_information(summary: str, missing_information: list[str]) -> str:
        if not missing_information:
            return summary
        return (
            f"{summary} Cần bổ sung thông tin: "
            f"{', '.join(missing_information)}."
        )

    def _build_report(
        self,
        *,
        status: str,
        overall_risk_level: str,
        summary: str,
        patient_context: dict[str, Any] | None,
        medication_summary: list[dict[str, Any]],
        medications_requiring_review: list[dict[str, Any]],
        risk_items: list[dict[str, Any]],
        missing_information: list[str],
        evidence_bundle: dict[str, Any] | None,
        evidence_sources: list[dict[str, Any]],
        warnings: list[str],
        errors: list[str],
    ) -> dict[str, Any]:
        safe_risk_items = []
        for item in risk_items:
            copied = dict(item)
            for field in ("title", "explanation", "recommendation"):
                if copied.get(field):
                    copied[field] = _doctor_safe_text(copied[field])
            safe_risk_items.append(copied)

        deduped_warnings = _deduplicate(warnings)
        report = {
            "status": status,
            "report_type": "prescription_audit",
            "audience": "doctor_or_pharmacist",
            "overall_risk_level": overall_risk_level,
            "summary": _doctor_safe_text(summary),
            "patient_context": patient_context or {},
            "medication_summary": medication_summary,
            "medications_requiring_review": medications_requiring_review,
            "risk_items": safe_risk_items,
            "missing_information": missing_information,
            "evidence_sources": evidence_sources,
            "source_count": len(evidence_sources),
            "checked_query_types": _checked_query_types(evidence_bundle),
            "safety_disclaimer": SAFETY_DISCLAIMER,
            "warnings": deduped_warnings,
            "doctor_facing_warnings": _doctor_facing_warnings(deduped_warnings),
            "errors": _deduplicate(errors),
        }
        report["markdown_report"] = self.build_markdown_report(report)
        return report

    def generate_report(
        self,
        normalized_result: dict[str, Any],
        evidence_bundle: dict[str, Any] | None,
        risk_analysis: dict[str, Any],
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence_index = self.build_evidence_index(evidence_bundle)
        evidence_sources = list(evidence_index.values())
        medication_summary = (
            self.build_medication_summary(normalized_result)
            if isinstance(normalized_result, dict)
            else []
        )
        medications_requiring_review = self._medications_requiring_review(
            medication_summary
        )

        if not isinstance(risk_analysis, dict):
            return self._build_report(
                status="cannot_generate_report",
                overall_risk_level="unknown",
                summary="Không thể tạo báo cáo từ kết quả phân tích nguy cơ không hợp lệ.",
                patient_context=patient_context,
                medication_summary=medication_summary,
                medications_requiring_review=medications_requiring_review,
                risk_items=[],
                missing_information=[],
                evidence_bundle=evidence_bundle,
                evidence_sources=evidence_sources,
                warnings=[],
                errors=["invalid_risk_analysis"],
            )

        analysis_status = risk_analysis.get("status")
        missing_information = risk_analysis.get("missing_information")
        if not isinstance(missing_information, list):
            missing_information = []
        missing_information = [
            str(item) for item in missing_information if item
        ]

        risk_warnings = risk_analysis.get("warnings")
        if not isinstance(risk_warnings, list):
            risk_warnings = []
        risk_errors = risk_analysis.get("errors")
        if not isinstance(risk_errors, list):
            risk_errors = []

        overall_risk_level = "unknown"
        report_status = "cannot_generate_report"
        risk_items_with_evidence: list[dict[str, Any]] = []
        warnings = [str(warning) for warning in risk_warnings if warning]
        errors = [str(error) for error in risk_errors if error]

        raw_risk_items = risk_analysis.get("risk_items")
        if not isinstance(raw_risk_items, list):
            raw_risk_items = []

        if analysis_status == "analysis_context_ready":
            report_status = "report_context_ready"
        elif analysis_status == "analysis_ready":
            report_status = "report_ready"
            overall = risk_analysis.get("overall_risk_level") or "unknown"
            overall_risk_level = str(overall)
            risk_items_with_evidence = self.attach_evidence_to_risk_items(
                raw_risk_items, evidence_index
            )
            warnings.extend(
                self._missing_evidence_ref_warning(raw_risk_items, evidence_index)
            )
        elif analysis_status == "insufficient_evidence":
            report_status = "report_insufficient_evidence"
        elif analysis_status == "analysis_failed":
            report_status = "report_analysis_failed"
        else:
            errors.append("unsupported_risk_analysis_status")

        summary = self._base_summary(
            str(analysis_status),
            risk_items_with_evidence,
            overall_risk_level,
        )
        summary = self._append_missing_information(summary, missing_information)

        return self._build_report(
            status=report_status,
            overall_risk_level=overall_risk_level,
            summary=summary,
            patient_context=patient_context,
            medication_summary=medication_summary,
            medications_requiring_review=medications_requiring_review,
            risk_items=risk_items_with_evidence,
            missing_information=missing_information,
            evidence_bundle=evidence_bundle,
            evidence_sources=evidence_sources,
            warnings=warnings,
            errors=errors,
        )

    @staticmethod
    def _display_medication_name(medication: dict[str, Any]) -> str:
        return (
            medication.get("raw_line")
            or medication.get("raw_name")
            or medication.get("generic_text")
            or medication.get("brand_text")
            or "Không rõ tên thuốc"
        )

    def build_markdown_report(self, report: dict[str, Any]) -> str:
        lines = [
            "# Báo cáo kiểm tra đơn thuốc",
            "",
            "## Kết luận tổng quan",
            "",
            f"- Trạng thái: {report.get('status')}",
            f"- Mức nguy cơ tổng quan: {report.get('overall_risk_level')}",
            f"- Tóm tắt: {report.get('summary')}",
            "",
            "## Danh sách thuốc",
            "",
        ]

        medications = report.get("medication_summary")
        if isinstance(medications, list) and medications:
            for medication in medications:
                if not isinstance(medication, dict):
                    continue
                lines.append(f"- {self._display_medication_name(medication)}")
                ingredients = medication.get("active_ingredients")
                if isinstance(ingredients, list) and ingredients:
                    ingredient_text = ", ".join(
                        str(ingredient.get("name"))
                        for ingredient in ingredients
                        if isinstance(ingredient, dict) and ingredient.get("name")
                    )
                    if ingredient_text:
                        lines.append(f"  - Hoạt chất: {ingredient_text}")
                if medication.get("requires_review"):
                    lines.append("  - Cần rà soát mapping/chuẩn hóa.")
        else:
            lines.append("- Không có dữ liệu thuốc.")

        lines.extend(["", "## Cảnh báo cần xem xét", ""])
        risk_items = report.get("risk_items")
        if isinstance(risk_items, list) and risk_items:
            for item in risk_items:
                if not isinstance(item, dict):
                    continue
                title = item.get("title") or "Cảnh báo không có tiêu đề"
                lines.append(
                    f"- [{item.get('severity', 'unknown')}] {title}"
                )
                if item.get("explanation"):
                    lines.append(f"  - Giải thích: {item.get('explanation')}")
                if item.get("recommendation"):
                    lines.append(f"  - Khuyến nghị: {item.get('recommendation')}")
                evidence = item.get("evidence")
                if isinstance(evidence, list) and evidence:
                    refs = ", ".join(
                        str(source.get("chunk_id"))
                        for source in evidence
                        if isinstance(source, dict) and source.get("chunk_id")
                    )
                    if refs:
                        lines.append(f"  - Bằng chứng: {refs}")
        elif report.get("status") == "report_context_ready":
            lines.append(f"- {CONTEXT_READY_SUMMARY}")
        elif report.get("status") == "report_insufficient_evidence":
            lines.append("- Chưa đủ bằng chứng để phân tích nguy cơ trong đơn thuốc.")
        elif report.get("status") == "report_analysis_failed":
            lines.append("- Quá trình phân tích nguy cơ không thành công nên chưa thể tạo cảnh báo.")
        elif report.get("status") == "cannot_generate_report":
            lines.append("- Chưa có kết quả phân tích nguy cơ hợp lệ để tạo cảnh báo.")
        elif report.get("status") == "report_ready":
            lines.append(f"- {NO_SUPPORTED_RISK_SUMMARY}")
        else:
            lines.append("- Chưa có kết quả phân tích nguy cơ hợp lệ để tạo cảnh báo.")

        lines.extend(["", "## Thông tin còn thiếu", ""])
        missing_information = report.get("missing_information")
        if isinstance(missing_information, list) and missing_information:
            for item in missing_information:
                lines.append(f"- {item}")
        else:
            lines.append("- Không ghi nhận thông tin còn thiếu trong kết quả phân tích.")

        lines.extend(["", "## Nguồn bằng chứng", ""])
        evidence_sources = report.get("evidence_sources")
        if isinstance(evidence_sources, list) and evidence_sources:
            for source in evidence_sources:
                if not isinstance(source, dict):
                    continue
                label = source.get("chunk_id") or "unknown"
                section = source.get("section") or source.get("section_title") or ""
                url = source.get("url") or ""
                lines.append(f"- {label} — {source.get('slug')} / {section}")
                if url:
                    lines.append(f"  - URL: {url}")
        else:
            lines.append("- Không có nguồn bằng chứng đi kèm.")

        lines.extend(
            [
                "",
                "## Lưu ý an toàn",
                "",
                str(report.get("safety_disclaimer") or SAFETY_DISCLAIMER),
                "",
            ]
        )
        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        return {
            "service": "ReportGeneratorService",
            "report_type": "prescription_audit",
            "audience": "doctor_or_pharmacist",
            "max_evidence_snippet_length": self.max_evidence_snippet_length,
        }
