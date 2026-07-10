from __future__ import annotations

from typing import Any, Callable, Iterable

from backend.app.services.doctor_memory_service import (
    DOCTOR_MEMORY_LABEL,
    DoctorMemoryService,
)
from backend.app.core.config import get_settings
from backend.app.services.doctor_report_text_safety import sanitize_doctor_report_text
from backend.app.services.doctor_report_composer_service import (
    DoctorReportComposerService,
    has_full_doctor_facing_sections,
    render_doctor_facing_response_from_sections,
)
from backend.app.services.gemini_risk_analyzer_client import GeminiRiskAnalyzerClient
from backend.app.services.prescription_document_parser import PrescriptionDocumentParser
from backend.app.services.prescription_check_service import PrescriptionCheckService
from backend.app.services.report_generator_service import ReportGeneratorService
from backend.app.services.risk_analyzer_service import RiskAnalyzerService
from backend.app.services.safety_layer_service import SafetyLayerService


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _list_from_result(result: dict[str, Any] | None, key: str) -> list[str]:
    if not isinstance(result, dict):
        return []
    values = result.get(key)
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if value]


def _empty_doctor_memory() -> dict[str, list[dict[str, Any]]]:
    return {"matched_notes": []}


class PrescriptionAuditService:
    """Orchestrate prescription check, risk analysis, and report generation."""

    def __init__(
        self,
        prescription_check_service: Any | None = None,
        report_generator_service: Any | None = None,
        doctor_report_composer_service: Any | None = None,
        safety_layer_service: Any | None = None,
        prescription_document_parser: Any | None = None,
        doctor_memory_service: Any | None = None,
        risk_analyzer_service_factory: Callable[..., Any] | None = None,
        gemini_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.prescription_check_service = (
            prescription_check_service or PrescriptionCheckService()
        )
        self.report_generator_service = (
            report_generator_service or ReportGeneratorService()
        )
        self.doctor_report_composer_service = (
            doctor_report_composer_service or DoctorReportComposerService()
        )
        self.safety_layer_service = safety_layer_service or SafetyLayerService()
        self.prescription_document_parser = (
            prescription_document_parser or PrescriptionDocumentParser()
        )
        self.doctor_memory_service = doctor_memory_service
        self.risk_analyzer_service_factory = (
            risk_analyzer_service_factory or RiskAnalyzerService
        )
        self.gemini_client_factory = gemini_client_factory or GeminiRiskAnalyzerClient
        self.use_langgraph_audit = get_settings().use_langgraph_audit
        self._graph_service: Any | None = None

    @staticmethod
    def _meaningful_context_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().casefold() not in {"", "unknown", "not provided"}
        if isinstance(value, (list, dict)):
            return bool(value)
        return True

    @classmethod
    def _merge_patient_context(
        cls,
        parsed_context: dict[str, Any],
        incoming_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(parsed_context)
        if not isinstance(incoming_context, dict):
            return merged
        for key, value in incoming_context.items():
            if cls._meaningful_context_value(value):
                merged[key] = value
        return merged

    @staticmethod
    def _top_level_status(report: dict[str, Any] | None) -> str:
        if not isinstance(report, dict):
            return "failed"
        report_status = report.get("status")
        if report_status == "report_ready":
            return "success"
        if report_status in {
            "report_context_ready",
            "report_insufficient_evidence",
            "report_analysis_failed",
        }:
            return "partial_success"
        return "failed"

    @staticmethod
    def _base_response(
        *,
        status: str,
        prescription_check: dict[str, Any] | None = None,
        risk_analysis: dict[str, Any] | None = None,
        report: dict[str, Any] | None = None,
        doctor_memory: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> dict[str, Any]:
        merged_warnings: list[str] = []
        merged_errors: list[str] = []
        for result in (prescription_check, risk_analysis, report):
            merged_warnings.extend(_list_from_result(result, "warnings"))
            merged_errors.extend(_list_from_result(result, "errors"))
        merged_warnings.extend(warnings or [])
        merged_errors.extend(errors or [])
        return {
            "status": status,
            "prescription_check": prescription_check,
            "risk_analysis": risk_analysis,
            "report": report,
            "doctor_memory": doctor_memory or _empty_doctor_memory(),
            "warnings": _deduplicate(merged_warnings),
            "errors": _deduplicate(merged_errors),
        }

    @staticmethod
    def _safe_memory_note_item(note: dict[str, Any]) -> dict[str, str] | None:
        content = sanitize_doctor_report_text(
            str(note.get("note_text") or note.get("content") or "").strip()
        )
        title = sanitize_doctor_report_text(
            DoctorMemoryService.display_title(note.get("title"), content)
        )
        if not title and not content:
            return None
        return {
            "title": title or "Ghi chú riêng",
            "content": content,
        }

    @classmethod
    def _doctor_memory_section(cls, matched_notes: list[Any]) -> dict[str, Any]:
        items: list[dict[str, str]] = []
        for note in matched_notes[:3]:
            if not isinstance(note, dict):
                continue
            item = cls._safe_memory_note_item(note)
            if item is not None:
                items.append(item)
        return {
            "title": DOCTOR_MEMORY_LABEL.upper(),
            "summary": "Có ghi chú liên quan từ Doctor Memory."
            if items
            else "Chưa có ghi chú liên quan.",
            "items": items,
        }

    @classmethod
    def _attach_doctor_memory_to_report(
        cls,
        report: dict[str, Any],
        doctor_memory: dict[str, Any],
    ) -> dict[str, Any]:
        output = dict(report)
        output["doctor_memory"] = doctor_memory
        matched_notes = doctor_memory.get("matched_notes")
        if not isinstance(matched_notes, list):
            matched_notes = []

        sections = output.get("doctor_facing_sections")
        if isinstance(sections, dict):
            updated_sections = dict(sections)
            updated_sections["doctor_memory"] = cls._doctor_memory_section(
                matched_notes
            )
            output["doctor_facing_sections"] = updated_sections
            if has_full_doctor_facing_sections(updated_sections):
                output["doctor_facing_response"] = (
                    render_doctor_facing_response_from_sections(updated_sections)
                )
            return output

        output["doctor_facing_sections"] = {
            "doctor_memory": cls._doctor_memory_section(matched_notes)
        }
        return output

    def _retrieve_doctor_memory(
        self,
        *,
        doctor_id: str | None,
        normalized_result: dict[str, Any],
        patient_context: dict[str, Any],
        risk_analysis: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        if self.doctor_memory_service is None:
            return _empty_doctor_memory(), []
        try:
            memory = self.doctor_memory_service.retrieve_for_audit_context(
                doctor_id=doctor_id,
                normalized_result=normalized_result,
                patient_context=patient_context,
                risk_analysis=None,
                max_notes=3,
            )
        except Exception:
            return _empty_doctor_memory(), ["doctor_memory_retrieval_failed"]
        if not isinstance(memory, dict):
            return _empty_doctor_memory(), []
        notes = memory.get("matched_notes")
        if not isinstance(notes, list):
            return _empty_doctor_memory(), []
        return {"matched_notes": notes[:3]}, []

    def _create_risk_analyzer(self, use_gemini: bool | None = None) -> Any:
        """Create Gemini-backed risk analyzer.

        ``use_gemini`` is kept for request/API backward compatibility.
        Prescription audit always uses Gemini risk analysis.
        """
        gemini_client = self.gemini_client_factory()
        return self.risk_analyzer_service_factory(llm_client=gemini_client)

    def audit_text(
        self,
        prescription_text: str,
        doctor_id: str | None = None,
        patient_context: dict[str, Any] | None = None,
        use_gemini: bool = False,
        query_types: list[str] | None = None,
        top_k_per_type: int = 8,
    ) -> dict[str, Any]:
        if self.use_langgraph_audit:
            if self._graph_service is None:
                from backend.app.services.prescription_audit_graph import (
                    PrescriptionAuditGraphService,
                )

                self._graph_service = PrescriptionAuditGraphService(self)
            return self._graph_service.audit_text(
                prescription_text=prescription_text,
                doctor_id=doctor_id,
                patient_context=patient_context,
                use_gemini=use_gemini,
                query_types=query_types,
                top_k_per_type=top_k_per_type,
            )

        return self._audit_text_legacy(
            prescription_text=prescription_text,
            doctor_id=doctor_id,
            patient_context=patient_context,
            use_gemini=use_gemini,
            query_types=query_types,
            top_k_per_type=top_k_per_type,
        )

    def _audit_text_legacy(
        self,
        prescription_text: str,
        doctor_id: str | None = None,
        patient_context: dict[str, Any] | None = None,
        use_gemini: bool = False,
        query_types: list[str] | None = None,
        top_k_per_type: int = 8,
    ) -> dict[str, Any]:
        if top_k_per_type <= 0:
            raise ValueError("top_k_per_type must be greater than 0")

        context = patient_context or {}
        parser_warnings: list[str] = []
        parsed_document = self.prescription_document_parser.parse(prescription_text)
        if parsed_document.get("applied") is True:
            parser_warnings.append("prescription_document_parser_applied")
            parser_warnings.extend(
                str(warning)
                for warning in parsed_document.get("warnings", [])
                if warning
            )
            parsed_prescription_text = str(
                parsed_document.get("prescription_text") or ""
            ).strip()
            if not parsed_prescription_text:
                return self._base_response(
                    status="failed",
                    warnings=parser_warnings,
                )
            prescription_text = parsed_prescription_text
            parsed_context = parsed_document.get("patient_context")
            context = self._merge_patient_context(
                parsed_context if isinstance(parsed_context, dict) else {},
                context,
            )

        try:
            prescription_check = self.prescription_check_service.check_text(
                prescription_text=prescription_text,
                doctor_id=doctor_id,
                patient_context=context,
                query_types=query_types,
                top_k_per_type=top_k_per_type,
            )
        except Exception:
            return self._base_response(
                status="failed",
                warnings=parser_warnings,
                errors=["prescription_check_failed"],
            )

        if not isinstance(prescription_check, dict):
            return self._base_response(
                status="failed",
                prescription_check=None,
                warnings=parser_warnings,
                errors=["invalid_prescription_check_result"],
            )

        check_status = prescription_check.get("status")
        normalized_result = prescription_check.get("normalized_result")
        if check_status in {"invalid_input", "error"}:
            return self._base_response(
                status="failed",
                prescription_check=prescription_check,
                warnings=parser_warnings,
            )
        if not isinstance(normalized_result, dict):
            return self._base_response(
                status="failed",
                prescription_check=prescription_check,
                warnings=parser_warnings,
                errors=["normalized_result_missing"],
            )

        try:
            analyzer = self._create_risk_analyzer(use_gemini)
            risk_analysis = analyzer.analyze(
                normalized_result=normalized_result,
                evidence_bundle=prescription_check.get("evidence_bundle"),
                patient_context=context,
            )
        except Exception:
            risk_analysis = {
                "status": "analysis_failed",
                "overall_risk_level": "unknown",
                "risk_items": [],
                "evidence_context": None,
                "missing_information": [],
                "warnings": [],
                "errors": ["risk_analysis_failed"],
            }

        try:
            risk_analysis = self.safety_layer_service.apply(
                risk_analysis=risk_analysis,
                normalized_result=normalized_result,
                evidence_bundle=prescription_check.get("evidence_bundle"),
                patient_context=context,
            )
        except Exception:
            risk_analysis = {
                "status": "analysis_failed",
                "overall_risk_level": "unknown",
                "risk_items": [],
                "evidence_context": None,
                "missing_information": [],
                "warnings": [],
                "errors": ["safety_layer_failed"],
            }

        try:
            report = self.report_generator_service.generate_report(
                normalized_result=normalized_result,
                evidence_bundle=prescription_check.get("evidence_bundle"),
                risk_analysis=risk_analysis,
                patient_context=context,
            )
            report = self.doctor_report_composer_service.compose(report)
        except Exception:
            return self._base_response(
                status="failed",
                prescription_check=prescription_check,
                risk_analysis=risk_analysis,
                warnings=parser_warnings,
                errors=["report_generation_failed"],
            )

        doctor_memory, memory_warnings = self._retrieve_doctor_memory(
            doctor_id=doctor_id,
            normalized_result=normalized_result,
            patient_context=context,
        )
        report = self._attach_doctor_memory_to_report(report, doctor_memory)

        return self._base_response(
            status=self._top_level_status(report),
            prescription_check=prescription_check,
            risk_analysis=risk_analysis,
            report=report,
            doctor_memory=doctor_memory,
            warnings=[*parser_warnings, *memory_warnings],
        )

    def get_stats(self) -> dict[str, Any]:
        checker_stats = (
            self.prescription_check_service.get_stats()
            if hasattr(self.prescription_check_service, "get_stats")
            else None
        )
        reporter_stats = (
            self.report_generator_service.get_stats()
            if hasattr(self.report_generator_service, "get_stats")
            else None
        )
        safety_stats = (
            self.safety_layer_service.get_stats()
            if hasattr(self.safety_layer_service, "get_stats")
            else None
        )
        composer_stats = (
            self.doctor_report_composer_service.get_stats()
            if hasattr(self.doctor_report_composer_service, "get_stats")
            else None
        )
        memory_stats = (
            self.doctor_memory_service.get_stats()
            if self.doctor_memory_service is not None
            and hasattr(self.doctor_memory_service, "get_stats")
            else None
        )
        return {
            "service": "PrescriptionAuditService",
            "prescription_check_service": checker_stats,
            "report_generator_service": reporter_stats,
            "doctor_report_composer_service": composer_stats,
            "safety_layer_service": safety_stats,
            "doctor_memory_service": memory_stats,
            "gemini_supported": self.gemini_client_factory is not None,
            "langgraph_audit_enabled": self.use_langgraph_audit,
        }
