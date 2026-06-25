from __future__ import annotations

from typing import Any, Callable, Iterable

from backend.app.services.gemini_risk_analyzer_client import GeminiRiskAnalyzerClient
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


class PrescriptionAuditService:
    """Orchestrate prescription check, risk analysis, and report generation."""

    def __init__(
        self,
        prescription_check_service: Any | None = None,
        report_generator_service: Any | None = None,
        safety_layer_service: Any | None = None,
        risk_analyzer_service_factory: Callable[..., Any] | None = None,
        gemini_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.prescription_check_service = (
            prescription_check_service or PrescriptionCheckService()
        )
        self.report_generator_service = (
            report_generator_service or ReportGeneratorService()
        )
        self.safety_layer_service = safety_layer_service or SafetyLayerService()
        self.risk_analyzer_service_factory = (
            risk_analyzer_service_factory or RiskAnalyzerService
        )
        self.gemini_client_factory = gemini_client_factory or GeminiRiskAnalyzerClient

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
            "warnings": _deduplicate(merged_warnings),
            "errors": _deduplicate(merged_errors),
        }

    def _create_risk_analyzer(self, use_gemini: bool) -> Any:
        if use_gemini:
            gemini_client = self.gemini_client_factory()
            return self.risk_analyzer_service_factory(llm_client=gemini_client)
        return self.risk_analyzer_service_factory()

    def audit_text(
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
                errors=["prescription_check_failed"],
            )

        if not isinstance(prescription_check, dict):
            return self._base_response(
                status="failed",
                prescription_check=None,
                errors=["invalid_prescription_check_result"],
            )

        check_status = prescription_check.get("status")
        normalized_result = prescription_check.get("normalized_result")
        if check_status in {"invalid_input", "error"}:
            return self._base_response(
                status="failed",
                prescription_check=prescription_check,
            )
        if not isinstance(normalized_result, dict):
            return self._base_response(
                status="failed",
                prescription_check=prescription_check,
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
        except Exception:
            return self._base_response(
                status="failed",
                prescription_check=prescription_check,
                risk_analysis=risk_analysis,
                errors=["report_generation_failed"],
            )

        return self._base_response(
            status=self._top_level_status(report),
            prescription_check=prescription_check,
            risk_analysis=risk_analysis,
            report=report,
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
        return {
            "service": "PrescriptionAuditService",
            "prescription_check_service": checker_stats,
            "report_generator_service": reporter_stats,
            "safety_layer_service": safety_stats,
            "gemini_supported": self.gemini_client_factory is not None,
        }
