from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph


ANALYSIS_CONTEXT_RULES = {
    "medical_evidence_priority": "authoritative",
}


class PrescriptionAuditState(TypedDict, total=False):
    prescription_text: str
    doctor_id: str | None
    patient_context: dict[str, Any]
    use_gemini: bool
    query_types: list[str] | None
    top_k_per_type: int
    parser_warnings: list[str]
    prescription_check: dict[str, Any] | None
    normalized_result: dict[str, Any] | None
    medical_evidence_bundle: dict[str, Any] | None
    doctor_memory: dict[str, Any]
    memory_warnings: list[str]
    risk_analysis: dict[str, Any] | None
    report: dict[str, Any] | None
    final_response: dict[str, Any] | None


class PrescriptionAuditGraphService:
    """LangGraph orchestration over the existing PrescriptionAuditService pieces."""

    def __init__(self, audit_service: Any) -> None:
        self.audit_service = audit_service
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(PrescriptionAuditState)
        graph.add_node("validate_request", self._validate_request)
        graph.add_node("parse_prescription", self._parse_prescription)
        graph.add_node("normalize_drugs", self._normalize_drugs)
        graph.add_node("retrieve_medical_evidence", self._retrieve_medical_evidence)
        graph.add_node("analyze_risks", self._analyze_risks)
        graph.add_node("retrieve_doctor_memory", self._retrieve_doctor_memory)
        graph.add_node("compose_report", self._compose_report)
        graph.add_node("safety_check", self._safety_check)
        graph.add_node("return_response", self._return_response)

        graph.set_entry_point("validate_request")
        graph.add_edge("validate_request", "parse_prescription")
        graph.add_edge("parse_prescription", "normalize_drugs")
        graph.add_edge("normalize_drugs", "retrieve_medical_evidence")
        graph.add_edge("retrieve_medical_evidence", "analyze_risks")
        graph.add_edge("analyze_risks", "retrieve_doctor_memory")
        graph.add_edge("retrieve_doctor_memory", "compose_report")
        graph.add_edge("compose_report", "safety_check")
        graph.add_edge("safety_check", "return_response")
        graph.add_edge("return_response", END)
        return graph.compile()

    def audit_text(
        self,
        prescription_text: str,
        doctor_id: str | None = None,
        patient_context: dict[str, Any] | None = None,
        use_gemini: bool = False,
        query_types: list[str] | None = None,
        top_k_per_type: int = 8,
    ) -> dict[str, Any]:
        state = self.graph.invoke(
            {
                "prescription_text": prescription_text,
                "doctor_id": doctor_id,
                "patient_context": patient_context or {},
                "use_gemini": use_gemini,
                "query_types": query_types,
                "top_k_per_type": top_k_per_type,
                "parser_warnings": [],
                "doctor_memory": self.audit_service._empty_doctor_memory()
                if hasattr(self.audit_service, "_empty_doctor_memory")
                else {"matched_notes": []},
                "memory_warnings": [],
            }
        )
        final_response = state.get("final_response")
        if isinstance(final_response, dict):
            return final_response
        return self.audit_service._base_response(status="failed")

    @staticmethod
    def _has_final_response(state: PrescriptionAuditState) -> bool:
        return isinstance(state.get("final_response"), dict)

    @staticmethod
    def build_analysis_context(
        *,
        medical_evidence_bundle: dict[str, Any] | None,
        doctor_memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        medical_evidence: list[Any] = []
        if isinstance(medical_evidence_bundle, dict):
            unique_chunks = medical_evidence_bundle.get("unique_chunks")
            all_chunks = medical_evidence_bundle.get("all_chunks")
            if isinstance(unique_chunks, list):
                medical_evidence = unique_chunks
            elif isinstance(all_chunks, list):
                medical_evidence = all_chunks

        return {
            "medical_evidence": medical_evidence,
            "context_rules": dict(ANALYSIS_CONTEXT_RULES),
        }

    def _validate_request(
        self, state: PrescriptionAuditState
    ) -> dict[str, Any]:
        top_k_per_type = state.get("top_k_per_type", 8)
        if top_k_per_type <= 0:
            raise ValueError("top_k_per_type must be greater than 0")
        return {}

    def _parse_prescription(
        self, state: PrescriptionAuditState
    ) -> dict[str, Any]:
        if self._has_final_response(state):
            return {}

        prescription_text = state["prescription_text"]
        context = state.get("patient_context") or {}
        parser_warnings: list[str] = []
        parsed_document = self.audit_service.prescription_document_parser.parse(
            prescription_text
        )
        if parsed_document.get("applied") is not True:
            return {
                "prescription_text": prescription_text,
                "patient_context": context,
                "parser_warnings": parser_warnings,
            }

        parser_warnings.append("prescription_document_parser_applied")
        parser_warnings.extend(
            str(warning) for warning in parsed_document.get("warnings", []) if warning
        )
        parsed_prescription_text = str(
            parsed_document.get("prescription_text") or ""
        ).strip()
        if not parsed_prescription_text:
            return {
                "parser_warnings": parser_warnings,
                "final_response": self.audit_service._base_response(
                    status="failed",
                    warnings=parser_warnings,
                ),
            }

        parsed_context = parsed_document.get("patient_context")
        merged_context = self.audit_service._merge_patient_context(
            parsed_context if isinstance(parsed_context, dict) else {},
            context,
        )
        return {
            "prescription_text": parsed_prescription_text,
            "patient_context": merged_context,
            "parser_warnings": parser_warnings,
        }

    def _normalize_drugs(
        self, state: PrescriptionAuditState
    ) -> dict[str, Any]:
        if self._has_final_response(state):
            return {}

        parser_warnings = state.get("parser_warnings") or []
        try:
            prescription_check = self.audit_service.prescription_check_service.check_text(
                prescription_text=state["prescription_text"],
                doctor_id=state.get("doctor_id"),
                patient_context=state.get("patient_context") or {},
                query_types=state.get("query_types"),
                top_k_per_type=state.get("top_k_per_type", 8),
            )
        except Exception:
            return {
                "final_response": self.audit_service._base_response(
                    status="failed",
                    warnings=parser_warnings,
                    errors=["prescription_check_failed"],
                )
            }

        if not isinstance(prescription_check, dict):
            return {
                "final_response": self.audit_service._base_response(
                    status="failed",
                    prescription_check=None,
                    warnings=parser_warnings,
                    errors=["invalid_prescription_check_result"],
                )
            }

        check_status = prescription_check.get("status")
        normalized_result = prescription_check.get("normalized_result")
        if check_status in {"invalid_input", "error"}:
            return {
                "prescription_check": prescription_check,
                "final_response": self.audit_service._base_response(
                    status="failed",
                    prescription_check=prescription_check,
                    warnings=parser_warnings,
                ),
            }
        if not isinstance(normalized_result, dict):
            return {
                "prescription_check": prescription_check,
                "final_response": self.audit_service._base_response(
                    status="failed",
                    prescription_check=prescription_check,
                    warnings=parser_warnings,
                    errors=["normalized_result_missing"],
                ),
            }

        return {
            "prescription_check": prescription_check,
            "normalized_result": normalized_result,
        }

    def _retrieve_medical_evidence(
        self, state: PrescriptionAuditState
    ) -> dict[str, Any]:
        if self._has_final_response(state):
            return {}
        prescription_check = state.get("prescription_check") or {}
        evidence_bundle = (
            prescription_check.get("evidence_bundle")
            if isinstance(prescription_check, dict)
            else None
        )
        return {"medical_evidence_bundle": evidence_bundle}

    def _retrieve_doctor_memory(
        self, state: PrescriptionAuditState
    ) -> dict[str, Any]:
        if self._has_final_response(state):
            return {}
        normalized_result = state.get("normalized_result")
        if not isinstance(normalized_result, dict):
            return {}
        doctor_memory, memory_warnings = self.audit_service._retrieve_doctor_memory(
            doctor_id=state.get("doctor_id"),
            normalized_result=normalized_result,
            patient_context=state.get("patient_context") or {},
            risk_analysis=None,
        )
        return {
            "doctor_memory": doctor_memory,
            "memory_warnings": memory_warnings,
        }

    def _analyze_risks(
        self, state: PrescriptionAuditState
    ) -> dict[str, Any]:
        if self._has_final_response(state):
            return {}
        normalized_result = state.get("normalized_result")
        if not isinstance(normalized_result, dict):
            return {}
        try:
            analyzer = self.audit_service._create_risk_analyzer(
                bool(state.get("use_gemini"))
            )
            risk_analysis = analyzer.analyze(
                normalized_result=normalized_result,
                evidence_bundle=state.get("medical_evidence_bundle"),
                patient_context=state.get("patient_context") or {},
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
        return {"risk_analysis": risk_analysis}

    def _compose_report(
        self, state: PrescriptionAuditState
    ) -> dict[str, Any]:
        if self._has_final_response(state):
            return {}
        normalized_result = state.get("normalized_result")
        risk_analysis = state.get("risk_analysis")
        prescription_check = state.get("prescription_check")
        parser_warnings = state.get("parser_warnings") or []
        if not isinstance(normalized_result, dict) or not isinstance(
            risk_analysis, dict
        ):
            return {}
        try:
            safe_risk_analysis = self.audit_service.safety_layer_service.apply(
                risk_analysis=risk_analysis,
                normalized_result=normalized_result,
                evidence_bundle=state.get("medical_evidence_bundle"),
                patient_context=state.get("patient_context") or {},
            )
        except Exception:
            safe_risk_analysis = {
                "status": "analysis_failed",
                "overall_risk_level": "unknown",
                "risk_items": [],
                "evidence_context": None,
                "missing_information": [],
                "warnings": [],
                "errors": ["safety_layer_failed"],
            }

        try:
            report = self.audit_service.report_generator_service.generate_report(
                normalized_result=normalized_result,
                evidence_bundle=state.get("medical_evidence_bundle"),
                risk_analysis=safe_risk_analysis,
                patient_context=state.get("patient_context") or {},
            )
            report = self.audit_service.doctor_report_composer_service.compose(report)
        except Exception:
            return {
                "risk_analysis": safe_risk_analysis,
                "final_response": self.audit_service._base_response(
                    status="failed",
                    prescription_check=prescription_check,
                    risk_analysis=safe_risk_analysis,
                    warnings=parser_warnings,
                    errors=["report_generation_failed"],
                ),
            }

        return {
            "risk_analysis": safe_risk_analysis,
            "report": report,
        }

    def _safety_check(
        self, state: PrescriptionAuditState
    ) -> dict[str, Any]:
        if self._has_final_response(state):
            return {}
        report = state.get("report")
        doctor_memory = state.get("doctor_memory") or {"matched_notes": []}
        if isinstance(report, dict):
            report = self.audit_service._attach_doctor_memory_to_report(
                report,
                doctor_memory,
            )
        return {"report": report}

    def _return_response(
        self, state: PrescriptionAuditState
    ) -> dict[str, Any]:
        if self._has_final_response(state):
            return {}
        report = state.get("report")
        return {
            "final_response": self.audit_service._base_response(
                status=self.audit_service._top_level_status(report),
                prescription_check=state.get("prescription_check"),
                risk_analysis=state.get("risk_analysis"),
                report=report,
                doctor_memory=state.get("doctor_memory"),
                warnings=[
                    *(state.get("parser_warnings") or []),
                    *(state.get("memory_warnings") or []),
                ],
            )
        }
