from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.graph import StateGraph

from backend.app.api.routes import chat as chat_route
from backend.app.api.routes import prescriptions as prescriptions_route
from backend.app.core.dependencies import (
    get_clinical_workflow_graph_service,
    get_current_doctor_id,
    get_prescription_workflow_graph_service,
    get_unified_clinical_workflow_graph_service,
)
from backend.app.db.session import get_db
from backend.app.main import app
from backend.app.schemas.prescription import PrescriptionAuditRequest
from backend.app.services.clinical_intent import ClinicalIntent
from backend.app.services.clinical_workflow_graph import ClinicalWorkflowGraphService
from backend.app.services.prescription_audit_graph import PrescriptionAuditGraphService
from backend.app.services.prescription_audit_service import PrescriptionAuditService


def _read_source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


class FakeChatQueryService:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.calls: list[Any] = []
        self.payload = payload or {
            "message": "fake chat response",
            "answer": "fake chat response",
            "intent": "single_drug_query",
            "normalized_drugs": [],
            "sources": [],
            "warnings": [],
        }

    def answer(self, request: Any) -> dict[str, Any]:
        self.calls.append(request)
        return self.payload


class FakePrescriptionAuditService:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.payload = payload or {
            "status": "ok",
            "doctor_memory": {"matched_notes": []},
        }

    def audit_text(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.payload


class FakeRouteGraph:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(state)
        return self.payload


class FakeDb:
    def __init__(self) -> None:
        self.added: list[Any] = []
        self.committed = False
        self.rolled_back = False

    def add(self, value: Any) -> None:
        self.added.append(value)

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


@pytest.fixture()
def route_client() -> TestClient:
    app.dependency_overrides[get_current_doctor_id] = lambda: "doctor-architecture"
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_doctor_id, None)
        app.dependency_overrides.pop(get_clinical_workflow_graph_service, None)
        app.dependency_overrides.pop(get_prescription_workflow_graph_service, None)
        app.dependency_overrides.pop(get_db, None)


def test_clinical_intent_has_exactly_three_top_level_intents() -> None:
    assert set(ClinicalIntent) == {
        ClinicalIntent.PRESCRIPTION_CHECK,
        ClinicalIntent.DRUG_INFORMATION_QUERY,
        ClinicalIntent.OUT_OF_SCOPE,
    }
    assert len(ClinicalIntent) == 3


def test_clinical_workflow_graph_service_uses_langgraph_state_graph() -> None:
    service = ClinicalWorkflowGraphService()

    assert isinstance(service.state_graph, StateGraph)


def test_clinical_workflow_graph_has_expected_node_trace_behavior() -> None:
    service = ClinicalWorkflowGraphService()

    prescription_state = service._invoke_state(
        {
            "request_type": "prescription_audit",
            "input_text": "Paracetamol 500mg",
            "trace": [],
        }
    )
    drug_state = service._invoke_state(
        {
            "request_type": "chat",
            "input_text": "Aspirin interaction with warfarin?",
            "trace": [],
        }
    )
    out_of_scope_state = service._invoke_state(
        {
            "request_type": "chat",
            "input_text": "Write a poem about summer",
            "trace": [],
        }
    )

    assert prescription_state["trace"] == [
        "classify_intent",
        "run_prescription_audit_branch",
        "safety_check",
        "finalize_response",
    ]
    assert drug_state["trace"] == [
        "classify_intent",
        "run_drug_information_branch",
        "safety_check",
        "finalize_response",
    ]
    assert out_of_scope_state["trace"] == [
        "classify_intent",
        "run_out_of_scope_branch",
        "safety_check",
        "finalize_response",
    ]


def test_clinical_workflow_graph_routes_the_three_expected_branches() -> None:
    service = ClinicalWorkflowGraphService()

    prescription_state = service._invoke_state(
        {
            "request_type": "prescription_audit",
            "input_text": "Paracetamol 500mg",
            "trace": [],
        }
    )
    drug_state = service._invoke_state(
        {
            "request_type": "chat",
            "input_text": "Paracetamol dose information",
            "trace": [],
        }
    )
    out_of_scope_state = service._invoke_state(
        {
            "request_type": "chat",
            "input_text": "Write a travel itinerary",
            "trace": [],
        }
    )

    assert prescription_state["intent"] == ClinicalIntent.PRESCRIPTION_CHECK
    assert drug_state["intent"] == ClinicalIntent.DRUG_INFORMATION_QUERY
    assert out_of_scope_state["intent"] == ClinicalIntent.OUT_OF_SCOPE


def test_chat_route_goes_through_clinical_workflow_graph_service() -> None:
    source = inspect.getsource(chat_route.chat)

    assert "get_clinical_workflow_graph_service" in source
    assert "ClinicalWorkflowGraphService" in source
    assert "graph.run(" in source
    assert ".answer(" not in source
    assert "ChatQueryService" not in source


def test_prescription_audit_route_goes_through_clinical_workflow_graph_service() -> None:
    source = inspect.getsource(prescriptions_route.audit_prescription)

    assert "get_prescription_workflow_graph_service" in source
    assert "ClinicalWorkflowGraphService" in source
    assert "graph.run(" in source
    assert ".audit_text(" not in source
    assert "_save_audit_history(" in source
    assert source.index("graph.run(") < source.index("_save_audit_history(")


@pytest.mark.parametrize(
    "legacy_intent",
    ["drug_interaction_query", "single_drug_query", "out_of_scope"],
)
def test_public_chat_response_keeps_legacy_lowercase_intents(
    route_client: TestClient,
    legacy_intent: str,
) -> None:
    fake_graph = FakeRouteGraph(
        {
            "message": f"fake {legacy_intent}",
            "answer": f"fake {legacy_intent}",
            "intent": legacy_intent,
            "normalized_drugs": [],
            "sources": [],
            "warnings": [],
        }
    )
    app.dependency_overrides[get_clinical_workflow_graph_service] = lambda: fake_graph

    response = route_client.post(
        "/api/v1/chat",
        json={"message": "Paracetamol dose information"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == legacy_intent
    assert body["intent"] != ClinicalIntent.DRUG_INFORMATION_QUERY.value
    assert len(fake_graph.calls) == 1
    assert fake_graph.calls[0]["request_type"] == "chat"
    assert "doctor_memory" not in fake_graph.calls[0]


def test_public_prescription_audit_response_returns_graph_result_unchanged(
    route_client: TestClient,
) -> None:
    audit_result = {
        "status": "partial_success",
        "prescription_check": {"status": "evidence_ready"},
        "risk_analysis": {"status": "analysis_context_ready"},
        "report": {"status": "report_context_ready"},
        "doctor_memory": {"matched_notes": []},
        "warnings": [],
        "errors": [],
    }
    fake_graph = FakeRouteGraph(audit_result)
    fake_db = FakeDb()
    app.dependency_overrides[get_prescription_workflow_graph_service] = (
        lambda: fake_graph
    )
    app.dependency_overrides[get_db] = lambda: fake_db

    response = route_client.post(
        "/api/v1/prescriptions/audit",
        json={"prescription_text": "1. Paracetamol 500mg"},
    )

    assert response.status_code == 200
    assert response.json() == audit_result
    assert "trace" not in response.json()
    assert "intent" not in response.json()
    assert "final_result" not in response.json()
    assert fake_db.committed is True
    assert len(fake_graph.calls) == 1
    assert fake_graph.calls[0]["request_type"] == "prescription_audit"


def test_drug_information_query_chat_does_not_use_doctor_memory() -> None:
    clinical_graph_source = _read_source("backend/app/services/clinical_workflow_graph.py")
    chat_route_source = _read_source("backend/app/api/routes/chat.py")
    chat_service_source = _read_source("backend/app/services/chat_query_service.py")

    for source in (clinical_graph_source, chat_route_source, chat_service_source):
        assert "DoctorMemoryService" not in source
        assert "doctor_memory_service" not in source

    service = ClinicalWorkflowGraphService(chat_query_service=FakeChatQueryService())
    state = service._invoke_state(
        {
            "request_type": "chat",
            "input_text": "Aspirin dose information",
            "trace": [],
        }
    )

    assert state["intent"] == ClinicalIntent.DRUG_INFORMATION_QUERY
    assert "doctor_memory" not in state


def test_prescription_check_preserves_doctor_memory_from_audit_result() -> None:
    doctor_memory = {"matched_notes": []}
    audit_service = FakePrescriptionAuditService(
        {
            "status": "ok",
            "doctor_memory": doctor_memory,
        }
    )
    service = ClinicalWorkflowGraphService(prescription_audit_service=audit_service)

    result = service.run(
        {
            "request_type": "prescription_audit",
            "prescription_request": PrescriptionAuditRequest(
                prescription_text="1. Paracetamol 500mg"
            ),
            "input_text": "1. Paracetamol 500mg",
            "doctor_id": "doctor-architecture",
            "trace": [],
        }
    )

    assert result["doctor_memory"] == doctor_memory
    assert len(audit_service.calls) == 1


def test_prescription_audit_graph_retrieves_doctor_memory_after_risk_analysis() -> None:
    source = _read_source("backend/app/services/prescription_audit_graph.py")

    assert 'graph.add_edge("retrieve_medical_evidence", "analyze_risks")' in source
    assert 'graph.add_edge("analyze_risks", "retrieve_doctor_memory")' in source
    assert 'graph.add_edge("retrieve_doctor_memory", "compose_report")' in source
    assert 'graph.add_edge("retrieve_medical_evidence", "retrieve_doctor_memory")' not in source
    assert 'graph.add_edge("retrieve_doctor_memory", "build_analysis_context")' not in source
    assert 'graph.add_edge("build_analysis_context", "analyze_risks")' not in source


def test_prescription_audit_graph_analyzer_does_not_receive_doctor_memory() -> None:
    analyze_source = inspect.getsource(PrescriptionAuditGraphService._analyze_risks)

    assert "doctor_memory" not in analyze_source
    assert "doctor_memory_notes" not in analyze_source
    assert "analysis_context" not in analyze_source


def test_prescription_audit_risk_analyzer_is_gemini_backed_always() -> None:
    source = inspect.getsource(PrescriptionAuditService._create_risk_analyzer)

    assert "self.gemini_client_factory()" in source
    assert "gemini_client =" in source
    assert "self.risk_analyzer_service_factory(llm_client=gemini_client)" in source
    assert "if use_gemini" not in source
    assert "return self.risk_analyzer_service_factory()" not in source


def test_doctor_memory_retrieval_is_display_only_for_risk_analysis() -> None:
    source = inspect.getsource(PrescriptionAuditService._retrieve_doctor_memory)

    assert "risk_analysis=None" in source
    assert "risk_analysis=risk_analysis" not in source


def test_prescription_audit_analysis_context_excludes_doctor_memory_notes() -> None:
    source = inspect.getsource(PrescriptionAuditGraphService.build_analysis_context)

    assert "doctor_memory_notes" not in source
    assert '"doctor_memory"' not in source
    assert '"medical_evidence"' in source


def test_safety_check_does_not_rewrite_payload() -> None:
    payload = {
        "message": "known payload",
        "answer": "known payload",
        "intent": "single_drug_query",
        "normalized_drugs": [{"name": "Aspirin"}],
        "sources": [{"title": "source"}],
        "warnings": ["review"],
    }
    service = ClinicalWorkflowGraphService(
        chat_query_service=FakeChatQueryService(payload)
    )

    state = service._invoke_state(
        {
            "request_type": "chat",
            "input_text": "Aspirin dose information",
            "trace": [],
        }
    )

    assert state["safety_status"] == "applied"
    assert state["draft_output"] == payload
    assert state["final_result"] == payload


def test_chat_graph_run_returns_public_payload_without_internal_fields() -> None:
    payload = {
        "message": "grounded chat answer",
        "answer": "grounded chat answer",
        "intent": "single_drug_query",
        "normalized_drugs": [],
        "sources": [],
        "warnings": [],
    }
    service = ClinicalWorkflowGraphService(
        chat_query_service=FakeChatQueryService(payload)
    )

    result = service.run(
        {
            "request_type": "chat",
            "input_text": "Paracetamol adverse effects",
            "trace": [],
        }
    )

    assert result == payload
    for internal_field in (
        "ClinicalIntent",
        "trace",
        "selected_intent",
        "final_result",
        "doctor_memory",
    ):
        assert internal_field not in result


def test_prescription_audit_service_does_not_import_clinical_workflow_graph() -> None:
    source = _read_source("backend/app/services/prescription_audit_service.py")

    assert "ClinicalWorkflowGraphService" not in source
    assert "clinical_workflow_graph" not in source


def test_clinical_workflow_graph_does_not_import_or_reference_intent_router() -> None:
    source = _read_source("backend/app/services/clinical_workflow_graph.py")

    assert "IntentRouter" not in source
    assert "intent_router" not in source


def test_prescription_audit_request_type_does_not_call_intent_router() -> None:
    class FailingIntentRouterGraphService(ClinicalWorkflowGraphService):
        @staticmethod
        def _looks_like_medical_query(input_text: str) -> bool:
            raise AssertionError("IntentRouter-like chat classification was called")

    state = FailingIntentRouterGraphService()._invoke_state(
        {
            "request_type": "prescription_audit",
            "input_text": "not relevant for prescription routing",
            "trace": [],
        }
    )

    assert state["intent"] == ClinicalIntent.PRESCRIPTION_CHECK


def test_dependency_boundaries_for_chat_prescription_and_unified_graphs() -> None:
    chat_dependency_source = inspect.getsource(get_clinical_workflow_graph_service)
    unified_dependency_source = inspect.getsource(
        get_unified_clinical_workflow_graph_service
    )
    prescription_dependency_source = inspect.getsource(
        get_prescription_workflow_graph_service
    )

    assert "get_prescription_audit_service" not in chat_dependency_source
    assert "prescription_audit_service" not in chat_dependency_source
    assert "chat_query_service" in chat_dependency_source
    assert "get_prescription_audit_service" in unified_dependency_source
    assert "prescription_audit_service" in unified_dependency_source
    assert "chat_query_service" in unified_dependency_source
    assert "chat_query_service" not in prescription_dependency_source


def test_history_saving_boundary_remains_in_prescription_route() -> None:
    route_source = _read_source("backend/app/api/routes/prescriptions.py")
    audit_source = inspect.getsource(prescriptions_route.audit_prescription)
    graph_source = _read_source("backend/app/services/clinical_workflow_graph.py")

    assert "def _save_audit_history(" in route_source
    assert "_save_audit_history(" in audit_source
    assert audit_source.index("graph.run(") < audit_source.index("_save_audit_history(")
    assert "PrescriptionHistory" not in graph_source
    assert "ReportHistory" not in graph_source
