from pathlib import Path
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from langgraph.graph import StateGraph

from backend.app.core.dependencies import (
    build_clinical_workflow_graph_service,
    get_clinical_workflow_graph_service,
    get_prescription_workflow_graph_service,
)
from backend.app.services.clinical_intent import ClinicalIntent
from backend.app.services.clinical_workflow_graph import (
    ClinicalWorkflowGraphService,
    ClinicalWorkflowState,
)


class FakePrescriptionAuditService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def audit_text(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "status": "fake_prescription_audit_ok",
            "source": "fake_audit_service",
            "echo": kwargs,
        }


class FakeChatQueryService:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def answer(self, request: Any) -> dict[str, Any]:
        self.calls.append(request)
        return {
            "status": "fake_chat_ok",
            "intent": request.intent or "single_drug_query",
            "message": f"fake answer for {request.message}",
            "source": "fake_chat_service",
        }


def _invoke(request_type: str, input_text: str = ""):
    return ClinicalWorkflowGraphService()._invoke_state(
        {
            "request_type": request_type,
            "input_text": input_text,
            "trace": [],
        }
    )


def test_clinical_workflow_graph_service_can_be_instantiated() -> None:
    service = ClinicalWorkflowGraphService()

    assert service is not None


def test_build_clinical_workflow_graph_service_returns_graph_service() -> None:
    service = build_clinical_workflow_graph_service(FakeChatQueryService())

    assert isinstance(service, ClinicalWorkflowGraphService)


def test_built_graph_service_can_run_with_injected_fake_chat_service() -> None:
    fake_chat = FakeChatQueryService()
    service = build_clinical_workflow_graph_service(fake_chat)

    result = service.run(
        {
            "request_type": "chat",
            "input_text": "Aspirin dose information",
            "trace": [],
        }
    )

    assert result["status"] == "fake_chat_ok"
    assert result["source"] == "fake_chat_service"
    assert len(fake_chat.calls) == 1


def test_clinical_workflow_graph_dependency_can_be_overridden() -> None:
    app = FastAPI()
    fake_graph = object()

    @app.get("/test-graph")
    def test_graph(
        graph=Depends(get_clinical_workflow_graph_service),
    ) -> dict[str, bool]:
        return {"received_fake_graph": graph is fake_graph}

    app.dependency_overrides[get_clinical_workflow_graph_service] = lambda: fake_graph
    try:
        response = TestClient(app).get("/test-graph")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"received_fake_graph": True}


def test_clinical_workflow_graph_service_uses_state_graph() -> None:
    service = ClinicalWorkflowGraphService()

    assert isinstance(service.state_graph, StateGraph)


def test_clinical_workflow_state_supports_doctor_memory_field() -> None:
    assert "doctor_memory" in ClinicalWorkflowState.__annotations__


def test_prescription_audit_request_routes_to_prescription_check() -> None:
    state = _invoke("prescription_audit", "Paracetamol 500mg")

    assert state["intent"] == ClinicalIntent.PRESCRIPTION_CHECK


def test_chat_drug_question_routes_to_drug_information_query() -> None:
    state = _invoke("chat", "Paracetamol co tac dung phu gi?")

    assert state["intent"] == ClinicalIntent.DRUG_INFORMATION_QUERY


@pytest.mark.parametrize(
    "message",
    [
        "Thuốc aspirin có tương tác với warfarin không?",
        "Liều dùng paracetamol như thế nào?",
        "Tác dụng phụ của thuốc này là gì?",
        "Chống chỉ định của aspirin là gì?",
    ],
)
def test_vietnamese_accented_medical_questions_route_to_drug_information_query(
    message: str,
) -> None:
    state = _invoke("chat", message)

    assert state["intent"] == ClinicalIntent.DRUG_INFORMATION_QUERY


def test_medical_query_matching_folds_vietnamese_accents() -> None:
    assert ClinicalWorkflowGraphService._looks_like_medical_query("thuốc") is True
    assert ClinicalWorkflowGraphService._looks_like_medical_query("tương tác") is True


def test_chat_unrelated_text_routes_to_out_of_scope() -> None:
    state = _invoke("chat", "Viet giup toi mot bai tho ve mua he")

    assert state["intent"] == ClinicalIntent.OUT_OF_SCOPE


def test_unknown_request_type_routes_to_out_of_scope() -> None:
    state = _invoke("unknown", "Paracetamol")

    assert state["intent"] == ClinicalIntent.OUT_OF_SCOPE


def test_prescription_branch_returns_fake_prescription_result() -> None:
    state = _invoke("prescription_audit", "Paracetamol 500mg")

    assert state["prescription_result"] == {
        "status": "skeleton_prescription_audit_ready",
        "message": "Prescription audit branch placeholder.",
    }
    assert state["final_result"] == state["prescription_result"]


def test_prescription_branch_calls_fake_audit_service_exactly_once() -> None:
    fake_audit = FakePrescriptionAuditService()
    service = ClinicalWorkflowGraphService(prescription_audit_service=fake_audit)

    state = service._invoke_state(
        {
            "request_type": "prescription_audit",
            "doctor_id": "doctor-graph",
            "input_text": "fallback text",
            "prescription_request": {
                "prescription_text": "Aspirin 81mg",
                "patient_context": {"age": 70},
                "use_gemini": True,
                "query_types": ["interaction"],
                "top_k_per_type": 4,
            },
            "trace": [],
        }
    )

    assert len(fake_audit.calls) == 1
    assert fake_audit.calls[0] == {
        "prescription_text": "Aspirin 81mg",
        "doctor_id": "doctor-graph",
        "patient_context": {"age": 70},
        "use_gemini": True,
        "query_types": ["interaction"],
        "top_k_per_type": 4,
    }
    assert state["prescription_result"]["source"] == "fake_audit_service"
    assert state["draft_output"] == state["prescription_result"]
    assert state["final_result"] == state["prescription_result"]


def test_prescription_branch_reads_object_prescription_request_fields() -> None:
    fake_audit = FakePrescriptionAuditService()
    service = ClinicalWorkflowGraphService(prescription_audit_service=fake_audit)
    request = SimpleNamespace(
        prescription_text="Metformin 500mg",
        patient_context={"renal": "normal"},
        use_gemini=False,
        query_types=["dose"],
        top_k_per_type=2,
    )

    service._invoke_state(
        {
            "request_type": "prescription_audit",
            "doctor_id": "doctor-object",
            "prescription_request": request,
            "trace": [],
        }
    )

    assert len(fake_audit.calls) == 1
    assert fake_audit.calls[0] == {
        "prescription_text": "Metformin 500mg",
        "doctor_id": "doctor-object",
        "patient_context": {"renal": "normal"},
        "use_gemini": False,
        "query_types": ["dose"],
        "top_k_per_type": 2,
    }


def test_drug_information_branch_returns_fake_chat_result() -> None:
    state = _invoke("chat", "Aspirin interaction with warfarin?")

    assert state["chat_result"] == {
        "status": "skeleton_drug_information_ready",
        "message": "Drug information branch placeholder.",
    }
    assert state["final_result"] == state["chat_result"]


def test_drug_information_branch_calls_fake_chat_service_exactly_once() -> None:
    fake_chat = FakeChatQueryService()
    service = ClinicalWorkflowGraphService(chat_query_service=fake_chat)

    state = service._invoke_state(
        {
            "request_type": "chat",
            "input_text": "fallback chat text",
            "chat_request": {
                "message": "Aspirin dose information",
                "intent": "single_drug_query",
            },
            "trace": [],
        }
    )

    assert len(fake_chat.calls) == 1
    assert fake_chat.calls[0].message == "Aspirin dose information"
    assert fake_chat.calls[0].intent == "single_drug_query"
    assert state["chat_result"]["source"] == "fake_chat_service"
    assert state["draft_output"] == state["chat_result"]
    assert state["final_result"] == state["chat_result"]


def test_drug_information_branch_reads_object_chat_request_fields() -> None:
    fake_chat = FakeChatQueryService()
    service = ClinicalWorkflowGraphService(chat_query_service=fake_chat)
    request = SimpleNamespace(
        message="Paracetamol adverse effects",
        intent="single_drug_query",
    )

    service._invoke_state(
        {
            "request_type": "chat",
            "chat_request": request,
            "trace": [],
        }
    )

    assert len(fake_chat.calls) == 1
    assert fake_chat.calls[0].message == "Paracetamol adverse effects"
    assert fake_chat.calls[0].intent == "single_drug_query"


def test_out_of_scope_branch_returns_fake_refusal_result() -> None:
    state = ClinicalWorkflowGraphService()._invoke_state(
        {
            "request_type": "chat",
            "input_text": "Tell me a joke",
            "doctor_id": "doctor-scope",
            "trace": [],
        }
    )

    assert state["out_of_scope_result"] == {
        "message": "This request is outside the supported clinical workflow.",
        "answer": "This request is outside the supported clinical workflow.",
        "intent": "out_of_scope",
        "normalized_drugs": [],
        "sources": [],
        "warnings": [],
        "doctor_id": "doctor-scope",
    }
    assert state["final_result"] == state["out_of_scope_result"]


def test_out_of_scope_branch_calls_neither_fake_service() -> None:
    fake_audit = FakePrescriptionAuditService()
    fake_chat = FakeChatQueryService()
    service = ClinicalWorkflowGraphService(
        prescription_audit_service=fake_audit,
        chat_query_service=fake_chat,
    )

    state = service._invoke_state(
        {
            "request_type": "chat",
            "input_text": "Tell me a joke",
            "doctor_id": "doctor-scope",
            "trace": [],
        }
    )

    assert fake_audit.calls == []
    assert fake_chat.calls == []
    assert state["final_result"]["intent"] == "out_of_scope"


def test_trace_contains_prescription_branch_flow() -> None:
    state = _invoke("prescription_audit", "Paracetamol 500mg")

    assert state["trace"] == [
        "classify_intent",
        "run_prescription_audit_branch",
        "safety_check",
        "finalize_response",
    ]


def test_trace_contains_drug_information_branch_flow() -> None:
    state = _invoke("chat", "Paracetamol co tac dung phu gi?")

    assert state["trace"] == [
        "classify_intent",
        "run_drug_information_branch",
        "safety_check",
        "finalize_response",
    ]


def test_trace_contains_out_of_scope_branch_flow() -> None:
    state = _invoke("chat", "Tell me a joke")

    assert state["trace"] == [
        "classify_intent",
        "run_out_of_scope_branch",
        "safety_check",
        "finalize_response",
    ]


def test_safety_check_does_not_rewrite_final_result() -> None:
    state = _invoke("chat", "Aspirin dose information")

    assert state["safety_status"] == "not_applied_skeleton"
    assert state["final_result"] == state["draft_output"]


def test_chat_branch_does_not_populate_doctor_memory() -> None:
    state = _invoke("chat", "Aspirin dose information")

    assert "doctor_memory" not in state


def test_drug_information_branch_with_fake_service_does_not_populate_doctor_memory() -> None:
    fake_chat = FakeChatQueryService()
    state = ClinicalWorkflowGraphService(chat_query_service=fake_chat)._invoke_state(
        {
            "request_type": "chat",
            "input_text": "Aspirin dose information",
            "trace": [],
        }
    )

    assert len(fake_chat.calls) == 1
    assert "doctor_memory" not in state


def test_run_returns_only_final_result() -> None:
    result = ClinicalWorkflowGraphService().run(
        {
            "request_type": "chat",
            "input_text": "Aspirin dose information",
            "trace": [],
        }
    )

    assert result == {
        "status": "skeleton_drug_information_ready",
        "message": "Drug information branch placeholder.",
    }
    assert "trace" not in result
    assert "intent" not in result
    assert "final_result" not in result


def test_run_returns_non_dict_final_result_as_is() -> None:
    class TextFinalResultGraphService(ClinicalWorkflowGraphService):
        def _invoke_state(self, initial_state):
            return {"final_result": "final text"}

    assert TextFinalResultGraphService().run({"request_type": "chat"}) == "final text"


def test_run_raises_controlled_error_when_final_result_is_missing() -> None:
    class MissingFinalResultGraphService(ClinicalWorkflowGraphService):
        def _invoke_state(self, initial_state):
            return {}

    with pytest.raises(
        RuntimeError,
        match="clinical_workflow_final_result_missing",
    ):
        MissingFinalResultGraphService().run({"request_type": "chat"})


def test_run_raises_controlled_error_when_final_result_is_none() -> None:
    class NoneFinalResultGraphService(ClinicalWorkflowGraphService):
        def _invoke_state(self, initial_state):
            return {"final_result": None}

    with pytest.raises(
        RuntimeError,
        match="clinical_workflow_final_result_missing",
    ):
        NoneFinalResultGraphService().run({"request_type": "chat"})


def test_clinical_workflow_graph_does_not_import_real_services() -> None:
    source = Path("backend/app/services/clinical_workflow_graph.py").read_text()

    forbidden_imports = [
        "ChatQueryService",
        "PrescriptionAuditService",
        "DoctorMemoryService",
        "doctor_memory_service",
    ]
    for forbidden_import in forbidden_imports:
        assert forbidden_import not in source


def test_clinical_workflow_graph_does_not_import_intent_router() -> None:
    source = Path("backend/app/services/clinical_workflow_graph.py").read_text()

    assert "IntentRouter" not in source


def test_prescription_audit_service_does_not_import_clinical_workflow_graph() -> None:
    source = Path("backend/app/services/prescription_audit_service.py").read_text()

    assert "ClinicalWorkflowGraphService" not in source
    assert "clinical_workflow_graph" not in source


def test_chat_graph_dependency_stays_isolated_from_prescription_audit() -> None:
    source = inspect.getsource(get_clinical_workflow_graph_service)

    assert "get_prescription_audit_service" not in source
    assert "PrescriptionAuditService" not in source
    assert "prescription_audit_service" not in source


def test_prescription_workflow_graph_dependency_uses_prescription_service() -> None:
    source = inspect.getsource(get_prescription_workflow_graph_service)

    assert "prescription_audit_service" in source
    assert "chat_query_service" not in source
