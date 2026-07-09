from __future__ import annotations

import unicodedata
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.app.schemas.chat import ChatRequest
from backend.app.services.clinical_intent import ClinicalIntent


MEDICAL_QUERY_KEYWORDS = (
    "adr",
    "adverse",
    "aspirin",
    "chong chi dinh",
    "contraindication",
    "dose",
    "drug",
    "interaction",
    "lieu",
    "medicine",
    "paracetamol",
    "prescription",
    "side effect",
    "tac dung",
    "than trong",
    "thuoc",
    "tuong tac",
)


def _fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).replace(
        "đ",
        "d",
    )


class ClinicalWorkflowState(TypedDict, total=False):
    input: dict[str, Any]
    input_text: str
    request_type: str
    doctor_id: str | None
    intent: ClinicalIntent
    sub_intent: str
    parsed_prescription: dict[str, Any]
    entities: list[dict[str, Any]]
    normalized_drugs: list[dict[str, Any]]
    evidence: dict[str, Any]
    doctor_memory: dict[str, Any]
    draft_output: Any
    safety_status: str
    prescription_request: Any
    chat_request: Any
    prescription_result: Any
    chat_result: Any
    out_of_scope_result: Any
    final_result: Any
    errors: list[str]
    trace: list[str]


class ClinicalWorkflowGraphService:
    """LangGraph skeleton for future clinical workflow orchestration."""

    def __init__(
        self,
        prescription_audit_service: Any | None = None,
        chat_query_service: Any | None = None,
    ) -> None:
        self.prescription_audit_service = prescription_audit_service
        self.chat_query_service = chat_query_service
        self.state_graph = self._build_state_graph()
        self.graph = self.state_graph.compile()

    def _build_state_graph(self) -> StateGraph:
        graph = StateGraph(ClinicalWorkflowState)
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node(
            "run_prescription_audit_branch",
            self._run_prescription_audit_branch,
        )
        graph.add_node(
            "run_drug_information_branch",
            self._run_drug_information_branch,
        )
        graph.add_node("run_out_of_scope_branch", self._run_out_of_scope_branch)
        graph.add_node("safety_check", self._safety_check)
        graph.add_node("finalize_response", self._finalize_response)

        graph.add_edge(START, "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self._branch_for_intent,
            {
                "run_prescription_audit_branch": "run_prescription_audit_branch",
                "run_drug_information_branch": "run_drug_information_branch",
                "run_out_of_scope_branch": "run_out_of_scope_branch",
            },
        )
        graph.add_edge("run_prescription_audit_branch", "safety_check")
        graph.add_edge("run_drug_information_branch", "safety_check")
        graph.add_edge("run_out_of_scope_branch", "safety_check")
        graph.add_edge("safety_check", "finalize_response")
        graph.add_edge("finalize_response", END)
        return graph

    def run(self, initial_state: ClinicalWorkflowState) -> Any:
        state = self._invoke_state(initial_state)
        if "final_result" not in state or state["final_result"] is None:
            raise RuntimeError("clinical_workflow_final_result_missing")
        return state["final_result"]

    def _invoke_state(
        self,
        initial_state: ClinicalWorkflowState,
    ) -> ClinicalWorkflowState:
        return self.graph.invoke(initial_state)

    @staticmethod
    def _append_trace(
        state: ClinicalWorkflowState,
        node_name: str,
    ) -> list[str]:
        return [*(state.get("trace") or []), node_name]

    @staticmethod
    def _get_field(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @staticmethod
    def _looks_like_medical_query(input_text: str) -> bool:
        folded = _fold_text(input_text)
        return any(keyword in folded for keyword in MEDICAL_QUERY_KEYWORDS)

    @classmethod
    def _intent_for_state(cls, state: ClinicalWorkflowState) -> ClinicalIntent:
        request_type = state.get("request_type")
        if request_type == "prescription_audit":
            return ClinicalIntent.PRESCRIPTION_CHECK
        if request_type == "chat":
            input_text = str(
                cls._get_field(state.get("chat_request"), "message", None)
                or state.get("input_text")
                or ""
            )
            if cls._looks_like_medical_query(input_text):
                return ClinicalIntent.DRUG_INFORMATION_QUERY
            return ClinicalIntent.OUT_OF_SCOPE
        return ClinicalIntent.OUT_OF_SCOPE

    def _classify_intent(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        return {
            "intent": self._intent_for_state(state),
            "trace": self._append_trace(state, "classify_intent"),
        }

    @staticmethod
    def _branch_for_intent(state: ClinicalWorkflowState) -> str:
        intent = state.get("intent")
        if intent == ClinicalIntent.PRESCRIPTION_CHECK:
            return "run_prescription_audit_branch"
        if intent == ClinicalIntent.DRUG_INFORMATION_QUERY:
            return "run_drug_information_branch"
        return "run_out_of_scope_branch"

    def _run_prescription_audit_branch(
        self,
        state: ClinicalWorkflowState,
    ) -> dict[str, Any]:
        if self.prescription_audit_service is not None:
            request = state.get("prescription_request")
            prescription_result = self.prescription_audit_service.audit_text(
                prescription_text=self._get_field(
                    request,
                    "prescription_text",
                    state.get("input_text", ""),
                ),
                doctor_id=state.get("doctor_id"),
                patient_context=self._get_field(request, "patient_context", {}),
                use_gemini=self._get_field(request, "use_gemini", False),
                query_types=self._get_field(request, "query_types", None),
                top_k_per_type=self._get_field(request, "top_k_per_type", 8),
            )
        else:
            prescription_result = {
                "status": "skeleton_prescription_audit_ready",
                "message": "Prescription audit branch placeholder.",
            }
        return {
            "prescription_result": prescription_result,
            "draft_output": prescription_result,
            "trace": self._append_trace(state, "run_prescription_audit_branch"),
        }

    def _run_drug_information_branch(
        self,
        state: ClinicalWorkflowState,
    ) -> dict[str, Any]:
        if self.chat_query_service is not None:
            request = state.get("chat_request")
            chat_request = ChatRequest(
                message=self._get_field(request, "message", state.get("input_text", "")),
                intent=self._get_field(request, "intent", state.get("sub_intent")),
            )
            chat_result = self.chat_query_service.answer(chat_request)
        else:
            chat_result = {
                "status": "skeleton_drug_information_ready",
                "message": "Drug information branch placeholder.",
            }
        return {
            "chat_result": chat_result,
            "draft_output": chat_result,
            "trace": self._append_trace(state, "run_drug_information_branch"),
        }

    def _run_out_of_scope_branch(
        self,
        state: ClinicalWorkflowState,
    ) -> dict[str, Any]:
        refusal = "This request is outside the supported clinical workflow."
        out_of_scope_result = {
            "message": refusal,
            "answer": refusal,
            "intent": "out_of_scope",
            "normalized_drugs": [],
            "sources": [],
            "warnings": [],
        }
        doctor_id = state.get("doctor_id")
        if doctor_id is not None:
            out_of_scope_result["doctor_id"] = doctor_id
        return {
            "out_of_scope_result": out_of_scope_result,
            "draft_output": out_of_scope_result,
            "trace": self._append_trace(state, "run_out_of_scope_branch"),
        }

    def _safety_check(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        return {
            "safety_status": "not_applied_skeleton",
            "trace": self._append_trace(state, "safety_check"),
        }

    def _finalize_response(self, state: ClinicalWorkflowState) -> dict[str, Any]:
        return {
            "final_result": state.get("draft_output"),
            "trace": self._append_trace(state, "finalize_response"),
        }
