from fastapi import APIRouter, Depends

from backend.app.core.dependencies import (
    get_chat_query_service,
    get_clinical_workflow_graph_service,
    get_current_doctor_id,
)
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.clinical_workflow_graph import ClinicalWorkflowGraphService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    doctor_id: str = Depends(get_current_doctor_id),
    graph: ClinicalWorkflowGraphService = Depends(get_clinical_workflow_graph_service),
) -> ChatResponse:
    result = graph.run(
        {
            "request_type": "chat",
            "input_text": request.message,
            "chat_request": request,
            "doctor_id": doctor_id,
            "trace": [],
        }
    )
    if isinstance(result, dict) and "doctor_id" in result:
        result = {key: value for key, value in result.items() if key != "doctor_id"}
    return ChatResponse(doctor_id=doctor_id, **result)
