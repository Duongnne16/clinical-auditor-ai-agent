from fastapi import APIRouter, Depends

from backend.app.core.dependencies import get_doctor_id
from backend.app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    doctor_id: str = Depends(get_doctor_id),
) -> ChatResponse:
    return ChatResponse(
        doctor_id=doctor_id,
        message="Gemini integration is disabled in this project skeleton.",
        intent=request.intent,
    )
