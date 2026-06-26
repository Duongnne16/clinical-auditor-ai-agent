from functools import lru_cache

from fastapi import APIRouter, Depends

from backend.app.core.dependencies import get_doctor_id
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.chat_query_service import ChatQueryService

router = APIRouter(prefix="/chat", tags=["chat"])


@lru_cache
def get_chat_query_service() -> ChatQueryService:
    return ChatQueryService()


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    doctor_id: str = Depends(get_doctor_id),
    service: ChatQueryService = Depends(get_chat_query_service),
) -> ChatResponse:
    result = service.answer(request)
    return ChatResponse(doctor_id=doctor_id, **result)
