from typing import Any

from pydantic import BaseModel, Field

CHAT_DISCLAIMER = (
    "Thông tin chỉ có mục đích hỗ trợ tra cứu chuyên môn, không thay thế "
    "quyết định của bác sĩ/dược sĩ."
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    intent: str | None = None


class ChatResponse(BaseModel):
    doctor_id: str
    message: str
    answer: str | None = None
    intent: str | None = None
    normalized_drugs: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = CHAT_DISCLAIMER
