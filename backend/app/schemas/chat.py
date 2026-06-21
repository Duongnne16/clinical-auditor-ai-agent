from pydantic import BaseModel, Field

from backend.app.schemas.report import CLINICAL_DISCLAIMER


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    intent: str | None = None


class ChatResponse(BaseModel):
    doctor_id: str
    message: str
    intent: str | None = None
    disclaimer: str = CLINICAL_DISCLAIMER
