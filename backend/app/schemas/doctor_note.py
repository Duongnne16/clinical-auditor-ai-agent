from datetime import datetime

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.report import CLINICAL_DISCLAIMER


class DoctorNoteCreate(BaseModel):
    content: str | None = Field(default=None, min_length=1)
    title: str | None = None
    note_text: str | None = Field(default=None, min_length=1)
    note_type: str | None = None
    source_context: str | None = None
    active_ingredients: list[str] = Field(default_factory=list)
    drug_pair_keys: list[str] = Field(default_factory=list)
    diagnosis_keywords: list[str] = Field(default_factory=list)
    patient_tags: list[str] = Field(default_factory=list)
    applicability: dict[str, Any] = Field(default_factory=dict)
    priority: str | None = None

    @model_validator(mode="after")
    def require_text(self) -> "DoctorNoteCreate":
        content = (self.content or "").strip()
        note_text = (self.note_text or "").strip()
        if not content and not note_text:
            raise ValueError("content or note_text is required")
        if not self.content:
            self.content = note_text
        if not self.note_text:
            self.note_text = content
        return self


class DoctorNoteRead(BaseModel):
    id: int
    doctor_id: str
    content: str
    created_at: datetime
    disclaimer: str = CLINICAL_DISCLAIMER
    title: str | None = None
    note_text: str | None = None
    note_type: str | None = None
    source_context: str | None = None
    active_ingredients: list[str] = Field(default_factory=list)
    drug_pair_keys: list[str] = Field(default_factory=list)
    diagnosis_keywords: list[str] = Field(default_factory=list)
    patient_tags: list[str] = Field(default_factory=list)
    applicability: dict[str, Any] = Field(default_factory=dict)
    priority: str | None = None

    model_config = ConfigDict(from_attributes=True)


class DoctorNoteSearchItem(BaseModel):
    note_id: str | None = None
    title: str | None = None
    note_text: str | None = None
    note_type: str | None = None
    source_context: str | None = None
    active_ingredients: list[str] = Field(default_factory=list)
    drug_pair_keys: list[str] = Field(default_factory=list)
    diagnosis_keywords: list[str] = Field(default_factory=list)
    patient_tags: list[str] = Field(default_factory=list)
    applicability: dict[str, Any] = Field(default_factory=dict)
    priority: str | None = None
    score: float = 0.0
    match_reason: str | None = None
    created_at: str | None = None
