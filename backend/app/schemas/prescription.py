from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PrescriptionAuditRequest(BaseModel):
    prescription_text: str
    doctor_id: str | None = None
    patient_context: dict[str, Any] = Field(default_factory=dict)
    use_gemini: bool = False
    query_types: list[str] | None = None
    top_k_per_type: int = Field(default=8, gt=0)

    @field_validator("prescription_text")
    @classmethod
    def prescription_text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prescription_text must not be empty")
        return stripped


class ReportHistoryRead(BaseModel):
    id: int
    prescription_history_id: int
    report_status: str | None = None
    summary: str | None = None
    doctor_facing_response: str | None = None
    report_payload: dict[str, Any] | None = None
    created_at: datetime


class PrescriptionHistoryListItem(BaseModel):
    id: int
    status: str | None = None
    overall_risk_level: str | None = None
    report_status: str | None = None
    created_at: datetime


class PrescriptionHistoryDetail(BaseModel):
    id: int
    prescription_text: str
    patient_context: dict[str, Any] | None = None
    query_types: list[str] | None = None
    use_gemini: bool
    top_k_per_type: int
    status: str | None = None
    overall_risk_level: str | None = None
    warnings: list[Any] | None = None
    errors: list[Any] | None = None
    audit_payload: dict[str, Any] | None = None
    report: ReportHistoryRead | None = None
    created_at: datetime
