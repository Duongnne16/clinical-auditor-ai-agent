from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ClinicalWorkflowRequest(BaseModel):
    input_text: str = Field(min_length=1)
    patient_context: dict[str, Any] = Field(default_factory=dict)
    use_gemini: bool = False
    query_types: list[str] | None = None
    top_k_per_type: int = Field(default=8, gt=0)

    @field_validator("input_text")
    @classmethod
    def input_text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("input_text must not be empty")
        return stripped
