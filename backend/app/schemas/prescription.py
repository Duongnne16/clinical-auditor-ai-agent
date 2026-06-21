from pydantic import BaseModel, Field

from backend.app.schemas.drug import Drug


class PrescriptionAuditRequest(BaseModel):
    prescription_id: str | None = None
    patient_context: str | None = None
    drugs: list[Drug] = Field(min_length=1)
