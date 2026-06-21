from pydantic import BaseModel


CLINICAL_DISCLAIMER = (
    "Placeholder only. This response is not a clinical conclusion or medical advice."
)


class Warning(BaseModel):
    code: str
    message: str
    severity: str = "unknown"


class AuditReport(BaseModel):
    prescription_id: str | None = None
    doctor_id: str
    status: str
    warnings: list[Warning]
    disclaimer: str = CLINICAL_DISCLAIMER
