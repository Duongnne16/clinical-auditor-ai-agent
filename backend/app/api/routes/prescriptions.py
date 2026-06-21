from fastapi import APIRouter, Depends

from backend.app.core.dependencies import get_doctor_id
from backend.app.schemas.prescription import PrescriptionAuditRequest
from backend.app.schemas.report import AuditReport

router = APIRouter(prefix="/prescriptions", tags=["prescriptions"])


@router.post("/audit", response_model=AuditReport)
def audit_prescription(
    prescription: PrescriptionAuditRequest,
    doctor_id: str = Depends(get_doctor_id),
) -> AuditReport:
    return AuditReport(
        prescription_id=prescription.prescription_id,
        doctor_id=doctor_id,
        status="placeholder",
        warnings=[],
    )
