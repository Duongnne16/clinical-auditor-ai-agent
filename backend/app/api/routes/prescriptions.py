from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from backend.app.core.dependencies import get_doctor_id
from backend.app.schemas.prescription import PrescriptionAuditRequest
from backend.app.services.prescription_audit_service import PrescriptionAuditService

router = APIRouter(prefix="/prescriptions", tags=["prescriptions"])


def get_prescription_audit_service() -> PrescriptionAuditService:
    return PrescriptionAuditService()


@router.post("/audit")
def audit_prescription(
    request: PrescriptionAuditRequest,
    development_doctor_id: str = Depends(get_doctor_id),
    service: PrescriptionAuditService = Depends(get_prescription_audit_service),
) -> dict[str, Any]:
    return service.audit_text(
        prescription_text=request.prescription_text,
        doctor_id=request.doctor_id or development_doctor_id,
        patient_context=request.patient_context,
        use_gemini=request.use_gemini,
        query_types=request.query_types,
        top_k_per_type=request.top_k_per_type,
    )
