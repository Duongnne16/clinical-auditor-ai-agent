from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.dependencies import (
    get_current_doctor_id,
    get_unified_clinical_workflow_graph_service,
)
from backend.app.db.session import get_db
from backend.app.api.routes.prescriptions import _save_audit_history
from backend.app.schemas.chat import ChatRequest
from backend.app.schemas.clinical_workflow import ClinicalWorkflowRequest
from backend.app.schemas.prescription import PrescriptionAuditRequest
from backend.app.services.clinical_workflow_graph import ClinicalWorkflowGraphService

router = APIRouter(prefix="/clinical-workflow", tags=["clinical-workflow"])


@router.post("/run")
def run_clinical_workflow(
    request: ClinicalWorkflowRequest,
    doctor_id: str = Depends(get_current_doctor_id),
    db: Session = Depends(get_db),
    graph: ClinicalWorkflowGraphService = Depends(
        get_unified_clinical_workflow_graph_service
    ),
) -> dict[str, Any]:
    result = graph.run(
        {
            "request_type": "clinical_workflow",
            "input_text": request.input_text,
            "chat_request": ChatRequest(message=request.input_text),
            "prescription_request": {
                "prescription_text": request.input_text,
                "patient_context": request.patient_context,
                "use_gemini": request.use_gemini,
                "query_types": request.query_types,
                "top_k_per_type": request.top_k_per_type,
            },
            "doctor_id": doctor_id,
            "trace": [],
        }
    )
    if isinstance(result, dict) and result.get("result_type") == "audit":
        audit_result = result.get("audit_result")
        if isinstance(audit_result, dict):
            _save_audit_history(
                db=db,
                doctor_id=doctor_id,
                request=PrescriptionAuditRequest(
                    prescription_text=request.input_text,
                    patient_context=request.patient_context,
                    use_gemini=request.use_gemini,
                    query_types=request.query_types,
                    top_k_per_type=request.top_k_per_type,
                ),
                audit_result=audit_result,
            )
    if isinstance(result, dict) and "doctor_id" not in result:
        return {"doctor_id": doctor_id, **result}
    return result
