from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.dependencies import (
    get_current_doctor_id,
    get_prescription_audit_service,
    get_prescription_workflow_graph_service,
)
from backend.app.db.models import PrescriptionHistory, ReportHistory
from backend.app.db.session import get_db
from backend.app.schemas.prescription import (
    PrescriptionAuditRequest,
    PrescriptionHistoryDetail,
    PrescriptionHistoryListItem,
    ReportHistoryRead,
)
from backend.app.services.clinical_workflow_graph import ClinicalWorkflowGraphService

router = APIRouter(prefix="/prescriptions", tags=["prescriptions"])
logger = logging.getLogger(__name__)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _extract_overall_risk_level(audit_result: dict[str, Any]) -> str | None:
    report = _as_dict(audit_result.get("report"))
    risk_analysis = _as_dict(audit_result.get("risk_analysis"))
    value = report.get("overall_risk_level") or risk_analysis.get("overall_risk_level")
    return str(value) if value is not None else None


def _save_audit_history(
    *,
    db: Session,
    doctor_id: str,
    request: PrescriptionAuditRequest,
    audit_result: dict[str, Any],
) -> None:
    report = audit_result.get("report")
    report_payload = _as_dict(report)

    history = PrescriptionHistory(
        doctor_id=doctor_id,
        prescription_text=request.prescription_text,
        patient_context=jsonable_encoder(request.patient_context),
        query_types=jsonable_encoder(request.query_types),
        use_gemini=request.use_gemini,
        top_k_per_type=request.top_k_per_type,
        status=str(audit_result.get("status"))
        if audit_result.get("status") is not None
        else None,
        overall_risk_level=_extract_overall_risk_level(audit_result),
        warnings=jsonable_encoder(audit_result.get("warnings")),
        errors=jsonable_encoder(audit_result.get("errors")),
        audit_payload=jsonable_encoder(audit_result),
    )
    db.add(history)
    db.flush()

    if isinstance(report, dict):
        db.add(
            ReportHistory(
                doctor_id=doctor_id,
                prescription_history_id=history.id,
                report_status=str(report.get("status"))
                if report.get("status") is not None
                else None,
                summary=str(report.get("summary"))
                if report.get("summary") is not None
                else None,
                doctor_facing_response=str(report.get("doctor_facing_response"))
                if report.get("doctor_facing_response") is not None
                else None,
                report_payload=jsonable_encoder(report_payload),
            )
        )

    db.commit()


def _report_read(report: ReportHistory | None) -> ReportHistoryRead | None:
    if report is None:
        return None
    return ReportHistoryRead(
        id=report.id,
        prescription_history_id=report.prescription_history_id,
        report_status=report.report_status,
        summary=report.summary,
        doctor_facing_response=report.doctor_facing_response,
        report_payload=report.report_payload,
        created_at=report.created_at,
    )


@router.post("/audit")
def audit_prescription(
    request: PrescriptionAuditRequest,
    doctor_id: str = Depends(get_current_doctor_id),
    db: Session = Depends(get_db),
    graph: ClinicalWorkflowGraphService = Depends(
        get_prescription_workflow_graph_service
    ),
) -> dict[str, Any]:
    audit_result = graph.run(
        {
            "request_type": "prescription_audit",
            "prescription_request": request,
            "input_text": request.prescription_text,
            "doctor_id": doctor_id,
            "trace": [],
        }
    )
    try:
        _save_audit_history(
            db=db,
            doctor_id=doctor_id,
            request=request,
            audit_result=audit_result,
        )
    except Exception:
        db.rollback()
        logger.exception("prescription_audit_history_save_failed")
    return audit_result


@router.get("/history", response_model=list[PrescriptionHistoryListItem])
def list_prescription_history(
    limit: int = Query(default=20, gt=0, le=100),
    offset: int = Query(default=0, ge=0),
    doctor_id: str = Depends(get_current_doctor_id),
    db: Session = Depends(get_db),
) -> list[PrescriptionHistoryListItem]:
    statement = (
        select(PrescriptionHistory, ReportHistory.report_status)
        .outerjoin(
            ReportHistory,
            ReportHistory.prescription_history_id == PrescriptionHistory.id,
        )
        .where(PrescriptionHistory.doctor_id == doctor_id)
        .order_by(PrescriptionHistory.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return [
        PrescriptionHistoryListItem(
            id=history.id,
            status=history.status,
            overall_risk_level=history.overall_risk_level,
            report_status=report_status,
            created_at=history.created_at,
        )
        for history, report_status in db.execute(statement)
    ]


@router.get("/history/{history_id}", response_model=PrescriptionHistoryDetail)
def get_prescription_history(
    history_id: int,
    doctor_id: str = Depends(get_current_doctor_id),
    db: Session = Depends(get_db),
) -> PrescriptionHistoryDetail:
    history = db.scalar(
        select(PrescriptionHistory).where(
            PrescriptionHistory.id == history_id,
            PrescriptionHistory.doctor_id == doctor_id,
        )
    )
    if history is None:
        raise HTTPException(status_code=404, detail="Prescription history not found")

    report = db.scalar(
        select(ReportHistory).where(
            ReportHistory.prescription_history_id == history.id,
            ReportHistory.doctor_id == doctor_id,
        )
    )
    return PrescriptionHistoryDetail(
        id=history.id,
        prescription_text=history.prescription_text,
        patient_context=history.patient_context,
        query_types=history.query_types,
        use_gemini=history.use_gemini,
        top_k_per_type=history.top_k_per_type,
        status=history.status,
        overall_risk_level=history.overall_risk_level,
        warnings=history.warnings,
        errors=history.errors,
        audit_payload=history.audit_payload,
        report=_report_read(report),
        created_at=history.created_at,
    )
