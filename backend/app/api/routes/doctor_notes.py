import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.dependencies import get_current_doctor_id
from backend.app.db.models import DoctorNote
from backend.app.db.session import get_db
from backend.app.schemas.doctor_note import (
    DoctorNoteCreate,
    DoctorNoteRead,
    DoctorNoteSearchItem,
)
from backend.app.services.doctor_memory_service import (
    DoctorMemoryValidationError,
    DoctorMemoryService,
    NOTE_VALIDATION_ERROR_MESSAGE,
    get_doctor_memory_service,
    validate_doctor_note_content,
)

router = APIRouter(prefix="/doctor-notes", tags=["doctor-notes"])
logger = logging.getLogger(__name__)


@router.post("", response_model=DoctorNoteRead, status_code=201)
def create_doctor_note(
    payload: DoctorNoteCreate,
    doctor_id: str = Depends(get_current_doctor_id),
    db: Session = Depends(get_db),
    memory_service: DoctorMemoryService = Depends(get_doctor_memory_service),
) -> DoctorNote:
    try:
        validate_doctor_note_content(payload.note_text or payload.content or "")
    except DoctorMemoryValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc) or NOTE_VALIDATION_ERROR_MESSAGE,
        ) from exc

    note = DoctorNote(doctor_id=doctor_id, content=payload.content or "")
    db.add(note)
    db.commit()
    db.refresh(note)
    try:
        memory_service.save_note(
            doctor_id=doctor_id,
            note_id=str(note.id),
            title=payload.title,
            note_text=payload.note_text or payload.content or "",
            note_type=payload.note_type,
            source_context=payload.source_context,
            active_ingredients=payload.active_ingredients,
            drug_pair_keys=payload.drug_pair_keys,
            diagnosis_keywords=payload.diagnosis_keywords,
            patient_tags=payload.patient_tags,
            applicability=payload.applicability,
            priority=payload.priority,
            created_at=note.created_at.isoformat() if note.created_at else None,
        )
    except DoctorMemoryValidationError as exc:
        db.delete(note)
        db.commit()
        raise HTTPException(
            status_code=422,
            detail=str(exc) or NOTE_VALIDATION_ERROR_MESSAGE,
        ) from exc
    except Exception:
        logger.exception("doctor_memory_save_failed")
        pass
    return note


@router.get("/search", response_model=list[DoctorNoteSearchItem])
def search_doctor_notes(
    q: str = Query(min_length=1),
    top_k: int = Query(default=5, gt=0, le=20),
    doctor_id: str = Depends(get_current_doctor_id),
    memory_service: DoctorMemoryService = Depends(get_doctor_memory_service),
) -> list[dict]:
    try:
        return memory_service.search_notes(doctor_id=doctor_id, query=q, top_k=top_k)
    except Exception:
        logger.exception("doctor_memory_search_failed")
        return []


@router.get("", response_model=list[DoctorNoteRead])
def list_doctor_notes(
    doctor_id: str = Depends(get_current_doctor_id),
    db: Session = Depends(get_db),
) -> list[DoctorNote]:
    statement = (
        select(DoctorNote)
        .where(DoctorNote.doctor_id == doctor_id)
        .order_by(DoctorNote.created_at.desc())
    )
    return list(db.scalars(statement))
