from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.dependencies import get_doctor_id
from backend.app.db.models import DoctorNote
from backend.app.db.session import get_db
from backend.app.schemas.doctor_note import DoctorNoteCreate, DoctorNoteRead

router = APIRouter(prefix="/doctor-notes", tags=["doctor-notes"])


@router.post("", response_model=DoctorNoteRead, status_code=201)
def create_doctor_note(
    payload: DoctorNoteCreate,
    doctor_id: str = Depends(get_doctor_id),
    db: Session = Depends(get_db),
) -> DoctorNote:
    note = DoctorNote(doctor_id=doctor_id, content=payload.content)
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


@router.get("", response_model=list[DoctorNoteRead])
def list_doctor_notes(
    doctor_id: str = Depends(get_doctor_id),
    db: Session = Depends(get_db),
) -> list[DoctorNote]:
    statement = (
        select(DoctorNote)
        .where(DoctorNote.doctor_id == doctor_id)
        .order_by(DoctorNote.created_at.desc())
    )
    return list(db.scalars(statement))
