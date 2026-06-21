from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.report import CLINICAL_DISCLAIMER


class DoctorNoteCreate(BaseModel):
    content: str = Field(min_length=1)


class DoctorNoteRead(BaseModel):
    id: int
    doctor_id: str
    content: str
    created_at: datetime
    disclaimer: str = CLINICAL_DISCLAIMER

    model_config = ConfigDict(from_attributes=True)
