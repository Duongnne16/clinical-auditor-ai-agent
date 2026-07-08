from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DoctorNote(Base):
    __tablename__ = "doctor_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    doctor_id: Mapped[str] = mapped_column(String(100), index=True)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    doctor_id: Mapped[str] = mapped_column(
        String(100), unique=True, index=True
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class PrescriptionHistory(Base):
    __tablename__ = "prescription_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    doctor_id: Mapped[str] = mapped_column(String(100), index=True)
    prescription_text: Mapped[str] = mapped_column(Text)
    patient_context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    query_types: Mapped[list | None] = mapped_column(JSON, nullable=True)
    use_gemini: Mapped[bool] = mapped_column(Boolean, default=False)
    top_k_per_type: Mapped[int] = mapped_column(default=8)
    status: Mapped[str | None] = mapped_column(String(100), nullable=True)
    overall_risk_level: Mapped[str | None] = mapped_column(String(100), nullable=True)
    warnings: Mapped[list | None] = mapped_column(JSON, nullable=True)
    errors: Mapped[list | None] = mapped_column(JSON, nullable=True)
    audit_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class ReportHistory(Base):
    __tablename__ = "report_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    doctor_id: Mapped[str] = mapped_column(String(100), index=True)
    prescription_history_id: Mapped[int] = mapped_column(
        ForeignKey("prescription_history.id"),
        index=True,
    )
    report_status: Mapped[str | None] = mapped_column(String(100), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    doctor_facing_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
