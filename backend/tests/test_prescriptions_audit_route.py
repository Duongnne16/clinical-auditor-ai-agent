from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api.routes.prescriptions import get_prescription_audit_service
from backend.app.core.security import create_access_token, hash_password
from backend.app.db.models import Base, User
from backend.app.db.session import get_db
from backend.app.main import app


class FakePrescriptionAuditService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def audit_text(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "status": "partial_success",
            "prescription_check": {"status": "evidence_ready"},
            "risk_analysis": {"status": "analysis_context_ready"},
            "report": {"status": "report_context_ready"},
            "doctor_memory": {"matched_notes": []},
            "warnings": [],
            "errors": [],
            "echo": kwargs,
        }


def _override_service(fake_service: FakePrescriptionAuditService) -> None:
    app.dependency_overrides[get_prescription_audit_service] = lambda: fake_service


def _clear_overrides() -> None:
    app.dependency_overrides.pop(get_prescription_audit_service, None)


@pytest.fixture()
def prescription_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "prescriptions.sqlite3"
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    testing_session = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False
    )
    Base.metadata.create_all(bind=engine)

    def override_get_db() -> Generator[Session, None, None]:
        with testing_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)
        client.testing_session = testing_session  # type: ignore[attr-defined]
        yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_prescription_audit_service, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _create_user(
    client: TestClient,
    *,
    doctor_id: str = "doctor-token",
    email: str = "prescription@example.test",
) -> User:
    testing_session = client.testing_session  # type: ignore[attr-defined]
    with testing_session() as session:
        user = User(
            doctor_id=doctor_id,
            email=email,
            full_name="Prescription Doctor",
            hashed_password=hash_password("secure-password-123"),
            is_active=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def _auth_headers(user: User) -> dict[str, str]:
    token = create_access_token(
        subject=str(user.id),
        doctor_id=user.doctor_id,
        email=user.email,
    )
    return {"Authorization": f"Bearer {token}"}


def test_prescription_audit_service_dependency_is_cached() -> None:
    get_prescription_audit_service.cache_clear()
    try:
        first = get_prescription_audit_service()
        second = get_prescription_audit_service()
    finally:
        get_prescription_audit_service.cache_clear()

    assert first is second


def test_prescription_audit_without_token_returns_401(
    prescription_client: TestClient,
) -> None:
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        json={"prescription_text": "1. Omeprazol 20mg"},
    )

    assert response.status_code == 401
    assert fake_service.calls == []


def test_prescription_audit_route_returns_fake_service_response_with_token(
    prescription_client: TestClient,
) -> None:
    user = _create_user(prescription_client, doctor_id="doctor-from-token")
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        headers=_auth_headers(user),
        json={
            "prescription_text": "1. Omeprazol 20mg",
            "doctor_id": "doctor-from-body",
            "patient_context": {"age": 60},
            "use_gemini": False,
            "query_types": ["interaction"],
            "top_k_per_type": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial_success"
    assert body["report"]["status"] == "report_context_ready"
    assert {
        "status",
        "prescription_check",
        "risk_analysis",
        "report",
        "doctor_memory",
        "warnings",
        "errors",
    }.issubset(body)
    assert fake_service.calls == [
        {
            "prescription_text": "1. Omeprazol 20mg",
            "doctor_id": "doctor-from-token",
            "patient_context": {"age": 60},
            "use_gemini": False,
            "query_types": ["interaction"],
            "top_k_per_type": 3,
        }
    ]


def test_prescription_audit_route_uses_token_doctor_when_body_missing(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-token-missing-body",
        email="missing-body@example.test",
    )
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        headers=_auth_headers(user),
        json={"prescription_text": "1. Paracetamol 500mg"},
    )

    assert response.status_code == 200
    assert fake_service.calls[0]["doctor_id"] == "doctor-token-missing-body"
    assert fake_service.calls[0]["patient_context"] == {}
    assert fake_service.calls[0]["use_gemini"] is False
    assert fake_service.calls[0]["top_k_per_type"] == 8


def test_prescription_audit_route_passes_use_gemini_true_to_fake_service(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-gemini",
        email="gemini@example.test",
    )
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        headers=_auth_headers(user),
        json={
            "prescription_text": "1. Omeprazol 20mg",
            "use_gemini": True,
        },
    )

    assert response.status_code == 200
    assert fake_service.calls[0]["use_gemini"] is True


def test_prescription_text_empty_returns_422(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-validation-empty",
        email="validation-empty@example.test",
    )
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        headers=_auth_headers(user),
        json={"prescription_text": ""},
    )

    assert response.status_code == 422
    assert fake_service.calls == []


def test_prescription_text_whitespace_only_returns_422(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-validation-whitespace",
        email="validation-whitespace@example.test",
    )
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        headers=_auth_headers(user),
        json={"prescription_text": "   "},
    )

    assert response.status_code == 422
    assert fake_service.calls == []


def test_top_k_per_type_zero_returns_422(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-validation-top-k",
        email="validation-top-k@example.test",
    )
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        headers=_auth_headers(user),
        json={
            "prescription_text": "1. Omeprazol 20mg",
            "top_k_per_type": 0,
        },
    )

    assert response.status_code == 422
    assert fake_service.calls == []
