from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api.routes.prescriptions import get_prescription_audit_service
from backend.app.core.dependencies import get_prescription_workflow_graph_service
from backend.app.core.security import create_access_token, hash_password
from backend.app.db.models import Base, PrescriptionHistory, ReportHistory, User
from backend.app.db.session import get_db
from backend.app.main import app


class FakePrescriptionWorkflowGraphService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.states: list[dict[str, Any]] = []

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        self.states.append(state)
        request = state["prescription_request"]
        kwargs = {
            "prescription_text": request.prescription_text,
            "doctor_id": state["doctor_id"],
            "patient_context": request.patient_context,
            "use_gemini": request.use_gemini,
            "query_types": request.query_types,
            "top_k_per_type": request.top_k_per_type,
        }
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


def _override_service(fake_service: FakePrescriptionWorkflowGraphService) -> None:
    app.dependency_overrides[get_prescription_workflow_graph_service] = (
        lambda: fake_service
    )


def _clear_overrides() -> None:
    app.dependency_overrides.pop(get_prescription_workflow_graph_service, None)


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
        app.dependency_overrides.pop(get_prescription_workflow_graph_service, None)
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


def _insert_history(
    client: TestClient,
    *,
    doctor_id: str,
    prescription_text: str,
    status: str = "success",
    created_at: datetime | None = None,
    report_status: str | None = "report_ready",
) -> PrescriptionHistory:
    testing_session = client.testing_session  # type: ignore[attr-defined]
    with testing_session() as session:
        history = PrescriptionHistory(
            doctor_id=doctor_id,
            prescription_text=prescription_text,
            patient_context={"age": 60},
            query_types=["interaction"],
            use_gemini=False,
            top_k_per_type=5,
            status=status,
            overall_risk_level="moderate",
            warnings=["review_needed"],
            errors=[],
            audit_payload={
                "status": status,
                "report": {"status": report_status},
            },
            created_at=created_at or datetime.now(timezone.utc),
        )
        session.add(history)
        session.flush()
        if report_status is not None:
            session.add(
                ReportHistory(
                    doctor_id=doctor_id,
                    prescription_history_id=history.id,
                    report_status=report_status,
                    summary=f"Summary for {prescription_text}",
                    doctor_facing_response="Doctor-facing response",
                    report_payload={
                        "status": report_status,
                        "summary": f"Summary for {prescription_text}",
                    },
                )
            )
        session.commit()
        session.refresh(history)
        return history


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
    fake_service = FakePrescriptionWorkflowGraphService()
    _override_service(fake_service)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        json={"prescription_text": "1. Omeprazol 20mg"},
    )

    assert response.status_code == 401
    assert fake_service.calls == []
    assert fake_service.states == []


def test_prescription_audit_route_returns_fake_service_response_with_token(
    prescription_client: TestClient,
) -> None:
    user = _create_user(prescription_client, doctor_id="doctor-from-token")
    fake_service = FakePrescriptionWorkflowGraphService()
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
    assert len(fake_service.states) == 1
    state = fake_service.states[0]
    assert state["request_type"] == "prescription_audit"
    assert state["prescription_request"].prescription_text == "1. Omeprazol 20mg"
    assert state["input_text"] == "1. Omeprazol 20mg"
    assert state["doctor_id"] == "doctor-from-token"
    assert state["trace"] == []


def test_prescription_audit_with_token_saves_history_and_report(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-history",
        email="history@example.test",
    )
    fake_service = FakePrescriptionWorkflowGraphService()
    _override_service(fake_service)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        headers=_auth_headers(user),
        json={
            "prescription_text": "1. Omeprazol 20mg",
            "patient_context": {"age": 60},
            "query_types": ["interaction"],
            "top_k_per_type": 3,
        },
    )

    assert response.status_code == 200
    assert response.json()["echo"]["doctor_id"] == "doctor-history"

    testing_session = prescription_client.testing_session  # type: ignore[attr-defined]
    with testing_session() as session:
        histories = list(session.scalars(select(PrescriptionHistory)))
        reports = list(session.scalars(select(ReportHistory)))

    assert len(histories) == 1
    assert histories[0].doctor_id == "doctor-history"
    assert histories[0].prescription_text == "1. Omeprazol 20mg"
    assert histories[0].patient_context == {"age": 60}
    assert histories[0].query_types == ["interaction"]
    assert histories[0].status == "partial_success"
    assert histories[0].audit_payload["echo"]["doctor_id"] == "doctor-history"
    assert len(reports) == 1
    assert reports[0].doctor_id == "doctor-history"
    assert reports[0].prescription_history_id == histories[0].id
    assert reports[0].report_status == "report_context_ready"


def test_prescription_audit_returns_original_response_when_history_save_fails(
    prescription_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-history-failure",
        email="history-failure@example.test",
    )
    fake_service = FakePrescriptionWorkflowGraphService()
    _override_service(fake_service)

    def fail_commit(self: Session) -> None:
        raise RuntimeError("commit failed")

    monkeypatch.setattr(Session, "commit", fail_commit)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        headers=_auth_headers(user),
        json={"prescription_text": "1. Omeprazol 20mg"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial_success"
    assert body["echo"]["doctor_id"] == "doctor-history-failure"


def test_prescription_audit_route_uses_token_doctor_when_body_missing(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-token-missing-body",
        email="missing-body@example.test",
    )
    fake_service = FakePrescriptionWorkflowGraphService()
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
    fake_service = FakePrescriptionWorkflowGraphService()
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
    fake_service = FakePrescriptionWorkflowGraphService()
    _override_service(fake_service)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        headers=_auth_headers(user),
        json={"prescription_text": ""},
    )

    assert response.status_code == 422
    assert fake_service.calls == []
    assert fake_service.states == []


def test_prescription_history_without_token_returns_401(
    prescription_client: TestClient,
) -> None:
    response = prescription_client.get("/api/v1/prescriptions/history")

    assert response.status_code == 401


def test_prescription_history_returns_only_current_doctor_with_limit_offset(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-list",
        email="history-list@example.test",
    )
    now = datetime.now(timezone.utc)
    older = _insert_history(
        prescription_client,
        doctor_id="doctor-list",
        prescription_text="older",
        created_at=now - timedelta(minutes=2),
    )
    newer = _insert_history(
        prescription_client,
        doctor_id="doctor-list",
        prescription_text="newer",
        created_at=now - timedelta(minutes=1),
    )
    _insert_history(
        prescription_client,
        doctor_id="doctor-other",
        prescription_text="other",
        created_at=now,
    )

    first_page = prescription_client.get(
        "/api/v1/prescriptions/history?limit=1",
        headers=_auth_headers(user),
    )
    second_page = prescription_client.get(
        "/api/v1/prescriptions/history?limit=1&offset=1",
        headers=_auth_headers(user),
    )

    assert first_page.status_code == 200
    assert second_page.status_code == 200
    assert [item["id"] for item in first_page.json()] == [newer.id]
    assert [item["id"] for item in second_page.json()] == [older.id]
    assert first_page.json()[0]["report_status"] == "report_ready"


def test_prescription_history_detail_returns_own_record(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-detail",
        email="history-detail@example.test",
    )
    history = _insert_history(
        prescription_client,
        doctor_id="doctor-detail",
        prescription_text="1. Paracetamol 500mg",
    )

    response = prescription_client.get(
        f"/api/v1/prescriptions/history/{history.id}",
        headers=_auth_headers(user),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == history.id
    assert body["prescription_text"] == "1. Paracetamol 500mg"
    assert body["patient_context"] == {"age": 60}
    assert body["query_types"] == ["interaction"]
    assert body["audit_payload"]["status"] == "success"
    assert body["report"]["report_status"] == "report_ready"
    assert body["report"]["doctor_facing_response"] == "Doctor-facing response"


def test_prescription_history_detail_for_other_doctor_returns_404(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-detail-owner",
        email="history-detail-owner@example.test",
    )
    history = _insert_history(
        prescription_client,
        doctor_id="doctor-detail-other",
        prescription_text="private",
    )

    response = prescription_client.get(
        f"/api/v1/prescriptions/history/{history.id}",
        headers=_auth_headers(user),
    )

    assert response.status_code == 404


def test_prescription_text_whitespace_only_returns_422(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-validation-whitespace",
        email="validation-whitespace@example.test",
    )
    fake_service = FakePrescriptionWorkflowGraphService()
    _override_service(fake_service)

    response = prescription_client.post(
        "/api/v1/prescriptions/audit",
        headers=_auth_headers(user),
        json={"prescription_text": "   "},
    )

    assert response.status_code == 422
    assert fake_service.calls == []
    assert fake_service.states == []


def test_top_k_per_type_zero_returns_422(
    prescription_client: TestClient,
) -> None:
    user = _create_user(
        prescription_client,
        doctor_id="doctor-validation-top-k",
        email="validation-top-k@example.test",
    )
    fake_service = FakePrescriptionWorkflowGraphService()
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
    assert fake_service.states == []
