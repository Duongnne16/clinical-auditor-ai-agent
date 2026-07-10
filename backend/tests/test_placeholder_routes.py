from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.dependencies import get_clinical_workflow_graph_service
from backend.app.core.security import create_access_token, hash_password
from backend.app.db.models import Base, User
from backend.app.db.session import get_db
from backend.app.main import app
from backend.app.services.doctor_memory_service import get_doctor_memory_service


class FakeClinicalWorkflowGraphService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(state)
        request = state["chat_request"]
        intent = request.intent or "single_drug_query"
        answer = (
            "Với dữ liệu hiện có, hệ thống ghi nhận thông tin cần rà soát."
        )
        if "bài văn" in request.message:
            intent = "out_of_scope"
            answer = (
                "Hệ thống này chỉ hỗ trợ tra cứu và rà soát thông tin liên quan "
                "đến thuốc, đơn thuốc, tương tác thuốc và một số lưu ý sử dụng "
                "thuốc. Vui lòng nhập câu hỏi trong phạm vi này."
            )
        elif "tương tác" in request.message:
            intent = "drug_interaction_query"
            answer = (
                "Với dữ liệu hiện có, hệ thống ghi nhận cần rà soát khả năng "
                "tương tác giữa hai thuốc."
            )
        return {
            "message": answer,
            "answer": answer,
            "intent": intent,
            "normalized_drugs": [],
            "sources": [],
            "warnings": [],
        }


class FakeDoctorMemoryService:
    def save_note(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def search_notes(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []


@pytest.fixture()
def placeholder_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "placeholder.sqlite3"
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
        app.dependency_overrides.pop(get_clinical_workflow_graph_service, None)
        app.dependency_overrides.pop(get_doctor_memory_service, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _override_graph_service(
    fake: FakeClinicalWorkflowGraphService | None = None,
) -> FakeClinicalWorkflowGraphService:
    service = fake or FakeClinicalWorkflowGraphService()
    app.dependency_overrides[get_clinical_workflow_graph_service] = lambda: service
    return service


def _override_memory_service() -> None:
    app.dependency_overrides[get_doctor_memory_service] = FakeDoctorMemoryService


def _override_failing_memory_service() -> None:
    def fail() -> None:
        raise AssertionError("Doctor Memory must not be used by /chat")

    app.dependency_overrides[get_doctor_memory_service] = fail


def _create_user(
    client: TestClient,
    *,
    doctor_id: str = "doctor-chat",
    email: str = "chat@example.test",
) -> User:
    testing_session = client.testing_session  # type: ignore[attr-defined]
    with testing_session() as session:
        user = User(
            doctor_id=doctor_id,
            email=email,
            full_name="Chat Doctor",
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


def test_chat_route_without_token_returns_401(
    placeholder_client: TestClient,
) -> None:
    _override_graph_service()

    response = placeholder_client.post(
        "/api/v1/chat",
        json={"message": "Paracetamol có tác dụng phụ gì?"},
    )

    assert response.status_code == 401


def test_chat_route_returns_backward_compatible_answer_fields_with_jwt(
    placeholder_client: TestClient,
) -> None:
    user = _create_user(placeholder_client, doctor_id="doctor-chat-token")
    fake = _override_graph_service()
    _override_failing_memory_service()

    response = placeholder_client.post(
        "/api/v1/chat",
        headers=_auth_headers(user),
        json={"message": "Paracetamol có tác dụng phụ gì?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "doctor_id",
        "message",
        "answer",
        "intent",
        "normalized_drugs",
        "sources",
        "warnings",
        "disclaimer",
    }
    assert body["doctor_id"] == "doctor-chat-token"
    assert body["doctor_id"] != "dev-doctor-001"
    assert body["intent"] == "single_drug_query"
    assert body["message"] == body["answer"]
    assert body["answer"]
    assert "Placeholder" not in body["disclaimer"]
    assert "không thay thế quyết định của bác sĩ/dược sĩ" in body["disclaimer"]
    assert len(fake.calls) == 1
    assert fake.calls[0]["request_type"] == "chat"
    assert fake.calls[0]["input_text"] == fake.calls[0]["chat_request"].message
    assert fake.calls[0]["doctor_id"] == "doctor-chat-token"
    assert fake.calls[0]["trace"] == []
    assert "doctor_memory" not in fake.calls[0]


def test_chat_route_can_return_out_of_scope_refusal_with_jwt(
    placeholder_client: TestClient,
) -> None:
    user = _create_user(
        placeholder_client,
        doctor_id="doctor-out-of-scope",
        email="scope@example.test",
    )
    fake = _override_graph_service()

    response = placeholder_client.post(
        "/api/v1/chat",
        headers=_auth_headers(user),
        json={"message": "Viết giúp tôi bài văn"},
    )

    assert response.status_code == 200
    assert response.json()["intent"] == "out_of_scope"
    assert "chỉ hỗ trợ tra cứu" in response.json()["answer"]
    assert len(fake.calls) == 1


def test_chat_route_can_return_interaction_answer_with_jwt(
    placeholder_client: TestClient,
) -> None:
    user = _create_user(
        placeholder_client,
        doctor_id="doctor-interaction",
        email="interaction@example.test",
    )
    fake = _override_graph_service()

    response = placeholder_client.post(
        "/api/v1/chat",
        headers=_auth_headers(user),
        json={
            "message": "Omeprazole có tương tác với Clopidogrel không?"
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["doctor_id"] == "doctor-interaction"
    assert body["intent"] == "drug_interaction_query"
    assert body["answer"]
    assert len(fake.calls) == 1


def test_doctor_notes_require_jwt_after_auth_migration(
    placeholder_client: TestClient,
) -> None:
    _override_memory_service()

    created = placeholder_client.post(
        "/api/v1/doctor-notes",
        json={"content": "Placeholder note for skeleton test"},
    )
    listed = placeholder_client.get("/api/v1/doctor-notes")

    assert created.status_code == 401
    assert listed.status_code == 401
