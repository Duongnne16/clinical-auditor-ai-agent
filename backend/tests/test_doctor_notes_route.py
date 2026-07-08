from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.security import create_access_token, hash_password
from backend.app.db.models import Base, DoctorNote, User
from backend.app.db.session import get_db
from backend.app.main import app
from backend.app.services.doctor_memory_service import get_doctor_memory_service


class FakeDoctorMemoryService:
    def __init__(
        self,
        notes: list[dict[str, Any]] | None = None,
        raises: bool = False,
    ) -> None:
        self.notes = notes or []
        self.raises = raises
        self.save_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []

    def save_note(self, **kwargs: Any) -> dict[str, Any]:
        self.save_calls.append(kwargs)
        if self.raises:
            raise RuntimeError("memory unavailable")
        return kwargs

    def search_notes(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.search_calls.append(kwargs)
        if self.raises:
            raise RuntimeError("memory unavailable")
        doctor_id = kwargs["doctor_id"]
        return [
            note
            for note in self.notes
            if note.get("doctor_id", doctor_id) == doctor_id
        ]


@pytest.fixture()
def doctor_notes_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "doctor_notes.sqlite3"
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
        app.dependency_overrides.pop(get_doctor_memory_service, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _override_memory(fake: FakeDoctorMemoryService) -> None:
    app.dependency_overrides[get_doctor_memory_service] = lambda: fake


def _create_user(
    client: TestClient,
    *,
    doctor_id: str = "doctor-token",
    email: str = "doctor@example.test",
    is_active: bool = True,
) -> User:
    testing_session = client.testing_session  # type: ignore[attr-defined]
    with testing_session() as session:
        user = User(
            doctor_id=doctor_id,
            email=email,
            full_name="JWT Doctor",
            hashed_password=hash_password("secure-password-123"),
            is_active=is_active,
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


def _insert_note(
    client: TestClient,
    *,
    doctor_id: str,
    content: str,
) -> DoctorNote:
    testing_session = client.testing_session  # type: ignore[attr-defined]
    with testing_session() as session:
        note = DoctorNote(doctor_id=doctor_id, content=content)
        session.add(note)
        session.commit()
        session.refresh(note)
        return note


def test_create_doctor_note_without_token_returns_401(
    doctor_notes_client: TestClient,
) -> None:
    response = doctor_notes_client.post(
        "/api/v1/doctor-notes",
        json={"content": "Requires auth"},
    )

    assert response.status_code == 401


def test_search_and_list_without_token_return_401(
    doctor_notes_client: TestClient,
) -> None:
    search = doctor_notes_client.get("/api/v1/doctor-notes/search?q=abc")
    listing = doctor_notes_client.get("/api/v1/doctor-notes")

    assert search.status_code == 401
    assert listing.status_code == 401


def test_create_doctor_note_uses_doctor_id_from_token(
    doctor_notes_client: TestClient,
) -> None:
    user = _create_user(doctor_notes_client, doctor_id="doctor-jwt")
    fake = FakeDoctorMemoryService()
    _override_memory(fake)

    response = doctor_notes_client.post(
        "/api/v1/doctor-notes",
        headers=_auth_headers(user),
        json={
            "content": "Fallback content",
            "title": "Levofloxacin + Sucralfate",
            "note_text": "Rà soát thời điểm dùng.",
            "note_type": "drug_interaction_note",
            "source_context": "prescription_audit",
            "active_ingredients": ["levofloxacin", "sucralfate"],
            "drug_pair_keys": ["levofloxacin|sucralfate"],
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["doctor_id"] == "doctor-jwt"
    assert body["content"] == "Fallback content"
    assert len(fake.save_calls) == 1
    call = fake.save_calls[0]
    assert call["doctor_id"] == "doctor-jwt"
    assert call["note_text"] == "Rà soát thời điểm dùng."
    assert call["active_ingredients"] == ["levofloxacin", "sucralfate"]
    assert call["drug_pair_keys"] == ["levofloxacin|sucralfate"]


def test_create_doctor_note_ignores_body_doctor_id(
    doctor_notes_client: TestClient,
) -> None:
    user = _create_user(doctor_notes_client, doctor_id="doctor-token")
    fake = FakeDoctorMemoryService()
    _override_memory(fake)

    response = doctor_notes_client.post(
        "/api/v1/doctor-notes",
        headers=_auth_headers(user),
        json={
            "doctor_id": "doctor-from-body",
            "content": "Body doctor id must be ignored",
        },
    )

    assert response.status_code == 201
    assert response.json()["doctor_id"] == "doctor-token"
    assert fake.save_calls[0]["doctor_id"] == "doctor-token"


def test_search_doctor_notes_uses_doctor_id_from_token(
    doctor_notes_client: TestClient,
) -> None:
    user = _create_user(doctor_notes_client, doctor_id="doctor-a")
    fake = FakeDoctorMemoryService(
        notes=[
            {
                "note_id": "n1",
                "doctor_id": "doctor-a",
                "title": "Mine",
                "note_text": "Private note",
                "score": 0.9,
                "match_reason": "semantic_match",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "note_id": "n2",
                "doctor_id": "doctor-b",
                "title": "Other",
                "note_text": "Other private note",
                "score": 0.95,
                "match_reason": "semantic_match",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
        ]
    )
    _override_memory(fake)

    response = doctor_notes_client.get(
        "/api/v1/doctor-notes/search?q=levofloxacin&doctor_id=doctor-b",
        headers=_auth_headers(user),
    )

    assert response.status_code == 200
    body = response.json()
    assert [note["note_id"] for note in body] == ["n1"]
    assert fake.search_calls == [
        {"doctor_id": "doctor-a", "query": "levofloxacin", "top_k": 5}
    ]


def test_list_doctor_notes_returns_only_token_doctor_notes(
    doctor_notes_client: TestClient,
) -> None:
    user = _create_user(doctor_notes_client, doctor_id="doctor-a")
    _insert_note(doctor_notes_client, doctor_id="doctor-a", content="Mine")
    _insert_note(doctor_notes_client, doctor_id="doctor-b", content="Other")

    response = doctor_notes_client.get(
        "/api/v1/doctor-notes",
        headers=_auth_headers(user),
    )

    assert response.status_code == 200
    body = response.json()
    assert [note["content"] for note in body] == ["Mine"]
    assert body[0]["doctor_id"] == "doctor-a"


def test_create_doctor_note_survives_memory_failure(
    doctor_notes_client: TestClient,
) -> None:
    user = _create_user(doctor_notes_client, doctor_id="doctor-memory")
    fake = FakeDoctorMemoryService(raises=True)
    _override_memory(fake)

    response = doctor_notes_client.post(
        "/api/v1/doctor-notes",
        headers=_auth_headers(user),
        json={"content": "SQLite note survives"},
    )

    assert response.status_code == 201
    assert response.json()["content"] == "SQLite note survives"
    assert len(fake.save_calls) == 1


def test_search_doctor_notes_returns_empty_on_memory_failure(
    doctor_notes_client: TestClient,
) -> None:
    user = _create_user(doctor_notes_client, doctor_id="doctor-memory")
    _override_memory(FakeDoctorMemoryService(raises=True))

    response = doctor_notes_client.get(
        "/api/v1/doctor-notes/search?q=levofloxacin",
        headers=_auth_headers(user),
    )

    assert response.status_code == 200
    assert response.json() == []
