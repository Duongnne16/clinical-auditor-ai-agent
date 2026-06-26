from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from backend.app.core.dependencies import get_doctor_id
from backend.app.main import app
from backend.app.services.doctor_memory_service import get_doctor_memory_service


class FakeDoctorMemoryService:
    def __init__(self, notes: list[dict[str, Any]] | None = None, raises: bool = False) -> None:
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


def _override_memory(fake: FakeDoctorMemoryService) -> None:
    app.dependency_overrides[get_doctor_memory_service] = lambda: fake


def _override_doctor_id(value: str) -> None:
    app.dependency_overrides[get_doctor_id] = lambda: value


def _clear_overrides() -> None:
    app.dependency_overrides.pop(get_doctor_memory_service, None)
    app.dependency_overrides.pop(get_doctor_id, None)


def test_create_doctor_note_stores_sqlite_and_upserts_memory() -> None:
    fake = FakeDoctorMemoryService()
    _override_memory(fake)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/doctor-notes",
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
    finally:
        _clear_overrides()

    assert response.status_code == 201
    body = response.json()
    assert body["doctor_id"] == "dev-doctor-001"
    assert body["content"] == "Fallback content"
    assert len(fake.save_calls) == 1
    call = fake.save_calls[0]
    assert call["doctor_id"] == "dev-doctor-001"
    assert call["note_text"] == "Rà soát thời điểm dùng."
    assert call["active_ingredients"] == ["levofloxacin", "sucralfate"]
    assert call["drug_pair_keys"] == ["levofloxacin|sucralfate"]


def test_create_doctor_note_survives_memory_failure() -> None:
    fake = FakeDoctorMemoryService(raises=True)
    _override_memory(fake)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/doctor-notes",
                json={"content": "SQLite note survives"},
            )
    finally:
        _clear_overrides()

    assert response.status_code == 201
    assert response.json()["content"] == "SQLite note survives"
    assert len(fake.save_calls) == 1


def test_search_doctor_notes_filters_to_current_doctor() -> None:
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
    _override_doctor_id("doctor-a")
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/doctor-notes/search?q=levofloxacin")
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert [note["note_id"] for note in body] == ["n1"]
    assert fake.search_calls[0]["doctor_id"] == "doctor-a"


def test_search_doctor_notes_returns_empty_on_memory_failure() -> None:
    _override_memory(FakeDoctorMemoryService(raises=True))
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/doctor-notes/search?q=levofloxacin")
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert response.json() == []
