from typing import Any

from fastapi.testclient import TestClient

from backend.app.api.routes.chat import get_chat_query_service
from backend.app.main import app
from backend.app.services.doctor_memory_service import get_doctor_memory_service


class FakeChatQueryService:
    def answer(self, request: Any) -> dict[str, Any]:
        intent = request.intent or "single_drug_query"
        answer = "Với dữ liệu hiện có, hệ thống ghi nhận thông tin cần rà soát."
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


def _override_chat_service() -> None:
    app.dependency_overrides[get_chat_query_service] = FakeChatQueryService


def _clear_chat_override() -> None:
    app.dependency_overrides.pop(get_chat_query_service, None)


def _override_memory_service() -> None:
    app.dependency_overrides[get_doctor_memory_service] = FakeDoctorMemoryService


def _clear_memory_override() -> None:
    app.dependency_overrides.pop(get_doctor_memory_service, None)


def test_chat_route_returns_backward_compatible_answer_fields() -> None:
    _override_chat_service()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat",
                json={"message": "Paracetamol có tác dụng phụ gì?"},
            )
    finally:
        _clear_chat_override()

    assert response.status_code == 200
    body = response.json()
    assert body["doctor_id"] == "dev-doctor-001"
    assert body["intent"] == "single_drug_query"
    assert body["message"] == body["answer"]
    assert body["answer"]
    assert "Placeholder" not in body["disclaimer"]
    assert "không thay thế quyết định của bác sĩ/dược sĩ" in body["disclaimer"]


def test_chat_route_can_return_out_of_scope_refusal() -> None:
    _override_chat_service()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat", json={"message": "Viết giúp tôi bài văn"}
            )
    finally:
        _clear_chat_override()

    assert response.status_code == 200
    assert response.json()["intent"] == "out_of_scope"
    assert "chỉ hỗ trợ tra cứu" in response.json()["answer"]


def test_chat_route_can_return_interaction_answer() -> None:
    _override_chat_service()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Omeprazole có tương tác với Clopidogrel không?"
                },
            )
    finally:
        _clear_chat_override()

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "drug_interaction_query"
    assert body["answer"]


def test_doctor_notes_use_the_development_doctor() -> None:
    _override_memory_service()
    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/doctor-notes",
                json={"content": "Placeholder note for skeleton test"},
            )
            listed = client.get("/api/v1/doctor-notes")
    finally:
        _clear_memory_override()

    assert created.status_code == 201
    assert created.json()["doctor_id"] == "dev-doctor-001"
    assert listed.status_code == 200
    assert any(
        note["id"] == created.json()["id"] and "not a clinical conclusion" in note["disclaimer"]
        for note in listed.json()
    )
