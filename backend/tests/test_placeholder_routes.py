from fastapi.testclient import TestClient

from backend.app.main import app


def test_prescription_audit_is_a_safe_placeholder() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/prescriptions/audit",
            json={"drugs": [{"name": "Paracetamol", "dosage": "500 mg"}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["doctor_id"] == "dev-doctor-001"
    assert body["status"] == "placeholder"
    assert "not a clinical conclusion" in body["disclaimer"]


def test_chat_does_not_call_gemini() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"message": "Check this"})

    assert response.status_code == 200
    assert "disabled" in response.json()["message"]


def test_prescription_requires_at_least_one_drug() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/prescriptions/audit", json={"drugs": []})

    assert response.status_code == 422


def test_doctor_notes_use_the_development_doctor() -> None:
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/doctor-notes",
            json={"content": "Placeholder note for skeleton test"},
        )
        listed = client.get("/api/v1/doctor-notes")

    assert created.status_code == 201
    assert created.json()["doctor_id"] == "dev-doctor-001"
    assert listed.status_code == 200
    assert any(
        note["id"] == created.json()["id"] and "not a clinical conclusion" in note["disclaimer"]
        for note in listed.json()
    )
