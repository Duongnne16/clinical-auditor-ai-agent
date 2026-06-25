from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from backend.app.api.routes.prescriptions import get_prescription_audit_service
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
            "warnings": [],
            "errors": [],
            "echo": kwargs,
        }


def _override_service(fake_service: FakePrescriptionAuditService) -> None:
    app.dependency_overrides[get_prescription_audit_service] = lambda: fake_service


def _clear_overrides() -> None:
    app.dependency_overrides.pop(get_prescription_audit_service, None)


def test_prescription_audit_route_returns_fake_service_response() -> None:
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/prescriptions/audit",
                json={
                    "prescription_text": "1. Omeprazol 20mg",
                    "doctor_id": "doctor-from-body",
                    "patient_context": {"age": 60},
                    "use_gemini": False,
                    "query_types": ["interaction"],
                    "top_k_per_type": 3,
                },
            )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial_success"
    assert body["report"]["status"] == "report_context_ready"
    assert fake_service.calls == [
        {
            "prescription_text": "1. Omeprazol 20mg",
            "doctor_id": "doctor-from-body",
            "patient_context": {"age": 60},
            "use_gemini": False,
            "query_types": ["interaction"],
            "top_k_per_type": 3,
        }
    ]


def test_prescription_audit_route_uses_development_doctor_when_missing() -> None:
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/prescriptions/audit",
                json={"prescription_text": "1. Paracetamol 500mg"},
            )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert fake_service.calls[0]["doctor_id"] == "dev-doctor-001"
    assert fake_service.calls[0]["patient_context"] == {}
    assert fake_service.calls[0]["use_gemini"] is False
    assert fake_service.calls[0]["top_k_per_type"] == 8


def test_prescription_audit_route_passes_use_gemini_true_to_fake_service() -> None:
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/prescriptions/audit",
                json={
                    "prescription_text": "1. Omeprazol 20mg",
                    "use_gemini": True,
                },
            )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert fake_service.calls[0]["use_gemini"] is True


def test_prescription_text_empty_returns_422() -> None:
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/prescriptions/audit",
                json={"prescription_text": ""},
            )
    finally:
        _clear_overrides()

    assert response.status_code == 422
    assert fake_service.calls == []


def test_prescription_text_whitespace_only_returns_422() -> None:
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/prescriptions/audit",
                json={"prescription_text": "   "},
            )
    finally:
        _clear_overrides()

    assert response.status_code == 422
    assert fake_service.calls == []


def test_top_k_per_type_zero_returns_422() -> None:
    fake_service = FakePrescriptionAuditService()
    _override_service(fake_service)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/prescriptions/audit",
                json={
                    "prescription_text": "1. Omeprazol 20mg",
                    "top_k_per_type": 0,
                },
            )
    finally:
        _clear_overrides()

    assert response.status_code == 422
    assert fake_service.calls == []
