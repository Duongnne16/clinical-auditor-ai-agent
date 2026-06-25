from __future__ import annotations

import sys

import pytest

from backend.app.core.config import get_settings
from backend.app.services.prescription_check_service import PrescriptionCheckService
from backend.app.services.risk_analyzer_service import RiskAnalyzerService


def _safe_console_text(text: object) -> str:
    value = str(text)
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="backslashreplace").decode(encoding)


def _skip_if_qdrant_config_missing() -> None:
    settings = get_settings()
    if not settings.qdrant_url:
        pytest.skip("QDRANT_URL is not configured")
    if not settings.qdrant_api_key:
        pytest.skip("QDRANT_API_KEY is not configured")
    if not settings.qdrant_medical_evidence_collection:
        pytest.skip("QDRANT_MEDICAL_EVIDENCE_COLLECTION is not configured")


def test_risk_analyzer_context_smoke_from_real_prescription_pipeline() -> None:
    pytest.importorskip("qdrant_client")
    pytest.importorskip("sentence_transformers")
    _skip_if_qdrant_config_missing()

    prescription_result = PrescriptionCheckService().check_text(
        """
        1. Omeprazol (Kagascdine) 20mg x 14 viên
        2. Metformin (Panfor SR) 750mg x 30 viên
        3. Paracetamol (Hapacol) 500mg x 10 viên
        """,
        doctor_id="dev-doctor-001",
    )
    analyzer_result = RiskAnalyzerService().analyze(
        prescription_result["normalized_result"],
        prescription_result["evidence_bundle"],
        patient_context={"age": 60, "sex": "male"},
    )

    context = analyzer_result["evidence_context"]
    print(_safe_console_text(f"\nprescription_status={prescription_result['status']}"))
    print(_safe_console_text(f"analyzer_status={analyzer_result['status']}"))
    print(
        _safe_console_text(
            f"missing_information={analyzer_result['missing_information']}"
        )
    )
    print(
        _safe_console_text(
            "evidence_by_query_type keys="
            f"{list(context['evidence_by_query_type'])}"
        )
    )
    print(
        _safe_console_text(
            f"valid_evidence_refs count={len(context['valid_evidence_refs'])}"
        )
    )
    print(
        _safe_console_text(
            f"interaction_candidates={context['interaction_candidates']}"
        )
    )

    assert prescription_result["status"] == "evidence_ready"
    assert analyzer_result["status"] == "analysis_context_ready"
    assert context["evidence_by_query_type"]
    assert len(context["valid_evidence_refs"]) > 0
    assert len(context["interaction_candidates"]) >= 3
