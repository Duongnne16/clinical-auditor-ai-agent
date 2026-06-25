from __future__ import annotations

import sys

import pytest

from backend.app.core.config import get_settings
from backend.app.services.prescription_check_service import (
    PrescriptionCheckService,
)


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


def test_prescription_check_service_real_pipeline_smoke() -> None:
    pytest.importorskip("qdrant_client")
    pytest.importorskip("sentence_transformers")
    _skip_if_qdrant_config_missing()

    service = PrescriptionCheckService()
    result = service.check_text(
        """
        Bệnh nhân: Nguyễn Văn A
        Chẩn đoán: Đái tháo đường type 2, đau dạ dày
        1. Omeprazol (Kagascdine) 20mg x 14 viên
        2. Metformin (Panfor SR) 750mg x 30 viên
        3. Paracetamol (Hapacol) 500mg x 10 viên
        Lời dặn: uống sau ăn
        """,
        doctor_id="dev-doctor-001",
    )

    normalized_result = result["normalized_result"]
    evidence_bundle = result["evidence_bundle"]
    unique_slugs = set(normalized_result["unique_evidence_slugs"])
    query_result_keys = set(evidence_bundle["query_results"])

    print(_safe_console_text(f"\nstatus={result['status']}"))
    print(
        _safe_console_text(
            f"medication_lines={result['input']['medication_lines']}"
        )
    )
    print(
        _safe_console_text(
            f"unique_evidence_slugs={normalized_result['unique_evidence_slugs']}"
        )
    )
    print(
        _safe_console_text(
            "evidence_bundle unique_chunks count="
            f"{len(evidence_bundle['unique_chunks'])}"
        )
    )
    print(_safe_console_text(f"query_results keys={list(query_result_keys)}"))
    print("top 2 chunks:")
    for chunk in evidence_bundle["unique_chunks"][:2]:
        snippet = " ".join((chunk.get("text") or "").split())[:180]
        print(
            _safe_console_text(
                "- "
                f"slug={chunk.get('slug')} "
                f"section={chunk.get('section')} "
                f"rerank_score={chunk.get('rerank_score')} "
                f"snippet={snippet} "
                f"url={chunk.get('url')}"
            )
        )

    assert result["status"] == "evidence_ready", result
    assert {"omeprazole", "metformin", "paracetamol"} <= unique_slugs
    assert evidence_bundle is not None
    assert len(evidence_bundle["unique_chunks"]) > 0
    assert {"interaction", "contraindication", "renal_hepatic"} <= query_result_keys
