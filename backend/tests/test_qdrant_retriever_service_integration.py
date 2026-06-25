from __future__ import annotations

import sys

import pytest

from backend.app.core.config import get_settings
from backend.app.services.qdrant_retriever_service import QdrantRetrieverService


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


@pytest.mark.parametrize(
    ("slugs", "query_type", "expected_sections_any"),
    [
        (
            ["paracetamol"],
            "overdose",
            ["qua_lieu_va_xu_tri", "qua_lieu", "trieu_chung", "xu_tri"],
        ),
        (
            ["metformin"],
            "renal_hepatic",
            ["than_trong", "lieu_luong_va_cach_dung", "duoc_dong_hoc"],
        ),
        (
            ["omeprazole"],
            "interaction",
            ["tuong_tac_thuoc", "than_trong"],
        ),
        (
            ["paracetamol"],
            "contraindication",
            ["chong_chi_dinh", "than_trong"],
        ),
    ],
)
def test_qdrant_retrieval_smoke(
    slugs: list[str],
    query_type: str,
    expected_sections_any: list[str],
) -> None:
    pytest.importorskip("qdrant_client")
    pytest.importorskip("sentence_transformers")
    _skip_if_qdrant_config_missing()
    settings = get_settings()
    assert settings.qdrant_medical_evidence_collection == "clinical_evidence"

    service = QdrantRetrieverService()
    result = service.retrieve(slugs, query_type=query_type, top_k=8)

    print(
        f"\n{slugs} + {query_type}: total_results={result['total_results']}"
    )
    for chunk in result["chunks"][:3]:
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

    assert result["total_results"] > 0, result
    assert any(chunk.get("slug") in slugs for chunk in result["chunks"]), result
    assert any(
        chunk.get("section") in expected_sections_any
        for chunk in result["chunks"]
    ), result
    assert any(chunk.get("text") for chunk in result["chunks"]), result
