from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.prescription_check_service import (
    PrescriptionCheckService,
)


class FakeNormalizer:
    def __init__(
        self,
        result: dict[str, Any] | None = None,
        *,
        should_raise: bool = False,
    ) -> None:
        self.result = result or _normalized_result(["metformin"])
        self.should_raise = should_raise
        self.calls: list[dict[str, Any]] = []

    def normalize_many(self, medications: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append({"medications": medications})
        if self.should_raise:
            raise RuntimeError("normalizer failed")
        return self.result

    def get_stats(self) -> dict[str, Any]:
        return {"service": "FakeNormalizer"}


class FakeRetriever:
    def __init__(
        self,
        bundle: dict[str, Any] | None = None,
        *,
        should_raise: bool = False,
    ) -> None:
        self.bundle = bundle or _evidence_bundle(["metformin"])
        self.should_raise = should_raise
        self.calls: list[dict[str, Any]] = []

    def build_prescription_evidence_bundle(
        self,
        normalized_result: dict[str, Any],
        query_types: list[str] | None = None,
        top_k_per_type: int = 8,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "normalized_result": normalized_result,
                "query_types": query_types,
                "top_k_per_type": top_k_per_type,
            }
        )
        if self.should_raise:
            raise RuntimeError("retriever failed")
        return self.bundle

    def get_stats(self) -> dict[str, Any]:
        return {"service": "FakeRetriever"}


def _normalized_result(
    slugs: list[str],
    *,
    requires_review: bool = False,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "medications": [
            {
                "raw_name": "Metformin 750mg",
                "mapping_status": "ingredient_only",
                "active_ingredients": [
                    {
                        "name": slug,
                        "evidence_slug": slug,
                        "evidence_status": "resolved",
                    }
                    for slug in slugs
                ],
                "requires_review": requires_review,
                "warnings": list(warnings or []),
            }
        ],
        "summary": {"requires_review": requires_review},
        "unique_evidence_slugs": list(slugs),
        "requires_review": requires_review,
        "warnings": list(warnings or []),
    }


def _normalized_without_unique_slugs() -> dict[str, Any]:
    return {
        "medications": [
            {
                "mapping_status": "ingredient_only",
                "active_ingredients": [
                    {"evidence_slug": "omeprazole"},
                    {"evidence_slug": "omeprazole"},
                    {"evidence_slug": None},
                ],
                "requires_review": False,
                "warnings": [],
            }
        ],
        "summary": {"requires_review": False},
        "requires_review": False,
        "warnings": [],
    }


def _evidence_bundle(
    slugs: list[str],
    *,
    chunks: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    if chunks is None:
        chunks = [
            {
                "chunk_id": f"{slug}:c1",
                "slug": slug,
                "section": "than_trong",
                "text": "evidence",
            }
            for slug in slugs
        ]
    return {
        "evidence_slugs": list(slugs),
        "query_results": {"interaction": {"chunks": chunks}},
        "all_chunks": list(chunks),
        "unique_chunks": list(chunks),
        "warnings": list(warnings or []),
    }


def _service(
    normalizer: FakeNormalizer | None = None,
    retriever: FakeRetriever | None = None,
) -> PrescriptionCheckService:
    return PrescriptionCheckService(
        normalizer=normalizer or FakeNormalizer(),
        retriever=retriever or FakeRetriever(),
    )


def test_extract_medication_lines_skips_clear_headers_only() -> None:
    service = _service()
    text = """
    Bệnh nhân: Nguyễn Văn A
    Chẩn đoán: Đái tháo đường type 2
    1. Metformin (Panfor SR) 750mg x 30 viên
    2. Omeprazol (Kagascdine) 20mg x 14 viên
    Lời dặn: uống sau ăn
    Có thể là dòng thuốc không rõ format
    """

    assert service.extract_medication_lines(text) == [
        "1. Metformin (Panfor SR) 750mg x 30 viên",
        "2. Omeprazol (Kagascdine) 20mg x 14 viên",
        "Có thể là dòng thuốc không rõ format",
    ]


def test_check_lines_strips_empty_lines_and_does_not_filter_headers() -> None:
    normalizer = FakeNormalizer()
    service = _service(normalizer=normalizer)

    result = service.check_lines(
        ["  Bệnh nhân: Nguyễn Văn A  ", "", "  Metformin 750mg  "]
    )

    assert result["input"]["medication_lines"] == [
        "Bệnh nhân: Nguyễn Văn A",
        "Metformin 750mg",
    ]
    assert normalizer.calls[0]["medications"] == [
        {"raw_line": "Bệnh nhân: Nguyễn Văn A"},
        {"raw_line": "Metformin 750mg"},
    ]


def test_empty_text_returns_invalid_input() -> None:
    service = _service()

    result = service.check_text("  ")

    assert result["status"] == "invalid_input"
    assert result["warnings"] == ["empty_prescription_text"]
    assert result["evidence_bundle"] is None


def test_no_lines_returns_invalid_input_without_retriever_call() -> None:
    retriever = FakeRetriever()
    service = _service(retriever=retriever)

    result = service.check_lines(["", "   "])

    assert result["status"] == "invalid_input"
    assert result["warnings"] == ["no_medication_lines_found"]
    assert retriever.calls == []


def test_normalizer_exception_returns_error() -> None:
    retriever = FakeRetriever()
    service = _service(
        normalizer=FakeNormalizer(should_raise=True), retriever=retriever
    )

    result = service.check_lines(["Metformin 750mg"])

    assert result["status"] == "error"
    assert result["errors"] == ["normalization_failed"]
    assert retriever.calls == []


def test_unique_evidence_slugs_calls_retriever_and_returns_evidence_ready() -> None:
    retriever = FakeRetriever(bundle=_evidence_bundle(["metformin"]))
    service = _service(
        normalizer=FakeNormalizer(result=_normalized_result(["metformin"])),
        retriever=retriever,
    )

    result = service.check_lines(
        ["Metformin 750mg"],
        doctor_id="doctor-1",
        patient_context={"age": 60},
        query_types=["interaction"],
        top_k_per_type=3,
    )

    assert result["status"] == "evidence_ready"
    assert result["doctor_id"] == "doctor-1"
    assert result["patient_context"] == {"age": 60}
    assert retriever.calls[0]["query_types"] == ["interaction"]
    assert retriever.calls[0]["top_k_per_type"] == 3


def test_slug_fallback_scans_medication_active_ingredients() -> None:
    retriever = FakeRetriever(bundle=_evidence_bundle(["omeprazole"]))
    service = _service(
        normalizer=FakeNormalizer(result=_normalized_without_unique_slugs()),
        retriever=retriever,
    )

    result = service.check_lines(["Omeprazol 20mg"])

    assert result["status"] == "evidence_ready"
    assert retriever.calls[0]["normalized_result"]["medications"][0][
        "active_ingredients"
    ][0]["evidence_slug"] == "omeprazole"


def test_no_evidence_slugs_skips_retriever() -> None:
    retriever = FakeRetriever()
    service = _service(
        normalizer=FakeNormalizer(result=_normalized_result([])),
        retriever=retriever,
    )

    result = service.check_lines(["Unknown drug"])

    assert result["status"] == "insufficient_information"
    assert "no_evidence_slugs_available" in result["warnings"]
    assert result["evidence_bundle"] is None
    assert retriever.calls == []


def test_requires_review_adds_warning_but_still_retrieves() -> None:
    retriever = FakeRetriever(bundle=_evidence_bundle(["levofloxacin"]))
    service = _service(
        normalizer=FakeNormalizer(
            result=_normalized_result(
                ["levofloxacin"],
                requires_review=True,
                warnings=["fuzzy_match_requires_review"],
            )
        ),
        retriever=retriever,
    )

    result = service.check_lines(["Levofloxacine 500mg"])

    assert result["status"] == "evidence_ready"
    assert "some_medications_require_review" in result["warnings"]
    assert "fuzzy_match_requires_review" in result["warnings"]
    assert len(retriever.calls) == 1


def test_retriever_exception_returns_evidence_retrieval_failed() -> None:
    service = _service(retriever=FakeRetriever(should_raise=True))

    result = service.check_lines(["Metformin 750mg"])

    assert result["status"] == "evidence_retrieval_failed"
    assert result["errors"] == ["evidence_retrieval_failed"]


def test_empty_retrieved_chunks_returns_evidence_unavailable() -> None:
    service = _service(
        retriever=FakeRetriever(
            bundle=_evidence_bundle(["metformin"], chunks=[])
        )
    )

    result = service.check_lines(["Metformin 750mg"])

    assert result["status"] == "evidence_unavailable"
    assert "no_evidence_chunks_retrieved" in result["warnings"]


def test_warning_merge_deduplicates_normalizer_and_retriever_warnings() -> None:
    service = _service(
        normalizer=FakeNormalizer(
            result=_normalized_result(
                ["metformin"],
                warnings=["shared_warning", "normalizer_warning"],
            )
        ),
        retriever=FakeRetriever(
            bundle=_evidence_bundle(
                ["metformin"], warnings=["shared_warning", "retriever_warning"]
            )
        ),
    )

    result = service.check_lines(["Metformin 750mg"])

    assert result["warnings"] == [
        "shared_warning",
        "normalizer_warning",
        "retriever_warning",
    ]


def test_medication_line_order_is_preserved() -> None:
    service = _service()

    result = service.check_lines(
        ["  Paracetamol 500mg ", " Metformin 750mg ", " Omeprazol 20mg "]
    )

    assert result["input"]["medication_lines"] == [
        "Paracetamol 500mg",
        "Metformin 750mg",
        "Omeprazol 20mg",
    ]


def test_invalid_top_k_per_type_raises() -> None:
    service = _service()

    with pytest.raises(ValueError):
        service.check_lines(["Metformin"], top_k_per_type=0)
    with pytest.raises(ValueError):
        service.check_text("Metformin", top_k_per_type=0)


def test_get_stats_uses_injected_dependencies() -> None:
    service = _service()

    assert service.get_stats() == {
        "service": "PrescriptionCheckService",
        "normalizer": {"service": "FakeNormalizer"},
        "retriever": {"service": "FakeRetriever"},
    }
