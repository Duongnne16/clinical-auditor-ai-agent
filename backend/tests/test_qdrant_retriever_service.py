from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from backend.app.services.qdrant_retriever_service import (
    QdrantRetrieverService,
    build_query_text,
    detect_query_type,
    get_preferred_sections,
)


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(self, text: str, normalize_embeddings: bool = False) -> list[float]:
        self.calls.append(
            {"text": text, "normalize_embeddings": normalize_embeddings}
        )
        return [0.1, 0.2, 0.3]


@dataclass
class FakePoint:
    payload: dict[str, Any]
    score: float


class FakeResponse:
    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points


class FakeQdrantClient:
    def __init__(self, responses: list[list[FakePoint]] | None = None) -> None:
        self.responses = responses or [[]]
        self.calls: list[dict[str, Any]] = []

    def query_points(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return FakeResponse(self.responses[index])


def _service(client: FakeQdrantClient) -> tuple[QdrantRetrieverService, FakeEmbeddingModel]:
    model = FakeEmbeddingModel()
    return (
        QdrantRetrieverService(
            qdrant_client=client,
            embedding_model=model,
            collection_name="clinical_evidence",
        ),
        model,
    )


def _must_values(query_filter: Any) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    conditions = (
        query_filter["must"]
        if isinstance(query_filter, dict)
        else getattr(query_filter, "must", [])
    )
    for condition in conditions:
        if isinstance(condition, dict):
            output[condition["key"]] = condition["match"]["any"]
            continue
        match = getattr(condition, "match")
        output[getattr(condition, "key")] = list(getattr(match, "any"))
    return output


def test_detect_query_type() -> None:
    cases = {
        "tương tác thuốc khi dùng chung": "interaction",
        "chống chỉ định của thuốc": "contraindication",
        "phụ nữ mang thai dùng được không": "pregnancy_lactation",
        "quá liều phải xử trí sao": "overdose",
        "liều dùng người lớn": "dose",
        "suy thận có cần hiệu chỉnh không": "renal_hepatic",
    }

    for query, expected in cases.items():
        assert detect_query_type(query) == expected


def test_overdose_is_detected_before_dose() -> None:
    assert detect_query_type("liều và xử trí quá liều") == "overdose"


def test_get_preferred_sections() -> None:
    assert get_preferred_sections("interaction") == [
        "tuong_tac_thuoc",
        "tuong_ky",
        "than_trong",
    ]
    assert get_preferred_sections("unknown") == []


def test_build_query_text_prefix_and_keywords() -> None:
    query = build_query_text(["omeprazole", "clopidogrel"], "interaction")

    assert query.startswith("query:")
    assert "omeprazole clopidogrel" in query
    assert "tương tác thuốc" in query
    assert build_query_text(["x"], query_text="query: custom") == "query: custom"


def test_retrieve_with_no_slugs_returns_warning() -> None:
    service, _ = _service(FakeQdrantClient())

    result = service.retrieve([], query_type="general")

    assert result["chunks"] == []
    assert result["total_results"] == 0
    assert result["warnings"] == ["no_evidence_slugs_provided"]


def test_retrieve_uses_flat_slug_and_section_filter() -> None:
    client = FakeQdrantClient(
        responses=[
            [
                FakePoint(
                    {
                        "chunk_id": "c1",
                        "slug": "omeprazole",
                        "section": "tuong_tac_thuoc",
                        "source": "trungtamthuoc",
                        "content": "interaction content",
                    },
                    0.7,
                )
            ]
        ]
    )
    service, model = _service(client)

    result = service.retrieve(["omeprazole"], query_type="interaction", top_k=3)

    values = _must_values(client.calls[0]["query_filter"])
    assert values["slug"] == ["omeprazole"]
    assert values["section"] == ["tuong_tac_thuoc", "tuong_ky", "than_trong"]
    assert client.calls[0]["limit"] == 20
    assert model.calls[0]["normalize_embeddings"] is True
    assert model.calls[0]["text"].startswith("query:")
    assert result["chunks"][0]["text"] == "interaction content"


def test_retrieve_falls_back_to_slug_only_when_section_has_no_results() -> None:
    client = FakeQdrantClient(
        responses=[
            [],
            [
                FakePoint(
                    {
                        "chunk_id": "c1",
                        "slug": "omeprazole",
                        "section": "duoc_luc_hoc",
                        "content": "fallback content",
                    },
                    0.5,
                )
            ],
        ]
    )
    service, _ = _service(client)

    result = service.retrieve(["omeprazole"], query_type="interaction")

    assert len(client.calls) == 2
    assert "section" in _must_values(client.calls[0]["query_filter"])
    assert "section" not in _must_values(client.calls[1]["query_filter"])
    assert result["warnings"] == [
        "section_filter_no_results_fallback_to_slug_only"
    ]
    assert result["chunks"][0]["text"] == "fallback content"


def test_retrieve_uses_top_k_times_three_when_larger_than_twenty() -> None:
    client = FakeQdrantClient(
        responses=[
            [
                FakePoint(
                    {
                        "chunk_id": "c1",
                        "slug": "paracetamol",
                        "section": "qua_lieu",
                        "content": "content",
                    },
                    0.4,
                )
            ]
        ]
    )
    service, _ = _service(client)

    service.retrieve(["paracetamol"], query_type="overdose", top_k=9)

    assert client.calls[0]["limit"] == 27


def test_rerank_boosts_preferred_section_and_source() -> None:
    client = FakeQdrantClient(
        responses=[
            [
                FakePoint(
                    {
                        "chunk_id": "general",
                        "slug": "omeprazole",
                        "section": "duoc_luc_hoc",
                        "source": "other",
                        "content": "general",
                    },
                    0.9,
                ),
                FakePoint(
                    {
                        "chunk_id": "interaction",
                        "slug": "omeprazole",
                        "section": "tuong_tac_thuoc",
                        "source": "trungtamthuoc",
                        "content": "interaction",
                    },
                    0.7,
                ),
            ]
        ]
    )
    service, _ = _service(client)

    result = service.retrieve(["omeprazole"], query_type="interaction", top_k=2)

    assert result["chunks"][0]["chunk_id"] == "interaction"
    assert result["chunks"][0]["rank"] == 1


def test_payload_falls_back_to_nested_metadata_but_prefers_content() -> None:
    client = FakeQdrantClient(
        responses=[
            [
                FakePoint(
                    {
                        "content": "flat content",
                        "text": "text should lose",
                        "metadata": {
                            "chunk_id": "nested-c1",
                            "slug": "metformin",
                            "section": "than_trong",
                            "source": "trungtamthuoc",
                        },
                    },
                    0.4,
                )
            ]
        ]
    )
    service, _ = _service(client)

    result = service.retrieve(["metformin"], query_type="renal_hepatic")

    chunk = result["chunks"][0]
    assert chunk["chunk_id"] == "nested-c1"
    assert chunk["slug"] == "metformin"
    assert chunk["text"] == "flat content"


def test_deduplicates_by_chunk_id_and_fallback_key() -> None:
    duplicate_without_id = {
        "slug": "metformin",
        "section": "than_trong",
        "content": "same content",
    }
    client = FakeQdrantClient(
        responses=[
            [
                FakePoint(
                    {
                        "chunk_id": "same",
                        "slug": "metformin",
                        "section": "than_trong",
                        "content": "first",
                    },
                    0.8,
                ),
                FakePoint(
                    {
                        "chunk_id": "same",
                        "slug": "metformin",
                        "section": "than_trong",
                        "content": "second",
                    },
                    0.7,
                ),
                FakePoint(duplicate_without_id, 0.6),
                FakePoint(duplicate_without_id, 0.5),
            ]
        ]
    )
    service, _ = _service(client)

    result = service.retrieve(["metformin"], query_type="renal_hepatic", top_k=10)

    assert len(result["chunks"]) == 2


def test_retrieve_for_normalized_result_prefers_unique_slugs() -> None:
    client = FakeQdrantClient(
        responses=[
            [
                FakePoint(
                    {
                        "chunk_id": "c1",
                        "slug": "paracetamol",
                        "section": "qua_lieu",
                        "content": "content",
                    },
                    0.5,
                )
            ]
        ]
    )
    service, _ = _service(client)

    result = service.retrieve_for_normalized_result(
        {"unique_evidence_slugs": ["paracetamol", "paracetamol"]},
        query_type="overdose",
    )

    assert result["evidence_slugs"] == ["paracetamol"]
    assert _must_values(client.calls[0]["query_filter"])["slug"] == ["paracetamol"]


def test_retrieve_for_normalized_result_extracts_medication_ingredients() -> None:
    client = FakeQdrantClient(responses=[[]])
    service, _ = _service(client)

    service.retrieve_for_normalized_result(
        {
            "medications": [
                {
                    "mapping_status": "ingredient_only",
                    "active_ingredients": [
                        {"evidence_slug": "metformin"},
                        {"evidence_slug": None},
                    ],
                },
                {
                    "mapping_status": "unmatched",
                    "active_ingredients": [{"evidence_slug": "ignored"}],
                },
            ]
        },
        query_type="renal_hepatic",
    )

    assert _must_values(client.calls[0]["query_filter"])["slug"] == ["metformin"]


def test_build_prescription_evidence_bundle_deduplicates_chunks() -> None:
    client = FakeQdrantClient(
        responses=[
            [
                FakePoint(
                    {
                        "chunk_id": "shared",
                        "slug": "metformin",
                        "section": "tuong_tac_thuoc",
                        "content": "a",
                    },
                    0.6,
                )
            ],
            [
                FakePoint(
                    {
                        "chunk_id": "shared",
                        "slug": "metformin",
                        "section": "than_trong",
                        "content": "b",
                    },
                    0.6,
                )
            ],
        ]
    )
    service, _ = _service(client)

    result = service.build_prescription_evidence_bundle(
        {"unique_evidence_slugs": ["metformin"]},
        query_types=["interaction", "renal_hepatic"],
        top_k_per_type=2,
    )

    assert set(result["query_results"]) == {"interaction", "renal_hepatic"}
    assert len(result["all_chunks"]) == 2
    assert len(result["unique_chunks"]) == 1


def test_invalid_top_k_raises() -> None:
    service, _ = _service(FakeQdrantClient())

    with pytest.raises(ValueError):
        service.retrieve(["metformin"], top_k=0)
    with pytest.raises(ValueError):
        service.build_prescription_evidence_bundle(
            {"unique_evidence_slugs": ["metformin"]}, top_k_per_type=0
        )


def test_import_is_safe_without_real_dependencies() -> None:
    service = QdrantRetrieverService(
        qdrant_client=FakeQdrantClient(),
        embedding_model=FakeEmbeddingModel(),
        collection_name="clinical_evidence",
    )

    stats = service.get_stats()

    assert stats["service"] == "QdrantRetrieverService"
    assert stats["collection_name"] == "clinical_evidence"
