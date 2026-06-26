from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.services.doctor_memory_service import (
    DOCTOR_MEMORY_VECTOR_SIZE,
    DoctorMemoryService,
    get_doctor_memory_service,
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
    def __init__(
        self,
        responses: list[list[FakePoint]] | None = None,
        scroll_points: list[FakePoint] | None = None,
        collection_exists: bool = True,
        query_raises: bool = False,
        scroll_rejects_with_vectors: bool = False,
    ) -> None:
        self.responses = responses or [[]]
        self.scroll_points = scroll_points or []
        self.collection_exists_value = collection_exists
        self.query_raises = query_raises
        self.scroll_rejects_with_vectors = scroll_rejects_with_vectors
        self.collection_exists_calls: list[dict[str, Any]] = []
        self.create_collection_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_points_calls: list[dict[str, Any]] = []
        self.scroll_calls: list[dict[str, Any]] = []

    def collection_exists(self, **kwargs: Any) -> bool:
        self.collection_exists_calls.append(kwargs)
        return self.collection_exists_value

    def create_collection(self, **kwargs: Any) -> None:
        self.create_collection_calls.append(kwargs)
        self.collection_exists_value = True

    def upsert(self, **kwargs: Any) -> None:
        self.upsert_calls.append(kwargs)

    def query_points(self, **kwargs: Any) -> FakeResponse:
        self.query_points_calls.append(kwargs)
        if self.query_raises:
            raise RuntimeError("query_points unavailable")
        index = min(len(self.query_points_calls) - 1, len(self.responses) - 1)
        return FakeResponse(self.responses[index])

    def scroll(self, **kwargs: Any) -> tuple[list[FakePoint], None]:
        self.scroll_calls.append(kwargs)
        if self.scroll_rejects_with_vectors and "with_vectors" in kwargs:
            raise TypeError("unexpected keyword argument with_vectors")
        return self.scroll_points, None


def _service(
    client: FakeQdrantClient | None = None,
    model: FakeEmbeddingModel | None = None,
) -> tuple[DoctorMemoryService, FakeQdrantClient, FakeEmbeddingModel]:
    final_client = client or FakeQdrantClient()
    final_model = model or FakeEmbeddingModel()
    return (
        DoctorMemoryService(
            qdrant_client=final_client,
            embedding_model=final_model,
            collection_name="doctor_memory",
        ),
        final_client,
        final_model,
    )


def _filter_values(query_filter: Any) -> dict[str, Any]:
    conditions = (
        query_filter["must"]
        if isinstance(query_filter, dict)
        else getattr(query_filter, "must", [])
    )
    values: dict[str, Any] = {}
    for condition in conditions:
        if isinstance(condition, dict):
            match = condition["match"]
            values[condition["key"]] = match.get("value") if isinstance(match, dict) else match
            continue
        match = getattr(condition, "match")
        values[getattr(condition, "key")] = getattr(match, "value")
    return values


def _payload(point: Any) -> dict[str, Any]:
    if isinstance(point, dict):
        return point["payload"]
    return point.payload


def _vector_size(vectors_config: Any) -> int:
    if isinstance(vectors_config, dict):
        return int(vectors_config["size"])
    return int(getattr(vectors_config, "size"))


def _distance_text(vectors_config: Any) -> str:
    if isinstance(vectors_config, dict):
        return str(vectors_config["distance"]).casefold()
    return str(getattr(vectors_config, "distance")).casefold()


def test_ensure_collection_creates_missing_collection_with_cosine_768() -> None:
    service, client, _ = _service(FakeQdrantClient(collection_exists=False))

    service.ensure_collection()
    service.ensure_collection()

    assert len(client.collection_exists_calls) == 1
    assert len(client.create_collection_calls) == 1
    vectors_config = client.create_collection_calls[0]["vectors_config"]
    assert _vector_size(vectors_config) == DOCTOR_MEMORY_VECTOR_SIZE
    assert "cosine" in _distance_text(vectors_config)


def test_save_note_upserts_doctor_scoped_payload() -> None:
    service, client, model = _service(FakeQdrantClient(collection_exists=True))

    payload = service.save_note(
        doctor_id="doctor-1",
        note_id="note-1",
        title="Levofloxacin + Sucralfate",
        note_text="Rà soát thời điểm dùng.",
        note_type="drug_interaction_note",
        source_context="prescription_audit",
        active_ingredients=["Levofloxacin", "Sucralfate"],
        drug_pair_keys=["levofloxacin|sucralfate"],
        diagnosis_keywords=["Loét dạ dày"],
        patient_tags=[],
        applicability={},
        priority="normal",
    )

    assert model.calls[0]["normalize_embeddings"] is True
    assert len(client.upsert_calls) == 1
    point_payload = _payload(client.upsert_calls[0]["points"][0])
    assert point_payload["doctor_id"] == "doctor-1"
    assert point_payload["note_text"] == "Rà soát thời điểm dùng."
    assert point_payload["active_ingredients"] == ["levofloxacin", "sucralfate"]
    assert point_payload["drug_pair_keys"] == ["levofloxacin|sucralfate"]
    assert point_payload["status"] == "active"
    assert payload["note_id"] == "note-1"


def test_search_notes_filters_by_doctor_id_and_status() -> None:
    matching = {
        "note_id": "n1",
        "doctor_id": "doctor-1",
        "title": "Match",
        "note_text": "Private note",
        "status": "active",
    }
    other_doctor = matching | {"note_id": "n2", "doctor_id": "doctor-2"}
    inactive = matching | {"note_id": "n3", "status": "deleted"}
    service, client, _ = _service(
        FakeQdrantClient(
            responses=[
                [
                    FakePoint(matching, 0.9),
                    FakePoint(other_doctor, 0.95),
                    FakePoint(inactive, 0.99),
                ]
            ]
        )
    )

    notes = service.search_notes(doctor_id="doctor-1", query="levofloxacin", top_k=5)

    assert [note["note_id"] for note in notes] == ["n1"]
    values = _filter_values(client.query_points_calls[0]["query_filter"])
    assert values["doctor_id"] == "doctor-1"
    assert values["status"] == "active"


def test_search_finds_saved_note_by_mixed_case_active_ingredients_metadata() -> None:
    point = FakePoint(
        {
            "note_id": "n1",
            "doctor_id": "dev-doctor-001",
            "title": "Levofloxacin + Sucralfate",
            "note_text": "Private note",
            "status": "active",
            "active_ingredients": ["Levofloxacin", "sucralfate"],
            "drug_pair_keys": [],
        },
        0.0,
    )
    service, client, _ = _service(
        FakeQdrantClient(responses=[[]], scroll_points=[point])
    )

    notes = service.search_notes(
        doctor_id="dev-doctor-001",
        query="levofloxacin sucralfate",
        top_k=5,
    )

    assert [note["note_id"] for note in notes] == ["n1"]
    assert notes[0]["active_ingredients"] == ["levofloxacin", "sucralfate"]
    assert notes[0]["match_reason"] == "metadata_match"
    assert client.scroll_calls


def test_search_finds_saved_note_by_mixed_case_drug_pair_key_metadata() -> None:
    point = FakePoint(
        {
            "note_id": "n1",
            "doctor_id": "dev-doctor-001",
            "title": "Levofloxacin + Sucralfate",
            "note_text": "Private note",
            "status": "active",
            "active_ingredients": [],
            "drug_pair_keys": ["Levofloxacin|sucralfate"],
        },
        0.0,
    )
    service, _, _ = _service(
        FakeQdrantClient(responses=[[]], scroll_points=[point])
    )

    notes = service.search_notes(
        doctor_id="dev-doctor-001",
        query="levofloxacin sucralfate",
        top_k=5,
    )

    assert [note["note_id"] for note in notes] == ["n1"]
    assert notes[0]["drug_pair_keys"] == ["levofloxacin|sucralfate"]
    assert notes[0]["match_reason"] == "metadata_match"


def test_search_metadata_fallback_does_not_return_other_doctor_or_inactive_notes() -> None:
    mine = {
        "note_id": "mine",
        "doctor_id": "doctor-1",
        "title": "Mine",
        "note_text": "Private note",
        "status": "active",
        "active_ingredients": ["Levofloxacin", "sucralfate"],
    }
    other_doctor = mine | {"note_id": "other", "doctor_id": "doctor-2"}
    inactive = mine | {"note_id": "inactive", "status": "deleted"}
    service, _, _ = _service(
        FakeQdrantClient(
            responses=[[]],
            scroll_points=[
                FakePoint(other_doctor, 0.0),
                FakePoint(inactive, 0.0),
                FakePoint(mine, 0.0),
            ],
        )
    )

    notes = service.search_notes(
        doctor_id="doctor-1",
        query="levofloxacin sucralfate",
        top_k=5,
    )

    assert [note["note_id"] for note in notes] == ["mine"]


def test_search_metadata_fallback_runs_when_vector_query_fails() -> None:
    point = FakePoint(
        {
            "note_id": "n1",
            "doctor_id": "dev-doctor-001",
            "title": "Levofloxacin + Sucralfate",
            "note_text": "Private note",
            "status": "active",
            "active_ingredients": ["Levofloxacin", "sucralfate"],
            "drug_pair_keys": ["Levofloxacin|sucralfate"],
        },
        0.0,
    )
    service, client, _ = _service(
        FakeQdrantClient(
            scroll_points=[point],
            query_raises=True,
            scroll_rejects_with_vectors=True,
        )
    )

    notes = service.search_notes(
        doctor_id="dev-doctor-001",
        query="Levofloxacin + Sucralfate",
        top_k=5,
    )

    assert [note["note_id"] for note in notes] == ["n1"]
    assert notes[0]["match_reason"] == "metadata_match"
    assert len(client.scroll_calls) == 2


def test_retrieve_for_audit_context_reranks_matches_and_excludes_conflict() -> None:
    matching = {
        "note_id": "match",
        "doctor_id": "doctor-1",
        "title": "Levofloxacin + Sucralfate",
        "note_text": "Rà soát thời điểm dùng.",
        "status": "active",
        "source_context": "prescription_audit",
        "active_ingredients": ["levofloxacin", "sucralfate"],
        "drug_pair_keys": ["levofloxacin|sucralfate"],
        "diagnosis_keywords": ["loet da day"],
        "patient_tags": [],
        "applicability": {},
    }
    conflict = matching | {
        "note_id": "pregnancy-only",
        "applicability": {"pregnancy_status": "pregnant"},
    }
    service, _, _ = _service(
        FakeQdrantClient(
            responses=[
                [
                    FakePoint(conflict, 0.9),
                    FakePoint(matching, 0.5),
                ]
            ]
        )
    )

    result = service.retrieve_for_audit_context(
        doctor_id="doctor-1",
        normalized_result={
            "medications": [
                {"active_ingredients": [{"evidence_slug": "levofloxacin"}]},
                {"active_ingredients": [{"evidence_slug": "sucralfate"}]},
            ]
        },
        patient_context={
            "pregnancy_status": "not pregnant",
            "diagnoses": ["Loét dạ dày"],
        },
        risk_analysis={"risk_items": [{"risk_type": "interaction"}]},
    )

    assert [note["note_id"] for note in result["matched_notes"]] == ["match"]
    assert result["matched_notes"][0]["match_reason"] == "audit_context_match"


def test_embedding_model_is_reused_for_multiple_calls() -> None:
    service, _, model = _service()

    service.search_notes(doctor_id="doctor-1", query="one", top_k=1)
    service.search_notes(doctor_id="doctor-1", query="two", top_k=1)

    assert service.embedding_model is model
    assert len(model.calls) == 2


def test_get_doctor_memory_service_is_cached() -> None:
    get_doctor_memory_service.cache_clear()
    try:
        service_1 = get_doctor_memory_service()
        service_2 = get_doctor_memory_service()
    finally:
        get_doctor_memory_service.cache_clear()

    assert service_1 is service_2
