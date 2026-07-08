import pytest

from data_pipeline.processing.embed_and_ingest import (
    CLINICAL_EVIDENCE_COLLECTION,
    embed_and_ingest,
    flatten_evidence_payload,
    prepare_payloads,
)


def _record() -> dict:
    return {
        "chunk_id": "longchau:ingredient:omeprazole:tuong_tac_thuoc:0001",
        "content": "Hoạt chất: Omeprazole\nNội dung:\nTương tác thuốc.",
        "metadata": {
            "source": "longchau",
            "source_name": "Dược chất Long Châu",
            "source_type": "supplementary",
            "entity_type": "ingredient",
            "entity_name": "Omeprazole",
            "slug": "omeprazole",
            "section": "tuong_tac_thuoc",
            "url": "https://nhathuoclongchau.com.vn/thanh-phan/omeprazole",
        },
    }


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.5] for index, _ in enumerate(texts)]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def upsert(self, *, collection_name: str, points: list) -> None:
        self.calls.append(
            {"collection_name": collection_name, "points": points}
        )


def test_flatten_evidence_payload_matches_backend_retrieval_schema() -> None:
    payload = flatten_evidence_payload(_record())

    assert payload == {
        "chunk_id": "longchau:ingredient:omeprazole:tuong_tac_thuoc:0001",
        "source": "longchau",
        "source_name": "Dược chất Long Châu",
        "source_type": "supplementary",
        "entity_type": "ingredient",
        "entity_name": "Omeprazole",
        "slug": "omeprazole",
        "section": "tuong_tac_thuoc",
        "url": "https://nhathuoclongchau.com.vn/thanh-phan/omeprazole",
        "text": "Hoạt chất: Omeprazole\nNội dung:\nTương tác thuốc.",
    }


def test_prepare_payloads_rejects_missing_required_field() -> None:
    record = _record()
    record["metadata"]["slug"] = ""

    with pytest.raises(ValueError, match="slug"):
        prepare_payloads([record])


def test_flatten_payload_fills_known_source_name_when_missing() -> None:
    record = _record()
    record["metadata"]["source"] = "trungtamthuoc"
    record["metadata"].pop("source_name")

    payload = flatten_evidence_payload(record)

    assert payload["source_name"] == (
        "Trung Tâm Thuốc / Dược thư Quốc gia Việt Nam"
    )


def test_embed_and_ingest_uses_clinical_evidence_collection() -> None:
    client = FakeQdrantClient()

    written = embed_and_ingest(
        [_record()],
        qdrant_client=client,
        embedding_model=FakeEmbeddingModel(),
    )

    assert written == 1
    assert client.calls[0]["collection_name"] == CLINICAL_EVIDENCE_COLLECTION
    point = client.calls[0]["points"][0]
    assert point.payload["slug"] == "omeprazole"
    assert point.payload["section"] == "tuong_tac_thuoc"
    assert point.payload["source"] == "longchau"
