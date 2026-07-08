"""Prepare and ingest evidence chunks into Qdrant."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid5, NAMESPACE_URL


CLINICAL_EVIDENCE_COLLECTION = "clinical_evidence"
SOURCE_NAMES = {
    "trungtamthuoc": "Trung Tâm Thuốc / Dược thư Quốc gia Việt Nam",
    "longchau": "Dược chất Long Châu",
}
DEFAULT_INPUT_PATHS = (
    Path("data/processed/trungtamthuoc_v2/ingredients_chunks.jsonl"),
    Path("data/processed/longchau_ingredients_v2/ingredients_chunks.jsonl"),
)
REQUIRED_PAYLOAD_FIELDS = (
    "chunk_id",
    "source",
    "source_name",
    "source_type",
    "entity_type",
    "entity_name",
    "slug",
    "section",
    "url",
    "text",
)


@dataclass
class PreparedPoint:
    id: str
    vector: Any
    payload: dict[str, Any]


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, dict) else {}


def _field(record: dict[str, Any], *names: str) -> Any:
    metadata = _metadata(record)
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            return value
        value = metadata.get(name)
        if value not in (None, ""):
            return value
    return None


def flatten_evidence_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a standard evidence chunk to the flat Qdrant payload schema."""
    text = _field(record, "content", "text")
    source = str(_field(record, "source") or "")
    payload = {
        "chunk_id": str(_field(record, "chunk_id") or ""),
        "source": source,
        "source_name": str(
            _field(record, "source_name") or SOURCE_NAMES.get(source, "")
        ),
        "source_type": str(_field(record, "source_type") or ""),
        "entity_type": str(_field(record, "entity_type") or "ingredient"),
        "entity_name": str(_field(record, "entity_name") or ""),
        "slug": str(_field(record, "slug") or ""),
        "section": str(_field(record, "section") or ""),
        "url": str(_field(record, "url", "source_url") or ""),
        "text": str(text or ""),
    }
    missing = [
        field for field in REQUIRED_PAYLOAD_FIELDS if not payload.get(field)
    ]
    if missing:
        raise ValueError(
            "Evidence payload missing required fields: " + ", ".join(missing)
        )
    return payload


def iter_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path} line {line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL record must be an object in {path} line "
                        f"{line_number}"
                    )
                yield record


def prepare_payloads(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [flatten_evidence_payload(record) for record in records]


def _point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))


def _batches(values: list[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    for index in range(0, len(values), batch_size):
        yield values[index:index + batch_size]


def embed_and_ingest(
    records: Iterable[dict[str, Any]],
    *,
    qdrant_client: Any,
    embedding_model: Any,
    collection_name: str = CLINICAL_EVIDENCE_COLLECTION,
    batch_size: int = 64,
) -> int:
    """Embed records and upsert flat payloads into Qdrant."""
    if qdrant_client is None:
        raise ValueError("qdrant_client is required")
    if embedding_model is None:
        raise ValueError("embedding_model is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    payloads = prepare_payloads(records)
    if not payloads:
        return 0

    try:
        from qdrant_client.http import models
    except ModuleNotFoundError:
        models = None

    written = 0
    for batch in _batches(payloads, batch_size):
        vectors = embedding_model.encode([item["text"] for item in batch])
        points = []
        for payload, vector in zip(batch, vectors, strict=True):
            point_data = {
                "id": _point_id(payload["chunk_id"]),
                "vector": (
                    vector.tolist() if hasattr(vector, "tolist") else vector
                ),
                "payload": payload,
            }
            points.append(
                models.PointStruct(**point_data)
                if models is not None
                else PreparedPoint(**point_data)
            )
        qdrant_client.upsert(collection_name=collection_name, points=points)
        written += len(points)
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare flat Qdrant payloads for clinical evidence chunks"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        nargs="+",
        default=list(DEFAULT_INPUT_PATHS),
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    records = iter_jsonl(args.input_path)
    if args.limit:
        records = (record for index, record in enumerate(records) if index < args.limit)
    payloads = prepare_payloads(records)
    print(f"collection: {CLINICAL_EVIDENCE_COLLECTION}")
    print(f"payloads_prepared: {len(payloads)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
