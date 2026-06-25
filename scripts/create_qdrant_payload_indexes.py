from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings


INDEX_FIELDS = ["slug", "section", "source", "entity_type"]


def _keyword_schema():
    try:
        from qdrant_client.models import PayloadSchemaType

        return PayloadSchemaType.KEYWORD
    except ImportError:
        from qdrant_client.http.models import PayloadSchemaType

        return PayloadSchemaType.KEYWORD


def _is_already_exists_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "already exists" in message
        or "already has" in message
        or "already created" in message
    )


def _print_payload_schema(client, collection_name: str) -> None:
    try:
        collection_info = client.get_collection(collection_name)
    except Exception as exc:  # pragma: no cover - depends on Qdrant version
        print(f"Could not read collection payload schema: {exc}")
        return

    config = getattr(collection_info, "config", None)
    params = getattr(config, "params", None)
    payload_schema = (
        getattr(collection_info, "payload_schema", None)
        or getattr(config, "payload_schema", None)
        or getattr(params, "payload_schema", None)
    )

    if not payload_schema:
        print("Payload schema/index info is unavailable from this qdrant-client.")
        return

    print("Current payload indexes:")
    if isinstance(payload_schema, dict):
        for field_name in sorted(payload_schema):
            print(f"- {field_name}: {payload_schema[field_name]}")
        return

    print(payload_schema)


def main() -> None:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise SystemExit(
            "qdrant-client is not installed in this Python environment. "
            "Install requirements or run this script with the project .venv."
        ) from exc

    settings = get_settings()
    collection_name = settings.qdrant_medical_evidence_collection
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
    keyword_schema = _keyword_schema()

    print(f"Collection: {collection_name}")
    for field_name in INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=keyword_schema,
            )
            print(f"Created payload index: {field_name}")
        except Exception as exc:
            if _is_already_exists_error(exc):
                print(f"Payload index already exists: {field_name}")
                continue
            raise

    _print_payload_schema(client, collection_name)


if __name__ == "__main__":
    main()
