from collections.abc import Iterable
from typing import Any


def embed_and_ingest(records: Iterable[dict[str, Any]]) -> int:
    """Explicit future entry point; does not load a model or connect on import."""
    raise NotImplementedError("Embedding and Qdrant ingestion are not implemented.")
