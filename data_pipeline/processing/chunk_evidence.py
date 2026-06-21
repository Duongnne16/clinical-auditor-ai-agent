from typing import Any


def chunk_evidence(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve source metadata while chunking is not implemented."""
    return [dict(record)]
