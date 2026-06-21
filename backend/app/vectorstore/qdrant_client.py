from qdrant_client import QdrantClient

from backend.app.core.config import Settings


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """Build a client on explicit request; this module never connects on import."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
