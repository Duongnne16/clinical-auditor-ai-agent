from typing import Any


class Retriever:
    """Retrieval boundary for evidence and future semantic doctor memory."""

    def retrieve_evidence(self, query: str) -> list[dict[str, Any]]:
        return []

    def retrieve_doctor_memory(
        self, doctor_id: str, query: str
    ) -> list[dict[str, Any]]:
        return []
