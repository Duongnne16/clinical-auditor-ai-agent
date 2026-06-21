from collections.abc import Iterable
from typing import Any


def build_drug_mapping(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Return an empty mapping until normalization rules are defined."""
    return {}
