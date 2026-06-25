from pathlib import Path

import pytest

from backend.app.services.drug_mapping_service import DrugMappingService


MAPPING_PATH = Path(
    "data/processed/longchau_mapping/drug_mapping.jsonl"
)


@pytest.mark.skipif(
    not MAPPING_PATH.exists(), reason="Full Long Châu mapping is unavailable"
)
def test_full_mapping_smoke_queries() -> None:
    service = DrugMappingService(MAPPING_PATH)
    queries = {
        "Hapacol 650 Extra": "matched",
        "Acyclovir 400mg Stella": "matched",
        "Metformin": "matched",
        "Paracetamol": "matched",
        "Povidone Iodine": "matched",
        "Mibetel Plus": "matched",
        "Thuốc ABC không có thật": "unmatched",
    }

    results = service.lookup_many(list(queries))

    assert service.get_stats()["records_loaded"] == 2359
    assert [result["status"] for result in results] == list(queries.values())
