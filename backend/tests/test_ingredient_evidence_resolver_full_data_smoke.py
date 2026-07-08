from pathlib import Path

import pytest

from backend.app.services.ingredient_evidence_resolver import (
    IngredientEvidenceResolver,
)


CATALOG_PATH = Path(
    "data/processed/evidence_ingredients_v2/evidence_ingredient_catalog.jsonl"
)


@pytest.mark.skipif(
    not CATALOG_PATH.exists(),
    reason="Full evidence ingredient catalog is unavailable",
)
def test_full_catalog_smoke_queries() -> None:
    service = IngredientEvidenceResolver(CATALOG_PATH)
    expected = {
        "Paracetamol": "paracetamol",
        "Metformin": "metformin",
        "Amlodipine": "amlodipin",
        "Acyclovir": "aciclovir",
        "Povidone Iodine": "povidone-iodine",
        "Diclofenac sodium": "diclofenac",
        "Sodium chloride": "natri-clorid",
    }
    results = service.resolve_many(
        [*expected, "Thuốc không tồn tại xyz"]
    )

    assert service.get_stats()["records_loaded"] > 1000
    for query, result in zip(expected, results):
        assert result["status"] == "resolved", query
        assert result["best_match"]["evidence_slug"] == expected[query]
    assert results[-1]["status"] == "unresolved"


@pytest.mark.skipif(
    not CATALOG_PATH.exists(),
    reason="Full evidence ingredient catalog is unavailable",
)
def test_optional_manual_aliases_report_actual_slug() -> None:
    service = IngredientEvidenceResolver(CATALOG_PATH)

    for query in [
        "Cetirizine",
        "Caffeine",
        "Clavulanic acid",
        "Folic acid",
        "Ascorbic acid",
    ]:
        result = service.resolve(query)
        if result["status"] == "resolved":
            assert result["best_match"]["evidence_slug"]
