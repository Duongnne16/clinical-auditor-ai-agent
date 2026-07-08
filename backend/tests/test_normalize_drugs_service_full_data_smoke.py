from pathlib import Path

import pytest

from backend.app.services.drug_mapping_service import DrugMappingService
from backend.app.services.ingredient_evidence_resolver import (
    IngredientEvidenceResolver,
)
from backend.app.services.normalize_drugs_service import (
    NormalizeDrugsService,
)


MAPPING_PATH = Path(
    "data/processed/longchau_mapping/drug_mapping.jsonl"
)
CATALOG_PATH = Path(
    "data/processed/evidence_ingredients_v2/evidence_ingredient_catalog.jsonl"
)


@pytest.mark.skipif(
    not MAPPING_PATH.exists() or not CATALOG_PATH.exists(),
    reason="Full mapping or evidence catalog is unavailable",
)
def test_full_data_normalization_smoke() -> None:
    service = NormalizeDrugsService(
        DrugMappingService(MAPPING_PATH),
        IngredientEvidenceResolver(CATALOG_PATH),
    )
    result = service.normalize_many(
        [
            {"raw_name": "Hapacol 650 Extra", "dose": "1 viên"},
            {"raw_name": "Metformin", "dose": "500mg"},
            {"raw_name": "Mibetel Plus", "dose": "1 viên"},
            {"raw_name": "Acyclovir 400mg Stella"},
            {"raw_name": "Povidone Iodine"},
            {"raw_name": "Thuốc ABC không có thật"},
        ]
    )

    medications = result["medications"]
    slugs = [
        {
            ingredient["evidence_slug"]
            for ingredient in medication["active_ingredients"]
            if ingredient["evidence_slug"]
        }
        for medication in medications
    ]
    assert "paracetamol" in slugs[0]
    assert "metformin" in slugs[1]
    assert "aciclovir" in slugs[3]
    assert "povidone-iodine" in slugs[4]
    assert medications[5]["mapping_status"] == "unmatched"

    mibetel_unresolved = [
        item
        for item in result["unresolved_ingredients"]
        if item["medication_index"] == 2
    ]
    if mibetel_unresolved:
        assert medications[2]["requires_review"] is True
    else:
        assert "telmisartan" in slugs[2]


@pytest.mark.skipif(
    not MAPPING_PATH.exists() or not CATALOG_PATH.exists(),
    reason="Full mapping or evidence catalog is unavailable",
)
def test_safe_ingredient_only_fallback_full_data() -> None:
    service = NormalizeDrugsService(
        DrugMappingService(MAPPING_PATH),
        IngredientEvidenceResolver(CATALOG_PATH),
    )
    result = service.normalize_many(
        [
            "Losec 20mg",
            "Losec 20mg (Omeprazole)",
            "Levofloxacine 500mg",
            "Levofloxacin 500mg",
            "Omeprazole",
            "Kaflovo 500",
        ]
    )
    medications = result["medications"]

    assert medications[0]["mapping_status"] == "unmatched"
    assert medications[1]["mapping_status"] == "ingredient_only"
    assert medications[1]["active_ingredients"][0][
        "evidence_slug"
    ] == "omeprazole"
    assert medications[2]["mapping_status"] == "ingredient_only"
    assert medications[2]["active_ingredients"][0][
        "evidence_slug"
    ] == "levofloxacin"
    assert medications[3]["mapping_status"] == "ingredient_only"
    assert medications[3]["active_ingredients"][0][
        "strength_raw"
    ] == "500mg"
    assert medications[4]["mapping_status"] == "ingredient_only"
    assert medications[4]["active_ingredients"][0][
        "evidence_slug"
    ] == "omeprazole"
    assert medications[5]["mapping_status"] == "matched"
    assert medications[5]["mapping_match_type"] == "exact_alias"

    assert all(
        medication["matched_brand"] is None
        or "Esomeprazol" not in medication["matched_brand"]
        for medication in medications
    )
    assert all(
        medication["matched_brand"] is None
        or "LevoDHG 750" not in medication["matched_brand"]
        for medication in medications
    )


@pytest.mark.skipif(
    not MAPPING_PATH.exists() or not CATALOG_PATH.exists(),
    reason="Full mapping or evidence catalog is unavailable",
)
def test_generic_first_real_prescription_lines() -> None:
    service = NormalizeDrugsService(
        DrugMappingService(MAPPING_PATH),
        IngredientEvidenceResolver(CATALOG_PATH),
    )
    result = service.normalize_many(
        [
            {
                "raw_line": (
                    "Omeprazol (Kagascdine) 20mg x 56 Viên"
                )
            },
            {
                "raw_line": (
                    "Metformin (Panfor SR) 750mg x 112 Viên"
                )
            },
            {
                "raw_line": (
                    "Paracetamol (Hapacol Caplet) 500mg x 10 Viên"
                )
            },
            {
                "raw_line": (
                    "Bisoprolol (Concor) 5mg x 56 Viên"
                )
            },
            {
                "raw_line": (
                    "Spiramycin + metronidazol (Spirastad Plus) "
                    "0,75MUI + 125mg x 20 Viên"
                )
            },
            {
                "raw_line": (
                    "Tobramycin + dexamethason (Tobradex) "
                    "(0,3%+0,1%)/5ml x 1 Lọ"
                )
            },
            {"raw_line": "Levofloxacine 500mg"},
            {"raw_line": "Losec 20mg"},
        ]
    )
    medications = result["medications"]
    slugs = [
        {
            ingredient["evidence_slug"]
            for ingredient in medication["active_ingredients"]
            if ingredient.get("evidence_slug")
        }
        for medication in medications
    ]

    assert medications[0]["mapping_status"] == "ingredient_with_brand"
    assert "omeprazole" in slugs[0]
    assert "metformin" in slugs[1]
    assert "paracetamol" in slugs[2]
    assert "bisoprolol" in slugs[3]
    assert {"spiramycin", "metronidazole"} <= slugs[4]
    assert {"tobramycin", "dexamethasone"} <= slugs[5]
    assert medications[6]["mapping_status"] == "ingredient_only"
    assert "levofloxacin" in slugs[6]
    assert medications[6]["requires_review"] is True
    assert medications[7]["mapping_status"] == "unmatched"
    assert result["summary"]["usable_medications"] == 7
