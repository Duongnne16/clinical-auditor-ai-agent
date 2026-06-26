import builtins
import importlib
import sys

import pytest

from backend.app.services.normalize_drugs_service import (
    NormalizeDrugsService,
    extract_strengths_and_clean_name,
)


def _ingredient(name: str, strength: str = "100mg") -> dict:
    return {
        "name": name,
        "normalized_name": name.lower(),
        "strength_raw": strength,
        "strength_value": 100.0,
        "strength_unit": "mg",
    }


class FakeDrugMappingService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def lookup(self, raw_name: str, top_k: int = 5) -> dict:
        self.calls.append((raw_name, top_k))
        if raw_name in {"Kagascdine", "BrandFuzzy"}:
            if raw_name == "Kagascdine":
                return {
                    "status": "unmatched",
                    "best_match": None,
                    "candidates": [],
                    "requires_review": True,
                    "warnings": ["no_mapping_found"],
                }
            candidate = {
                "mapping_id": "mapping:fuzzy-brand",
                "matched_brand": "Unrelated Brand",
                "match_type": "fuzzy_alias",
                "score": 0.9,
                "confidence": "high",
                "active_ingredients": [_ingredient("Cafein")],
                "url": "https://example.test/fuzzy",
                "warnings": [],
            }
            return {
                "status": "matched",
                "best_match": candidate,
                "candidates": [candidate],
                "requires_review": True,
                "warnings": ["mapping_requires_review"],
            }
        if raw_name in {"BrandExact", "BrandConflict"}:
            ingredient_name = (
                "Paracetamol"
                if raw_name == "BrandExact"
                else "Cafein"
            )
            candidate = {
                "mapping_id": f"mapping:{raw_name}",
                "matched_brand": raw_name,
                "match_type": "exact_alias",
                "score": 1.0,
                "confidence": "high",
                "active_ingredients": [_ingredient(ingredient_name)],
                "url": f"https://example.test/{raw_name}",
                "warnings": [],
            }
            return {
                "status": "matched",
                "best_match": candidate,
                "candidates": [candidate],
                "requires_review": False,
                "warnings": [],
            }
        if raw_name in {
            "Unknown",
            "Losec 20mg",
            "Losec 20mg (Omeprazole)",
            "Levofloxacine 500mg",
            "Amoxicillin 875/125mg",
            "Losec (20mg)",
            "Sucrate Gel",
        }:
            return {
                "status": "unmatched",
                "best_match": None,
                "candidates": [],
                "requires_review": True,
                "warnings": ["no_mapping_found"],
            }
        if raw_name == "Omeprazole":
            ingredients = [_ingredient("Esomeprazole")]
            candidate = {
                "mapping_id": "mapping:esomeprazole",
                "matched_brand": "Esomeprazole Product",
                "match_type": "fuzzy_alias",
                "score": 0.86,
                "confidence": "high",
                "active_ingredients": ingredients,
                "url": "https://example.test/esomeprazole",
                "warnings": [],
            }
            return {
                "status": "matched",
                "best_match": candidate,
                "candidates": [candidate],
                "requires_review": True,
                "warnings": ["mapping_requires_review"],
            }
        if raw_name == "Levofloxacin 500mg":
            ingredients = [_ingredient("Levofloxacin", "750mg")]
            candidate = {
                "mapping_id": "mapping:levo-750",
                "matched_brand": "Levo Product 750mg",
                "match_type": "ingredient_name",
                "score": 0.95,
                "confidence": "high",
                "active_ingredients": ingredients,
                "url": "https://example.test/levo-750",
                "warnings": [],
            }
            return {
                "status": "matched",
                "best_match": candidate,
                "candidates": [candidate],
                "requires_review": False,
                "warnings": [],
            }
        if raw_name == "Review Drug":
            ingredients = [_ingredient("Paracetamol")]
            warning = ["mapping_data_warning"]
            requires_review = True
            confidence = "medium"
        elif raw_name == "Mystery Drug":
            ingredients = [_ingredient("Unknown Ingredient")]
            warning = []
            requires_review = False
            confidence = "high"
        elif raw_name == "Evidence Review Drug":
            ingredients = [_ingredient("Review Ingredient")]
            warning = []
            requires_review = False
            confidence = "high"
        elif raw_name == "Paracetamol Duplicate":
            ingredients = [_ingredient("Paracetamol")]
            warning = []
            requires_review = False
            confidence = "high"
        else:
            ingredients = [
                _ingredient("Paracetamol", "650mg"),
                _ingredient("Cafein", "65mg"),
            ]
            warning = []
            requires_review = False
            confidence = "high"
        candidate = {
            "mapping_id": f"mapping:{raw_name}",
            "matched_brand": f"Brand {raw_name}",
            "match_type": "exact_alias",
            "score": 1.0,
            "confidence": confidence,
            "active_ingredients": ingredients,
            "url": "https://example.test/drug",
            "warnings": warning,
        }
        return {
            "status": "matched",
            "best_match": candidate,
            "candidates": [candidate],
            "requires_review": requires_review,
            "warnings": (
                ["mapping_requires_review"] if requires_review else []
            ),
        }

    def get_stats(self) -> dict:
        return {"records_loaded": 4}


class FakeIngredientResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def resolve(self, name: str, top_k: int = 5) -> dict:
        self.calls.append((name, top_k))
        if name in {
            "Unknown",
            "Losec",
            "Unknown Ingredient",
            "Amoxicillin",
        }:
            return {
                "status": "unresolved",
                "best_match": None,
                "requires_review": True,
                "warnings": ["no_evidence_ingredient_found"],
            }
        review = name in {"Review Ingredient", "Levofloxacine"}
        slug = {
            "Levofloxacine": "levofloxacin",
            "Omeprazol": "omeprazole",
        }.get(name, name.lower().replace(" ", "-"))
        return {
            "status": "resolved",
            "best_match": {
                "evidence_name": f"Evidence {name}",
                "evidence_slug": slug,
                "match_type": "fuzzy" if review else "exact",
                "score": 0.9 if review else 1.0,
            },
            "requires_review": review,
            "warnings": (
                ["evidence_resolution_requires_review"] if review else []
            ),
        }

    def get_stats(self) -> dict:
        return {"records_loaded": 3}


@pytest.fixture
def service() -> NormalizeDrugsService:
    return NormalizeDrugsService(
        FakeDrugMappingService(), FakeIngredientResolver()
    )


def test_normalize_string_medication(service: NormalizeDrugsService) -> None:
    result = service.normalize_medication("Hapacol 650 Extra")

    assert result["mapping_status"] == "matched"
    assert [
        ingredient["evidence_slug"]
        for ingredient in result["active_ingredients"]
    ] == ["paracetamol", "cafein"]
    assert result["requires_review"] is False
    assert result["mapping_candidates"][0][
        "active_ingredient_names"
    ] == ["Paracetamol", "Cafein"]


def test_dict_preserves_standard_and_original_fields(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication(
        {
            "name": "Hapacol 650 Extra",
            "dose": "1 viên",
            "frequency": "2 lần/ngày",
            "route": "uống",
            "duration": "3 ngày",
            "note": "sau ăn",
            "prescription_line": 2,
            "parser_confidence": 0.94,
        }
    )

    assert result["raw_name"] == "Hapacol 650 Extra"
    assert result["dose"] == "1 viên"
    assert result["frequency"] == "2 lần/ngày"
    assert result["route"] == "uống"
    assert result["duration"] == "3 ngày"
    assert result["note"] == "sau ăn"
    assert result["original_fields"] == {
        "prescription_line": 2,
        "parser_confidence": 0.94,
    }


def test_unmatched_drug(service: NormalizeDrugsService) -> None:
    result = service.normalize_medication("Unknown")

    assert result["mapping_status"] == "unmatched"
    assert result["active_ingredients"] == []
    assert result["requires_review"] is True
    assert "drug_mapping_not_found" in result["warnings"]


def test_unresolved_ingredient(service: NormalizeDrugsService) -> None:
    result = service.normalize_medication("Mystery Drug")

    assert result["requires_review"] is True
    assert "ingredient_evidence_unresolved" in result["warnings"]
    ingredient = result["active_ingredients"][0]
    assert ingredient["evidence_status"] == "unresolved"
    assert ingredient["evidence_slug"] is None


def test_mapping_requires_review_and_deduplicates_warning(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication("Review Drug")

    assert result["requires_review"] is True
    assert result["warnings"].count("mapping_requires_review") == 1
    assert "mapping_data_warning" in result["warnings"]


def test_evidence_requires_review(service: NormalizeDrugsService) -> None:
    result = service.normalize_medication("Evidence Review Drug")

    assert result["requires_review"] is True
    assert result["active_ingredients"][0][
        "evidence_requires_review"
    ] is True


def test_normalize_many_summary_and_slug_order(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_many(
        [
            "Hapacol 650 Extra",
            "Paracetamol Duplicate",
            "Mystery Drug",
            "Unknown",
        ]
    )

    assert result["summary"] == {
        "total_medications": 4,
        "matched_medications": 3,
        "product_matched_medications": 3,
        "ingredient_with_brand_medications": 0,
        "ingredient_only_medications": 0,
        "unmatched_medications": 1,
        "usable_medications": 3,
        "total_active_ingredients": 4,
        "resolved_ingredients": 3,
        "unresolved_ingredients": 1,
        "requires_review": True,
    }
    assert result["resolved_evidence_slugs"] == [
        "paracetamol",
        "cafein",
        "paracetamol",
    ]
    assert result["unique_evidence_slugs"] == [
        "cafein",
        "paracetamol",
    ]
    assert result["unresolved_ingredients"] == [
        {
            "medication_index": 2,
            "raw_name": "Mystery Drug",
            "ingredient_name": "Unknown Ingredient",
            "evidence_warnings": ["no_evidence_ingredient_found"],
        }
    ]


def test_empty_and_invalid_input_skip_dependencies(
    service: NormalizeDrugsService,
) -> None:
    mapping = service.drug_mapping_service
    empty = service.normalize_medication({"raw_name": ""})
    invalid = service.normalize_medication(123)  # type: ignore[arg-type]

    assert empty["warnings"] == ["empty_drug_name"]
    assert invalid["warnings"] == [
        "invalid_medication_input",
        "empty_drug_name",
    ]
    assert mapping.calls == []


def test_top_k_validation_and_stats(
    service: NormalizeDrugsService,
) -> None:
    with pytest.raises(ValueError):
        service.normalize_medication("Drug", top_k=0)
    with pytest.raises(ValueError):
        service.normalize_many([], top_k=0)
    assert service.get_stats() == {
        "service": "NormalizeDrugsService",
        "drug_mapping": {"records_loaded": 4},
        "ingredient_resolver": {"records_loaded": 3},
    }


def test_import_does_not_instantiate_default_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = builtins.open

    def guarded_open(*args, **kwargs):
        path = str(args[0]) if args else ""
        if "drug_mapping.jsonl" in path or (
            "evidence_ingredient_catalog.jsonl" in path
        ):
            raise AssertionError("data file read during import")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    sys.modules.pop("backend.app.services.normalize_drugs_service", None)
    importlib.import_module("backend.app.services.normalize_drugs_service")


def test_extract_strengths_and_clean_name() -> None:
    parsed = extract_strengths_and_clean_name(
        "Levofloxacine 500 mg"
    )
    decimal = extract_strengths_and_clean_name("Drug 0.5mg")
    iu = extract_strengths_and_clean_name("Vitamin D 1000 IU")

    assert parsed == {
        "clean_name": "Levofloxacine",
        "strengths": [
            {
                "strength_raw": "500 mg",
                "strength_value": 500.0,
                "strength_unit": "mg",
                "is_combination": False,
            }
        ],
        "has_combination_strength": False,
    }
    assert decimal["strengths"][0]["strength_value"] == 0.5
    assert iu["strengths"][0]["strength_unit"] == "IU"


def test_extract_ratio_strengths_and_preserves_slash_drug_names() -> None:
    parsed = extract_strengths_and_clean_name(
        "Sucralfate (Sucrate Gel) 1g/5mL"
    )
    spaced = extract_strengths_and_clean_name("Drug 250 mg / 5 mL")
    slash_name = extract_strengths_and_clean_name(
        "Amoxicillin/Clavulanate"
    )

    assert parsed == {
        "clean_name": "Sucralfate (Sucrate Gel)",
        "strengths": [
            {
                "strength_raw": "1g/5mL",
                "strength_value": 1.0,
                "strength_unit": "g",
                "is_combination": False,
            }
        ],
        "has_combination_strength": False,
    }
    assert "1g/" not in parsed["clean_name"]
    assert not parsed["clean_name"].strip().endswith("/")
    assert spaced["clean_name"] == "Drug"
    assert spaced["strengths"][0]["strength_raw"] == "250 mg / 5 mL"
    assert slash_name["clean_name"] == "Amoxicillin/Clavulanate"
    assert slash_name["strengths"] == []


def test_parenthetical_ingredient_fallback(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication(
        "Losec 20mg (Omeprazole)"
    )

    assert result["mapping_status"] == "ingredient_only"
    assert result["matched_brand"] is None
    assert result["active_ingredients"][0]["name"] == "Omeprazole"
    assert result["active_ingredients"][0]["strength_raw"] == "20mg"
    assert result["active_ingredients"][0][
        "evidence_slug"
    ] == "omeprazole"
    assert result["requires_review"] is True
    assert result["warnings"] == [
        "drug_product_mapping_not_found",
        "ingredient_resolved_without_product_mapping",
    ]


def test_brand_only_without_ingredient_remains_unmatched(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication("Losec 20mg")

    assert result["mapping_status"] == "unmatched"
    assert result["active_ingredients"] == []


def test_fuzzy_ingredient_fallback_preserves_strength(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication("Levofloxacine 500mg")

    assert result["mapping_status"] == "ingredient_only"
    ingredient = result["active_ingredients"][0]
    assert ingredient["strength_raw"] == "500mg"
    assert ingredient["strength_value"] == 500.0
    assert ingredient["evidence_slug"] == "levofloxacin"
    assert ingredient["evidence_requires_review"] is True


def test_generic_ingredient_strength_rejects_wrong_product(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication("Levofloxacin 500mg")

    assert result["mapping_status"] == "ingredient_only"
    assert result["matched_brand"] is None
    assert result["active_ingredients"][0][
        "strength_raw"
    ] == "500mg"


def test_dangerous_fuzzy_product_is_rejected(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication("Omeprazole")

    assert result["mapping_status"] == "ingredient_only"
    assert result["matched_brand"] is None
    assert result["active_ingredients"][0][
        "evidence_slug"
    ] == "omeprazole"
    assert "unsafe_product_fuzzy_match_rejected" in result["warnings"]


def test_exact_product_still_wins(service: NormalizeDrugsService) -> None:
    result = service.normalize_medication("Kaflovo 500")

    assert result["mapping_status"] == "matched"
    assert result["mapping_match_type"] == "exact_alias"


def test_combination_strength_does_not_create_single_fallback(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication("Amoxicillin 875/125mg")

    assert result["mapping_status"] == "unmatched"
    assert result["active_ingredients"] == []


def test_parenthetical_strength_is_not_ingredient(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication("Losec (20mg)")

    assert result["mapping_status"] == "unmatched"


def test_batch_separates_ingredient_only_status(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_many(
        ["Kaflovo 500", "Levofloxacine 500mg", "Losec 20mg"]
    )

    assert result["summary"]["matched_medications"] == 1
    assert result["summary"]["product_matched_medications"] == 1
    assert result["summary"]["ingredient_with_brand_medications"] == 0
    assert result["summary"]["ingredient_only_medications"] == 1
    assert result["summary"]["unmatched_medications"] == 1
    assert result["summary"]["usable_medications"] == 2
    assert result["ingredient_only_medications"] == [
        {
            "medication_index": 1,
            "raw_name": "Levofloxacine 500mg",
            "evidence_slugs": ["levofloxacin"],
        }
    ]
    assert result["unmapped_medications"] == [
        {
            "raw_name": "Losec 20mg",
            "reason": "drug_mapping_not_found",
        }
    ]


def test_generic_first_brand_not_found_is_non_blocking(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication(
        {
            "raw_line": "Paracetamol (Kagascdine) 500mg x 10 Viên",
            "instruction": "Uống sau ăn",
        }
    )

    assert result["mapping_status"] == "ingredient_with_brand"
    assert result["brand_text"] == "Kagascdine"
    assert result["brand_verification_status"] == "not_found"
    assert result["active_ingredients"][0][
        "evidence_slug"
    ] == "paracetamol"
    assert result["requires_review"] is False
    assert result["instruction"] == "Uống sau ăn"


def test_generic_first_sucralfate_ratio_strength(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication(
        {"raw_line": "2. Sucralfate (Sucrate Gel) 1g/5mL x 15 goi"}
    )

    assert result["generic_text"] == "Sucralfate"
    assert result["brand_text"] == "Sucrate Gel"
    assert result["strength_text"] == "1g/5mL"
    assert result["mapping_status"] == "ingredient_with_brand"
    ingredient = result["active_ingredients"][0]
    assert ingredient["name"] == "Sucralfate"
    assert ingredient["strength_raw"] == "1g/5mL"
    assert ingredient["strength_value"] == 1.0
    assert ingredient["strength_unit"] == "g"
    assert ingredient["evidence_slug"] == "sucralfate"
    candidates = [
        result["generic_text"],
        ingredient["name"],
        *[
            candidate["matched_brand"]
            for candidate in result["mapping_candidates"]
        ],
    ]
    assert all("1g/" not in str(candidate) for candidate in candidates)
    assert all(not str(candidate).strip().endswith("/") for candidate in candidates)


def test_generic_first_exact_brand_verified(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication(
        {"raw_line": "Paracetamol (BrandExact) 500mg"}
    )

    assert result["mapping_status"] == "ingredient_with_brand"
    assert result["brand_verification_status"] == "verified"
    assert result["requires_review"] is False


def test_generic_first_fuzzy_brand_is_not_accepted(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication(
        {"raw_line": "Paracetamol (BrandFuzzy) 500mg"}
    )

    assert result["brand_verification_status"] == "not_found"
    assert result["requires_review"] is False


def test_generic_first_exact_brand_conflict_requires_review(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication(
        {"raw_line": "Paracetamol (BrandConflict) 500mg"}
    )

    assert result["brand_verification_status"] == "conflict"
    assert result["requires_review"] is True
    assert "ingredient_brand_conflict" in result["warnings"]


def test_generic_first_fuzzy_evidence_requires_review(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_medication(
        {"raw_line": "Levofloxacine 500mg"}
    )

    assert result["mapping_status"] == "ingredient_only"
    assert result["active_ingredients"][0][
        "evidence_slug"
    ] == "levofloxacin"
    assert result["requires_review"] is True


def test_generic_first_batch_summary(
    service: NormalizeDrugsService,
) -> None:
    result = service.normalize_many(
        [
            {"raw_line": "Paracetamol (BrandExact) 500mg"},
            {"raw_line": "Metformin 750mg"},
            "Kaflovo 500",
            "Unknown",
        ]
    )

    assert result["summary"]["product_matched_medications"] == 1
    assert result["summary"]["matched_medications"] == 1
    assert result["summary"]["ingredient_with_brand_medications"] == 1
    assert result["summary"]["ingredient_only_medications"] == 1
    assert result["summary"]["unmatched_medications"] == 1
    assert result["summary"]["usable_medications"] == 3
