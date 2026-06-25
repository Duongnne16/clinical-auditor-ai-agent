import importlib
import json
import sys
from pathlib import Path

import pytest

from backend.app.services.drug_mapping_service import DrugMappingService


def _ingredient(name: str, strength: str = "500mg") -> dict:
    return {
        "name": name,
        "normalized_name": name.lower(),
        "normalized_name_no_diacritics": name.lower(),
        "strength_raw": strength,
        "strength_value": 500.0,
        "strength_unit": "mg",
    }


def _record(
    mapping_id: str,
    brand: str,
    aliases: list[str],
    ingredients: list[dict],
    *,
    confidence: str = "high",
    warnings: list[str] | None = None,
) -> dict:
    return {
        "mapping_id": mapping_id,
        "brand_name": brand,
        "normalized_brand_name": brand.lower(),
        "normalized_brand_name_no_diacritics": brand.lower(),
        "brand_aliases": aliases,
        "active_ingredients": ingredients,
        "category": "Thuốc thử",
        "url": f"https://example.test/{mapping_id}",
        "confidence": confidence,
        "warnings": warnings or [],
    }


def _mapping_file(tmp_path: Path) -> Path:
    records = [
        _record(
            "drug:acyclovir",
            "Thuốc Acyclovir 400mg Stella Pharm điều trị nhiễm Herpes simplex",
            ["acyclovir 400mg stella pharm", "acyclovir 400mg", "acyclovir"],
            [_ingredient("Acyclovir", "400mg")],
        ),
        _record(
            "drug:hapacol",
            "Thuốc Hapacol 650 Extra DHG điều trị đau đầu",
            ["hapacol 650 extra dhg", "hapacol 650 extra", "hapacol"],
            [_ingredient("Paracetamol", "650mg"), _ingredient("Cafein", "65mg")],
        ),
        _record(
            "drug:metformin-single",
            "Thuốc Metformin 500",
            ["metformin 500"],
            [_ingredient("Metformin", "500mg")],
        ),
        _record(
            "drug:metformin-multi",
            "Thuốc Metformin Plus",
            ["metformin plus"],
            [_ingredient("Metformin", "500mg"), _ingredient("Gliclazide", "80mg")],
            confidence="medium",
        ),
        _record(
            "drug:natri",
            "Thuốc Natri Clorid",
            ["natri clorid", "natri cloríd"],
            [_ingredient("Natri clorid", "0.9%")],
        ),
        _record(
            "drug:warning",
            "Thuốc Warning",
            ["warning"],
            [_ingredient("Warning")],
            warnings=["source_warning"],
        ),
    ]
    path = tmp_path / "mapping.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        + "\n{broken json\n"
        + json.dumps(["not-object"])
        + "\n"
        + json.dumps({"mapping_id": "missing-brand"})
        + "\n",
        encoding="utf-8",
    )
    return path


def test_load_mapping_stats_and_optional_fields(tmp_path: Path) -> None:
    service = DrugMappingService(_mapping_file(tmp_path), enable_fuzzy=False)
    stats = service.get_stats()

    assert stats["records_loaded"] == 6
    assert stats["invalid_line_count"] == 3
    assert stats["records_with_active_ingredients"] == 6
    assert stats["confidence_counts"] == {"high": 5, "medium": 1, "low": 0}
    assert stats["alias_count"] > 0
    assert stats["ingredient_index_count"] > 0


def test_exact_brand_match(tmp_path: Path) -> None:
    service = DrugMappingService(_mapping_file(tmp_path), enable_fuzzy=False)
    result = service.lookup(
        "Thuốc Acyclovir 400mg Stella Pharm điều trị nhiễm Herpes simplex"
    )

    assert result["status"] == "matched"
    assert result["best_match"]["match_type"] == "exact_brand"
    assert result["best_match"]["active_ingredients"][0]["name"] == "Acyclovir"
    assert result["requires_review"] is False


def test_exact_alias_and_no_diacritics_match(tmp_path: Path) -> None:
    service = DrugMappingService(_mapping_file(tmp_path), enable_fuzzy=False)

    hapacol = service.lookup("hapacol 650 extra")
    natri = service.lookup("Natri Clorid")

    assert hapacol["best_match"]["match_type"] == "exact_alias"
    assert [item["name"] for item in hapacol["best_match"]["active_ingredients"]] == [
        "Paracetamol",
        "Cafein",
    ]
    assert natri["status"] == "matched"


def test_ingredient_match_ranks_single_high_before_multi_medium(
    tmp_path: Path,
) -> None:
    service = DrugMappingService(_mapping_file(tmp_path), enable_fuzzy=False)
    result = service.lookup("metformin", top_k=5)

    assert result["best_match"]["match_type"] == "ingredient_name"
    assert result["best_match"]["mapping_id"] == "drug:metformin-single"
    assert result["requires_review"] is False
    assert len(result["candidates"]) == 2


def test_fuzzy_match_always_requires_review(tmp_path: Path) -> None:
    service = DrugMappingService(
        _mapping_file(tmp_path), enable_fuzzy=True, fuzzy_threshold=70
    )
    result = service.lookup("hapacol extra")

    assert result["status"] == "matched"
    assert result["best_match"]["match_type"].startswith("fuzzy_")
    assert result["requires_review"] is True
    assert "mapping_requires_review" in result["warnings"]


def test_fuzzy_rejects_unrelated_multi_token_partial_matches(
    tmp_path: Path,
) -> None:
    service = DrugMappingService(
        _mapping_file(tmp_path), enable_fuzzy=True, fuzzy_threshold=70
    )

    assert service.lookup("thuoc abc khong co that")["status"] == "unmatched"


def test_warning_and_medium_matches_require_review(tmp_path: Path) -> None:
    service = DrugMappingService(_mapping_file(tmp_path), enable_fuzzy=False)

    warning = service.lookup("warning")
    medium = service.lookup("metformin plus")

    assert warning["requires_review"] is True
    assert medium["requires_review"] is True


def test_unmatched_empty_input_and_top_k_validation(tmp_path: Path) -> None:
    service = DrugMappingService(_mapping_file(tmp_path), enable_fuzzy=False)

    unknown = service.lookup("thuoc abc khong co that")
    empty = service.lookup(" ")

    assert unknown["status"] == "unmatched"
    assert unknown["warnings"] == ["no_mapping_found"]
    assert empty["warnings"] == ["empty_input", "no_mapping_found"]
    with pytest.raises(ValueError, match="top_k"):
        service.lookup("hapacol", top_k=0)


def test_lookup_many_preserves_input_order(tmp_path: Path) -> None:
    service = DrugMappingService(_mapping_file(tmp_path), enable_fuzzy=False)
    results = service.lookup_many(["hapacol", "metformin", "unknown"])

    assert [result["input_name"] for result in results] == [
        "hapacol",
        "metformin",
        "unknown",
    ]
    assert [result["status"] for result in results] == [
        "matched",
        "matched",
        "unmatched",
    ]


def test_difflib_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = importlib.import_module(
        "backend.app.services.drug_mapping_service"
    )
    monkeypatch.setattr(module, "rapidfuzz_fuzz", None)
    service = module.DrugMappingService(
        _mapping_file(tmp_path), enable_fuzzy=True, fuzzy_threshold=65
    )

    assert service.get_stats()["fuzzy_backend"] == "difflib"
    assert service.lookup("hapacol extra")["status"] == "matched"


def test_import_does_not_read_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "backend.app.services.drug_mapping_service"
    sys.modules.pop(module_name, None)
    calls: list[object] = []
    original_open = Path.open

    def tracking_open(self: Path, *args: object, **kwargs: object):
        calls.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    importlib.import_module(module_name)

    assert calls == []
