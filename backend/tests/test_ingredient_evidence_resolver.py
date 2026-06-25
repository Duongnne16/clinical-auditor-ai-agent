import builtins
import importlib
import json
import sys
from pathlib import Path

import pytest

import backend.app.services.ingredient_evidence_resolver as resolver_module
from backend.app.services.ingredient_evidence_resolver import (
    IngredientEvidenceResolver,
    normalize_text,
)


def _record(
    slug: str,
    name: str,
    *,
    aliases: list[str] | None = None,
    chunk_count: int = 10,
    sections: list[str] | None = None,
    warnings: list[str] | None = None,
    catalog_id: str | None = None,
) -> dict:
    return {
        "catalog_id": catalog_id or f"ingredient:{slug}",
        "entity_name": name,
        "slug": slug,
        "normalized_name": normalize_text(name),
        "aliases": aliases or [],
        "url": f"https://example.test/{slug}",
        "sections": sections or ["chi_dinh", "tuong_tac_thuoc"],
        "chunk_count": chunk_count,
        "warnings": warnings or [],
    }


def _catalog_file(tmp_path: Path) -> Path:
    records = [
        _record("paracetamol", "Paracetamol"),
        _record("metformin", "Metformin"),
        _record("amlodipin", "Amlodipin"),
        _record("aciclovir", "Aciclovir"),
        _record("diclofenac", "Diclofenac"),
        _record("methionine", "Methionine"),
        _record("natri-clorid", "Natri clorid"),
        _record("kali-clorid", "Kali clorid"),
        _record("natri-bicarbonat", "Natri bicarbonat"),
        _record("kali-bicarbonat", "Kali bicarbonat"),
        _record("cafein", "Cafein"),
        _record("acid-folic", "Acid folic"),
        _record(
            "bupivacaine-weak",
            "Bupivacaine",
            chunk_count=2,
            sections=["mo_ta_chung"],
            catalog_id="ingredient:bupivacaine-weak",
        ),
        _record(
            "bupivacaine",
            "Bupivacaine",
            chunk_count=20,
            sections=["chi_dinh", "than_trong", "tuong_tac_thuoc"],
            catalog_id="ingredient:bupivacaine",
        ),
        _record(
            "warning-drug",
            "Warning drug",
            warnings=["multiple_entity_names_for_slug"],
        ),
    ]
    path = tmp_path / "catalog.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.write("{invalid json\n")
        handle.write("[]\n")
        handle.write(json.dumps({"catalog_id": "missing"}) + "\n")
    return path


def test_load_stats_and_invalid_records(tmp_path: Path) -> None:
    service = IngredientEvidenceResolver(_catalog_file(tmp_path))
    stats = service.get_stats()

    assert stats["records_loaded"] == 15
    assert stats["invalid_line_count"] == 1
    assert stats["invalid_record_count"] == 2
    assert stats["manual_alias_count"] >= 6


def test_exact_and_catalog_warning_review(tmp_path: Path) -> None:
    service = IngredientEvidenceResolver(_catalog_file(tmp_path))

    exact = service.resolve("Paracetamol")
    warning = service.resolve("Warning drug")

    assert exact["best_match"]["evidence_slug"] == "paracetamol"
    assert exact["best_match"]["match_type"] == "exact"
    assert exact["requires_review"] is False
    assert warning["requires_review"] is True


@pytest.mark.parametrize(
    ("query", "slug"),
    [
        ("Acyclovir", "aciclovir"),
        ("Amlodipine", "amlodipin"),
        ("Caffeine", "cafein"),
        ("Folic acid", "acid-folic"),
    ],
)
def test_manual_aliases(tmp_path: Path, query: str, slug: str) -> None:
    service = IngredientEvidenceResolver(_catalog_file(tmp_path))
    result = service.resolve(query)

    assert result["best_match"]["evidence_slug"] == slug
    assert result["best_match"]["match_type"] == "manual_alias"
    assert result["requires_review"] is False


def test_manual_alias_target_must_exist(tmp_path: Path) -> None:
    service = IngredientEvidenceResolver(_catalog_file(tmp_path))

    assert service.resolve("Metronidazole")["status"] == "unresolved"


@pytest.mark.parametrize(
    ("query", "slug"),
    [
        ("Metformin HCl", "metformin"),
        ("Amlodipine besylate", "amlodipin"),
        ("Diclofenac sodium", "diclofenac"),
        ("Natri diclofenac", "diclofenac"),
    ],
)
def test_safe_salt_stripping(tmp_path: Path, query: str, slug: str) -> None:
    service = IngredientEvidenceResolver(_catalog_file(tmp_path))
    result = service.resolve(query)

    assert result["best_match"]["evidence_slug"] == slug
    assert result["best_match"]["match_type"] == "salt_stripped_exact"
    assert result["requires_review"] is False


@pytest.mark.parametrize(
    ("query", "slug", "match_type"),
    [
        ("Natri clorid", "natri-clorid", "exact"),
        ("Sodium chloride", "natri-clorid", "manual_alias"),
        ("Kali clorid", "kali-clorid", "exact"),
        ("Potassium chloride", "kali-clorid", "manual_alias"),
        ("Natri bicarbonat", "natri-bicarbonat", "exact"),
        ("Sodium bicarbonate", "natri-bicarbonat", "manual_alias"),
    ],
)
def test_protected_electrolytes_are_not_stripped(
    tmp_path: Path, query: str, slug: str, match_type: str
) -> None:
    service = IngredientEvidenceResolver(_catalog_file(tmp_path))
    result = service.resolve(query)

    assert result["best_match"]["evidence_slug"] == slug
    assert result["best_match"]["match_type"] == match_type


def test_stereochemical_prefix_requires_review(tmp_path: Path) -> None:
    service = IngredientEvidenceResolver(_catalog_file(tmp_path))
    result = service.resolve("L-Methionine")

    assert result["best_match"]["evidence_slug"] == "methionine"
    assert result["best_match"]["match_type"] == "prefix_stripped_exact"
    assert result["requires_review"] is True


def test_fuzzy_match_and_false_positive_guard(tmp_path: Path) -> None:
    service = IngredientEvidenceResolver(_catalog_file(tmp_path))

    fuzzy = service.resolve("Metformn")
    unknown = service.resolve("Sodium imaginary")

    assert fuzzy["best_match"]["evidence_slug"] == "metformin"
    assert fuzzy["best_match"]["match_type"] == "fuzzy"
    assert fuzzy["requires_review"] is True
    assert unknown["status"] == "unresolved"


def test_ranking_prefers_more_evidence(tmp_path: Path) -> None:
    service = IngredientEvidenceResolver(_catalog_file(tmp_path))

    result = service.resolve("Bupivacaine")

    assert result["best_match"]["catalog_id"] == "ingredient:bupivacaine"


def test_resolve_many_empty_input_and_top_k(tmp_path: Path) -> None:
    service = IngredientEvidenceResolver(_catalog_file(tmp_path))
    results = service.resolve_many(["Paracetamol", "", "Unknown xyz"])

    assert [result["status"] for result in results] == [
        "resolved",
        "unresolved",
        "unresolved",
    ]
    assert "empty_input" in results[1]["warnings"]
    with pytest.raises(ValueError):
        service.resolve("Paracetamol", top_k=0)


def test_difflib_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(resolver_module, "rapidfuzz_fuzz", None)
    service = IngredientEvidenceResolver(_catalog_file(tmp_path))

    assert service.get_stats()["fuzzy_backend"] == "difflib"
    assert service.resolve("Metformn")["status"] == "resolved"


def test_import_does_not_read_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = builtins.open

    def guarded_open(*args, **kwargs):
        path = str(args[0]) if args else ""
        if "evidence_ingredient_catalog" in path:
            raise AssertionError("catalog read during import")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    sys.modules.pop(
        "backend.app.services.ingredient_evidence_resolver", None
    )
    importlib.import_module(
        "backend.app.services.ingredient_evidence_resolver"
    )
