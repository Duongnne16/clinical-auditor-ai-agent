import json
from pathlib import Path

import pytest

from data_pipeline.processing.build_longchau_drug_mapping import (
    build_aliases,
    build_mapping_dataset,
    normalize_text,
    normalize_unit,
    parse_ingredients,
    parse_strength,
    product_slug_from_path,
    split_outside_parentheses,
    strip_vietnamese_diacritics,
)


def _product(
    name: str,
    ingredients: str | None,
    product_id: str,
    category: str = "Thuốc thử",
) -> dict:
    record = {
        "name": name,
        "_metadata": {
            "id": product_id,
            "url": f"https://example.test/{product_id}",
            "category": category,
        },
    }
    if ingredients is not None:
        record["ingredients"] = ingredients
    return record


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def test_split_outside_parentheses() -> None:
    assert split_outside_parentheses("A (1mg), B (2mg)") == [
        "A (1mg)",
        "B (2mg)",
    ]
    assert split_outside_parentheses(
        "Bán hạ (Thân, Rễ), None (0.1g)"
    ) == ["Bán hạ (Thân, Rễ)", "None (0.1g)"]


@pytest.mark.parametrize(
    ("raw", "value", "unit", "warnings"),
    [
        ("400mg", 400.0, "mg", []),
        ("12.5mg", 12.5, "mg", []),
        ("1% w/v", 1.0, "%", []),
        ("5000iu", 5000.0, "iu", []),
        ("125mcg", 125.0, "mcg", []),
        ("2.5Mcg", 2.5, "mcg", []),
        ("8", 8.0, None, ["missing_strength_unit"]),
        ("natural", None, None, ["unparsed_strength"]),
    ],
)
def test_parse_strength(
    raw: str,
    value: float | None,
    unit: str | None,
    warnings: list[str],
) -> None:
    assert parse_strength(raw) == (value, unit, warnings)


def test_parse_single_multi_repair_and_povidone() -> None:
    single, warnings, excluded = parse_ingredients(
        "Thành phần: Acyclovir (400mg)"
    )
    assert single[0]["name"] == "Acyclovir"
    assert single[0]["strength_value"] == 400
    assert warnings == []
    assert excluded == 0

    multi, warnings, _ = parse_ingredients(
        "Thành phần: Paracetamol (650mg), Cafein (65mg)"
    )
    assert [item["name"] for item in multi] == ["Paracetamol", "Cafein"]
    assert warnings == []

    repaired, warnings, excluded = parse_ingredients(
        "Thành phần: Bán hạ (Thân, Rễ), None (0.1g)"
    )
    assert repaired[0]["name"] == "Bán hạ (Thân, Rễ)"
    assert repaired[0]["strength_raw"] == "0.1g"
    assert warnings == ["repaired_none_strength_pair"]
    assert excluded == 0

    povidone, warnings, _ = parse_ingredients(
        "Thành phần: Povidone Iodine (1% w/v)"
    )
    assert povidone[0]["strength_value"] == 1
    assert povidone[0]["strength_unit"] == "%"
    assert warnings == []


def test_excipient_exclusion_does_not_remove_topical_actives() -> None:
    ingredients, warnings, excluded = parse_ingredients(
        "Thành phần: Menthol (8mg), Camphor (4mg), "
        "Methyl salicylate (2%), Eucalyptol (1mg), "
        "Povidone iodine (1%), Tá dược vừa đủ (10g), "
        "Parafin (2g), Mineral oil (3ml)"
    )

    assert [item["name"] for item in ingredients] == [
        "Menthol",
        "Camphor",
        "Methyl salicylate",
        "Eucalyptol",
        "Povidone iodine",
    ]
    assert warnings == ["excluded_excipient_or_vehicle"]
    assert excluded == 3


def test_equivalent_leading_strength_is_parsed_medium_quality() -> None:
    ingredients, warnings, _ = parse_ingredients(
        "Thành phần: 22 mg Olopatadin hydroclorid "
        "(tương đương với 2 mg olopatadin)"
    )

    assert ingredients[0]["name"] == "Olopatadin hydroclorid"
    assert ingredients[0]["strength_value"] == 22
    assert ingredients[0]["strength_unit"] == "mg"
    assert warnings == ["contains_equivalent_phrase"]


def test_normalization_helpers() -> None:
    assert normalize_text("  Thuốc® A–B  ") == "thuốc a-b"
    assert strip_vietnamese_diacritics("Thuốc Đỏ") == "thuoc do"
    assert normalize_unit("McG") == "mcg"
    assert normalize_unit("μg") == "µg"


def test_aliases_are_useful_and_filter_generic_single_tokens() -> None:
    hapacol = build_aliases(
        "Thuốc Hapacol 650 Extra DHG điều trị đau đầu "
        "(10 vỉ x 10 viên)"
    )
    assert "hapacol 650 extra dhg" in hapacol
    assert "hapacol 650 extra" in hapacol
    assert "hapacol 650" in hapacol
    assert "hapacol" in hapacol

    acyclovir = build_aliases(
        "Thuốc Acyclovir 400mg Stella Pharm điều trị nhiễm Herpes"
    )
    assert "acyclovir 400mg stella pharm" in acyclovir
    assert "acyclovir 400mg" in acyclovir
    assert "acyclovir" in acyclovir

    for forbidden in (
        "plus", "extra", "forte", "stella", "dhg", "opc", "500mg",
        "xoa", "phong",
    ):
        assert forbidden not in build_aliases(f"Thuốc {forbidden}")


def test_product_slug_only_strips_matching_numeric_id() -> None:
    assert product_slug_from_path(Path("hapacol-28113.json"), "28113") == "hapacol"
    assert (
        product_slug_from_path(
            Path("calquence-100-mg-astrazeneca-10-x6.json"),
            "calquence-100-mg-astrazeneca-10-x6",
        )
        == "calquence-100-mg-astrazeneca-10-x6"
    )


def test_end_to_end_counts_duplicate_id_fallback_and_atomic_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "input"
    _write(
        root / "Category" / "single-10.json",
        _product(
            "Thuốc Single 10mg ABC điều trị bệnh",
            "Thành phần: Alpha (10mg)",
            "10",
        ),
    )
    _write(
        root / "Category" / "multi-1.json",
        _product(
            "Thuốc Multi Plus điều trị bệnh",
            "Thành phần: Beta (5mg), Menthol (2mg)",
            "1",
        ),
    )
    _write(
        root / "Category" / "missing-1.json",
        _product("Thuốc Missing", None, "1"),
    )
    before = {
        path: path.read_bytes() for path in root.rglob("*.json")
    }
    output = tmp_path / "output"

    report = build_mapping_dataset(root, output)
    mappings = [
        json.loads(line)
        for line in (output / "drug_mapping.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert {path: path.read_bytes() for path in root.rglob("*.json")} == before
    assert report["mappings_written"] == 3
    assert report["records_with_active_ingredients"] == 2
    assert report["records_without_active_ingredients"] == 1
    assert report["single_ingredient_count"] == 1
    assert report["multi_ingredient_count"] == 1
    assert report["duplicate_product_id_fallback_count"] == 2
    assert len({mapping["mapping_id"] for mapping in mappings}) == 3
    assert any(
        item["normalized_name"] == "menthol"
        and item["name"] == "Menthol"
        for item in report["top_ingredient_names"]
    )
    assert json.loads(
        (output / "failed_or_low_confidence_samples.json").read_text(
            encoding="utf-8"
        )
    )
    assert not list(output.glob("*.tmp"))


def test_strict_mode_fails_on_structural_error(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "bad.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="root"):
        build_mapping_dataset(root, tmp_path / "output", strict=True)
