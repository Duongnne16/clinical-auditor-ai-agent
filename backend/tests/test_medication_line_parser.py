import pytest

from backend.app.services.medication_line_parser import (
    MedicationLineParser,
)


@pytest.fixture
def parser() -> MedicationLineParser:
    return MedicationLineParser()


def test_parses_number_brand_strength_and_quantity(
    parser: MedicationLineParser,
) -> None:
    result = parser.parse_line(
        "1. Omeprazol (Kagascdine) 20mg x 56 Viên"
    )

    assert result["order_index"] == 1
    assert result["generic_text"] == "Omeprazol"
    assert result["brand_text"] == "Kagascdine"
    assert result["strength_text"] == "20mg"
    assert result["quantity"] == {"value": 56, "unit": "Viên"}
    assert result["ingredients"] == [
        {
            "name": "Omeprazol",
            "strength_raw": "20mg",
            "strength_value": 20.0,
            "strength_unit": "mg",
        }
    ]
    assert result["warnings"] == []


@pytest.mark.parametrize(
    ("prefix", "index"),
    [("2)", 2), ("3-", 3), ("4.", 4)],
)
def test_numbering_variants(
    parser: MedicationLineParser, prefix: str, index: int
) -> None:
    result = parser.parse_line(
        f"{prefix} Metformin (Panfor SR) 750mg x 10 Viên"
    )

    assert result["order_index"] == index


def test_split_combination_only_on_plus(
    parser: MedicationLineParser,
) -> None:
    result = parser.parse_line(
        "Calcium carbonate + Vitamin D (Bonclum) "
        "500mg + 250IU x 56 Viên"
    )

    assert [item["name"] for item in result["ingredients"]] == [
        "Calcium carbonate",
        "Vitamin D",
    ]
    assert [item["strength_unit"] for item in result["ingredients"]] == [
        "mg",
        "IU",
    ]


def test_does_not_split_slash_comma_or_vietnamese_and(
    parser: MedicationLineParser,
) -> None:
    result = parser.parse_line(
        "Vitamin A, D và E (Brand) 10mg x 1 Hộp"
    )

    assert len(result["ingredients"]) == 1
    assert result["ingredients"][0]["name"] == "Vitamin A, D và E"


def test_dosage_parenthesis_is_not_brand(
    parser: MedicationLineParser,
) -> None:
    result = parser.parse_line(
        "Tobramycin + dexamethason "
        "(0,3%+0,1%)/5ml x 1 Lọ"
    )

    assert result["brand_text"] is None
    assert result["strength_text"] == "(0,3%+0,1%)/5ml"


def test_complex_shared_denominator_strengths(
    parser: MedicationLineParser,
) -> None:
    result = parser.parse_line(
        "Betamethason + Acid salicylic (Asosalic) "
        "(0,5+30)mg/g x 30g"
    )

    assert [item["strength_raw"] for item in result["ingredients"]] == [
        "0,5mg/g",
        "30mg/g",
    ]
    assert [item["strength_value"] for item in result["ingredients"]] == [
        0.5,
        30.0,
    ]


def test_uncertain_strength_alignment_preserves_medication_text(
    parser: MedicationLineParser,
) -> None:
    result = parser.parse_line(
        "Ingredient A + Ingredient B (Brand) 500mg x 10 Viên"
    )

    assert result["strength_text"] == "500mg"
    assert all(
        item["strength_raw"] is None for item in result["ingredients"]
    )
    assert result["warnings"] == [
        "strength_ingredient_alignment_uncertain"
    ]


def test_empty_line_is_unparsed(parser: MedicationLineParser) -> None:
    result = parser.parse_line("")

    assert result["parse_status"] == "unparsed"
    assert "generic_text_not_found" in result["warnings"]


def test_parse_many_preserves_order(
    parser: MedicationLineParser,
) -> None:
    results = parser.parse_many(
        ["Paracetamol 500mg", "Metformin 750mg"]
    )

    assert [result["generic_text"] for result in results] == [
        "Paracetamol",
        "Metformin",
    ]


def test_non_string_raises(parser: MedicationLineParser) -> None:
    with pytest.raises(TypeError):
        parser.parse_line(123)  # type: ignore[arg-type]
