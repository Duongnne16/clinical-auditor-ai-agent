import json
from pathlib import Path

import pytest

from data_pipeline.processing.build_drug_mapping import (
    BRAND_REVIEW_REASON,
    DESCRIPTION_REVIEW_REASON,
    NO_INGREDIENTS_REVIEW_REASON,
    build_drug_products,
    build_product,
    extract_brand_name,
    parse_ingredients_text,
    repair_ingredients,
    validate_ingredients,
    write_drug_products,
)


@pytest.mark.parametrize(
    ("full_name", "expected"),
    [
        (
            "Thuốc Antacil 250mg Thai Nakorn Patana điều trị bệnh dạ dày",
            "Antacil",
        ),
        (
            "Hỗn dịch uống Apigel-Plus Apimed điều trị viêm dạ dày",
            "Apigel-Plus",
        ),
        ("Thuốc Trixlazi DaviPharm bổ sung canxi", "Trixlazi"),
        ("Thuốc\u00a0  Trixlazi   DaviPharm", "Trixlazi"),
    ],
)
def test_extract_brand_name(full_name: str, expected: str) -> None:
    brand_name, used_fallback = extract_brand_name(full_name)

    assert brand_name == expected
    assert used_fallback is False


def test_extract_brand_name_has_a_safe_fallback() -> None:
    brand_name, used_fallback = extract_brand_name("123mg sản phẩm khó tách")

    assert brand_name
    assert used_fallback is True


def test_generic_leading_word_uses_brand_fallback() -> None:
    brand_name, used_fallback = extract_brand_name(
        "Thuốc đặt âm đạo Canvey điều trị viêm nhiễm"
    )

    assert brand_name
    assert used_fallback is True


def test_parse_ingredients_text_uses_last_marker() -> None:
    text = (
        "Thuốc: Có nhắc Thành phần: không phải payload | "
        "Phần: Thành phần | Nội dung:\n"
        "Thành phần: A (10 mg), B complex (20mg), C dạng muối (5%, w/w)"
    )

    assert parse_ingredients_text(text) == [
        {"name": "A", "strength": "10 mg"},
        {"name": "B complex", "strength": "20mg"},
        {"name": "C dạng muối", "strength": "5%, w/w"},
    ]


def test_parse_ingredient_with_multiple_parenthetical_groups() -> None:
    assert parse_ingredients_text(
        "Thành phần: Vitamin B6 (Pyridoxine HCl) (25MG)"
    ) == [
        {
            "name": "Vitamin B6 (Pyridoxine HCl)",
            "strength": "25MG",
        }
    ]


def test_repair_split_alias_and_strength() -> None:
    ingredients = [
        {"name": "Vitamin B6", "strength": "Pyridoxine HCl"},
        {"name": "None", "strength": "25mg"},
    ]

    assert repair_ingredients(ingredients) == [
        {
            "name": "Vitamin B6 (Pyridoxine HCl)",
            "strength": "25mg",
        }
    ]


def test_repair_requires_adjacent_numeric_strength_with_unit() -> None:
    ingredients = [
        {"name": "Vitamin B6", "strength": "Pyridoxine HCl"},
        {"name": "Other", "strength": "10mg"},
        {"name": "None", "strength": "unknown"},
    ]

    assert repair_ingredients(ingredients) == ingredients


def test_validate_ingredients_detects_invalid_names_and_strengths() -> None:
    reasons = validate_ingredients(
        [
            {"name": "None", "strength": "25mg"},
            {"name": "", "strength": "10mg"},
            {"name": "Vitamin B6", "strength": "Pyridoxine HCl"},
            {"name": "***", "strength": "2mg"},
        ]
    )

    assert "Suspicious ingredient parsed: name=None" in reasons
    assert "Suspicious ingredient parsed: name=" in reasons
    assert "Suspicious strength without numeric value" in reasons
    assert "Suspicious ingredient name" in reasons


def test_validate_ingredients_only_flags_exact_duplicate_pair() -> None:
    reasons = validate_ingredients(
        [
            {"name": "Vitamin C", "strength": "100mg"},
            {"name": "vitamin c", "strength": "100MG"},
            {"name": "Vitamin C", "strength": "500mg"},
        ]
    )

    assert reasons == ["Duplicate ingredient parsed: name=vitamin c"]


def test_description_strength_none_is_not_suspicious() -> None:
    assert validate_ingredients(
        [{"name": "magnesi hydroxyd", "strength": None}]
    ) == []


def _chunks(
    *,
    name: str = "Thuốc Antacil 250mg Thai Nakorn điều trị dạ dày",
    ingredients_text: str | None = None,
    description: str = "Không có thông tin thành phần.",
) -> list[dict]:
    base_metadata = {
        "name": name,
        "id": "4712",
        "url": "https://example.test/antacil",
        "category": "Thuốc dạ dày",
        "type": "thuốc",
        "source": "Thuốc Long Châu",
    }
    chunks = [
        {
            "text": description,
            "metadata": {**base_metadata, "field": "describe", "chunk_index": 0},
        }
    ]
    if ingredients_text is not None:
        chunks.append(
            {
                "text": ingredients_text,
                "metadata": {
                    **base_metadata,
                    "field": "ingredients",
                    "chunk_index": 0,
                },
            }
        )
    return chunks


def test_build_product_with_complete_ingredients() -> None:
    product = build_product(
        _chunks(ingredients_text="Thành phần: A (10mg), B (2%)")
    )

    assert product["ingredient_source"] == "ingredients"
    assert product["needs_review"] is False
    assert product["review_reason"] is None


def test_suspicious_product_is_marked_for_review() -> None:
    product = build_product(
        _chunks(ingredients_text="Thành phần: Vitamin B6 (Pyridoxine HCl)")
    )

    assert product["ingredients"] == [
        {"name": "Vitamin B6", "strength": "Pyridoxine HCl"}
    ]
    assert product["needs_review"] is True
    assert (
        product["review_reason"]
        == "Suspicious strength without numeric value"
    )


def test_repaired_product_is_not_marked_partial() -> None:
    product = build_product(
        _chunks(
            ingredients_text=(
                "Thành phần: Vitamin B6 (Pyridoxine HCl), None (25mg)"
            )
        )
    )

    assert product["ingredients"] == [
        {
            "name": "Vitamin B6 (Pyridoxine HCl)",
            "strength": "25mg",
        }
    ]
    assert product["needs_review"] is False
    assert product["review_reason"] is None


def test_validation_reason_is_appended_to_existing_parse_reason() -> None:
    product = build_product(
        _chunks(
            ingredients_text=(
                "Thành phần: Vitamin B6 (Pyridoxine HCl), malformed ingredient"
            )
        )
    )

    assert product["needs_review"] is True
    assert product["review_reason"] == (
        "Ingredients parsed partially; "
        "Suspicious strength without numeric value"
    )


def test_build_product_falls_back_to_description() -> None:
    product = build_product(
        _chunks(
            description=(
                "Sản phẩm có thành phần chính là nhôm hydroxyd, "
                "magnesi hydroxyd và simethicon. Thuốc được sử dụng..."
            )
        )
    )

    assert product["ingredients"] == [
        {"name": "nhôm hydroxyd", "strength": None},
        {"name": "magnesi hydroxyd", "strength": None},
        {"name": "simethicon", "strength": None},
    ]
    assert product["ingredient_source"] == "describe"
    assert product["needs_review"] is True
    assert product["review_reason"] == DESCRIPTION_REVIEW_REASON


def test_build_product_marks_missing_ingredients() -> None:
    product = build_product(_chunks())

    assert product["ingredients"] == []
    assert product["ingredient_source"] is None
    assert product["needs_review"] is True
    assert product["review_reason"] == NO_INGREDIENTS_REVIEW_REASON


def test_ingredient_review_reason_has_priority_over_brand_fallback() -> None:
    product = build_product(
        _chunks(
            name="123mg sản phẩm khó tách",
            description="Có thành phần chính là hoạt chất A và hoạt chất B.",
        )
    )

    assert product["review_reason"] == DESCRIPTION_REVIEW_REASON
    assert product["review_reason"] != BRAND_REVIEW_REASON


def test_recursive_build_and_utf8_atomic_write(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    category_dir = input_dir / "Thuốc dạ dày"
    category_dir.mkdir(parents=True)
    source_path = category_dir / "antacil.json"
    source_path.write_text(
        json.dumps(
            _chunks(ingredients_text="Thành phần: Nhôm hydroxyd (250mg)"),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    products = build_drug_products(input_dir)
    output_path = tmp_path / "output" / "products.json"
    write_drug_products(products, output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(written) == 1
    assert written[0]["product_id"] == "4712"
    assert written[0]["ingredients"][0]["name"] == "Nhôm hydroxyd"


def test_invalid_root_reports_its_path(tmp_path: Path) -> None:
    source_path = tmp_path / "invalid.json"
    source_path.write_text('{"not": "a list"}', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid.json"):
        build_drug_products(tmp_path)
