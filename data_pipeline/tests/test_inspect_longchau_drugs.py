import json
from pathlib import Path

from data_pipeline.inspection.inspect_longchau_drugs import (
    analyze_ingredients,
    classify_ingredients_prefix,
    inspect_longchau,
    nearest_rank_percentile,
    render_samples_text,
    write_outputs,
)


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def _record(
    name: str | None,
    ingredients: object,
    *,
    category: str | None = "Thuốc thử",
    nested_bad: bool = False,
) -> dict:
    metadata = {"url": "https://example.test/drug"}
    if category is not None:
        metadata["category"] = category
    return {
        "name": name,
        "describe": "Mô tả",
        "ingredients": ingredients,
        "usage_details": "bad" if nested_bad else {"chi_dinh": "Điều trị"},
        "dosage_details": {"lieu_dung": "Một viên"},
        "careful_details": {"than_trong": "Theo dõi"},
        "_metadata": metadata,
    }


def test_ingredients_helpers_cover_patterns_and_percentile() -> None:
    text = (
        "Mỗi viên chứa: Alpha (10mg) + Beta (5 mg); "
        "tương đương 2 µg equivalent"
    )
    analysis = analyze_ingredients(text)

    assert classify_ingredients_prefix(text) == "moi_vien_chua"
    assert classify_ingredients_prefix("Thành phần: A") == "thanh_phan"
    assert classify_ingredients_prefix("Tá dược vừa đủ") == "ta_duoc"
    assert classify_ingredients_prefix("Không có nhãn: A") == "other"
    assert analysis["contains_equivalent"] is True
    assert analysis["contains_parentheses"] is True
    assert analysis["contains_strength_unit"] is True
    assert analysis["strength_token_count"] == 3
    assert analysis["strength_units"] == ["mg", "mg", "µg"]
    assert analysis["possible_multi_ingredient"] is True
    assert nearest_rank_percentile([1, 2, 3, 4, 100], 0.95) == 100


def test_inspector_handles_invalid_schema_fallbacks_and_samples(
    tmp_path: Path,
) -> None:
    root = tmp_path / "input"
    category_dir = root / "Fallback Category"
    _write(
        category_dir / "a.json",
        _record(
            "Drug A",
            "Thành phần: Alpha (10mg), Beta (5mg)",
            category=None,
        ),
    )
    _write(
        category_dir / "b.json",
        _record(
            None,
            "Hoạt chất: Gamma tương đương 20 IU",
            nested_bad=True,
        ),
    )
    _write(category_dir / "c.json", _record("Drug C", ["Alpha", "Beta"]))
    _write(category_dir / "root-list.json", [{"name": "not-object"}])
    (category_dir / "broken.json").write_text("{broken", encoding="utf-8")
    before = {
        path: path.read_bytes() for path in root.rglob("*.json")
    }

    report, samples, schema = inspect_longchau(root, random_seed=42)

    assert {path: path.read_bytes() for path in root.rglob("*.json")} == before
    assert report["total_json_files"] == 5
    assert report["valid_json_objects"] == 3
    assert report["invalid_json_files"] == 1
    assert report["non_object_json_files"] == 1
    assert report["records_with_ingredients"] == 2
    assert report["records_without_ingredients"] == 1
    assert report["non_string_ingredients_count"] == 1
    assert report["records_without_name"] == 1
    assert {"category": "Fallback Category", "count": 1} in report[
        "top_categories"
    ]
    assert report["ingredients_prefix_counts"]["thanh_phan"] == 1
    assert report["ingredients_prefix_counts"]["hoat_chat"] == 1
    assert report["ingredients_stats"]["contains_equivalent_count"] == 1
    assert schema["warnings_count"] == 1
    assert schema["top_level_fields"]["ingredients"]["types"] == {
        "str": 2,
        "list": 1,
    }
    assert samples["contains_strength_unit_50"]
    assert samples["contains_parentheses_50"]
    assert samples["contains_equivalent_30"]
    non_string = samples["non_string_ingredients_samples"][0]
    assert non_string["ingredients_type"] == "list"
    assert non_string["ingredients"] is None
    assert "Alpha" in non_string["ingredients_repr"]


def test_random_samples_are_deterministic_and_outputs_are_utf8_atomic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "input"
    for index in range(60):
        _write(
            root / "Category" / f"{index}.json",
            _record(
                f"Thuốc {index}",
                f"Mỗi gói chứa: Hoạt chất {index} ({index + 1}mg)",
            ),
        )

    report_a, samples_a, schema_a = inspect_longchau(root, random_seed=7)
    report_b, samples_b, _ = inspect_longchau(root, random_seed=7)

    assert samples_a["random_50"] == samples_b["random_50"]
    assert len(samples_a["random_50"]) == 50
    assert len(samples_a["contains_strength_unit_50"]) == 50
    assert report_a["ingredients_stats"]["count"] == 60
    assert report_a["ingredients_prefix_counts"]["moi_goi_chua"] == 60
    output_dir = tmp_path / "output"
    paths = write_outputs(output_dir, report_a, samples_a, schema_a)

    assert len(paths) == 4
    assert all(path.exists() for path in paths)
    assert json.loads(paths[0].read_text(encoding="utf-8"))[
        "valid_json_objects"
    ] == 60
    assert json.loads(paths[1].read_text(encoding="utf-8"))["first_50"]
    assert json.loads(paths[3].read_text(encoding="utf-8"))[
        "top_level_fields"
    ]
    text = paths[2].read_text(encoding="utf-8")
    assert "GROUP: contains_strength_unit_50" in text
    assert "NAME: Thuốc" in text
    assert not list(output_dir.glob("*.tmp"))


def test_render_empty_sample_group() -> None:
    rendered = render_samples_text({"empty": []})
    assert "GROUP: empty" in rendered
    assert "(No samples)" in rendered
