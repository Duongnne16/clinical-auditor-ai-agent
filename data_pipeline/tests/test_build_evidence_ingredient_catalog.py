import json
from pathlib import Path

import pytest

from data_pipeline.processing.build_evidence_ingredient_catalog import (
    build_catalog,
    extract_chunk_identity,
    normalize_text,
    parse_slug_from_chunk_id,
    slugify,
    strip_vietnamese_diacritics,
)


def _chunk(
    chunk_id: str,
    *,
    name: str | None = "Paracetamol",
    slug: str | None = "paracetamol",
    section: str = "chi_dinh",
    nested: bool = True,
    title: str | None = None,
) -> dict:
    values = {
        "entity_name": name,
        "slug": slug,
        "section": section,
        "url": f"https://example.test/{slug or 'missing'}",
        "title": title,
        "language": "vi",
    }
    values = {key: value for key, value in values.items() if value is not None}
    if nested:
        return {"chunk_id": chunk_id, "content": "text", "metadata": values}
    return {"chunk_id": chunk_id, **values}


def _write_jsonl(path: Path, rows: list[object | str]) -> None:
    lines = [
        row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_normalization_helpers() -> None:
    assert normalize_text("  Paracetamol™  ") == "paracetamol"
    assert strip_vietnamese_diacritics("Natri clorid") == "natri clorid"
    assert strip_vietnamese_diacritics("Đường") == "duong"
    assert slugify("Natri clorid") == "natri-clorid"
    assert (
        parse_slug_from_chunk_id(
            "trungtamthuoc:ingredient:metformin:chi_dinh:0001"
        )
        == "metformin"
    )
    assert (
        parse_slug_from_chunk_id("trungtamthuoc:amlodipine:chi_dinh:001")
        == "amlodipine"
    )


def test_build_single_catalog_record_merges_sections(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "output"
    _write_jsonl(
        input_path,
        [
            _chunk("c1", section="chi_dinh"),
            _chunk("c2", section="than_trong"),
        ],
    )

    report = build_catalog(input_path, output_dir)
    records = [
        json.loads(line)
        for line in (output_dir / "evidence_ingredient_catalog.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert report["unique_ingredients"] == 1
    assert report["valid_chunks"] == 2
    assert records[0]["chunk_count"] == 2
    assert records[0]["sections"] == ["chi_dinh", "than_trong"]
    assert records[0]["section_counts"] == {
        "chi_dinh": 1,
        "than_trong": 1,
    }


def test_flat_schema_multiple_ingredients_and_slugify(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "output"
    _write_jsonl(
        input_path,
        [
            _chunk("m1", slug="metformin", name="Metformin", nested=False),
            _chunk(
                "a1",
                slug=None,
                name="Amlodipine",
                nested=False,
            ),
        ],
    )

    report = build_catalog(input_path, output_dir)
    records = {
        item["slug"]: item
        for item in (
            json.loads(line)
            for line in (output_dir / "evidence_ingredient_catalog.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }

    assert set(records) == {"metformin", "amlodipine"}
    assert report["records_without_slug_but_slugified"] == 1
    assert records["amlodipine"]["catalog_id"].endswith(":amlodipine")


def test_title_fallback_and_long_title_uses_slug_name() -> None:
    short, warnings, _ = extract_chunk_identity(
        _chunk(
            "short",
            name=None,
            slug=None,
            title="Natri clorid",
        )
    )
    long, long_warnings, _ = extract_chunk_identity(
        _chunk(
            "trungtamthuoc:ingredient:povidone-iodine:chi_dinh:0001",
            name=None,
            slug="povidone-iodine",
            title=(
                "Povidone iodine có tác dụng gì và dùng điều trị "
                "những bệnh nhiễm khuẩn nào?"
            ),
        )
    )

    assert short is not None
    assert short["slug"] == "natri-clorid"
    assert short["entity_name"] == "Natri clorid"
    assert warnings == []
    assert long is not None
    assert long["entity_name"] == "povidone iodine"
    assert long_warnings == ["entity_name_derived_from_slug"]


def test_chunk_id_slug_fallback_and_multiple_names_warning(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "output"
    first = _chunk(
        "trungtamthuoc:ingredient:paracetamol:chi_dinh:0001",
        name="Paracetamol",
        slug=None,
    )
    second = _chunk(
        "trungtamthuoc:ingredient:paracetamol:than_trong:0001",
        name="Acetaminophen",
        slug=None,
        section="than_trong",
    )
    _write_jsonl(input_path, [first, second])

    report = build_catalog(input_path, output_dir)
    record = json.loads(
        (output_dir / "evidence_ingredient_catalog.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    assert report["records_with_slug_from_chunk_id"] == 2
    assert record["slug"] == "paracetamol"
    assert record["entity_name"] == "Paracetamol"
    assert "multiple_entity_names_for_slug" in record["warnings"]
    assert report["warning_counts"]["multiple_entity_names_for_slug"] == 1


def test_invalid_json_non_strict_and_strict(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    _write_jsonl(input_path, ["{bad json", _chunk("valid")])

    report = build_catalog(input_path, tmp_path / "non-strict")

    assert report["invalid_lines"] == 1
    assert report["unique_ingredients"] == 1
    with pytest.raises(ValueError, match="Invalid JSON"):
        build_catalog(input_path, tmp_path / "strict", strict=True)


def test_invalid_chunk_atomic_output_and_report_readback(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "output"
    _write_jsonl(
        input_path,
        [
            {"chunk_id": "missing"},
            _chunk("valid"),
        ],
    )
    before = input_path.read_bytes()

    report = build_catalog(input_path, output_dir)

    assert input_path.read_bytes() == before
    assert report["invalid_chunks"] == 1
    assert report["chunks_read"] == 2
    assert json.loads(
        (output_dir / "evidence_ingredient_catalog_report.json")
        .read_text(encoding="utf-8")
    )["unique_ingredients"] == 1
    assert not list(output_dir.glob("*.tmp"))

