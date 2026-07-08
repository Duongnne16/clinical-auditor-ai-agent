import json
import unicodedata
from pathlib import Path

import pytest

import data_pipeline.processing.normalize_longchau_ingredients as normalizer
from data_pipeline.processing.normalize_longchau_ingredients import (
    FIELD_TO_SECTION,
    create_normalized_chunk,
    normalize_directory,
)


def _item(
    *,
    name: str = "Omeprazole",
    slug: str = "omeprazole",
    field: str = "interaction",
    text: str = "Nội dung tương tác thuốc.",
    source: str = "longchau",
) -> dict:
    return {
        "text": text,
        "metadata": {
            "name": name,
            "id": slug,
            "url": f"https://nhathuoclongchau.com.vn/thanh-phan/{slug}",
            "field": field,
            "source": source,
        },
    }


def _write_array(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_field_mapping_matches_required_sections() -> None:
    assert FIELD_TO_SECTION["interaction"] == "tuong_tac_thuoc"
    assert FIELD_TO_SECTION["careful"] == "than_trong"
    assert FIELD_TO_SECTION["dosage"] == "lieu_luong_va_cach_dung"
    assert FIELD_TO_SECTION["describe"] == "mo_ta_chung"


def test_create_normalized_chunk_schema_and_unicode_nfc() -> None:
    item = _item(text=unicodedata.normalize("NFD", "Nội dung chuẩn."))

    chunk = create_normalized_chunk(
        item, section="tuong_tac_thuoc", chunk_index=1
    )

    assert chunk["chunk_id"] == (
        "longchau:ingredient:omeprazole:tuong_tac_thuoc:0001"
    )
    assert chunk["content"].startswith(
        "Hoạt chất: Omeprazole\nSlug: omeprazole\nMục: tuong_tac_thuoc"
    )
    assert unicodedata.is_normalized("NFC", chunk["content"])
    assert chunk["metadata"]["source"] == "longchau"
    assert chunk["metadata"]["source_name"] == "Dược chất Long Châu"
    assert chunk["metadata"]["source_type"] == "supplementary"
    assert chunk["metadata"]["entity_type"] == "ingredient"
    assert chunk["metadata"]["entity_name"] == "Omeprazole"
    assert chunk["metadata"]["slug"] == "omeprazole"
    assert chunk["metadata"]["section"] == "tuong_tac_thuoc"
    assert chunk["metadata"]["url"].endswith("/omeprazole")
    assert chunk["metadata"]["chunk_index"] == 1
    assert chunk["metadata"]["char_count"] == len(chunk["content"])


def test_normalize_directory_writes_stable_chunk_ids_and_report(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_array(
        input_dir / "omeprazole.json",
        [
            _item(field="interaction"),
            _item(field="interaction", text="Tương tác thứ hai."),
            _item(field="careful", text="Cần thận trọng."),
        ],
    )
    output = tmp_path / "out" / "ingredients_chunks.jsonl"
    report_path = tmp_path / "out" / "normalization_report.json"

    report = normalize_directory(input_dir, output, report_path)
    chunks = _read_jsonl(output)

    assert report["files_read"] == 1
    assert report["files_valid"] == 1
    assert report["chunks_read"] == 3
    assert report["chunks_written"] == 3
    assert report["duplicate_chunk_ids"] == []
    assert report["unknown_fields"] == []
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "longchau:ingredient:omeprazole:tuong_tac_thuoc:0001",
        "longchau:ingredient:omeprazole:tuong_tac_thuoc:0002",
        "longchau:ingredient:omeprazole:than_trong:0001",
    ]
    assert all(chunk["metadata"]["entity_name"] == "Omeprazole" for chunk in chunks)
    assert json.loads(report_path.read_text(encoding="utf-8"))[
        "chunks_written"
    ] == 3


def test_invalid_json_does_not_stop_non_strict_but_strict_fails(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "broken.json").write_text("{bad json", encoding="utf-8")
    _write_array(input_dir / "valid.json", [_item()])

    report = normalize_directory(
        input_dir,
        tmp_path / "out" / "chunks.jsonl",
        tmp_path / "out" / "report.json",
    )

    assert report["files_invalid"] == 1
    assert report["chunks_written"] == 1
    with pytest.raises(ValueError, match="Invalid JSON file"):
        normalize_directory(
            input_dir,
            tmp_path / "strict" / "chunks.jsonl",
            tmp_path / "strict" / "report.json",
            strict=True,
        )


def test_unknown_field_and_missing_required_are_reported(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    broken = _item(field="marketing")
    missing = _item()
    del missing["metadata"]["url"]
    _write_array(input_dir / "items.json", [broken, missing, _item()])

    report = normalize_directory(
        input_dir,
        tmp_path / "out" / "chunks.jsonl",
        tmp_path / "out" / "report.json",
    )

    assert report["chunks_read"] == 3
    assert report["chunks_written"] == 1
    assert report["invalid_chunks"] == 2
    assert report["unknown_fields"] == [{"field": "marketing", "count": 1}]
    assert report["missing_required_fields"]["metadata.url"] == 1


def test_strict_unknown_field_fails_without_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_array(input_dir / "items.json", [_item(field="marketing")])
    output = tmp_path / "out" / "chunks.jsonl"

    with pytest.raises(ValueError, match="Unknown Long Chau field"):
        normalize_directory(
            input_dir, output, tmp_path / "out" / "report.json", strict=True
        )
    assert not output.exists()


def test_duplicate_chunk_id_is_reported_and_strict_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_array(input_dir / "items.json", [_item(), _item(text="Hai.")])
    original = normalizer.create_normalized_chunk

    def duplicate_id(item: dict, *, section: str, chunk_index: int) -> dict:
        chunk = original(item, section=section, chunk_index=chunk_index)
        chunk["chunk_id"] = (
            "longchau:ingredient:omeprazole:tuong_tac_thuoc:0001"
        )
        return chunk

    monkeypatch.setattr(normalizer, "create_normalized_chunk", duplicate_id)
    report = normalize_directory(
        input_dir,
        tmp_path / "out" / "chunks.jsonl",
        tmp_path / "out" / "report.json",
    )

    assert report["chunks_written"] == 1
    assert report["duplicate_chunk_ids"] == [
        "longchau:ingredient:omeprazole:tuong_tac_thuoc:0001"
    ]
    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        normalize_directory(
            input_dir,
            tmp_path / "strict" / "chunks.jsonl",
            tmp_path / "strict" / "report.json",
            strict=True,
        )


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_array(input_dir / "items.json", [_item()])
    output = tmp_path / "out" / "chunks.jsonl"
    report = tmp_path / "out" / "report.json"
    output.parent.mkdir()
    output.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        normalize_directory(input_dir, output, report)

    result = normalize_directory(input_dir, output, report, overwrite=True)
    assert result["chunks_written"] == 1
