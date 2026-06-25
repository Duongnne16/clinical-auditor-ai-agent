import json
import unicodedata
from pathlib import Path

import pytest

from data_pipeline.processing.chunk_trungtamthuoc_ingredients import (
    SOURCE_NAME,
    build_prefix,
    chunk_jsonl,
    chunk_section_content,
    create_chunk,
    is_important_section,
    split_oversized_block,
)


def _record(
    slug: str = "paracetamol",
    sections: dict[str, str] | None = None,
) -> dict:
    return {
        "source": "trungtamthuoc",
        "entity_type": "ingredient",
        "name": "Paracetamol",
        "slug": slug,
        "url": f"https://trungtamthuoc.com/hoat-chat/{slug}",
        "title": "Paracetamol",
        "updated_at": "01/01/2026",
        "crawled_at": "2026-01-01T00:00:00+00:00",
        "sections": sections or {"chi_dinh": "Nội dung chỉ định."},
    }


def _run_job(
    tmp_path: Path,
    records: list[dict],
    *,
    max_chars: int = 350,
    min_chars: int = 20,
    overlap_chars: int = 20,
) -> tuple[dict, list[dict], Path, Path, Path]:
    input_path = tmp_path / "cleaned.jsonl"
    output_path = tmp_path / "out" / "ingredients_chunks.jsonl"
    report_path = tmp_path / "out" / "chunking_report.json"
    input_path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False) for record in records
        ),
        encoding="utf-8",
    )
    report = chunk_jsonl(
        input_path,
        output_path,
        report_path,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_chars=overlap_chars,
        overwrite=True,
    )
    chunks = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    return report, chunks, input_path, output_path, report_path


def test_prefix_schema_and_extended_chunk_id() -> None:
    record = _record()
    prefix = build_prefix(
        record["name"], record["slug"], "chi_dinh", record["url"]
    )
    chunk = create_chunk(
        record, "chi_dinh", 10_000, "Nội dung", "section"
    )

    assert chunk["chunk_id"].endswith(":10000")
    assert chunk["content"].startswith(prefix)
    assert f"Nguồn: {SOURCE_NAME}" in chunk["content"]
    assert chunk["metadata"]["char_count"] == len(chunk["content"])
    assert chunk["metadata"]["chunk_strategy"] == "section"
    assert chunk["metadata"]["section_title"] == "chi_dinh"


def test_oversized_split_respects_budget() -> None:
    text = (
        "Câu đầu tiên khá dài để kiểm tra tách câu. "
        "Câu thứ hai cũng dài để kiểm tra. "
        + ("x" * 90)
    )
    pieces = split_oversized_block(text, 50)

    assert pieces
    assert all(0 < len(piece) <= 50 for piece in pieces)


def test_split_preserves_newlines_and_overlap() -> None:
    chunks = chunk_section_content(
        "Block one text.\nBlock two text.\nBlock three text.",
        budget=35,
        overlap_chars=16,
    )

    assert all(strategy == "split" for _, strategy in chunks)
    assert "\n" in chunks[0][0]
    assert "Block two text." in chunks[1][0]


def test_table_uses_line_group_and_repeats_header() -> None:
    text = (
        "Nhóm thuốc | Hoạt chất | Tác động | Khuyến cáo\n"
        "A | A1 | tăng | theo dõi\n"
        "B | B1 | giảm | tránh dùng\n"
        "C | C1 | tăng | giảm liều"
    )
    chunks = chunk_section_content(text, budget=110, overlap_chars=0)

    assert len(chunks) >= 2
    assert all(strategy == "table_line_group" for _, strategy in chunks)
    assert chunks[1][0].startswith("Nhóm thuốc | Hoạt chất")


def test_streaming_job_short_sections_unicode_schema_and_report(
    tmp_path: Path,
) -> None:
    record = _record(
        sections={
            "chi_dinh": "ngắn",
            "non_important": "bỏ",
            "duoc_luc_hoc": (
                unicodedata.normalize("NFD", "Tiếng Việt")
                + " µ → ∞ α β γ "
                + ("nội dung dài. " * 30)
            ),
            "tuong_tac_thuoc": (
                "Nhóm thuốc | Hoạt chất | Tác động | Khuyến cáo\n"
                + "\n".join(
                    f"Thuốc {index} | tác động | theo dõi"
                    for index in range(20)
                )
            ),
        }
    )
    report, chunks, input_path, _, report_path = _run_job(
        tmp_path, [record], max_chars=350
    )

    assert report["records_read"] == 1
    assert report["records_written"] == 1
    assert report["sections_skipped_short"] == 1
    assert report["sections_chunked"] == 3
    assert report["chunks_written"] == len(chunks)
    assert report["duplicate_chunk_ids"] == []
    assert report["strategy_counts"]["section"] >= 1
    assert report["strategy_counts"]["split"] >= 1
    assert report["strategy_counts"]["table_line_group"] >= 1
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(chunk["metadata"]["char_count"] <= 350 for chunk in chunks)
    assert all(
        chunk["content"].startswith("Hoạt chất: Paracetamol\nSlug: paracetamol")
        for chunk in chunks
    )
    combined = "\n".join(chunk["content"] for chunk in chunks)
    for symbol in ("µ", "→", "∞", "α", "β", "γ"):
        assert symbol in combined
    assert all(
        unicodedata.is_normalized("NFC", chunk["content"])
        for chunk in chunks
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))[
        "chunks_written"
    ] == len(chunks)
    assert input_path.exists()


def test_short_nonimportant_is_skipped_but_important_is_kept(
    tmp_path: Path,
) -> None:
    _, chunks, _, _, _ = _run_job(
        tmp_path,
        [_record(sections={"chi_dinh": "x", "other": "y"})],
        min_chars=80,
    )

    assert [chunk["metadata"]["section"] for chunk in chunks] == ["chi_dinh"]


def test_pattern_based_short_medical_sections_are_kept(
    tmp_path: Path,
) -> None:
    sections = {
        "rat_hiem_gap": "Hiếm.",
        "chua_xac_dinh_duoc_tan_suat": "Chưa rõ.",
        "dang_thuoc_va_ham_luong": "Viên 5 mg.",
        "marketing_note": "Bỏ.",
    }

    report, chunks, _, _, _ = _run_job(
        tmp_path,
        [_record(sections=sections)],
        min_chars=80,
    )

    kept = {chunk["metadata"]["section"] for chunk in chunks}
    assert kept == {
        "rat_hiem_gap",
        "chua_xac_dinh_duoc_tan_suat",
        "dang_thuoc_va_ham_luong",
    }
    assert report["kept_short_important_sections_count"] == 3
    assert {
        item["section"]
        for item in report["kept_short_important_sections_sample"]
    } == kept
    assert report["sections_skipped_short"] == 1


def test_is_important_section_uses_exact_keys_and_patterns() -> None:
    assert is_important_section("chi_dinh")
    assert is_important_section("nguoi_suy_gan")
    assert is_important_section("canh_bao_tuong_ky_thuoc")
    assert is_important_section("rat_thuong_gap")
    assert not is_important_section("marketing_note")


def test_invalid_record_is_reported_and_valid_record_continues(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "cleaned.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    report_path = tmp_path / "report.json"
    input_path.write_text(
        '{"broken": true}\n' + json.dumps(_record(), ensure_ascii=False),
        encoding="utf-8",
    )

    report = chunk_jsonl(
        input_path,
        output_path,
        report_path,
        max_chars=500,
        min_chars=0,
        overlap_chars=0,
        overwrite=True,
    )

    assert report["records_read"] == 2
    assert report["records_written"] == 1
    assert report["invalid_records_count"] == 1
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1


def test_duplicate_chunk_id_is_fatal_and_output_is_not_replaced(
    tmp_path: Path,
) -> None:
    duplicate = _record(sections={"chi_dinh": "Nội dung đủ dài."})
    input_path = tmp_path / "cleaned.jsonl"
    output_path = tmp_path / "ingredients_chunks.jsonl"
    report_path = tmp_path / "report.json"
    input_path.write_text(
        json.dumps(duplicate, ensure_ascii=False)
        + "\n"
        + json.dumps(duplicate, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        chunk_jsonl(
            input_path,
            output_path,
            report_path,
            max_chars=500,
            min_chars=0,
            overlap_chars=0,
            overwrite=True,
        )
    assert not output_path.exists()
    assert not (tmp_path / "ingredients_chunks.tmp.jsonl").exists()


def test_existing_output_or_temp_requires_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "cleaned.jsonl"
    output_path = tmp_path / "ingredients_chunks.jsonl"
    report_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(_record()), encoding="utf-8")
    temp_path = tmp_path / "ingredients_chunks.tmp.jsonl"
    temp_path.write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        chunk_jsonl(
            input_path,
            output_path,
            report_path,
            max_chars=500,
            min_chars=0,
            overlap_chars=0,
            overwrite=False,
        )

    chunk_jsonl(
        input_path,
        output_path,
        report_path,
        max_chars=500,
        min_chars=0,
        overlap_chars=0,
        overwrite=True,
    )
    assert output_path.exists()
    assert not temp_path.exists()


def test_missing_input_is_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input file"):
        chunk_jsonl(
            tmp_path / "missing.jsonl",
            tmp_path / "chunks.jsonl",
            tmp_path / "report.json",
            max_chars=500,
            min_chars=0,
            overlap_chars=0,
            overwrite=True,
        )
