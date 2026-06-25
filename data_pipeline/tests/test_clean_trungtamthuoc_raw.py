import json
import unicodedata
from pathlib import Path

import pytest

from data_pipeline.cleaning.clean_trungtamthuoc_raw import (
    CHANGED_SECTIONS_LIMIT,
    clean_jsonl,
    clean_section_text,
    repair_repeated_table_suffix_text,
)


def _record(sections: dict[str, str], slug: str = "test") -> dict:
    return {
        "name": "Test ingredient",
        "slug": slug,
        "url": f"https://example.test/{slug}",
        "sections": sections,
    }


def test_clean_text_preserves_blocks_and_exact_deduplicates() -> None:
    text = (
        " First\xa0 block  \n"
        "Second\u200b block\n"
        " First  block \n"
        "Second block changed"
    )

    cleaned, actions, duplicate_count = clean_section_text(text)

    assert cleaned == "First block\nSecond block\nSecond block changed"
    assert duplicate_count == 1
    assert "replaced_nbsp" in actions
    assert "removed_zero_width_chars" in actions
    assert "deduplicated_blocks" in actions


def test_unicode_normalization_removes_only_noise_characters() -> None:
    decomposed = unicodedata.normalize("NFD", "Tiếng Việt")
    text = (
        f"{decomposed}\u00ad\n"
        "A\u200bB\u200cC\u200dD\ufeffE\n"
        "Ký hiệu hợp lệ: µ → ∞ α β γ"
    )

    cleaned, actions, duplicate_count = clean_section_text(text)

    assert cleaned == (
        "Tiếng Việt\n"
        "ABCDE\n"
        "Ký hiệu hợp lệ: µ → ∞ α β γ"
    )
    assert unicodedata.is_normalized("NFC", cleaned)
    assert duplicate_count == 0
    assert "unicode_nfc_normalized" in actions
    assert "removed_soft_hyphen" in actions
    assert "removed_zero_width_chars" in actions


def _suffix_chain_text(repeat_count: int = 700) -> str:
    first = (
        "Cơ quan Tần suất Rất phổ biến Thường gặp Không phổ biến "
        "Hiếm Không rõ Tác dụng phụ"
    )
    segments = [
        first,
        "Tần suất Rất phổ biến Thường gặp Không phổ biến Hiếm Không rõ Tác dụng phụ",
        "Rất phổ biến Thường gặp Không phổ biến Hiếm Không rõ Tác dụng phụ",
        "Thường gặp Không phổ biến Hiếm Không rõ Tác dụng phụ",
        "Không phổ biến Hiếm Không rõ Tác dụng phụ",
        "Hiếm Không rõ Tác dụng phụ",
        "Không rõ Tác dụng phụ",
    ]
    return "\n".join(
        " | ".join([f"Dòng {index} {segments[0]}", *segments[1:]])
        for index in range(repeat_count)
    )


def test_repair_repeated_table_suffix_and_metrics() -> None:
    text = _suffix_chain_text()
    repaired, metrics = repair_repeated_table_suffix_text(text)

    assert metrics["eligible"] is True
    assert metrics["repaired"] is True
    assert metrics["removed_segments_count"] > 100
    assert metrics["pipe_count"] > 100
    assert metrics["compression_ratio"] < 0.05
    assert metrics["unique_segment_ratio"] < 0.25
    assert metrics["reduction_ratio"] < 1
    assert repaired.count("|") == 0
    assert repaired.count("\n") == text.count("\n")


def test_valid_multicolumn_table_is_not_repaired() -> None:
    row = (
        "Cơ quan | Rất phổ biến | Thường gặp | Không phổ biến | "
        "Hiếm | Không rõ"
    )
    text = "\n".join(row for _ in range(1500))
    repaired, metrics = repair_repeated_table_suffix_text(text)

    assert repaired == text
    assert metrics["repaired"] is False
    assert metrics["removed_segments_count"] == 0


def test_repair_rolls_back_abnormally_small_result() -> None:
    text = _suffix_chain_text(130)
    # Make every first segment tiny while preserving eligibility and suffixes.
    text = text.replace(
        "Cơ quan Tần suất Rất phổ biến Thường gặp Không phổ biến Hiếm Không rõ Tác dụng phụ",
        "A B C D E F G",
    ).replace(
        "Tần suất Rất phổ biến Thường gặp Không phổ biến Hiếm Không rõ Tác dụng phụ",
        "B C D E F G",
    ).replace(
        "Rất phổ biến Thường gặp Không phổ biến Hiếm Không rõ Tác dụng phụ",
        "C D E F G",
    ).replace(
        "Thường gặp Không phổ biến Hiếm Không rõ Tác dụng phụ",
        "D E F G",
    ).replace(
        "Không phổ biến Hiếm Không rõ Tác dụng phụ",
        "E F G",
    ).replace("Hiếm Không rõ Tác dụng phụ", "F G").replace(
        "Không rõ Tác dụng phụ", "G"
    )
    # Extend only via repeated suffix columns so the repaired text is <1%.
    text = "\n".join(line + (" | G" * 1_000) for line in text.splitlines())
    repaired, metrics = repair_repeated_table_suffix_text(text)

    assert repaired == text
    assert metrics["rolled_back"] is True
    assert metrics["rollback_reason"] == "reduction_below_one_percent"


def test_clean_job_keeps_important_short_and_long_text(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "out" / "cleaned.jsonl"
    report_path = tmp_path / "out" / "report.json"
    long_text = "A" * 200_001
    records = [
        _record(
            {
                "chi_dinh": "short",
                "tiny_other": "tiny",
                "exact_ten": "0123456789",
                "tuong_tac_thuoc": long_text,
            }
        )
    ]
    input_path.write_text(
        "\n".join(json.dumps(item) for item in records),
        encoding="utf-8",
    )
    original = input_path.read_bytes()

    report = clean_jsonl(
        input_path,
        output_path,
        report_path,
        overwrite=True,
    )
    cleaned = json.loads(output_path.read_text(encoding="utf-8"))

    assert input_path.read_bytes() == original
    assert "chi_dinh" in cleaned["sections"]
    assert "tiny_other" not in cleaned["sections"]
    assert cleaned["sections"]["exact_ten"] == "0123456789"
    assert cleaned["sections"]["tuong_tac_thuoc"] == long_text
    assert report["removed_short_sections_count"] == 1
    assert report["long_sections_over_50000"]["count"] == 1
    assert report["long_sections_over_200000"]["count"] == 1
    special = report["needs_special_chunking_sections"]["sections"][0]
    assert set(
        (
            "name",
            "slug",
            "url",
            "section_key",
            "text_length",
            "preview",
            "needs_special_chunking",
        )
    ) <= set(special)
    assert special["needs_special_chunking"] is True
    assert report["important_sections"]


def test_clean_jsonl_reports_unicode_changes_and_stays_parseable(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "cleaned.jsonl"
    report_path = tmp_path / "report.json"
    decomposed = unicodedata.normalize("NFD", "Điều trị")
    input_path.write_text(
        json.dumps(
            _record(
                {
                    "chi_dinh": (
                        f"{decomposed}\u00ad A\u200bB\u200cC\u200dD\ufeffE "
                        "µ → ∞"
                    )
                }
            ),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = clean_jsonl(
        input_path,
        output_path,
        report_path,
        overwrite=True,
    )
    record = json.loads(output_path.read_text(encoding="utf-8"))

    assert record["sections"]["chi_dinh"] == "Điều trị ABCDE µ → ∞"
    assert report["unicode_nfc_normalized_sections_count"] == 1
    assert report["soft_hyphen_removed_count"] == 1
    assert report["zero_width_removed_count"] == 4
    actions = report["changed_sections"][0]["actions"]
    assert "unicode_nfc_normalized" in actions
    assert "removed_soft_hyphen" in actions
    assert "removed_zero_width_chars" in actions


def test_changed_section_details_are_capped(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "cleaned.jsonl"
    report_path = tmp_path / "report.json"
    sections = {
        f"key_{index}": "value  \nvalue"
        for index in range(CHANGED_SECTIONS_LIMIT + 5)
    }
    input_path.write_text(
        json.dumps(_record(sections)),
        encoding="utf-8",
    )

    report = clean_jsonl(
        input_path,
        output_path,
        report_path,
        overwrite=True,
    )

    assert report["changed_sections_total_count"] == CHANGED_SECTIONS_LIMIT + 5
    assert len(report["changed_sections"]) == CHANGED_SECTIONS_LIMIT
    assert report["changed_sections_truncated"] is True


def test_temp_or_output_requires_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "ingredients_cleaned.jsonl"
    report_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(_record({"chi_dinh": "valid"})), encoding="utf-8")
    temp_path = tmp_path / "ingredients_cleaned.tmp.jsonl"
    temp_path.write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        clean_jsonl(
            input_path,
            output_path,
            report_path,
            overwrite=False,
        )

    clean_jsonl(
        input_path,
        output_path,
        report_path,
        overwrite=True,
    )
    assert output_path.exists()
    assert not temp_path.exists()


def test_slug_filter_and_repair_report(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "natalizumab.jsonl"
    report_path = tmp_path / "report.json"
    records = [
        _record(
            {"tac_dung_khong_mong_muon": _suffix_chain_text()},
            slug="natalizumab",
        ),
        _record({"chi_dinh": "other record"}, slug="other"),
    ]
    input_path.write_text(
        "\n".join(json.dumps(item) for item in records),
        encoding="utf-8",
    )

    report = clean_jsonl(
        input_path,
        output_path,
        report_path,
        overwrite=True,
        slug_filter="natalizumab",
    )
    output_records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(output_records) == 1
    assert output_records[0]["slug"] == "natalizumab"
    assert report["written_records"] == 1
    assert report["repaired_repeated_table_suffix_sections_count"] == 1
    repaired_item = report["repaired_table_suffix_sections"][0]
    assert repaired_item["pipe_count"] > 100
    assert "preview_before" in repaired_item
    assert "preview_after" in repaired_item
