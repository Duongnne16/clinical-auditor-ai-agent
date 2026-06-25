import json
from pathlib import Path

from data_pipeline.cleaning.inspect_trungtamthuoc_raw import (
    calculate_length_stats,
    detect_repeated_text,
    inspect_jsonl,
    normalize_preview,
    save_report,
)


def _record(slug: str, sections: dict[str, str], url: str | None = None) -> dict:
    return {
        "name": f"Ingredient {slug}",
        "slug": slug,
        "url": url or f"https://example.test/{slug}",
        "sections": sections,
    }


def test_helpers() -> None:
    assert normalize_preview("a\n  b", 10) == "a b"
    assert calculate_length_stats([1, 2, 9]) == {
        "count": 3,
        "min": 1,
        "max": 9,
        "median": 2,
        "average": 4.0,
    }
    repeated = ("Repeated paragraph " * 10).strip()
    signals = detect_repeated_text(f"{repeated}\n{repeated}")
    assert signals["suspected"] is True
    assert signals["repeated_block_count"] == 1
    assert detect_repeated_text("Short unique text")["suspected"] is False


def test_inspection_report_statistics_and_anomalies(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    original_lines = [
        json.dumps(
            _record(
                "a",
                {
                    "chi_dinh": "A" * 10,
                    "duoc_luc_hoc": "B" * 50_001,
                },
                "https://example.test/shared",
            )
        ),
        json.dumps(
            _record(
                "a",
                {
                    "chi_dinh": "C" * 20,
                    "duoc_dong_hoc": "D" * 200_001,
                    **{f"key_{index}": "x" for index in range(49)},
                },
                "https://example.test/shared",
            )
        ),
        "{bad json",
        json.dumps({"slug": "bad-schema"}),
    ]
    input_path.write_text("\n".join(original_lines), encoding="utf-8")
    before = input_path.read_bytes()

    report = inspect_jsonl(input_path)

    assert input_path.read_bytes() == before
    assert report["summary"]["valid_record_count"] == 2
    assert report["summary"]["invalid_record_count"] == 2
    assert report["duplicate_slugs"]["extra_record_count"] == 1
    assert report["duplicate_urls"]["extra_record_count"] == 1
    assert report["important_section_coverage"]["chi_dinh"] == {
        "record_count": 2,
        "coverage_percent": 100.0,
    }
    assert report["section_key_length_stats"]["chi_dinh"] == {
        "count": 2,
        "min": 10,
        "max": 20,
        "median": 15.0,
        "average": 15.0,
    }
    assert report["longest_sections"][0]["text_length"] == 200_001
    assert report["anomaly_counts_by_reason"]["section_over_50000"] == 2
    assert report["anomaly_counts_by_reason"]["section_over_200000"] == 1
    record_anomaly = next(
        item
        for item in report["anomalies"]
        if "record_over_50_sections" in item["reasons"]
    )
    assert record_anomaly["section_count"] == 51
    assert len(report["longest_sections"]) <= 20


def test_report_write_is_utf8_and_atomic(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "report.json"
    report = {"message": "Dược lực học"}

    save_report(output_path, report)

    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert not list(output_path.parent.glob("*.tmp"))
