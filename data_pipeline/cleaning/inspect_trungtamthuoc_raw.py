"""Inspect Trung Tâm Thuốc raw JSONL without modifying source data."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import statistics
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data/raw/trungtamthuoc/ingredients_raw.jsonl")
DEFAULT_OUTPUT = Path("data/cleaned/trungtamthuoc/inspection_report.json")
IMPORTANT_SECTION_KEYS = (
    "chi_dinh",
    "chong_chi_dinh",
    "than_trong",
    "tuong_tac_thuoc",
    "thai_ky_cho_con_bu",
    "tac_dung_khong_mong_muon",
    "lieu_luong_va_cach_dung",
    "duoc_luc_hoc",
    "duoc_dong_hoc",
)


def normalize_preview(text: str, limit: int = 300) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def calculate_length_stats(lengths: Iterable[int]) -> dict[str, int | float | None]:
    values = list(lengths)
    if not values:
        return {"count": 0, "min": None, "max": None, "median": None, "average": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "median": round(statistics.median(values), 2),
        "average": round(statistics.fmean(values), 2),
    }


def detect_repeated_text(text: str) -> dict[str, Any]:
    """Return repeat signals using duplicate blocks and compression ratio."""
    normalized = re.sub(r"\s+", " ", text).strip()
    block_counts: Counter[str] = Counter()
    for block in re.split(r"(?:\r?\n)+", text):
        clean_block = re.sub(r"\s+", " ", block).strip()
        if len(clean_block) >= 100:
            block_counts[clean_block] += 1

    repeated_blocks = {
        block: count for block, count in block_counts.items() if count >= 2
    }
    repeated_chars = sum(
        len(block) * (count - 1) for block, count in repeated_blocks.items()
    )
    compression_ratio: float | None = None
    if len(normalized) >= 10_000:
        encoded = normalized.encode("utf-8")
        compression_ratio = len(zlib.compress(encoded)) / len(encoded)

    suspected = bool(repeated_blocks) or (
        compression_ratio is not None and compression_ratio <= 0.15
    )
    return {
        "suspected": suspected,
        "compression_ratio": (
            round(compression_ratio, 6) if compression_ratio is not None else None
        ),
        "repeated_block_count": len(repeated_blocks),
        "repeated_character_ratio": (
            round(repeated_chars / len(normalized), 6) if normalized else 0.0
        ),
    }


def _duplicate_report(counter: Counter[str]) -> dict[str, Any]:
    duplicates = [
        {"value": value, "count": count}
        for value, count in sorted(counter.items())
        if value and count > 1
    ]
    return {
        "group_count": len(duplicates),
        "extra_record_count": sum(item["count"] - 1 for item in duplicates),
        "values": duplicates,
    }


def _section_descriptor(
    record: dict[str, Any],
    section_key: str | None,
    text: str = "",
) -> dict[str, Any]:
    return {
        "name": record.get("name"),
        "slug": record.get("slug"),
        "url": record.get("url"),
        "section_key": section_key,
        "text_length": len(text),
        "preview": normalize_preview(text),
    }


def inspect_jsonl(input_path: Path) -> dict[str, Any]:
    valid_record_count = 0
    section_counts: list[int] = []
    all_section_lengths: list[int] = []
    section_frequency: Counter[str] = Counter()
    section_lengths_by_key: dict[str, list[int]] = defaultdict(list)
    coverage_counts: Counter[str] = Counter()
    slug_counts: Counter[str] = Counter()
    url_counts: Counter[str] = Counter()
    invalid_records: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    anomaly_counts: Counter[str] = Counter()
    longest_heap: list[tuple[int, int, dict[str, Any]]] = []
    sequence = 0

    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_records.append(
                    {
                        "line_number": line_number,
                        "reason": "invalid_json",
                        "detail": str(exc),
                    }
                )
                continue
            if not isinstance(record, dict) or not isinstance(
                record.get("sections"), dict
            ):
                invalid_records.append(
                    {
                        "line_number": line_number,
                        "reason": "invalid_schema",
                        "detail": "Record must be an object with a sections object",
                    }
                )
                continue

            valid_record_count += 1
            slug_counts[str(record.get("slug") or "")] += 1
            url_counts[str(record.get("url") or "")] += 1
            sections = record["sections"]
            section_count = len(sections)
            section_counts.append(section_count)

            if section_count > 50:
                anomaly = _section_descriptor(record, None)
                anomaly.update(
                    {
                        "section_count": section_count,
                        "reasons": ["record_over_50_sections"],
                    }
                )
                anomalies.append(anomaly)
                anomaly_counts["record_over_50_sections"] += 1

            present_important = set(sections) & set(IMPORTANT_SECTION_KEYS)
            coverage_counts.update(present_important)

            for section_key, text in sections.items():
                if not isinstance(section_key, str) or not isinstance(text, str):
                    invalid_records.append(
                        {
                            "line_number": line_number,
                            "reason": "invalid_section",
                            "detail": f"Section {section_key!r} must contain text",
                        }
                    )
                    continue

                text_length = len(text)
                all_section_lengths.append(text_length)
                section_counts_for_key = section_lengths_by_key[section_key]
                section_counts_for_key.append(text_length)
                section_frequency[section_key] += 1

                descriptor = _section_descriptor(record, section_key, text)
                sequence += 1
                heap_entry = (text_length, sequence, descriptor)
                if len(longest_heap) < 20:
                    heapq.heappush(longest_heap, heap_entry)
                elif text_length > longest_heap[0][0]:
                    heapq.heapreplace(longest_heap, heap_entry)

                reasons: list[str] = []
                if text_length > 50_000:
                    reasons.append("section_over_50000")
                if text_length > 200_000:
                    reasons.append("section_over_200000")

                repeat_signals = detect_repeated_text(text)
                if repeat_signals["suspected"]:
                    reasons.append("repeated_text_suspected")

                if reasons:
                    anomaly = dict(descriptor)
                    anomaly["reasons"] = reasons
                    anomaly["repeat_signals"] = repeat_signals
                    anomalies.append(anomaly)
                    anomaly_counts.update(reasons)

    important_coverage = {
        key: {
            "record_count": coverage_counts[key],
            "coverage_percent": (
                round(coverage_counts[key] * 100 / valid_record_count, 2)
                if valid_record_count
                else 0.0
            ),
        }
        for key in IMPORTANT_SECTION_KEYS
    }
    per_key_stats = {
        key: calculate_length_stats(lengths)
        for key, lengths in sorted(section_lengths_by_key.items())
    }
    longest_sections = [
        descriptor
        for _, _, descriptor in sorted(
            longest_heap, key=lambda item: item[0], reverse=True
        )
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "summary": {
            "valid_record_count": valid_record_count,
            "invalid_record_count": len(invalid_records),
            "duplicate_slug_record_count": _duplicate_report(slug_counts)[
                "extra_record_count"
            ],
            "duplicate_url_record_count": _duplicate_report(url_counts)[
                "extra_record_count"
            ],
            "average_sections_per_record": (
                round(statistics.fmean(section_counts), 2)
                if section_counts
                else 0.0
            ),
        },
        "duplicate_slugs": _duplicate_report(slug_counts),
        "duplicate_urls": _duplicate_report(url_counts),
        "section_length_stats": calculate_length_stats(all_section_lengths),
        "top_section_keys": [
            {"section_key": key, "count": count}
            for key, count in section_frequency.most_common(30)
        ],
        "important_section_coverage": important_coverage,
        "section_key_length_stats": per_key_stats,
        "longest_sections": longest_sections,
        "anomaly_counts_by_reason": dict(sorted(anomaly_counts.items())),
        "anomalies": anomalies,
        "invalid_records": invalid_records,
    }


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def print_report(report: dict[str, Any], output_path: Path) -> None:
    summary = report["summary"]
    lengths = report["section_length_stats"]
    print(f"Tổng record hợp lệ: {summary['valid_record_count']}")
    print(f"Record lỗi: {summary['invalid_record_count']}")
    print(f"Record trùng slug: {summary['duplicate_slug_record_count']}")
    print(f"Record trùng URL: {summary['duplicate_url_record_count']}")
    print(
        "Số section trung bình / record: "
        f"{summary['average_sections_per_record']}"
    )
    print(
        "Độ dài section min/max/median: "
        f"{lengths['min']}/{lengths['max']}/{lengths['median']}"
    )
    print("Top 30 section keys:")
    for item in report["top_section_keys"]:
        print(f"  {item['section_key']}: {item['count']}")
    print("Important section coverage:")
    for key, values in report["important_section_coverage"].items():
        print(
            f"  {key}: {values['record_count']} "
            f"({values['coverage_percent']}%)"
        )
    print("Anomaly counts:")
    for reason, count in report["anomaly_counts_by_reason"].items():
        print(f"  {reason}: {count}")
    for anomaly in report["anomalies"]:
        print("---")
        print(f"Name: {anomaly.get('name')}")
        print(f"Slug: {anomaly.get('slug')}")
        print(f"URL: {anomaly.get('url')}")
        print(f"Section: {anomaly.get('section_key')}")
        if "section_count" in anomaly:
            print(f"Section count: {anomaly['section_count']}")
        print(f"Text length: {anomaly.get('text_length')}")
        print(f"Reasons: {', '.join(anomaly['reasons'])}")
        print(f"Preview: {anomaly.get('preview')}")
    print(f"Report: {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Trung Tâm Thuốc raw ingredient JSONL"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    report = inspect_jsonl(args.input)
    save_report(args.output, report)
    print_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
