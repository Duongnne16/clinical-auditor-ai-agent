"""Safely clean Trung Tâm Thuốc raw JSONL without truncating content."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import sys
import tempfile
import unicodedata
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data/raw/trungtamthuoc/ingredients_raw.jsonl")
DEFAULT_OUTPUT = Path("data/cleaned/trungtamthuoc/ingredients_cleaned.jsonl")
DEFAULT_REPORT = Path("data/cleaned/trungtamthuoc/cleaning_report.json")
CHANGED_SECTIONS_LIMIT = 2_000
SHORT_SECTION_THRESHOLD = 10
IMPORTANT_SECTIONS = (
    "chi_dinh",
    "chong_chi_dinh",
    "than_trong",
    "tuong_tac_thuoc",
    "thai_ky_cho_con_bu",
    "thoi_ky_mang_thai",
    "tac_dung_khong_mong_muon",
    "lieu_luong_va_cach_dung",
    "duoc_luc_hoc",
    "duoc_dong_hoc",
    "qua_lieu_va_xu_tri",
    "trieu_chung",
    "xu_tri",
    "hap_thu",
    "phan_bo",
    "chuyen_hoa",
    "thai_tru",
    "thuong_gap",
    "it_gap",
    "hiem_gap",
)

_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")
_INLINE_WHITESPACE_RE = re.compile(r"[^\S\r\n]+")


def normalize_preview(text: str, limit: int = 300) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _normalize_unicode_text(
    text: str,
) -> tuple[str, set[str], dict[str, int]]:
    actions: set[str] = set()
    normalized = unicodedata.normalize("NFC", text)
    nfc_changed = normalized != text
    if nfc_changed:
        actions.add("unicode_nfc_normalized")

    soft_hyphen_count = normalized.count("\u00ad")
    if soft_hyphen_count:
        normalized = normalized.replace("\u00ad", "")
        actions.add("removed_soft_hyphen")

    zero_width_count = len(_ZERO_WIDTH_RE.findall(normalized))
    if zero_width_count:
        normalized = _ZERO_WIDTH_RE.sub("", normalized)
        actions.add("removed_zero_width_chars")

    return normalized, actions, {
        "unicode_nfc_normalized": int(nfc_changed),
        "soft_hyphen_removed": soft_hyphen_count,
        "zero_width_removed": zero_width_count,
    }


def _normalize_block(block: str) -> tuple[str, set[str]]:
    actions: set[str] = set()
    value = block
    if "\xa0" in value:
        value = value.replace("\xa0", " ")
        actions.add("replaced_nbsp")
    normalized = _INLINE_WHITESPACE_RE.sub(" ", value).strip()
    if normalized != value:
        actions.add("normalized_whitespace")
    return normalized, actions


def _clean_section_text_with_metrics(
    text: str,
) -> tuple[str, list[str], int, dict[str, int]]:
    """Normalize and exact-deduplicate blocks while preserving newlines."""
    prepared_text, actions, unicode_metrics = _normalize_unicode_text(text)
    cleaned_blocks: list[str] = []
    seen_blocks: set[str] = set()
    duplicate_count = 0

    for raw_block in prepared_text.splitlines():
        block, block_actions = _normalize_block(raw_block)
        actions.update(block_actions)
        if not block:
            if raw_block:
                actions.add("removed_empty_blocks")
            continue
        if block in seen_blocks:
            duplicate_count += 1
            actions.add("deduplicated_blocks")
            continue
        seen_blocks.add(block)
        cleaned_blocks.append(block)

    cleaned = "\n".join(cleaned_blocks)
    if cleaned != text and not actions:
        actions.add("normalized_whitespace")
    return cleaned, sorted(actions), duplicate_count, unicode_metrics


def clean_section_text(text: str) -> tuple[str, list[str], int]:
    cleaned, actions, duplicate_count, _ = _clean_section_text_with_metrics(
        text
    )
    return cleaned, actions, duplicate_count


def _table_repair_metrics(text: str) -> dict[str, int | float]:
    pipe_count = text.count("|")
    encoded = text.encode("utf-8")
    compression_ratio = (
        len(zlib.compress(encoded)) / len(encoded) if encoded else 1.0
    )
    segments = [
        re.sub(r"\s+", " ", segment).strip()
        for segment in text.split("|")
    ]
    segments = [segment for segment in segments if segment]
    unique_segment_ratio = (
        len(set(segments)) / len(segments) if segments else 1.0
    )
    return {
        "pipe_count": pipe_count,
        "compression_ratio": round(compression_ratio, 6),
        "unique_segment_ratio": round(unique_segment_ratio, 6),
    }


def _is_exact_suffix_segment(
    segment: str,
    first_segment: str,
    previous_segment: str,
) -> bool:
    if not segment or segment == first_segment:
        return False
    return (
        first_segment.endswith(segment)
        or segment in first_segment
        or previous_segment.endswith(segment)
    )


def repair_repeated_table_suffix_text(
    text: str,
) -> tuple[str, dict[str, Any]]:
    """Repair exact suffix chains in heavily repeated flattened tables."""
    metrics = _table_repair_metrics(text)
    eligible = (
        len(text) > 50_000
        and metrics["pipe_count"] > 100
        and (
            metrics["compression_ratio"] < 0.05
            or metrics["unique_segment_ratio"] < 0.25
        )
    )
    result: dict[str, Any] = {
        **metrics,
        "eligible": eligible,
        "removed_segments_count": 0,
        "reduction_ratio": 1.0,
        "high_reduction": False,
        "repaired": False,
        "rolled_back": False,
        "rollback_reason": None,
    }
    if not eligible:
        return text, result

    repaired_lines: list[str] = []
    removed_segments = 0
    for line in text.splitlines():
        segments = [
            re.sub(r"\s+", " ", segment).strip()
            for segment in line.split("|")
        ]
        segments = [segment for segment in segments if segment]
        if len(segments) < 4:
            repaired_lines.append(line)
            continue

        first_segment = segments[0]
        suffix_flags = [
            _is_exact_suffix_segment(
                segment,
                first_segment,
                segments[index - 1],
            )
            for index, segment in enumerate(segments[1:], 1)
        ]
        suffix_ratio = sum(suffix_flags) / len(suffix_flags)
        if suffix_ratio < 0.60:
            repaired_lines.append(line)
            continue

        kept = [first_segment]
        for index, segment in enumerate(segments[1:], 1):
            if _is_exact_suffix_segment(
                segment,
                first_segment,
                segments[index - 1],
            ):
                removed_segments += 1
            else:
                kept.append(segment)
        repaired_lines.append(" | ".join(kept))

    if not removed_segments:
        return text, result

    repaired_text = "\n".join(repaired_lines)
    reduction_ratio = len(repaired_text) / len(text) if text else 1.0
    result.update(
        {
            "removed_segments_count": removed_segments,
            "reduction_ratio": round(reduction_ratio, 6),
            "high_reduction": reduction_ratio < 0.05,
        }
    )
    if not repaired_text.strip():
        result.update(
            {"rolled_back": True, "rollback_reason": "empty_after_repair"}
        )
        return text, result
    if len(repaired_text) < 100:
        result.update(
            {"rolled_back": True, "rollback_reason": "too_short_after_repair"}
        )
        return text, result
    if reduction_ratio < 0.01:
        result.update(
            {"rolled_back": True, "rollback_reason": "reduction_below_one_percent"}
        )
        return text, result

    result["repaired"] = True
    return repaired_text, result


def _section_item(
    record: dict[str, Any],
    section_key: str,
    text: str,
    *,
    needs_special_chunking: bool | None = None,
) -> dict[str, Any]:
    item = {
        "name": record.get("name"),
        "slug": record.get("slug"),
        "url": record.get("url"),
        "section_key": section_key,
        "text_length": len(text),
        "preview": normalize_preview(text),
    }
    if needs_special_chunking is not None:
        item["needs_special_chunking"] = needs_special_chunking
    return item


def _validate_output_state(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output or temp file exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def _save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
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


def clean_jsonl(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    overwrite: bool,
    slug_filter: str | None = None,
) -> dict[str, Any]:
    temp_output = output_path.with_name("ingredients_cleaned.tmp.jsonl")
    _validate_output_state(
        (output_path, report_path, temp_output),
        overwrite,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_output.exists():
        temp_output.unlink()

    total_records = 0
    written_records = 0
    total_sections_before = 0
    total_sections_after = 0
    removed_short_sections_count = 0
    deduplicated_sections_count = 0
    normalized_sections_count = 0
    unicode_nfc_normalized_sections_count = 0
    soft_hyphen_removed_count = 0
    zero_width_removed_count = 0
    repaired_repeated_table_suffix_sections_count = 0
    repair_repeated_table_suffix_rollback_count = 0
    removed_table_suffix_segments_count = 0
    changed_sections_total_count = 0
    changed_sections: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    repaired_table_suffix_sections: list[dict[str, Any]] = []
    long_over_50k: list[dict[str, Any]] = []
    long_over_200k: list[dict[str, Any]] = []
    special_chunking: list[dict[str, Any]] = []
    longest_heap: list[tuple[int, int, dict[str, Any]]] = []
    sequence = 0

    try:
        with input_path.open(encoding="utf-8") as source, temp_output.open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
            for line_number, line in enumerate(source, 1):
                total_records += 1
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
                if slug_filter and record.get("slug") != slug_filter:
                    continue

                cleaned_sections: dict[str, str] = {}
                sections = record["sections"]
                total_sections_before += len(sections)

                for section_key, raw_text in sections.items():
                    if not isinstance(section_key, str) or not isinstance(
                        raw_text, str
                    ):
                        invalid_records.append(
                            {
                                "line_number": line_number,
                                "reason": "invalid_section",
                                "detail": f"Section {section_key!r} must contain text",
                            }
                        )
                        continue

                    (
                        cleaned_text,
                        actions,
                        duplicate_count,
                        unicode_metrics,
                    ) = _clean_section_text_with_metrics(raw_text)
                    unicode_nfc_normalized_sections_count += unicode_metrics[
                        "unicode_nfc_normalized"
                    ]
                    soft_hyphen_removed_count += unicode_metrics[
                        "soft_hyphen_removed"
                    ]
                    zero_width_removed_count += unicode_metrics[
                        "zero_width_removed"
                    ]
                    before_repair = cleaned_text
                    cleaned_text, repair = repair_repeated_table_suffix_text(
                        cleaned_text
                    )
                    if repair["repaired"]:
                        actions = sorted(
                            set(actions)
                            | {"repaired_repeated_table_suffix"}
                        )
                        repaired_repeated_table_suffix_sections_count += 1
                        removed_table_suffix_segments_count += int(
                            repair["removed_segments_count"]
                        )
                    elif repair["rolled_back"]:
                        actions = sorted(
                            set(actions)
                            | {"repair_repeated_table_suffix_rollback"}
                        )
                        repair_repeated_table_suffix_rollback_count += 1

                    if repair["repaired"] or repair["rolled_back"]:
                        repaired_table_suffix_sections.append(
                            {
                                "slug": record.get("slug"),
                                "name": record.get("name"),
                                "section_key": section_key,
                                "old_length": len(before_repair),
                                "new_length": len(cleaned_text),
                                "removed_segments_count": repair[
                                    "removed_segments_count"
                                ],
                                "pipe_count": repair["pipe_count"],
                                "compression_ratio": repair[
                                    "compression_ratio"
                                ],
                                "unique_segment_ratio": repair[
                                    "unique_segment_ratio"
                                ],
                                "reduction_ratio": repair[
                                    "reduction_ratio"
                                ],
                                "high_reduction": repair["high_reduction"],
                                "actions": [
                                    "repaired_repeated_table_suffix"
                                    if repair["repaired"]
                                    else "repair_repeated_table_suffix_rollback"
                                ],
                                "rollback_reason": repair["rollback_reason"],
                                "preview_before": normalize_preview(
                                    before_repair
                                ),
                                "preview_after": normalize_preview(
                                    cleaned_text
                                ),
                            }
                        )
                    removed = (
                        section_key not in IMPORTANT_SECTIONS
                        and len(cleaned_text) < SHORT_SECTION_THRESHOLD
                    )
                    if removed:
                        actions = sorted(set(actions) | {"removed_short_section"})
                        removed_short_sections_count += 1
                    else:
                        cleaned_sections[section_key] = cleaned_text
                        total_sections_after += 1

                        needs_special = len(cleaned_text) > 50_000
                        descriptor = _section_item(
                            record,
                            section_key,
                            cleaned_text,
                            needs_special_chunking=True if needs_special else None,
                        )
                        sequence += 1
                        heap_entry = (len(cleaned_text), sequence, descriptor)
                        if len(longest_heap) < 30:
                            heapq.heappush(longest_heap, heap_entry)
                        elif len(cleaned_text) > longest_heap[0][0]:
                            heapq.heapreplace(longest_heap, heap_entry)

                        if needs_special:
                            special_item = _section_item(
                                record,
                                section_key,
                                cleaned_text,
                                needs_special_chunking=True,
                            )
                            special_chunking.append(special_item)
                            long_over_50k.append(dict(special_item))
                        if len(cleaned_text) > 200_000:
                            long_over_200k.append(
                                _section_item(
                                    record,
                                    section_key,
                                    cleaned_text,
                                    needs_special_chunking=True,
                                )
                            )

                    if duplicate_count:
                        deduplicated_sections_count += 1
                    if actions and not (
                        len(actions) == 1 and actions[0] == "removed_short_section"
                    ):
                        normalized_sections_count += 1

                    if actions:
                        changed_sections_total_count += 1
                        if len(changed_sections) < CHANGED_SECTIONS_LIMIT:
                            changed_sections.append(
                                {
                                    "slug": record.get("slug"),
                                    "name": record.get("name"),
                                    "section_key": section_key,
                                    "old_length": len(raw_text),
                                    "new_length": 0 if removed else len(cleaned_text),
                                    "actions": actions,
                                    "duplicate_blocks_removed": duplicate_count,
                                    "removed_segments_count": repair[
                                        "removed_segments_count"
                                    ],
                                    "preview_before": normalize_preview(
                                        before_repair
                                    ),
                                    "preview": (
                                        ""
                                        if removed
                                        else normalize_preview(cleaned_text)
                                    ),
                                }
                            )

                output_record = dict(record)
                output_record["sections"] = cleaned_sections
                destination.write(
                    json.dumps(output_record, ensure_ascii=False) + "\n"
                )
                destination.flush()
                written_records += 1

        os.replace(temp_output, output_path)
    except Exception:
        # Keep the temp output for diagnosis; --overwrite can replace it later.
        raise

    longest_sections = [
        item
        for _, _, item in sorted(
            longest_heap, key=lambda entry: entry[0], reverse=True
        )
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "important_sections": list(IMPORTANT_SECTIONS),
        "total_records": total_records,
        "written_records": written_records,
        "invalid_records_count": len(invalid_records),
        "invalid_records": invalid_records,
        "total_sections_before": total_sections_before,
        "total_sections_after": total_sections_after,
        "removed_short_sections_count": removed_short_sections_count,
        "normalized_sections_count": normalized_sections_count,
        "unicode_nfc_normalized_sections_count": (
            unicode_nfc_normalized_sections_count
        ),
        "soft_hyphen_removed_count": soft_hyphen_removed_count,
        "zero_width_removed_count": zero_width_removed_count,
        "deduplicated_sections_count": deduplicated_sections_count,
        "repaired_repeated_table_suffix_sections_count": (
            repaired_repeated_table_suffix_sections_count
        ),
        "repair_repeated_table_suffix_rollback_count": (
            repair_repeated_table_suffix_rollback_count
        ),
        "removed_table_suffix_segments_count": (
            removed_table_suffix_segments_count
        ),
        "repaired_table_suffix_sections": repaired_table_suffix_sections,
        "long_sections_over_50000": {
            "count": len(long_over_50k),
            "sections": long_over_50k,
        },
        "long_sections_over_200000": {
            "count": len(long_over_200k),
            "sections": long_over_200k,
        },
        "needs_special_chunking_sections": {
            "count": len(special_chunking),
            "sections": special_chunking,
        },
        "top_30_longest_sections_after_clean": longest_sections,
        "changed_sections_total_count": changed_sections_total_count,
        "changed_sections_truncated": (
            changed_sections_total_count > CHANGED_SECTIONS_LIMIT
        ),
        "changed_sections": changed_sections,
    }
    _save_json_atomic(report_path, report)
    return report


def print_summary(report: dict[str, Any], report_path: Path) -> None:
    print(f"Total records: {report['total_records']}")
    print(f"Written records: {report['written_records']}")
    print(
        "Sections before/after: "
        f"{report['total_sections_before']}/{report['total_sections_after']}"
    )
    print(
        f"Removed short sections: {report['removed_short_sections_count']}"
    )
    print(
        f"Deduplicated sections: {report['deduplicated_sections_count']}"
    )
    print(
        "Unicode NFC normalized sections: "
        f"{report['unicode_nfc_normalized_sections_count']}"
    )
    print(
        f"Soft hyphens removed: {report['soft_hyphen_removed_count']}"
    )
    print(
        f"Zero-width characters removed: {report['zero_width_removed_count']}"
    )
    print(
        "Repaired repeated table suffix sections: "
        f"{report['repaired_repeated_table_suffix_sections_count']}"
    )
    print(
        "Table suffix repair rollbacks: "
        f"{report['repair_repeated_table_suffix_rollback_count']}"
    )
    print(
        "Long sections > 50,000: "
        f"{report['long_sections_over_50000']['count']}"
    )
    print(
        "Long sections > 200,000: "
        f"{report['long_sections_over_200000']['count']}"
    )
    print(
        "Needs special chunking: "
        f"{report['needs_special_chunking_sections']['count']}"
    )
    print(f"Changed sections: {report['changed_sections_total_count']}")
    print(f"Cleaned JSONL: {report['output_path']}")
    print(f"Report: {report_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely clean Trung Tâm Thuốc raw ingredient JSONL"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--slug")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    report = clean_jsonl(
        args.input,
        args.output,
        args.report,
        overwrite=args.overwrite,
        slug_filter=args.slug,
    )
    print_summary(report, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
