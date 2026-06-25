"""Inspect cleaned Long Châu drug products without modifying source data."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_ROOT = Path("data/cleaned/thuoc_long_chau_cleaned")
DEFAULT_OUTPUT_DIR = Path("data/processed/longchau_mapping")
NESTED_OBJECT_FIELDS = (
    "usage_details",
    "dosage_details",
    "careful_details",
    "_metadata",
)
SHORT_INGREDIENTS_MAX_LENGTH = 30
SCHEMA_WARNING_SAMPLE_LIMIT = 100
INVALID_SAMPLE_LIMIT = 100

STRENGTH_UNIT_RE = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|ml|iu|ui|%)(?!\w)",
    re.IGNORECASE,
)
STRENGTH_UNIT_CAPTURE_RE = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*(mg|mcg|µg|g|ml|iu|ui|%)(?!\w)",
    re.IGNORECASE,
)
PREFIX_RULES = (
    ("moi_vien_chua", re.compile(r"^\s*mỗi\s+viên\s+chứa\b", re.IGNORECASE)),
    ("moi_goi_chua", re.compile(r"^\s*mỗi\s+gói\s+chứa\b", re.IGNORECASE)),
    (
        "moi_ml_chua",
        re.compile(r"^\s*(?:mỗi\s+)?(?:\d+(?:[.,]\d+)?\s*)?ml\s+chứa\b", re.IGNORECASE),
    ),
    ("thanh_phan", re.compile(r"^\s*thành\s+phần\b", re.IGNORECASE)),
    ("hoat_chat", re.compile(r"^\s*hoạt\s+chất\b", re.IGNORECASE)),
    ("ta_duoc", re.compile(r"^\s*tá\s+dược\b", re.IGNORECASE)),
)


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def normalize_preview(value: Any, limit: int = 300) -> str:
    text = normalize_text(str(value))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def nearest_rank_percentile(values: Iterable[int], percentile: float) -> int | None:
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def classify_ingredients_prefix(text: str) -> str:
    normalized = normalize_text(text)
    for key, pattern in PREFIX_RULES:
        if pattern.search(normalized):
            return key
    return "other"


def analyze_ingredients(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    folded = normalized.casefold()
    strength_tokens = STRENGTH_UNIT_RE.findall(normalized)
    strength_units = []
    for unit in STRENGTH_UNIT_CAPTURE_RE.findall(normalized):
        normalized_unit = unit.casefold()
        if normalized_unit == "μg":
            normalized_unit = "µg"
        elif normalized_unit in {"iu", "ui"}:
            normalized_unit = normalized_unit.upper()
        strength_units.append(normalized_unit)
    contains_comma = "," in normalized
    contains_newline = "\n" in normalized or "\r" in normalized
    possible_multi = (
        "+" in normalized
        or ";" in normalized
        or contains_newline
        or len(strength_tokens) >= 2
        or (contains_comma and len(strength_tokens) >= 2)
    )
    return {
        "length": len(normalized),
        "multi_line": contains_newline,
        "contains_ta_duoc": "tá dược" in folded,
        "contains_thanh_phan": "thành phần" in folded,
        "contains_hoat_chat": "hoạt chất" in folded,
        "contains_moi_vien_chua": bool(
            re.search(r"mỗi\s+viên\s+chứa", folded)
        ),
        "contains_moi_goi_chua": bool(
            re.search(r"mỗi\s+gói\s+chứa", folded)
        ),
        "contains_equivalent": (
            "tương đương" in folded or "equivalent" in folded
        ),
        "contains_plus_separator": "+" in normalized,
        "contains_semicolon_separator": ";" in normalized,
        "contains_comma_separator": contains_comma,
        "contains_parentheses": "(" in normalized and ")" in normalized,
        "contains_strength_unit": bool(strength_tokens),
        "strength_token_count": len(strength_tokens),
        "strength_units": strength_units,
        "possible_multi_ingredient": possible_multi,
        "prefix_group": classify_ingredients_prefix(normalized),
    }


def _sample(
    record: dict[str, Any],
    path: Path,
    input_root: Path,
    ingredients: Any,
) -> dict[str, Any]:
    metadata = record.get("_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    category = metadata.get("category") or path.parent.name
    return {
        "name": record.get("name"),
        "category": category,
        "path": path.relative_to(input_root).as_posix(),
        "url": metadata.get("url"),
        "ingredients": ingredients,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            handle.write(content)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def save_json(path: Path, data: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    )


def _limited(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return items[:limit]


def _random_sample(
    items: list[dict[str, Any]], limit: int, rng: random.Random
) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return list(items)
    return rng.sample(items, limit)


def _ingredients_stats(
    analyses: list[dict[str, Any]],
    p95_length: int | None,
) -> dict[str, Any]:
    lengths = [item["length"] for item in analyses]
    strength_unit_counts: Counter[str] = Counter()
    for item in analyses:
        strength_unit_counts.update(item["strength_units"])
    return {
        "count": len(analyses),
        "empty_count": 0,
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "avg_length": (
            round(statistics.fmean(lengths), 2) if lengths else None
        ),
        "median_length": (
            round(statistics.median(lengths), 2) if lengths else None
        ),
        "p95_length": p95_length,
        "long_ingredients_count": (
            sum(length >= p95_length for length in lengths)
            if p95_length is not None
            else 0
        ),
        "multi_line_count": sum(item["multi_line"] for item in analyses),
        "contains_ta_duoc_count": sum(
            item["contains_ta_duoc"] for item in analyses
        ),
        "contains_thanh_phan_count": sum(
            item["contains_thanh_phan"] for item in analyses
        ),
        "contains_hoat_chat_count": sum(
            item["contains_hoat_chat"] for item in analyses
        ),
        "contains_moi_vien_chua_count": sum(
            item["contains_moi_vien_chua"] for item in analyses
        ),
        "contains_moi_goi_chua_count": sum(
            item["contains_moi_goi_chua"] for item in analyses
        ),
        "contains_equivalent_count": sum(
            item["contains_equivalent"] for item in analyses
        ),
        "contains_plus_separator_count": sum(
            item["contains_plus_separator"] for item in analyses
        ),
        "contains_semicolon_separator_count": sum(
            item["contains_semicolon_separator"] for item in analyses
        ),
        "contains_comma_separator_count": sum(
            item["contains_comma_separator"] for item in analyses
        ),
        "contains_parentheses_count": sum(
            item["contains_parentheses"] for item in analyses
        ),
        "contains_strength_unit_count": sum(
            item["contains_strength_unit"] for item in analyses
        ),
        "strength_unit_counts": dict(strength_unit_counts.most_common()),
        "possible_multi_ingredient_count": sum(
            item["possible_multi_ingredient"] for item in analyses
        ),
    }


def inspect_longchau(
    input_root: Path,
    *,
    random_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    files = sorted(input_root.rglob("*.json"))
    valid_records = 0
    invalid_json_count = 0
    non_object_count = 0
    nested_warning_count = 0
    non_string_ingredients_count = 0
    invalid_json_samples: list[dict[str, Any]] = []
    non_object_samples: list[dict[str, Any]] = []
    missing_name_samples: list[dict[str, Any]] = []
    non_string_ingredients_samples: list[dict[str, Any]] = []
    field_presence: Counter[str] = Counter()
    field_types: dict[str, Counter[str]] = defaultdict(Counter)
    nested_presence: dict[str, Counter[str]] = {
        key: Counter() for key in NESTED_OBJECT_FIELDS
    }
    nested_types: dict[str, dict[str, Counter[str]]] = {
        key: defaultdict(Counter) for key in NESTED_OBJECT_FIELDS
    }
    nested_container_types: dict[str, Counter[str]] = {
        key: Counter() for key in NESTED_OBJECT_FIELDS
    }
    nested_warnings: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    records_with_name = records_with_ingredients = 0
    records_with_describe = records_with_adverse_effect = records_with_prices = 0
    records_without_name = records_without_ingredients = 0
    ingredient_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing_ingredients: list[dict[str, Any]] = []

    for path in files:
        relative_path = path.relative_to(input_root).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            invalid_json_count += 1
            if len(invalid_json_samples) < INVALID_SAMPLE_LIMIT:
                invalid_json_samples.append(
                    {"path": relative_path, "detail": str(exc)}
                )
            continue
        if not isinstance(data, dict):
            non_object_count += 1
            if len(non_object_samples) < INVALID_SAMPLE_LIMIT:
                non_object_samples.append(
                    {
                        "path": relative_path,
                        "root_type": type(data).__name__,
                    }
                )
            continue

        valid_records += 1
        for key, value in data.items():
            field_presence[key] += 1
            field_types[key][type(value).__name__] += 1

        metadata = data.get("_metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        category = metadata_dict.get("category") or path.parent.name
        categories[str(category)] += 1

        if isinstance(data.get("name"), str) and data["name"].strip():
            records_with_name += 1
        else:
            records_without_name += 1
            if len(missing_name_samples) < INVALID_SAMPLE_LIMIT:
                missing_name_samples.append(
                    _sample(data, path, input_root, data.get("ingredients"))
                )
        records_with_describe += int(
            isinstance(data.get("describe"), str)
            and bool(data["describe"].strip())
        )
        records_with_adverse_effect += int(
            isinstance(data.get("adverse_effect"), str)
            and bool(data["adverse_effect"].strip())
        )
        records_with_prices += int(
            isinstance(data.get("prices"), str) and bool(data["prices"].strip())
        )

        for container_name in NESTED_OBJECT_FIELDS:
            container = data.get(container_name)
            nested_container_types[container_name][
                type(container).__name__
            ] += 1
            if container is None:
                continue
            if not isinstance(container, dict):
                nested_warning_count += 1
                if len(nested_warnings) < SCHEMA_WARNING_SAMPLE_LIMIT:
                    nested_warnings.append(
                        {
                            "path": relative_path,
                            "field": container_name,
                            "actual_type": type(container).__name__,
                        }
                    )
                continue
            for nested_key, nested_value in container.items():
                nested_presence[container_name][nested_key] += 1
                nested_types[container_name][nested_key][
                    type(nested_value).__name__
                ] += 1

        ingredients = data.get("ingredients")
        if isinstance(ingredients, str) and ingredients.strip():
            normalized = normalize_text(ingredients)
            records_with_ingredients += 1
            sample = _sample(data, path, input_root, normalized)
            ingredient_entries.append((sample, analyze_ingredients(normalized)))
        else:
            records_without_ingredients += 1
            sample = _sample(data, path, input_root, ingredients)
            missing_ingredients.append(sample)
            if ingredients is not None and not isinstance(ingredients, str):
                non_string_ingredients_count += 1
                non_string = dict(sample)
                non_string.update(
                    {
                        "ingredients": None,
                        "ingredients_type": type(ingredients).__name__,
                        "ingredients_repr": normalize_preview(
                            repr(ingredients), 300
                        ),
                    }
                )
                if len(non_string_ingredients_samples) < 30:
                    non_string_ingredients_samples.append(non_string)

    analyses = [analysis for _, analysis in ingredient_entries]
    p95_length = nearest_rank_percentile(
        (analysis["length"] for analysis in analyses), 0.95
    )
    prefix_counts = Counter(
        analysis["prefix_group"] for analysis in analyses
    )
    all_samples = [sample for sample, _ in ingredient_entries]
    rng = random.Random(random_seed)

    def selected(predicate: Any) -> list[dict[str, Any]]:
        return [
            sample
            for sample, analysis in ingredient_entries
            if predicate(analysis)
        ]

    long_samples = sorted(
        ingredient_entries,
        key=lambda item: item[1]["length"],
        reverse=True,
    )
    short_samples = sorted(
        (
            item
            for item in ingredient_entries
            if item[1]["length"] <= SHORT_INGREDIENTS_MAX_LENGTH
        ),
        key=lambda item: (item[1]["length"], item[0]["path"]),
    )
    possible_multi = selected(lambda item: item["possible_multi_ingredient"])
    ingredients_samples = {
        "first_50": _limited(all_samples, 50),
        "random_50": _random_sample(all_samples, 50, rng),
        "missing_ingredients_30": _limited(missing_ingredients, 30),
        "longest_30": [sample for sample, _ in long_samples[:30]],
        "multi_line_30": _limited(
            selected(lambda item: item["multi_line"]), 30
        ),
        "contains_ta_duoc_30": _limited(
            selected(lambda item: item["contains_ta_duoc"]), 30
        ),
        "possible_multi_ingredient_50": _limited(possible_multi, 50),
        "short_ingredients_30": [
            sample for sample, _ in short_samples[:30]
        ],
        "contains_strength_unit_50": _limited(
            selected(lambda item: item["contains_strength_unit"]), 50
        ),
        "contains_parentheses_50": _limited(
            selected(lambda item: item["contains_parentheses"]), 50
        ),
        "contains_equivalent_30": _limited(
            selected(lambda item: item["contains_equivalent"]), 30
        ),
        "non_string_ingredients_samples": non_string_ingredients_samples,
    }

    top_categories = [
        {"category": category, "count": count}
        for category, count in categories.most_common()
    ]
    inspection_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "total_json_files": len(files),
        "valid_json_objects": valid_records,
        "invalid_json_files": invalid_json_count,
        "non_object_json_files": non_object_count,
        "records_with_name": records_with_name,
        "records_without_name": records_without_name,
        "records_with_ingredients": records_with_ingredients,
        "records_without_ingredients": records_without_ingredients,
        "non_string_ingredients_count": non_string_ingredients_count,
        "records_with_describe": records_with_describe,
        "records_with_adverse_effect": records_with_adverse_effect,
        "records_with_prices": records_with_prices,
        "category_count": len(categories),
        "top_categories": top_categories,
        "field_presence_counts": dict(field_presence.most_common()),
        "nested_field_presence_counts": {
            key: dict(value.most_common())
            for key, value in nested_presence.items()
        },
        "ingredients_prefix_counts": {
            key: prefix_counts.get(key, 0)
            for key in (
                "thanh_phan",
                "hoat_chat",
                "moi_vien_chua",
                "moi_goi_chua",
                "moi_ml_chua",
                "ta_duoc",
                "other",
            )
        },
        "ingredients_stats": _ingredients_stats(analyses, p95_length),
        "thresholds": {
            "long_ingredients_min_length": p95_length,
            "short_ingredients_max_length": SHORT_INGREDIENTS_MAX_LENGTH,
            "p95_method": "nearest_rank",
        },
        "samples": {
            "missing_ingredients": _limited(missing_ingredients, 30),
            "long_ingredients": [
                sample
                for sample, analysis in long_samples
                if p95_length is not None and analysis["length"] >= p95_length
            ][:30],
            "multi_line_ingredients": ingredients_samples["multi_line_30"],
            "contains_ta_duoc": ingredients_samples["contains_ta_duoc_30"],
            "possible_multi_ingredient": ingredients_samples[
                "possible_multi_ingredient_50"
            ],
            "random_ingredients": ingredients_samples["random_50"],
            "non_string_ingredients_samples": non_string_ingredients_samples,
        },
        "invalid_json_samples": invalid_json_samples,
        "non_object_json_samples": non_object_samples,
        "missing_name_samples": missing_name_samples,
    }

    schema_report = {
        "generated_at": inspection_report["generated_at"],
        "input_root": str(input_root),
        "valid_json_objects": valid_records,
        "top_level_fields": {
            key: {
                "count": field_presence[key],
                "types": dict(field_types[key].most_common()),
            }
            for key in sorted(field_presence)
        },
        "nested_objects": {
            container: {
                "container_types": dict(
                    nested_container_types[container].most_common()
                ),
                "fields": {
                    nested_key: {
                        "count": nested_presence[container][nested_key],
                        "types": dict(
                            nested_types[container][nested_key].most_common()
                        ),
                    }
                    for nested_key in sorted(nested_presence[container])
                },
            }
            for container in NESTED_OBJECT_FIELDS
        },
        "warnings_count": nested_warning_count,
        "warnings": nested_warnings,
    }
    return inspection_report, ingredients_samples, schema_report


def render_samples_text(samples: dict[str, list[dict[str, Any]]]) -> str:
    parts: list[str] = []
    for group_name, items in samples.items():
        parts.append(f"\n{'#' * 80}\nGROUP: {group_name}\n{'#' * 80}\n")
        if not items:
            parts.append("(No samples)\n")
            continue
        for item in items:
            parts.extend(
                [
                    "=" * 80,
                    f"NAME: {item.get('name') or 'N/A'}",
                    f"CATEGORY: {item.get('category') or 'N/A'}",
                    f"URL: {item.get('url') or 'N/A'}",
                    f"PATH: {item.get('path') or 'N/A'}",
                ]
            )
            if "ingredients_type" in item:
                parts.append(
                    f"INGREDIENTS TYPE: {item['ingredients_type']}"
                )
                parts.append(
                    f"INGREDIENTS REPR: {item.get('ingredients_repr')}"
                )
            parts.append("INGREDIENTS:")
            value = item.get("ingredients")
            parts.append(str(value) if value is not None else "N/A")
            parts.append("")
    return "\n".join(parts).lstrip() + "\n"


def write_outputs(
    output_dir: Path,
    inspection_report: dict[str, Any],
    ingredients_samples: dict[str, Any],
    schema_report: dict[str, Any],
) -> list[Path]:
    paths = [
        output_dir / "inspection_report.json",
        output_dir / "ingredients_samples.json",
        output_dir / "ingredients_samples.txt",
        output_dir / "schema_field_report.json",
    ]
    save_json(paths[0], inspection_report)
    save_json(paths[1], ingredients_samples)
    _atomic_write_text(paths[2], render_samples_text(ingredients_samples))
    save_json(paths[3], schema_report)
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect cleaned Long Châu drug product JSON files"
    )
    parser.add_argument(
        "--input-root", type=Path, default=DEFAULT_INPUT_ROOT
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    if not args.input_root.is_dir():
        raise FileNotFoundError(
            f"Input root does not exist: {args.input_root}"
        )
    report, samples, schema_report = inspect_longchau(
        args.input_root,
        random_seed=args.random_seed,
    )
    write_outputs(args.output_dir, report, samples, schema_report)
    print(f"Total files: {report['total_json_files']}")
    print(f"Valid JSON objects: {report['valid_json_objects']}")
    print(f"With ingredients: {report['records_with_ingredients']}")
    print(f"Without ingredients: {report['records_without_ingredients']}")
    print(f"Output saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
