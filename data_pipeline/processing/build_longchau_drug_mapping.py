"""Build Long Châu product-to-active-ingredient mappings.

Importing this module performs no filesystem or external-service operations.
"""

from __future__ import annotations

import argparse
import json
import os
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
LONG_INGREDIENTS_THRESHOLD = 165
SAMPLE_LIMIT = 20

INGREDIENT_PREFIX_RE = re.compile(r"^\s*thành\s+phần\s*:\s*", re.IGNORECASE)
FINAL_PAREN_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<strength>[^()]*)\)\s*$")
LEADING_STRENGTH_RE = re.compile(
    r"^\s*(?P<strength>\d+(?:[.,]\d+)?\s*"
    r"(?:mg|mcg|µg|g|ml|iu|ui|%)(?:\s*w/v)?)\s+"
    r"(?P<name>.+?)\s*$",
    re.IGNORECASE,
)
STRENGTH_RE = re.compile(
    r"^\s*(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>mg|mcg|µg|g|ml|iu|ui|%)?"
    r"(?P<suffix>.*)\s*$",
    re.IGNORECASE,
)
NUMBER_ONLY_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
DOSAGE_TOKEN_RE = re.compile(
    r"^\d+(?:[.,/]\d+)*(?:mg|mcg|µg|g|ml|iu|ui|%)?$",
    re.IGNORECASE,
)
NUMERIC_ID_RE = re.compile(r"^\d+$")
TRAILING_PACKAGE_RE = re.compile(r"\s*\([^()]*\)\s*$")
ACTION_PHRASE_RE = re.compile(
    r"\s+(?:điều\s+trị|hỗ\s+trợ|bổ\s+sung|phòng|dùng|giảm|cung\s+cấp)\b",
    re.IGNORECASE,
)
FORM_PREFIXES = (
    "dung dịch tiêm truyền tĩnh mạch",
    "dung dịch truyền tĩnh mạch",
    "dung dịch tiêm truyền",
    "dung dịch súc miệng",
    "dung dịch vệ sinh phụ nữ",
    "thuốc nhỏ mắt",
    "thuốc nhỏ tai",
    "thuốc mỡ",
    "thuốc xịt mũi",
    "thuốc xịt",
    "viên nang mềm",
    "viên nang",
    "viên nén",
    "viên sủi",
    "viên ngậm",
    "hỗn dịch uống",
    "nhũ dịch",
    "dịch truyền",
    "dung dịch",
    "miếng dán",
    "bình xịt",
    "thuốc",
    "gel",
    "kem",
    "siro",
    "bột",
    "cao",
    "dầu",
    "ống",
    "viên",
)
GENERIC_SINGLE_TOKEN_ALIASES = {
    "plus",
    "extra",
    "forte",
    "stella",
    "dhg",
    "opc",
    "pharm",
    "pharma",
    "tablet",
    "capsule",
    "thuoc",
    "thuốc",
    "vien",
    "viên",
    "gel",
    "kem",
    "siro",
    "bot",
    "bột",
    "cao",
    "dau",
    "dầu",
    "ong",
    "ống",
    "hop",
    "hộp",
    "xoa",
    "phong",
    "dịch",
    "dich",
    "hỗn",
    "hon",
    "nhũ",
    "nhu",
    "chai",
    "lọ",
    "lo",
    "gói",
    "goi",
}
INVALID_INGREDIENT_NAMES = {"", "none", "null", "nan"}
EXACT_EXCLUDED_INGREDIENTS = {
    "parafin",
    "paraffinum perliquidum",
    "mineral oil",
}
MEDIUM_WARNINGS = {
    "contains_equivalent_phrase",
    "ingredients_length_at_or_above_p95",
    "repaired_none_strength_pair",
    "excluded_none_item",
    "excluded_excipient_or_vehicle",
    "missing_strength_unit",
    "unparsed_strength",
    "partial_ingredient_parse",
}


def normalize_text(text: str) -> str:
    """Normalize text for matching while preserving Vietnamese diacritics."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("®", "").replace("™", "")
    value = re.sub(r"[\u2010-\u2015\u2212]", "-", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def strip_vietnamese_diacritics(text: str) -> str:
    """Remove Vietnamese combining marks and normalize đ to d."""
    value = unicodedata.normalize("NFD", normalize_text(text))
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Mn"
    )
    return value.replace("đ", "d")


def normalize_unit(unit: str | None) -> str | None:
    """Normalize supported strength units without guessing unknown units."""
    if not unit:
        return None
    normalized = unicodedata.normalize("NFC", unit).strip().lower()
    if normalized == "μg":
        normalized = "µg"
    return normalized if normalized in {
        "mg", "g", "mcg", "µg", "ml", "%", "iu", "ui"
    } else None


def split_outside_parentheses(text: str) -> list[str]:
    """Split commas/semicolons outside parentheses and preserve inner commas."""
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        elif depth == 0 and character in {",", ";", "\n"}:
            part = text[start:index].strip(" ,;\n")
            if part:
                parts.append(part)
            start = index + 1
    final = text[start:].strip(" ,;\n")
    if final:
        parts.append(final)
    return parts


def parse_strength(
    strength_raw: str | None,
) -> tuple[float | None, str | None, list[str]]:
    """Parse a numeric strength and normalized unit."""
    raw = str(strength_raw or "").strip()
    if not raw:
        return None, None, ["unparsed_strength"]
    match = STRENGTH_RE.match(raw)
    if not match:
        return None, None, ["unparsed_strength"]
    try:
        value = float(match.group("value").replace(",", "."))
    except ValueError:
        return None, None, ["unparsed_strength"]
    unit = normalize_unit(match.group("unit"))
    warnings: list[str] = []
    if unit is None:
        warnings.append("missing_strength_unit")
    return value, unit, warnings


def _is_invalid_name(name: str) -> bool:
    return normalize_text(name) in INVALID_INGREDIENT_NAMES


def _is_excluded_excipient_or_vehicle(name: str) -> bool:
    normalized = normalize_text(name)
    return (
        normalized.startswith("tá dược")
        or normalized in EXACT_EXCLUDED_INGREDIENTS
    )


def _parse_raw_item(item: str) -> dict[str, Any]:
    match = FINAL_PAREN_RE.match(item)
    if not match:
        return {
            "name": "",
            "strength_raw": "",
            "raw_text": item.strip(),
            "parsed": False,
        }
    return {
        "name": match.group("name").strip(" .:-"),
        "strength_raw": match.group("strength").strip(),
        "raw_text": item.strip(),
        "parsed": True,
    }


def _repair_none_pairs(
    parsed_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    repaired: list[dict[str, Any]] = []
    repair_count = 0
    index = 0
    while index < len(parsed_items):
        current = parsed_items[index]
        following = (
            parsed_items[index + 1]
            if index + 1 < len(parsed_items)
            else None
        )
        if (
            current["parsed"]
            and following is not None
            and following["parsed"]
            and not _is_invalid_name(current["name"])
            and _is_invalid_name(following["name"])
            and not any(character.isdigit() for character in current["strength_raw"])
            and any(character.isdigit() for character in following["strength_raw"])
        ):
            repaired.append(
                {
                    "name": (
                        f"{current['name']} ({current['strength_raw']})"
                        if current["strength_raw"]
                        else current["name"]
                    ),
                    "strength_raw": following["strength_raw"],
                    "raw_text": (
                        f"{current['raw_text']}, {following['raw_text']}"
                    ),
                    "repaired_none_pair": True,
                    "parsed": True,
                }
            )
            repair_count += 1
            index += 2
            continue
        repaired.append(current)
        index += 1
    return repaired, repair_count


def _ingredient_record(item: dict[str, Any]) -> dict[str, Any]:
    name = item["name"].strip()
    value, unit, warnings = parse_strength(item["strength_raw"])
    return {
        "name": name,
        "normalized_name": normalize_text(name),
        "normalized_name_no_diacritics": strip_vietnamese_diacritics(name),
        "strength_raw": item["strength_raw"],
        "strength_value": value,
        "strength_unit": unit,
        "raw_text": item["raw_text"],
        "ingredient_type": "active",
        "warnings": warnings,
    }


def parse_ingredients(
    ingredients_raw: str | None,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Parse Long Châu ingredient text and return ingredients/warnings/exclusions."""
    if not isinstance(ingredients_raw, str) or not ingredients_raw.strip():
        return [], ["missing_ingredients"], 0
    content = INGREDIENT_PREFIX_RE.sub("", ingredients_raw, count=1).strip()
    record_warnings: list[str] = []
    if "tương đương" in normalize_text(content) or "equivalent" in normalize_text(content):
        record_warnings.append("contains_equivalent_phrase")

    raw_items = split_outside_parentheses(content)
    parsed_items = [_parse_raw_item(item) for item in raw_items]
    parsed_items, repair_count = _repair_none_pairs(parsed_items)
    if repair_count:
        record_warnings.append("repaired_none_strength_pair")

    active_ingredients: list[dict[str, Any]] = []
    excluded_count = 0
    for parsed in parsed_items:
        if not parsed["parsed"]:
            leading = LEADING_STRENGTH_RE.match(parsed["raw_text"])
            if leading:
                parsed = {
                    "name": leading.group("name").strip(" .:-"),
                    "strength_raw": leading.group("strength").strip(),
                    "raw_text": parsed["raw_text"],
                    "parsed": True,
                }
            else:
                record_warnings.append("partial_ingredient_parse")
                continue
        name = parsed["name"]
        if (
            "tương đương" in normalize_text(parsed["strength_raw"])
            or "equivalent" in normalize_text(parsed["strength_raw"])
        ):
            leading = LEADING_STRENGTH_RE.match(name)
            if leading:
                parsed = dict(parsed)
                parsed["name"] = leading.group("name").strip(" .:-")
                parsed["strength_raw"] = leading.group("strength").strip()
                name = parsed["name"]
        if _is_invalid_name(name):
            record_warnings.append("excluded_none_item")
            excluded_count += 1
            continue
        if _is_excluded_excipient_or_vehicle(name):
            record_warnings.append("excluded_excipient_or_vehicle")
            excluded_count += 1
            continue
        ingredient = _ingredient_record(parsed)
        active_ingredients.append(ingredient)
        record_warnings.extend(ingredient["warnings"])

    return active_ingredients, _deduplicate(record_warnings), excluded_count


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _strip_form_prefix(name: str) -> str:
    candidate = name.strip()
    folded = normalize_text(candidate)
    for prefix in FORM_PREFIXES:
        normalized_prefix = normalize_text(prefix)
        if folded == normalized_prefix:
            return ""
        if folded.startswith(normalized_prefix + " "):
            return candidate[len(prefix):].strip(" :-")
    return candidate


def _valid_alias(alias: str) -> bool:
    normalized = normalize_text(alias)
    if not normalized or not any(character.isalpha() for character in normalized):
        return False
    tokens = normalized.split()
    if len(tokens) == 1:
        token = tokens[0]
        if (
            token in GENERIC_SINGLE_TOKEN_ALIASES
            or DOSAGE_TOKEN_RE.match(token)
            or len(token) < 3
        ):
            return False
    return True


def build_aliases(brand_name: str) -> list[str]:
    """Build prioritized normalized and no-diacritics aliases."""
    candidate = TRAILING_PACKAGE_RE.sub("", brand_name).strip()
    action_match = ACTION_PHRASE_RE.search(candidate)
    if action_match:
        candidate = candidate[: action_match.start()].strip(" ,.-")
    candidate = _strip_form_prefix(candidate)
    normalized = normalize_text(candidate)
    tokens = normalized.split()
    base_aliases: list[str] = []
    if normalized:
        base_aliases.append(normalized)
    for size in range(min(4, len(tokens)), 0, -1):
        base_aliases.append(" ".join(tokens[:size]))

    aliases: list[str] = []
    for alias in base_aliases:
        for variant in (alias, strip_vietnamese_diacritics(alias)):
            if _valid_alias(variant):
                aliases.append(variant)
    return _deduplicate(aliases)


def product_slug_from_path(path: Path, product_id: str) -> str:
    """Derive a stable slug, removing only a matching numeric ID suffix."""
    stem = path.stem
    if (
        product_id
        and NUMERIC_ID_RE.fullmatch(product_id)
        and stem.endswith(f"-{product_id}")
    ):
        stripped = stem[: -(len(product_id) + 1)]
        return stripped or stem
    return stem


def _choose_confidence(
    active_ingredients: list[dict[str, Any]],
    ingredients_raw: str | None,
    warnings: list[str],
) -> str:
    if not active_ingredients:
        return "low"
    if (
        len(ingredients_raw or "") >= LONG_INGREDIENTS_THRESHOLD
        or any(warning in MEDIUM_WARNINGS for warning in warnings)
    ):
        return "medium"
    return "high"


def _sample_mapping(mapping: dict[str, Any], path: str | None = None) -> dict[str, Any]:
    sample = {
        "mapping_id": mapping["mapping_id"],
        "brand_name": mapping["brand_name"],
        "category": mapping["category"],
        "url": mapping["url"],
        "ingredients_raw": mapping["ingredients_raw"],
        "active_ingredients": mapping["active_ingredients"],
        "confidence": mapping["confidence"],
        "warnings": mapping["warnings"],
    }
    if path is not None:
        sample["path"] = path
    return sample


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


def _write_json(path: Path, data: Any) -> None:
    _atomic_write_text(
        path, json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )


def build_mapping_dataset(
    input_root: Path,
    output_dir: Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Build and atomically write the complete Long Châu mapping dataset."""
    paths = sorted(input_root.rglob("*.json"))
    product_id_counts: Counter[str] = Counter()
    structural_errors: list[dict[str, Any]] = []
    loaded: list[tuple[Path, dict[str, Any]]] = []

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            error = {"path": str(path), "reason": "invalid_json", "detail": str(exc)}
            structural_errors.append(error)
            if strict:
                raise ValueError(f"Invalid JSON {path}: {exc}") from exc
            continue
        if not isinstance(payload, dict):
            error = {"path": str(path), "reason": "non_object_json"}
            structural_errors.append(error)
            if strict:
                raise ValueError(f"JSON root must be an object: {path}")
            continue
        metadata = payload.get("_metadata")
        if metadata is not None and not isinstance(metadata, dict):
            error = {"path": str(path), "reason": "invalid_metadata_schema"}
            structural_errors.append(error)
            if strict:
                raise ValueError(f"_metadata must be an object: {path}")
            metadata = {}
        metadata = metadata or {}
        product_id = str(metadata.get("id") or "")
        if product_id:
            product_id_counts[product_id] += 1
        loaded.append((path, payload))

    mappings: list[dict[str, Any]] = []
    mapping_ids: set[str] = set()
    duplicate_id_fallback_count = 0
    ingredient_display_counts: dict[str, Counter[str]] = defaultdict(Counter)
    warning_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    normalized_brand_counts: Counter[str] = Counter()
    failed_samples: list[dict[str, Any]] = []
    excluded_items_count = 0
    records_with_ingredients = 0

    for path, payload in loaded:
        metadata = payload.get("_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        product_id = str(metadata.get("id") or "")
        product_slug = product_slug_from_path(path, product_id)
        record_warnings: list[str] = []
        if product_id and product_id_counts[product_id] == 1:
            identifier = product_id
        else:
            identifier = product_slug
            if product_id and product_id_counts[product_id] > 1:
                record_warnings.append("duplicate_product_id_used_slug")
                duplicate_id_fallback_count += 1
        mapping_id = f"longchau:drug:{identifier}"
        if mapping_id in mapping_ids:
            error = {
                "path": path.relative_to(input_root).as_posix(),
                "reason": "duplicate_mapping_id",
                "mapping_id": mapping_id,
            }
            structural_errors.append(error)
            if strict:
                raise ValueError(f"Duplicate mapping_id: {mapping_id}")
            continue
        mapping_ids.add(mapping_id)

        ingredients_raw = payload.get("ingredients")
        if isinstance(ingredients_raw, str) and ingredients_raw.strip():
            records_with_ingredients += 1
        active_ingredients, parse_warnings, excluded_count = parse_ingredients(
            ingredients_raw if isinstance(ingredients_raw, str) else None
        )
        excluded_items_count += excluded_count
        record_warnings.extend(parse_warnings)
        if (
            isinstance(ingredients_raw, str)
            and len(ingredients_raw) >= LONG_INGREDIENTS_THRESHOLD
        ):
            record_warnings.append("ingredients_length_at_or_above_p95")
        record_warnings = _deduplicate(record_warnings)

        brand_name = str(payload.get("name") or "")
        category = metadata.get("category") or path.parent.name
        normalized_brand = normalize_text(brand_name)
        normalized_brand_counts[normalized_brand] += 1
        confidence = _choose_confidence(
            active_ingredients,
            ingredients_raw if isinstance(ingredients_raw, str) else None,
            record_warnings,
        )
        mapping = {
            "mapping_id": mapping_id,
            "source": "longchau",
            "source_type": "drug_product",
            "brand_name": brand_name,
            "normalized_brand_name": normalized_brand,
            "normalized_brand_name_no_diacritics": (
                strip_vietnamese_diacritics(brand_name)
            ),
            "brand_aliases": build_aliases(brand_name),
            "active_ingredients": active_ingredients,
            "ingredients_raw": ingredients_raw if isinstance(ingredients_raw, str) else None,
            "category": category,
            "product_id": product_id,
            "product_slug": product_slug,
            "url": metadata.get("url"),
            "confidence": confidence,
            "warnings": record_warnings,
            "notes": [],
        }
        mappings.append(mapping)
        confidence_counts[confidence] += 1
        category_counts[str(category)] += 1
        warning_counts.update(record_warnings)
        for ingredient in active_ingredients:
            ingredient_display_counts[ingredient["normalized_name"]][
                ingredient["name"]
            ] += 1
        if confidence == "low" or record_warnings or not active_ingredients:
            failed_samples.append(
                _sample_mapping(
                    mapping, path.relative_to(input_root).as_posix()
                )
            )

    mapping_output = output_dir / "drug_mapping.jsonl"
    report_output = output_dir / "drug_mapping_report.json"
    failed_output = output_dir / "failed_or_low_confidence_samples.json"
    _atomic_write_text(
        mapping_output,
        "".join(
            json.dumps(mapping, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for mapping in mappings
        ),
    )

    ingredient_totals = {
        normalized: sum(display_counts.values())
        for normalized, display_counts in ingredient_display_counts.items()
    }
    top_ingredient_names = []
    for normalized, count in sorted(
        ingredient_totals.items(), key=lambda item: (-item[1], item[0])
    )[:50]:
        display_counts = ingredient_display_counts[normalized]
        display_name = sorted(
            display_counts.items(), key=lambda item: (-item[1], item[0])
        )[0][0]
        top_ingredient_names.append(
            {
                "name": display_name,
                "normalized_name": normalized,
                "count": count,
            }
        )

    single_count = sum(
        len(mapping["active_ingredients"]) == 1 for mapping in mappings
    )
    multi_count = sum(
        len(mapping["active_ingredients"]) > 1 for mapping in mappings
    )
    records_with_active = single_count + multi_count
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_path": str(mapping_output),
        "total_files": len(paths),
        "valid_json_objects": len(loaded),
        "mappings_written": len(mappings),
        "records_with_ingredients": records_with_ingredients,
        "records_without_ingredients": len(loaded) - records_with_ingredients,
        "records_with_active_ingredients": records_with_active,
        "records_without_active_ingredients": len(mappings) - records_with_active,
        "confidence_counts": {
            key: confidence_counts.get(key, 0)
            for key in ("high", "medium", "low")
        },
        "single_ingredient_count": single_count,
        "multi_ingredient_count": multi_count,
        "total_active_ingredient_items": sum(
            len(mapping["active_ingredients"]) for mapping in mappings
        ),
        "excluded_items_count": excluded_items_count,
        "warning_counts": dict(warning_counts.most_common()),
        "duplicate_mapping_id_count": sum(
            1
            for error in structural_errors
            if error["reason"] == "duplicate_mapping_id"
        ),
        "duplicate_product_id_fallback_count": duplicate_id_fallback_count,
        "duplicate_normalized_brand_name_count": sum(
            count - 1
            for name, count in normalized_brand_counts.items()
            if name and count > 1
        ),
        "top_categories": [
            {"category": category, "count": count}
            for category, count in category_counts.most_common()
        ],
        "top_ingredient_names": top_ingredient_names,
        "sample_high_confidence": [
            _sample_mapping(mapping)
            for mapping in mappings
            if mapping["confidence"] == "high"
        ][:SAMPLE_LIMIT],
        "sample_medium_confidence": [
            _sample_mapping(mapping)
            for mapping in mappings
            if mapping["confidence"] == "medium"
        ][:SAMPLE_LIMIT],
        "sample_low_confidence": [
            _sample_mapping(mapping)
            for mapping in mappings
            if mapping["confidence"] == "low"
        ][:SAMPLE_LIMIT],
        "sample_multi_ingredient": [
            _sample_mapping(mapping)
            for mapping in mappings
            if len(mapping["active_ingredients"]) > 1
        ][:SAMPLE_LIMIT],
        "sample_warnings": [
            _sample_mapping(mapping)
            for mapping in mappings
            if mapping["warnings"]
        ][:SAMPLE_LIMIT],
        "structural_error_count": len(structural_errors),
        "structural_error_samples": structural_errors[:SAMPLE_LIMIT],
    }
    _write_json(report_output, report)
    _write_json(failed_output, failed_samples)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Long Châu drug product mappings"
    )
    parser.add_argument(
        "--input-root", type=Path, default=DEFAULT_INPUT_ROOT
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--strict", action="store_true")
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
    report = build_mapping_dataset(
        args.input_root, args.output_dir, strict=args.strict
    )
    print(f"Mappings written: {report['mappings_written']}")
    print(f"Confidence counts: {report['confidence_counts']}")
    print(
        "Records with active ingredients: "
        f"{report['records_with_active_ingredients']}"
    )
    print(
        "Records without active ingredients: "
        f"{report['records_without_active_ingredients']}"
    )
    print(f"Output: {report['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
