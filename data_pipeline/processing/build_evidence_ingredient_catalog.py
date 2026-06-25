"""Build a unique ingredient catalog from Trung Tâm Thuốc evidence chunks."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_PATH = Path(
    "data/processed/trungtamthuoc_v2/ingredients_chunks.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/processed/evidence_ingredients")
SOURCE_NAME = "Trung Tâm Thuốc / Dược thư Quốc gia Việt Nam"
TITLE_DIRECT_MAX_LENGTH = 120
SAMPLE_LIMIT = 20
CHUNK_ID_SAMPLE_LIMIT = 5
ARTICLE_TITLE_SIGNALS = (
    "?",
    " là thuốc gì",
    " có tác dụng gì",
    " điều trị ",
    " dược thư ",
    " dùng cho ",
    " bà bầu ",
)


def normalize_text(text: str) -> str:
    """Normalize text conservatively for catalog matching."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("®", "").replace("™", "")
    value = re.sub(r"[\u2010-\u2015\u2212]", "-", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def strip_vietnamese_diacritics(text: str) -> str:
    """Remove Vietnamese combining marks and normalize đ."""
    value = unicodedata.normalize("NFD", normalize_text(text))
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Mn"
    )
    return value.replace("đ", "d")


def slugify(text: str) -> str:
    """Create a lowercase ASCII-like slug from text."""
    value = strip_vietnamese_diacritics(text)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    value = chunk.get("metadata")
    return value if isinstance(value, dict) else {}


def _field(chunk: dict[str, Any], *names: str) -> Any:
    metadata = _metadata(chunk)
    for name in names:
        value = metadata.get(name)
        if value not in (None, ""):
            return value
        value = chunk.get(name)
        if value not in (None, ""):
            return value
    return None


def parse_slug_from_chunk_id(chunk_id: str | None) -> str | None:
    """Extract slug from current or legacy Trung Tâm Thuốc chunk IDs."""
    parts = str(chunk_id or "").split(":")
    if len(parts) >= 4 and parts[0] == "trungtamthuoc":
        if parts[1] == "ingredient":
            return parts[2] or None
        return parts[1] or None
    return None


def _title_is_direct_name(title: str) -> bool:
    normalized = normalize_text(title)
    return (
        bool(normalized)
        and len(title) <= TITLE_DIRECT_MAX_LENGTH
        and not any(signal in normalized for signal in ARTICLE_TITLE_SIGNALS)
    )


def _name_from_slug(slug: str) -> str:
    return normalize_text(slug.replace("-", " "))


def extract_chunk_identity(
    chunk: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str], dict[str, int]]:
    """Extract normalized catalog identity and extraction counters."""
    warnings: list[str] = []
    counters = {
        "records_with_slug": 0,
        "records_with_slug_from_chunk_id": 0,
        "records_without_slug_but_slugified": 0,
        "records_without_entity_name": 0,
    }

    explicit_slug = _field(chunk, "slug")
    slug_source = "explicit"
    slug = normalize_text(str(explicit_slug or "")).replace(" ", "-")
    if slug:
        counters["records_with_slug"] = 1
    else:
        slug = normalize_text(
            parse_slug_from_chunk_id(str(chunk.get("chunk_id") or "")) or ""
        ).replace(" ", "-")
        if slug:
            slug_source = "chunk_id"
            counters["records_with_slug"] = 1
            counters["records_with_slug_from_chunk_id"] = 1

    entity_name = _field(chunk, "entity_name")
    if entity_name in (None, ""):
        counters["records_without_entity_name"] = 1
        entity_name = _field(chunk, "name", "active_ingredient")

    title = _field(chunk, "title")
    if entity_name in (None, "") and title not in (None, ""):
        title_text = str(title).strip()
        if _title_is_direct_name(title_text):
            entity_name = title_text
        elif slug:
            entity_name = _name_from_slug(slug)
            warnings.append("entity_name_derived_from_slug")

    if entity_name in (None, "") and slug:
        entity_name = _name_from_slug(slug)
        warnings.append("entity_name_derived_from_slug")

    if not slug and entity_name not in (None, ""):
        slug = slugify(str(entity_name))
        if slug:
            slug_source = "entity_name"
            counters["records_without_slug_but_slugified"] = 1

    if not slug or entity_name in (None, ""):
        return None, warnings, counters

    identity = {
        "slug": slugify(slug) or slug,
        "slug_source": slug_source,
        "entity_name": str(entity_name).strip(),
        "section": str(_field(chunk, "section") or "").strip(),
        "url": _field(chunk, "url", "source_url"),
        "language": str(_field(chunk, "language") or "vi"),
        "chunk_id": str(chunk.get("chunk_id") or ""),
    }
    return identity, warnings, counters


def _aliases(entity_name: str, slug: str) -> list[str]:
    normalized_name = normalize_text(entity_name)
    no_diacritics = strip_vietnamese_diacritics(entity_name)
    slug_spaces = slug.replace("-", " ")
    return _deduplicate(
        (normalized_name, no_diacritics, slug_spaces, slug)
    )


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


def build_catalog(
    input_path: Path,
    output_dir: Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Build catalog JSONL and report from evidence chunks."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    chunks_read = valid_chunks = invalid_lines = invalid_chunks = 0
    records_with_slug = records_with_slug_from_chunk_id = 0
    records_without_slug_but_slugified = records_without_entity_name = 0
    invalid_samples: list[dict[str, Any]] = []
    section_counts_global: Counter[str] = Counter()
    groups: dict[str, dict[str, Any]] = {}

    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            chunks_read += 1
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_lines += 1
                if len(invalid_samples) < SAMPLE_LIMIT:
                    invalid_samples.append(
                        {
                            "line_number": line_number,
                            "reason": "invalid_json",
                            "detail": str(exc),
                        }
                    )
                if strict:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {exc}"
                    ) from exc
                continue
            if not isinstance(chunk, dict):
                invalid_chunks += 1
                if len(invalid_samples) < SAMPLE_LIMIT:
                    invalid_samples.append(
                        {
                            "line_number": line_number,
                            "reason": "non_object_chunk",
                        }
                    )
                if strict:
                    raise ValueError(
                        f"Chunk on line {line_number} must be an object"
                    )
                continue

            identity, warnings, counters = extract_chunk_identity(chunk)
            records_with_slug += counters["records_with_slug"]
            records_with_slug_from_chunk_id += counters[
                "records_with_slug_from_chunk_id"
            ]
            records_without_slug_but_slugified += counters[
                "records_without_slug_but_slugified"
            ]
            records_without_entity_name += counters[
                "records_without_entity_name"
            ]
            if identity is None:
                invalid_chunks += 1
                if len(invalid_samples) < SAMPLE_LIMIT:
                    invalid_samples.append(
                        {
                            "line_number": line_number,
                            "reason": "missing_catalog_identity",
                            "chunk_id": chunk.get("chunk_id"),
                        }
                    )
                if strict:
                    raise ValueError(
                        f"Cannot create catalog identity on line {line_number}"
                    )
                continue

            valid_chunks += 1
            slug = identity["slug"]
            section = identity["section"]
            if section:
                section_counts_global[section] += 1
            group = groups.setdefault(
                slug,
                {
                    "name_counts": Counter(),
                    "name_first_seen": {},
                    "url": None,
                    "language_counts": Counter(),
                    "section_counts": Counter(),
                    "chunk_count": 0,
                    "sample_chunk_ids": [],
                    "warnings": [],
                },
            )
            name = identity["entity_name"]
            group["name_counts"][name] += 1
            group["name_first_seen"].setdefault(name, valid_chunks)
            if group["url"] is None and identity["url"]:
                group["url"] = identity["url"]
            group["language_counts"][identity["language"]] += 1
            if section:
                group["section_counts"][section] += 1
            group["chunk_count"] += 1
            if (
                identity["chunk_id"]
                and len(group["sample_chunk_ids"]) < CHUNK_ID_SAMPLE_LIMIT
            ):
                group["sample_chunk_ids"].append(identity["chunk_id"])
            group["warnings"].extend(warnings)

    catalog: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    duplicate_slug_groups = 0
    for slug in sorted(groups):
        group = groups[slug]
        if group["chunk_count"] > 1:
            duplicate_slug_groups += 1
        name_counts: Counter[str] = group["name_counts"]
        entity_name = sorted(
            name_counts,
            key=lambda name: (
                -name_counts[name],
                group["name_first_seen"][name],
            ),
        )[0]
        warnings = _deduplicate(group["warnings"])
        if len(name_counts) > 1:
            warnings.append("multiple_entity_names_for_slug")
        warnings = _deduplicate(warnings)
        warning_counts.update(warnings)
        language_counts: Counter[str] = group["language_counts"]
        language = (
            sorted(
                language_counts,
                key=lambda value: (-language_counts[value], value),
            )[0]
            if language_counts
            else "vi"
        )
        section_counts = {
            key: group["section_counts"][key]
            for key in sorted(group["section_counts"])
        }
        catalog.append(
            {
                "catalog_id": f"trungtamthuoc:ingredient:{slug}",
                "source": "trungtamthuoc",
                "source_name": SOURCE_NAME,
                "entity_type": "ingredient",
                "entity_name": entity_name,
                "slug": slug,
                "normalized_name": normalize_text(entity_name),
                "normalized_name_no_diacritics": (
                    strip_vietnamese_diacritics(entity_name)
                ),
                "aliases": _aliases(entity_name, slug),
                "url": group["url"],
                "sections": list(section_counts),
                "section_counts": section_counts,
                "chunk_count": group["chunk_count"],
                "sample_chunk_ids": group["sample_chunk_ids"],
                "language": language,
                "warnings": warnings,
            }
        )

    output_path = output_dir / "evidence_ingredient_catalog.jsonl"
    report_path = output_dir / "evidence_ingredient_catalog_report.json"
    _atomic_write_text(
        output_path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for record in catalog
        ),
    )
    top_ingredients = [
        {
            "slug": record["slug"],
            "entity_name": record["entity_name"],
            "chunk_count": record["chunk_count"],
        }
        for record in sorted(
            catalog,
            key=lambda item: (-item["chunk_count"], item["slug"]),
        )[:50]
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "chunks_read": chunks_read,
        "valid_chunks": valid_chunks,
        "invalid_lines": invalid_lines,
        "invalid_chunks": invalid_chunks,
        "unique_ingredients": len(catalog),
        "records_with_slug": records_with_slug,
        "records_with_slug_from_chunk_id": records_with_slug_from_chunk_id,
        "records_without_slug_but_slugified": (
            records_without_slug_but_slugified
        ),
        "records_without_entity_name": records_without_entity_name,
        "duplicate_slug_groups": duplicate_slug_groups,
        "section_counts_global": dict(section_counts_global.most_common()),
        "top_sections": [
            {"section": section, "count": count}
            for section, count in section_counts_global.most_common(30)
        ],
        "top_ingredients_by_chunk_count": top_ingredients,
        "warning_counts": dict(warning_counts.most_common()),
        "invalid_samples": invalid_samples,
        "sample_catalog_records": catalog[:SAMPLE_LIMIT],
    }
    _write_json(report_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Trung Tâm Thuốc evidence ingredient catalog"
    )
    parser.add_argument(
        "--input-path", type=Path, default=DEFAULT_INPUT_PATH
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
    report = build_catalog(
        args.input_path, args.output_dir, strict=args.strict
    )
    print(f"Chunks read: {report['chunks_read']}")
    print(f"Unique ingredients: {report['unique_ingredients']}")
    print(f"Invalid lines: {report['invalid_lines']}")
    print(f"Invalid chunks: {report['invalid_chunks']}")
    print(f"Output: {report['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
