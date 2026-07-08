"""Normalize Long Chau ingredient chunks into evidence chunk JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("data/processed/longchau_ingredients_chunked")
DEFAULT_OUTPUT = Path(
    "data/processed/longchau_ingredients_v2/ingredients_chunks.jsonl"
)
DEFAULT_REPORT = Path(
    "data/processed/longchau_ingredients_v2/normalization_report.json"
)
SOURCE_NAME = "Dược chất Long Châu"
SAMPLE_LIMIT = 50

FIELD_TO_SECTION = {
    "describe": "mo_ta_chung",
    "indication": "chi_dinh",
    "pharmacodynamics": "duoc_luc_hoc",
    "pharmacokinetics": "duoc_dong_hoc",
    "dosage": "lieu_luong_va_cach_dung",
    "interaction": "tuong_tac_thuoc",
    "careful": "than_trong",
    "overdose": "qua_lieu_va_xu_tri",
}

SECTION_TITLES = {
    "mo_ta_chung": "Mô tả chung",
    "chi_dinh": "Chỉ định",
    "duoc_luc_hoc": "Dược lực học",
    "duoc_dong_hoc": "Dược động học",
    "lieu_luong_va_cach_dung": "Liều lượng và cách dùng",
    "tuong_tac_thuoc": "Tương tác thuốc",
    "than_trong": "Thận trọng",
    "qua_lieu_va_xu_tri": "Quá liều và xử trí",
}

REQUIRED_METADATA_FIELDS = ("name", "id", "url", "field", "source")


def normalize_nfc(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "").strip())


def build_prefix(name: str, slug: str, section: str, url: str) -> str:
    return unicodedata.normalize(
        "NFC",
        (
            f"Hoạt chất: {name}\n"
            f"Slug: {slug}\n"
            f"Mục: {section}\n"
            f"Nguồn: {SOURCE_NAME}\n"
            f"URL: {url}\n\n"
            "Nội dung:\n"
        ),
    )


def _sample_append(samples: list[dict[str, Any]], item: dict[str, Any]) -> None:
    if len(samples) < SAMPLE_LIMIT:
        samples.append(item)


def _invalid_chunk(
    report: dict[str, Any],
    *,
    file_path: Path,
    item_index: int | None,
    reason: str,
    detail: Any = None,
) -> None:
    report["invalid_chunks"] += 1
    sample = {
        "file": str(file_path),
        "item_index": item_index,
        "reason": reason,
    }
    if detail not in (None, "", []):
        sample["detail"] = detail
    _sample_append(report["sample_invalid_chunks"], sample)


def _missing_required(item: Any) -> list[str]:
    if not isinstance(item, dict):
        return ["item"]
    missing: list[str] = []
    if not item.get("text"):
        missing.append("text")
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        missing.append("metadata")
        return missing
    for field in REQUIRED_METADATA_FIELDS:
        if not metadata.get(field):
            missing.append(f"metadata.{field}")
    return missing


def create_normalized_chunk(
    item: dict[str, Any],
    *,
    section: str,
    chunk_index: int,
) -> dict[str, Any]:
    metadata = item["metadata"]
    name = normalize_nfc(metadata["name"])
    slug = normalize_nfc(metadata["id"])
    url = normalize_nfc(metadata["url"])
    body = normalize_nfc(item["text"])
    prefix = build_prefix(name, slug, section, url)
    content = unicodedata.normalize("NFC", prefix + body)
    return {
        "chunk_id": (
            f"longchau:ingredient:{slug}:{section}:{chunk_index:04d}"
        ),
        "content": content,
        "metadata": {
            "source": "longchau",
            "source_name": SOURCE_NAME,
            "source_type": "supplementary",
            "entity_type": "ingredient",
            "entity_name": name,
            "slug": slug,
            "section": section,
            "section_title": SECTION_TITLES.get(section, section),
            "url": url,
            "title": name,
            "language": "vi",
            "chunk_index": chunk_index,
            "chunk_strategy": "section",
            "char_count": len(content),
        },
    }


def validate_normalized_chunk(chunk: dict[str, Any]) -> None:
    if not chunk.get("chunk_id") or not chunk.get("content"):
        raise ValueError("Chunk missing required chunk_id or content")
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Chunk metadata must be an object")
    for field in ("slug", "section", "entity_name", "url"):
        if not metadata.get(field):
            raise ValueError(f"Chunk metadata missing required field: {field}")
    if metadata.get("char_count") != len(chunk["content"]):
        raise ValueError("Chunk char_count does not match content length")
    if not unicodedata.is_normalized("NFC", chunk["content"]):
        raise ValueError("Chunk content must be Unicode NFC")


def _blank_report(input_dir: Path, output_path: Path) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "files_read": 0,
        "files_valid": 0,
        "files_invalid": 0,
        "chunks_read": 0,
        "chunks_written": 0,
        "invalid_chunks": 0,
        "field_counts": {},
        "section_counts": {},
        "source_counts": {},
        "duplicate_chunk_ids": [],
        "unknown_fields": [],
        "missing_required_fields": {},
        "sample_invalid_chunks": [],
    }


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
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
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def normalize_directory(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    *,
    overwrite: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not overwrite and (output_path.exists() or report_path.exists()):
        raise FileExistsError("Output exists; use --overwrite")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(f".{output_path.name}.tmp")
    if temp_output.exists():
        temp_output.unlink()

    report = _blank_report(input_dir, output_path)
    field_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    missing_required_fields: Counter[str] = Counter()
    unknown_fields: Counter[str] = Counter()
    section_indexes: defaultdict[tuple[str, str], int] = defaultdict(int)
    seen_chunk_ids: set[str] = set()
    duplicate_chunk_ids: list[str] = []

    try:
        with temp_output.open("w", encoding="utf-8", newline="\n") as out:
            for file_path in sorted(input_dir.glob("*.json")):
                report["files_read"] += 1
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    report["files_invalid"] += 1
                    _invalid_chunk(
                        report,
                        file_path=file_path,
                        item_index=None,
                        reason="invalid_json",
                        detail=str(exc),
                    )
                    if strict:
                        raise ValueError(
                            f"Invalid JSON file: {file_path}"
                        ) from exc
                    continue
                if not isinstance(data, list):
                    report["files_invalid"] += 1
                    _invalid_chunk(
                        report,
                        file_path=file_path,
                        item_index=None,
                        reason="file_not_json_array",
                    )
                    if strict:
                        raise ValueError(
                            f"File must contain a JSON array: {file_path}"
                        )
                    continue

                file_has_valid_chunk = False
                for item_index, item in enumerate(data, 1):
                    report["chunks_read"] += 1
                    missing = _missing_required(item)
                    if missing:
                        for field in missing:
                            missing_required_fields[field] += 1
                        _invalid_chunk(
                            report,
                            file_path=file_path,
                            item_index=item_index,
                            reason="missing_required_fields",
                            detail=missing,
                        )
                        if strict:
                            raise ValueError(
                                f"Missing required fields in {file_path}: "
                                + ", ".join(missing)
                            )
                        continue

                    metadata = item["metadata"]
                    raw_field = normalize_nfc(metadata["field"])
                    field_counts[raw_field] += 1
                    source_counts[normalize_nfc(metadata["source"])] += 1
                    section = FIELD_TO_SECTION.get(raw_field)
                    if not section:
                        unknown_fields[raw_field] += 1
                        _invalid_chunk(
                            report,
                            file_path=file_path,
                            item_index=item_index,
                            reason="unknown_field",
                            detail=raw_field,
                        )
                        if strict:
                            raise ValueError(
                                f"Unknown Long Chau field: {raw_field}"
                            )
                        continue

                    slug = normalize_nfc(metadata["id"])
                    section_indexes[(slug, section)] += 1
                    chunk = create_normalized_chunk(
                        item,
                        section=section,
                        chunk_index=section_indexes[(slug, section)],
                    )
                    validate_normalized_chunk(chunk)
                    chunk_id = chunk["chunk_id"]
                    if chunk_id in seen_chunk_ids:
                        duplicate_chunk_ids.append(chunk_id)
                        _invalid_chunk(
                            report,
                            file_path=file_path,
                            item_index=item_index,
                            reason="duplicate_chunk_id",
                            detail=chunk_id,
                        )
                        if strict:
                            raise ValueError(f"Duplicate chunk_id: {chunk_id}")
                        continue

                    seen_chunk_ids.add(chunk_id)
                    out.write(
                        json.dumps(
                            chunk, ensure_ascii=False, separators=(",", ":")
                        )
                        + "\n"
                    )
                    report["chunks_written"] += 1
                    section_counts[section] += 1
                    file_has_valid_chunk = True

                if file_has_valid_chunk:
                    report["files_valid"] += 1
                else:
                    report["files_invalid"] += 1

        os.replace(temp_output, output_path)
    except Exception:
        if temp_output.exists():
            temp_output.unlink()
        raise

    report["field_counts"] = dict(field_counts.most_common())
    report["section_counts"] = dict(section_counts.most_common())
    report["source_counts"] = dict(source_counts.most_common())
    report["duplicate_chunk_ids"] = duplicate_chunk_ids
    report["unknown_fields"] = [
        {"field": field, "count": count}
        for field, count in unknown_fields.most_common()
    ]
    report["missing_required_fields"] = dict(
        missing_required_fields.most_common()
    )
    _write_json_atomic(report_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize Long Chau ingredient chunks"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    report = normalize_directory(
        args.input_dir,
        args.output,
        args.report,
        overwrite=args.overwrite,
        strict=args.strict,
    )
    print(f"files_read: {report['files_read']}")
    print(f"chunks_read: {report['chunks_read']}")
    print(f"chunks_written: {report['chunks_written']}")
    print(f"invalid_chunks: {report['invalid_chunks']}")
    print(f"duplicate_chunk_ids: {report['duplicate_chunk_ids']}")
    print(f"unknown_fields: {report['unknown_fields']}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
