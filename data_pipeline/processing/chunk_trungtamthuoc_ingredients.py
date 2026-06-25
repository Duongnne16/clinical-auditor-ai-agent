"""Stream-chunk cleaned Trung Tâm Thuốc ingredient monographs."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import statistics
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data/cleaned/trungtamthuoc_v2/ingredients_cleaned.jsonl")
DEFAULT_OUTPUT = Path("data/processed/trungtamthuoc_v2/ingredients_chunks.jsonl")
DEFAULT_REPORT = Path("data/processed/trungtamthuoc_v2/chunking_report.json")
SOURCE_NAME = "Trung Tâm Thuốc / Dược thư Quốc gia Việt Nam"
IMPORTANT_SECTIONS = {
    "chi_dinh", "chong_chi_dinh", "than_trong", "tuong_tac_thuoc",
    "thai_ky_cho_con_bu", "thoi_ky_mang_thai",
    "tac_dung_khong_mong_muon", "lieu_luong_va_cach_dung",
    "duoc_luc_hoc", "duoc_dong_hoc", "qua_lieu_va_xu_tri",
    "trieu_chung", "xu_tri", "hap_thu", "phan_bo", "chuyen_hoa",
    "thai_tru", "thuong_gap", "it_gap", "hiem_gap",
}
IMPORTANT_SECTION_PATTERNS = (
    "chong_chi", "than_trong", "tuong_tac", "tuong_ky", "thai",
    "cho_con_bu", "tac_dung", "adr", "thuong_gap", "it_gap",
    "hiem_gap", "rat_thuong_gap", "rat_hiem_gap", "chua_xac_dinh",
    "qua_lieu", "xu_tri", "tre_em", "nguoi_lon", "nguoi_cao_tuoi",
    "suy_than", "suy_gan", "dang_thuoc", "ham_luong", "lieu_luong",
)
KEPT_SHORT_SAMPLE_LIMIT = 100
TABLE_HEADER_KEYWORDS = (
    "thuốc", "nhóm thuốc", "hoạt chất", "tác động", "khuyến cáo",
    "liều", "đối tượng", "tần suất", "tác dụng phụ",
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+")


def normalize_preview(text: str, limit: int = 300) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def is_important_section(section_key: str) -> bool:
    normalized = str(section_key or "").casefold()
    return (
        normalized in IMPORTANT_SECTIONS
        or any(pattern in normalized for pattern in IMPORTANT_SECTION_PATTERNS)
    )


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


def _hard_split(text: str, budget: int) -> list[str]:
    return [text[index:index + budget] for index in range(0, len(text), budget)]


def split_oversized_block(text: str, budget: int) -> list[str]:
    """Split by sentence, then words, then characters without exceeding budget."""
    if len(text) <= budget:
        return [text]

    sentences = [item.strip() for item in _SENTENCE_BOUNDARY.split(text) if item.strip()]
    if len(sentences) <= 1:
        sentences = [text]

    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > budget:
            if current:
                pieces.append(current)
                current = ""
            words = sentence.split()
            word_buffer = ""
            for word in words:
                if len(word) > budget:
                    if word_buffer:
                        pieces.append(word_buffer)
                        word_buffer = ""
                    pieces.extend(_hard_split(word, budget))
                    continue
                candidate = f"{word_buffer} {word}".strip()
                if len(candidate) <= budget:
                    word_buffer = candidate
                else:
                    pieces.append(word_buffer)
                    word_buffer = word
            if word_buffer:
                pieces.append(word_buffer)
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= budget:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece]


def _overlap_tail(text: str, limit: int) -> str:
    if limit <= 0 or not text:
        return ""
    blocks = text.splitlines()
    selected: list[str] = []
    size = 0
    for block in reversed(blocks):
        addition = len(block) + (1 if selected else 0)
        if size + addition > limit:
            break
        selected.append(block)
        size += addition
    if selected:
        return "\n".join(reversed(selected))
    return text[-limit:]


def group_blocks(
    blocks: list[str],
    budget: int,
    overlap_chars: int,
    strategy: str,
) -> list[tuple[str, str]]:
    """Return content and strategy while respecting the content budget."""
    output: list[tuple[str, str]] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            output.append((current, strategy))
            current = ""

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) > budget:
            flush()
            output.extend(
                (piece, strategy)
                for piece in split_oversized_block(block, budget)
            )
            continue
        candidate = f"{current}\n{block}".strip() if current else block
        if len(candidate) <= budget:
            current = candidate
            continue
        previous = current
        flush()
        overlap = _overlap_tail(previous, min(overlap_chars, budget))
        candidate = f"{overlap}\n{block}".strip() if overlap else block
        current = candidate if len(candidate) <= budget else block
    flush()
    return output


def detect_table_header(lines: list[str], budget: int) -> str:
    candidates: list[str] = []
    for line in lines[:3]:
        folded = line.casefold()
        if (
            len(line) <= 300
            and any(keyword in folded for keyword in TABLE_HEADER_KEYWORDS)
        ):
            candidates.append(line)
        elif candidates:
            break
    header = "\n".join(candidates)
    return header if header and len(header) <= min(600, budget // 2) else ""


def chunk_section_content(
    text: str,
    *,
    budget: int,
    overlap_chars: int,
) -> list[tuple[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines and text.strip():
        lines = [text.strip()]
    is_table = any("|" in line for line in lines)
    if not is_table:
        return group_blocks(lines, budget, overlap_chars, "split")

    header = detect_table_header(lines, budget)
    chunks = group_blocks(lines, budget, 0, "table_line_group")
    if not header:
        return chunks

    with_headers: list[tuple[str, str]] = []
    for index, (content, strategy) in enumerate(chunks):
        if index == 0 or content.startswith(header):
            with_headers.append((content, strategy))
            continue
        candidate = f"{header}\n{content}"
        if len(candidate) <= budget:
            with_headers.append((candidate, strategy))
        else:
            with_headers.append((content, strategy))
    return with_headers


def create_chunk(
    record: dict[str, Any],
    section: str,
    chunk_index: int,
    content: str,
    strategy: str,
) -> dict[str, Any]:
    name = str(record.get("name") or "")
    url = str(record.get("url") or "")
    slug = str(record.get("slug") or "")
    prefix = build_prefix(name, slug, section, url)
    full_content = unicodedata.normalize("NFC", prefix + content)
    return {
        "chunk_id": (
            f"trungtamthuoc:ingredient:{slug}:{section}:{chunk_index:04d}"
        ),
        "content": full_content,
        "metadata": {
            "source": "trungtamthuoc",
            "source_type": "duoc_thu",
            "entity_type": "ingredient",
            "entity_name": name,
            "slug": slug,
            "section": section,
            "section_title": section,
            "url": url,
            "title": record.get("title"),
            "language": "vi",
            "chunk_index": chunk_index,
            "chunk_strategy": strategy,
            "char_count": len(full_content),
        },
    }


def validate_chunk(chunk: dict[str, Any], prefix: str) -> None:
    if not chunk.get("chunk_id") or not chunk.get("content"):
        raise ValueError("Chunk missing required chunk_id or content")
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Chunk metadata must be an object")
    for field in ("url", "entity_name", "section", "slug"):
        if not metadata.get(field):
            raise ValueError(f"Chunk metadata missing required field: {field}")
    if not chunk["content"].startswith(prefix):
        raise ValueError("Chunk text is missing the required context prefix")
    if metadata.get("char_count") != len(chunk["content"]):
        raise ValueError("Chunk char_count does not match content length")
    if not unicodedata.is_normalized("NFC", chunk["content"]):
        raise ValueError("Chunk content must be Unicode NFC")


def calculate_stats(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "max": None, "median": None, "average": None}
    return {
        "min": min(values),
        "max": max(values),
        "median": round(statistics.median(values), 2),
        "average": round(statistics.fmean(values), 2),
    }


def _validate_paths(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output or temp file exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def _save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def chunk_jsonl(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    max_chars: int,
    min_chars: int,
    overlap_chars: int,
    overwrite: bool,
) -> dict[str, Any]:
    if max_chars <= 0 or min_chars < 0:
        raise ValueError("Require max_chars > 0 and min_chars >= 0")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("Require 0 <= overlap_chars < max_chars")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    temp_output = output_path.with_name("ingredients_chunks.tmp.jsonl")
    _validate_paths((output_path, report_path, temp_output), overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_output.exists():
        temp_output.unlink()

    records_read = records_written = sections_seen = sections_chunked = 0
    sections_skipped_short = chunks_written = 0
    kept_short_important_sections_count = 0
    kept_short_important_sections_sample: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    skipped_short: list[dict[str, Any]] = []
    chunk_ids: set[str] = set()
    duplicate_chunk_ids: list[str] = []
    strategy_counts: Counter[str] = Counter()
    chunks_by_section: Counter[str] = Counter()
    chunks_by_slug: Counter[str] = Counter()
    chunk_lengths: list[int] = []
    sample_chunks: list[dict[str, Any]] = []
    longest_heap: list[tuple[int, int, dict[str, Any]]] = []
    warnings: list[str] = []
    sequence = 0

    try:
        with input_path.open(encoding="utf-8") as source, temp_output.open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
          for line_number, line in enumerate(source, 1):
            records_read += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_records.append(
                    {"line_number": line_number, "reason": "invalid_json", "detail": str(exc)}
                )
                continue
            if not isinstance(record, dict) or not isinstance(record.get("sections"), dict):
                invalid_records.append(
                    {"line_number": line_number, "reason": "invalid_schema"}
                )
                continue
            required = ("source", "entity_type", "name", "slug", "url", "sections")
            missing = [field for field in required if not record.get(field)]
            if missing:
                invalid_records.append(
                    {
                        "line_number": line_number,
                        "reason": "missing_required_fields",
                        "detail": ", ".join(missing),
                    }
                )
                continue
            records_written += 1

            for section, raw_text in record["sections"].items():
                sections_seen += 1
                if not isinstance(section, str) or not isinstance(raw_text, str):
                    invalid_records.append(
                        {"line_number": line_number, "reason": "invalid_section", "section": section}
                    )
                    continue
                text = raw_text.strip()
                if not text:
                    continue
                if len(text) < min_chars:
                    if not is_important_section(section):
                        sections_skipped_short += 1
                        skipped_short.append(
                            {
                                "slug": record.get("slug"),
                                "name": record.get("name"),
                                "section": section,
                                "text_length": len(text),
                            }
                        )
                        continue
                    kept_short_important_sections_count += 1
                    if (
                        len(kept_short_important_sections_sample)
                        < KEPT_SHORT_SAMPLE_LIMIT
                    ):
                        kept_short_important_sections_sample.append(
                            {
                                "slug": record.get("slug"),
                                "name": record.get("name"),
                                "section": section,
                                "text_length": len(text),
                                "preview": normalize_preview(text),
                            }
                        )

                prefix = build_prefix(
                    str(record.get("name") or ""),
                    str(record.get("slug") or ""),
                    section,
                    str(record.get("url") or ""),
                )
                budget = max_chars - len(prefix)
                if budget <= 0:
                    raise ValueError(
                        f"Context prefix is not smaller than max_chars for "
                        f"{record.get('slug')}:{section}"
                    )
                if len(prefix) + len(text) <= max_chars:
                    pieces = [(text, "section")]
                else:
                    pieces = chunk_section_content(
                        text,
                        budget=budget,
                        overlap_chars=overlap_chars,
                    )
                if pieces:
                    sections_chunked += 1
                for index, (content, strategy) in enumerate(pieces, 1):
                    if not content:
                        continue
                    chunk = create_chunk(
                        record, section, index, content, strategy
                    )
                    validate_chunk(chunk, prefix)
                    if chunk["chunk_id"] in chunk_ids:
                        duplicate_chunk_ids.append(chunk["chunk_id"])
                        raise ValueError(
                            f"Duplicate chunk_id: {chunk['chunk_id']}"
                        )
                    chunk_ids.add(chunk["chunk_id"])
                    destination.write(
                        json.dumps(
                            chunk,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    destination.flush()
                    chunks_written += 1
                    strategy_counts[strategy] += 1
                    chunks_by_section[section] += 1
                    chunks_by_slug[str(record.get("slug") or "")] += 1
                    char_count = chunk["metadata"]["char_count"]
                    chunk_lengths.append(char_count)
                    if char_count > max_chars:
                        raise ValueError(
                            f"Chunk exceeds max_chars: {chunk['chunk_id']}"
                        )
                    preview = {
                        "chunk_id": chunk["chunk_id"],
                        "slug": chunk["metadata"]["slug"],
                        "section": section,
                        "char_count": char_count,
                        "preview": normalize_preview(chunk["content"]),
                    }
                    if len(sample_chunks) < 20:
                        sample_chunks.append(preview)
                    sequence += 1
                    entry = (char_count, sequence, preview)
                    if len(longest_heap) < 20:
                        heapq.heappush(longest_heap, entry)
                    elif char_count > longest_heap[0][0]:
                        heapq.heapreplace(longest_heap, entry)
        os.replace(temp_output, output_path)
    except Exception:
        if temp_output.exists():
            temp_output.unlink()
        raise

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "max_chars": max_chars,
            "min_chars": min_chars,
            "overlap_chars": overlap_chars,
        },
        "records_read": records_read,
        "records_written": records_written,
        "sections_seen": sections_seen,
        "sections_chunked": sections_chunked,
        "sections_skipped_short": sections_skipped_short,
        "kept_short_important_sections_count": (
            kept_short_important_sections_count
        ),
        "kept_short_important_sections_sample": (
            kept_short_important_sections_sample
        ),
        "chunks_written": chunks_written,
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "invalid_records_count": len(invalid_records),
        "invalid_records": invalid_records,
        "chunk_length_stats": calculate_stats(chunk_lengths),
        "strategy_counts": {
            key: strategy_counts.get(key, 0)
            for key in ("section", "split", "table_line_group")
        },
        "top_sections_by_chunk_count": [
            {"section": key, "count": count}
            for key, count in chunks_by_section.most_common(30)
        ],
        "top_slugs_by_chunk_count": [
            {"slug": key, "count": count}
            for key, count in chunks_by_slug.most_common(30)
        ],
        "skipped_short_sections": skipped_short,
        "longest_chunks": [
            item for _, _, item in sorted(longest_heap, reverse=True)
        ],
        "sample_chunks": sample_chunks,
        "warnings": warnings,
    }
    _save_json_atomic(report_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk Trung Tâm Thuốc cleaned ingredient JSONL"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--overlap-chars", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    report = chunk_jsonl(
        args.input, args.output, args.report,
        max_chars=args.max_chars, min_chars=args.min_chars,
        overlap_chars=args.overlap_chars,
        overwrite=args.overwrite,
    )
    print(f"records_read: {report['records_read']}")
    print(f"sections_chunked: {report['sections_chunked']}")
    print(f"chunks_written: {report['chunks_written']}")
    print(f"chunk_length_stats: {report['chunk_length_stats']}")
    print(f"strategy_counts: {report['strategy_counts']}")
    print(f"duplicate_chunk_ids: {report['duplicate_chunk_ids']}")
    print(f"warnings: {report['warnings']}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
