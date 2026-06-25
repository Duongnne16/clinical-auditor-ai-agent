"""Raw ingredient crawler for Trung Tâm Thuốc.

Importing this module performs no network or filesystem operations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


BASE_URL = "https://trungtamthuoc.com"
INDEX_URL = f"{BASE_URL}/hoat-chat"
OUTPUT_DIR = Path("data/raw/trungtamthuoc")
URLS_PATH = OUTPUT_DIR / "ingredient_urls.json"
RAW_PATH = OUTPUT_DIR / "ingredients_raw.jsonl"
FAILED_PATH = OUTPUT_DIR / "failed_urls.json"

USER_AGENT = (
    "ClinicalAuditorResearchCrawler/0.1 "
    "(educational medical-data collection; respectful rate limiting)"
)
NOISE_TAGS = (
    "script", "style", "noscript", "svg", "header", "footer", "nav",
    "form", "button", "iframe",
)
CONTENT_HINTS = ("content", "article", "detail", "post", "entry", "document")
STOP_HEADINGS = (
    "so sánh",
    "sự khác biệt",
    "paracetamol và ibuprofen",
    "tài liệu tham khảo",
    "bài viết liên quan",
    "sản phẩm liên quan",
    "câu hỏi thường gặp",
    "giải đáp các thắc mắc",
)
MEDICAL_SIGNALS = (
    "tên chung quốc tế", "mã atc", "chỉ định", "chống chỉ định",
    "thận trọng", "liều lượng và cách dùng", "tương tác thuốc",
    "tác dụng không mong muốn",
)


class CrawlError(Exception):
    """Structured crawler failure with a stable reason code."""

    def __init__(
        self,
        reason: str,
        message: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or reason)
        self.reason = reason
        self.detail = detail or {}


def clean_text(text: Any) -> str:
    value = str(text or "").replace("\xa0", " ").replace("\u200b", " ")
    return re.sub(r"\s+", " ", value).strip()


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.8"}
    )
    return session


def fetch_html(
    url: str,
    timeout: int = 25,
    session: requests.Session | None = None,
) -> str:
    client = session or create_session()
    try:
        response = client.get(url, timeout=timeout)
    except requests.Timeout as exc:
        raise CrawlError("timeout", str(exc)) from exc
    except requests.RequestException as exc:
        raise CrawlError("request_error", str(exc)) from exc
    if response.status_code != 200:
        raise CrawlError("invalid_status", f"HTTP {response.status_code}")
    response.encoding = response.encoding or response.apparent_encoding or "utf-8"
    return response.text


def normalize_url(href: str | None) -> str | None:
    if not href:
        return None
    parsed = urlparse(urljoin(BASE_URL, href.strip()))
    if parsed.scheme not in {"http", "https"} or (
        parsed.hostname or ""
    ).lower() not in {"trungtamthuoc.com", "www.trungtamthuoc.com"}:
        return None
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    return urlunparse(("https", "trungtamthuoc.com", path, "", "", ""))


def is_ingredient_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return bool(
        parsed.scheme == "https"
        and parsed.netloc == "trungtamthuoc.com"
        and re.fullmatch(r"/hoat-chat/[a-z0-9][a-z0-9-]*", parsed.path)
        and not parsed.query
        and not parsed.fragment
    )


def slug_from_url(url: str) -> str:
    if not is_ingredient_url(url):
        raise ValueError(f"Not an ingredient URL: {url}")
    return urlparse(url).path.rsplit("/", 1)[-1]


def _extract_key_urls(index_soup: BeautifulSoup) -> list[str]:
    keys: list[str] = []
    for anchor in index_soup.find_all("a", href=True):
        parsed = urlparse(urljoin(BASE_URL, anchor["href"]))
        query = parse_qs(parsed.query)
        if parsed.path.rstrip("/") == "/hoat-chat" and query.get("key"):
            key = query["key"][0].strip()
            if key and key not in keys:
                keys.append(key)
    if not keys:
        keys = [chr(code) for code in range(ord("a"), ord("z") + 1)] + ["0-9"]
    return [f"{INDEX_URL}?key={key}" for key in keys]


def extract_ingredient_links(
    session: requests.Session | None = None,
    timeout: int = 25,
    delay: Callable[[], None] | None = None,
) -> list[str]:
    """Prefer key links exposed by the index, with A-Z/0-9 as fallback."""
    client = session or create_session()
    index_soup = BeautifulSoup(
        fetch_html(INDEX_URL, timeout=timeout, session=client), "html.parser"
    )
    links: set[str] = set()
    for key_url in _extract_key_urls(index_soup):
        if delay:
            delay()
        soup = BeautifulSoup(
            fetch_html(key_url, timeout=timeout, session=client), "html.parser"
        )
        for anchor in soup.find_all("a", href=True):
            normalized = normalize_url(anchor["href"])
            if is_ingredient_url(normalized):
                links.add(normalized)
    return sorted(links)


def extract_title(soup: BeautifulSoup | Tag) -> str | None:
    heading = soup.find("h1")
    if heading:
        title = clean_text(heading.get_text(" ", strip=True))
        if title:
            return title
    if isinstance(soup, BeautifulSoup) and soup.title:
        return clean_text(soup.title.get_text(" ", strip=True)) or None
    return None


def extract_updated_at(text: str) -> str | None:
    match = re.search(
        r"(?:ngày\s+)?cập\s+nhật\s*:\s*(\d{1,2}/\d{1,2}/\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def remove_noise_tags(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()


def _tag_text_length(tag: Tag) -> int:
    return len(clean_text(tag.get_text(" ", strip=True)))


def extract_main_content(soup: BeautifulSoup) -> Tag:
    tiers: list[list[Tag]] = [
        list(soup.find_all("article")),
        list(soup.find_all("main")),
        [
            tag
            for tag in soup.find_all("div")
            if any(
                hint in " ".join(
                    (str(tag.get("id", "")), " ".join(tag.get("class", [])))
                ).casefold()
                for hint in CONTENT_HINTS
            )
        ],
    ]
    if soup.body:
        tiers.append([soup.body])
    for candidates in tiers:
        viable = [item for item in candidates if _tag_text_length(item) >= 100]
        if viable:
            return max(viable, key=_tag_text_length)
    raise CrawlError("parse_error", "No usable main content container")


def _fold_vietnamese(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", clean_text(text).casefold())
    folded = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    return folded.replace("đ", "d")


def normalize_heading(text: str) -> str:
    heading = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", clean_text(text))
    value = re.sub(r"[^a-z0-9]+", "_", _fold_vietnamese(heading))
    return re.sub(r"_+", "_", value).strip("_") or "section"


def map_section_name(heading: str) -> str:
    value = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", heading)
    folded = _fold_vietnamese(value)
    without_contraindication = folded.replace("chong chi dinh", "")
    if (
        "chong chi dinh" in folded
        and "chi dinh" in without_contraindication
    ):
        return "chi_dinh_chong_chi_dinh"
    mappings = (
        (("dang thuoc va ham luong",), "dang_thuoc_va_ham_luong"),
        (("duoc luc hoc",), "duoc_luc_hoc"),
        (("duoc dong hoc",), "duoc_dong_hoc"),
        (("chong chi dinh",), "chong_chi_dinh"),
        (("chi dinh",), "chi_dinh"),
        (("than trong",), "than_trong"),
        (
            ("thoi ky mang thai va cho con bu", "phu nu co thai", "cho con bu"),
            "thai_ky_cho_con_bu",
        ),
        (("tac dung khong mong muon", "adr"), "tac_dung_khong_mong_muon"),
        (
            ("lieu luong va cach dung", "lieu dung", "cach dung"),
            "lieu_luong_va_cach_dung",
        ),
        (("tuong tac thuoc", "tuong tac"), "tuong_tac_thuoc"),
        (("qua lieu va xu tri",), "qua_lieu_va_xu_tri"),
        (("bao quan",), "bao_quan"),
    )
    for needles, key in mappings:
        if any(needle in folded for needle in needles):
            return key
    return normalize_heading(heading)


def _is_stop_heading(heading: str) -> bool:
    folded = _fold_vietnamese(heading)
    return any(_fold_vietnamese(value) in folded for value in STOP_HEADINGS)


def get_cell_own_text(cell: Tag) -> str:
    """Read a cell without consuming nested table structure."""
    clone = copy.deepcopy(cell)
    for nested in list(clone.find_all(["table", "tr", "td", "th"])):
        nested.decompose()
    return clean_text(clone.get_text(" ", strip=True))


def _normalized_chain_segment(value: str) -> str:
    return clean_text(value).casefold()


def detect_suffix_chain_line(line: str) -> bool:
    """Detect exact nested suffix/substring chains in a pipe-separated row."""
    segments = [
        _normalized_chain_segment(segment)
        for segment in line.split("|")
        if clean_text(segment)
    ]
    if len(segments) < 3:
        return False

    chain_links = 0
    for previous, current in zip(segments, segments[1:]):
        if (
            current != previous
            and len(current) < len(previous)
            and current in previous
        ):
            chain_links += 1
        else:
            chain_links = 0
        if chain_links >= 2:
            return True
    return False


def _direct_row_cells(row: Tag) -> list[Tag]:
    return [
        child
        for child in row.children
        if isinstance(child, Tag) and child.name in {"th", "td"}
    ]


def _logical_row_cells(row: Tag) -> list[Tag]:
    return [
        cell
        for cell in row.find_all(["th", "td"])
        if cell.find_parent("tr") is row
    ]


def _has_stuck_descendant_text(
    direct_cell: Tag,
    logical_cells: list[Tag],
) -> bool:
    """Identify a malformed wrapper cell that swallowed sibling cells."""
    flattened = clean_text(direct_cell.get_text(" ", strip=True)).casefold()
    descendant_values: list[str] = []
    for cell in logical_cells:
        if cell is direct_cell:
            continue
        value = get_cell_own_text(cell).casefold()
        if value:
            descendant_values.append(value)
    contained_values = {
        value for value in descendant_values if value and value in flattened
    }
    required_matches = 1 if len(logical_cells) == 2 else 2
    return len(contained_values) >= required_matches


def _should_recover_logical_cells(
    direct_cells: list[Tag],
    logical_cells: list[Tag],
) -> bool:
    if len(direct_cells) != 1 or len(logical_cells) <= 1:
        return False

    direct_cell = direct_cells[0]
    legacy_line = " | ".join(
        clean_text(cell.get_text(" ", strip=True)) for cell in logical_cells
    )
    has_suffix_chain = detect_suffix_chain_line(legacy_line)

    if direct_cell.has_attr("colspan") and not has_suffix_chain:
        return False

    return has_suffix_chain or _has_stuck_descendant_text(
        direct_cell,
        logical_cells,
    )


def parse_table(
    table: Tag,
    diagnostics: dict[str, int] | None = None,
) -> list[str]:
    """Parse rows safely while recovering malformed nested logical cells."""
    rows: list[str] = []
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        direct_cells = _direct_row_cells(row)
        logical_cells = _logical_row_cells(row)
        cells = direct_cells
        if _should_recover_logical_cells(direct_cells, logical_cells):
            cells = logical_cells

        values = [get_cell_own_text(cell) for cell in cells]
        while values and not values[-1]:
            values.pop()
        if values and any(values):
            rows.append(" | ".join(values))

    if diagnostics is not None:
        diagnostics.update(
            {
                "row_count": len(rows),
                "max_pipe_count_per_row": max(
                    (row.count("|") for row in rows),
                    default=0,
                ),
                "suspected_suffix_chain_count": sum(
                    detect_suffix_chain_line(row) for row in rows
                ),
            }
        )
    return rows


_HEADING_TAGS = {"h2", "h3", "h4"}
_TEXT_BLOCK_TAGS = {"p", "li"}
_CONTAINER_TAGS = {
    "article", "main", "section", "div", "p", "li", "ul", "ol",
    "blockquote", "details",
}
_IGNORED_PREHEADING_PATTERNS = (
    "dược sĩ lâm sàng",
    "ước tính:",
    "ngày đăng:",
    "cập nhật:",
    "nếu phát hiện nội dung không chính xác",
)


def _contains_block_elements(tag: Tag) -> bool:
    return tag.find(
        list(_HEADING_TAGS | _TEXT_BLOCK_TAGS | {"table", "ul", "ol"}),
        recursive=True,
    ) is not None


def _iter_section_events(root: Tag):
    """Yield heading/text events without reading text from parent wrappers."""

    def walk(node: Tag):
        inline_parts: list[str] = []
        collects_direct_text = node.name in _TEXT_BLOCK_TAGS

        def flush_inline():
            if not inline_parts:
                return
            value = clean_text(" ".join(inline_parts))
            inline_parts.clear()
            if value:
                yield ("text", value)

        for child in node.children:
            if isinstance(child, NavigableString):
                if collects_direct_text:
                    value = clean_text(child)
                    if value:
                        inline_parts.append(value)
                continue
            if not isinstance(child, Tag):
                continue

            if child.name in _HEADING_TAGS:
                yield from flush_inline()
                yield ("heading", clean_text(child.get_text(" ", strip=True)))
                continue

            if child.name == "table":
                yield from flush_inline()
                for row in parse_table(child):
                    yield ("text", row)
                continue

            if child.name in _CONTAINER_TAGS or _contains_block_elements(child):
                yield from flush_inline()
                yield from walk(child)
                continue

            if collects_direct_text:
                value = clean_text(child.get_text(" ", strip=True))
                if value:
                    inline_parts.append(value)

        yield from flush_inline()

    yield from walk(root)


def _is_useful_preheading_text(text: str) -> bool:
    folded = _fold_vietnamese(text)
    return not any(
        _fold_vietnamese(pattern) in folded
        for pattern in _IGNORED_PREHEADING_PATTERNS
    )


def parse_sections(main_content: Tag) -> dict[str, str]:
    collected: dict[str, list[str]] = {"mo_ta_chung": []}
    current_key = "mo_ta_chung"
    seen_heading = False

    for event_type, value in _iter_section_events(main_content):
        if event_type == "heading":
            heading = value
            if _is_stop_heading(heading):
                break
            current_key = map_section_name(heading)
            collected.setdefault(current_key, [])
            seen_heading = True
            continue

        if not seen_heading and not _is_useful_preheading_text(value):
            continue
        if value and (
            not collected[current_key] or collected[current_key][-1] != value
        ):
            collected[current_key].append(value)

    return {
        key: "\n".join(values)
        for key, values in collected.items()
        if values
    }


def print_record_preview(record: dict[str, Any]) -> None:
    print(f"Name: {record['name']}")
    print(f"Slug: {record['slug']}")
    print("Sections:")
    for name, text in record["sections"].items():
        preview = clean_text(text)[:150]
        print(f"  {name}: {len(text)} chars")
        print(f"    {preview}")


def _page_diagnostics(
    title: str | None,
    full_text: str,
    sections: dict[str, str],
) -> dict[str, Any]:
    folded = _fold_vietnamese(full_text)
    return {
        "title": title,
        "text_length": len(full_text),
        "medical_signal_count": sum(
            _fold_vietnamese(signal) in folded for signal in MEDICAL_SIGNALS
        ),
        "section_count": len(sections),
    }


def is_valid_ingredient_page(
    title: str | None,
    full_text: str,
    sections: dict[str, str],
) -> tuple[bool, str | None, dict[str, Any]]:
    detail = _page_diagnostics(title, full_text, sections)
    if not title:
        return False, "missing_title", detail
    if detail["text_length"] <= 500:
        return False, "content_too_short", detail
    if detail["medical_signal_count"] < 2:
        return False, "not_enough_medical_signals", detail
    if detail["section_count"] < 2:
        return False, "too_few_sections", detail
    return True, None, detail


def parse_ingredient_page(
    url: str,
    session: requests.Session | None = None,
    timeout: int = 25,
) -> dict[str, Any]:
    try:
        soup = BeautifulSoup(
            fetch_html(url, timeout=timeout, session=session), "html.parser"
        )
        document_title = (
            clean_text(soup.title.get_text(" ", strip=True))
            if soup.title else None
        )
        remove_noise_tags(soup)
        main_content = extract_main_content(soup)
        title = extract_title(main_content) or extract_title(soup) or document_title
        full_text = clean_text(main_content.get_text(" ", strip=True))
        sections = parse_sections(main_content)
        valid, reason, detail = is_valid_ingredient_page(
            title, full_text, sections
        )
        if not valid:
            raise CrawlError(reason or "parse_error", detail=detail)
        return {
            "source": "trungtamthuoc",
            "entity_type": "ingredient",
            "name": title,
            "slug": slug_from_url(url),
            "url": url,
            "title": document_title or title,
            "updated_at": extract_updated_at(full_text),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "sections": sections,
        }
    except CrawlError:
        raise
    except Exception as exc:
        raise CrawlError("parse_error", str(exc)) from exc


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
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
            handle.write(serialized)
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_completed_urls(path: Path) -> set[str]:
    """Resume extension point; not used by the overwrite workflow yet."""
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("url"):
                completed.add(str(item["url"]))
    return completed


def _failure_record(url: str, error: CrawlError) -> dict[str, Any]:
    record: dict[str, Any] = {"url": url, "reason": error.reason}
    detail = dict(error.detail)
    if str(error) and str(error) != error.reason:
        detail.setdefault("message", str(error))
    if detail:
        record["detail"] = detail
    return record


def _validate_output_state(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def run_crawler(
    urls: list[str],
    *,
    session: requests.Session,
    timeout: int,
    delay_min: float,
    delay_max: float,
    raw_path: Path,
    failed_path: Path,
    overwrite: bool,
    sleeper: Callable[[float], None] = time.sleep,
    random_uniform: Callable[[float, float], float] = random.uniform,
) -> tuple[int, list[dict[str, Any]], Counter[str], Counter[str]]:
    temp_path = raw_path.with_name("ingredients_raw.tmp.jsonl")
    _validate_output_state((raw_path, failed_path, temp_path), overwrite)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_path.exists():
        temp_path.unlink()

    failures: list[dict[str, Any]] = []
    section_counts: Counter[str] = Counter()
    success_count = 0
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, url in enumerate(urls):
            if index:
                sleeper(random_uniform(delay_min, delay_max))
            try:
                record = parse_ingredient_page(
                    url, session=session, timeout=timeout
                )
            except CrawlError as error:
                failures.append(_failure_record(url, error))
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            success_count += 1
            section_counts.update(record["sections"].keys())
            print_record_preview(record)
    os.replace(temp_path, raw_path)

    save_json(failed_path, failures)
    failure_counts = Counter(item["reason"] for item in failures)
    return success_count, failures, section_counts, failure_counts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl Trung Tâm Thuốc ingredients"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--urls", nargs="+")
    parser.add_argument("--delay-min", type=float, default=1.0)
    parser.add_argument("--delay-max", type=float, default=2.5)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/trungtamthuoc_v2"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.delay_min < 0 or args.delay_max < args.delay_min:
        raise SystemExit("Require 0 <= --delay-min <= --delay-max")

    session = create_session()
    delay = lambda: time.sleep(random.uniform(args.delay_min, args.delay_max))
    if args.urls:
        normalized_urls = [normalize_url(url) for url in args.urls]
        invalid = [
            original
            for original, normalized in zip(args.urls, normalized_urls)
            if not is_ingredient_url(normalized)
        ]
        if invalid:
            raise SystemExit(
                f"Invalid ingredient URL(s): {', '.join(invalid)}"
            )
        discovered = sorted({url for url in normalized_urls if url})
    else:
        discovered = extract_ingredient_links(
            session=session, timeout=args.timeout, delay=delay
        )

    selected = discovered[: args.limit] if args.limit else discovered
    output_dir = args.output_dir
    urls_path = output_dir / "ingredient_urls.json"
    raw_path = output_dir / "ingredients_raw.jsonl"
    failed_path = output_dir / "failed_urls.json"
    temp_raw_path = output_dir / "ingredients_raw.tmp.jsonl"
    _validate_output_state(
        (urls_path, raw_path, failed_path, temp_raw_path),
        args.overwrite,
    )
    save_json(urls_path, discovered)
    success, failures, sections, failure_reasons = run_crawler(
        selected,
        session=session,
        timeout=args.timeout,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        raw_path=raw_path,
        failed_path=failed_path,
        overwrite=args.overwrite,
    )

    print(f"Tổng số URL tìm được: {len(discovered)}")
    print(f"Số URL thực sự crawl: {len(selected)}")
    print(f"Số thành công: {success}")
    print(f"Số thất bại: {len(failures)}")
    print(f"URL file: {urls_path}")
    print(f"Raw JSONL file: {raw_path}")
    print(f"Failed URL file: {failed_path}")
    print("Top sections:")
    for name, count in sections.most_common(10):
        print(f"  {name}: {count}")
    print("Failure reasons:")
    for reason, count in failure_reasons.most_common():
        print(f"  {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
