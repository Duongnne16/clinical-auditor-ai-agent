"""Build normalized Long Châu product records from chunked JSON files.

All filesystem operations are explicit. Importing this module never reads input
data, writes output data, or initializes any external service.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PROCESSED_INPUT = Path("data/processed/longchau_chunked")
DEFAULT_RAW_INPUT = Path("data/raw/longchau_chunked")
DEFAULT_OUTPUT = Path("data/processed/longchau_drug_products.json")

DESCRIPTION_REVIEW_REASON = (
    "Ingredients extracted from describe without strength information"
)
NO_INGREDIENTS_REVIEW_REASON = "No ingredients found"
BRAND_REVIEW_REASON = "Brand name extracted with fallback"
PARTIAL_INGREDIENTS_REVIEW_REASON = "Ingredients parsed partially"
METADATA_REVIEW_REASON = "Inconsistent product metadata"

_INGREDIENT_MARKER = re.compile(r"thành\s+phần\s*:", re.IGNORECASE)
_DESCRIPTION_MARKER = re.compile(
    r"(?:có\s+|với\s+)?thành\s+phần\s+chính\s+là\s+",
    re.IGNORECASE,
)
_INGREDIENT_WITH_STRENGTH = re.compile(
    r"^\s*(?P<name>.+)\s*\((?P<strength>[^()]*)\)\s*$"
)
_STRENGTH_WITH_UNIT = re.compile(
    r"\d(?:[\d.,/\s-]*)(?:mcg|µg|mg|g|ml|iu|ui|%)\b|"
    r"\d(?:[\d.,/\s-]*)%",
    re.IGNORECASE,
)
_INVALID_INGREDIENT_NAMES = {"", "none", "null", "nan"}
_DOSAGE_OR_STRENGTH = re.compile(
    r"^\d+(?:[.,/]\d+)*(?:mg|g|mcg|µg|ml|iu|ui|%|mg/ml|mcg/ml)\b",
    re.IGNORECASE,
)
_ACTION_WORDS = {
    "điều",
    "hỗ",
    "giúp",
    "dùng",
    "bổ",
    "phòng",
    "ngăn",
    "cung",
    "làm",
    "trị",
    "giảm",
}
_GENERIC_LEADING_WORDS = {
    "bôi",
    "bột",
    "cao",
    "dầu",
    "dịch",
    "đạn",
    "đặt",
    "gel",
    "ho",
    "hít",
    "nhỏ",
    "phun",
    "súc",
    "tiêm",
    "tra",
    "truyền",
    "xịt",
}
_FORM_PREFIXES = (
    "dung dịch tiêm truyền tĩnh mạch",
    "dung dịch truyền tĩnh mạch",
    "dung dịch tiêm truyền",
    "thuốc đặt trực tràng",
    "hỗn dịch uống",
    "dung dịch uống",
    "dung dịch tiêm",
    "dung dịch truyền",
    "dung dịch súc miệng",
    "dung dịch vệ sinh phụ nữ",
    "thuốc nhỏ mắt",
    "thuốc nhỏ tai",
    "thuốc xịt mũi",
    "thuốc xịt",
    "viên nén",
    "viên nang",
    "viên sủi",
    "viên ngậm",
    "bột sủi",
    "siro",
    "thuốc",
)
_METADATA_FIELDS = ("name", "id", "url", "category", "type", "source")


def normalize_text(value: str) -> str:
    """Normalize non-breaking spaces and repeated whitespace."""
    return " ".join(str(value).replace("\xa0", " ").split())


def resolve_input_directory(
    processed_dir: Path = DEFAULT_PROCESSED_INPUT,
    raw_dir: Path = DEFAULT_RAW_INPUT,
) -> Path:
    """Choose processed chunks first, then raw chunks."""
    for candidate in (processed_dir, raw_dir):
        if candidate.is_dir() and any(candidate.rglob("*.json")):
            return candidate
    raise FileNotFoundError(
        f"No Long Châu JSON files found in {processed_dir} or {raw_dir}"
    )


def _split_outside_parentheses(text: str, separators: set[str]) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        elif depth == 0 and character in separators:
            part = normalize_text(text[start:index]).strip(" ,;")
            if part:
                parts.append(part)
            start = index + 1
    final = normalize_text(text[start:]).strip(" ,;")
    if final:
        parts.append(final)
    return parts


def _parse_ingredients_with_status(
    text: str,
) -> tuple[list[dict[str, str | None]], bool]:
    matches = list(_INGREDIENT_MARKER.finditer(text or ""))
    if not matches:
        return [], False

    content = text[matches[-1].end() :].strip()
    candidates = _split_outside_parentheses(content, {",", ";", "\n"})
    ingredients: list[dict[str, str | None]] = []
    parsed_all = bool(candidates)

    for candidate in candidates:
        match = _INGREDIENT_WITH_STRENGTH.match(candidate)
        if not match:
            parsed_all = False
            continue
        name = normalize_text(match.group("name")).strip(" .:-")
        strength = normalize_text(match.group("strength")).strip(" .:-")
        if not name or not strength:
            parsed_all = False
            continue
        # With multiple parenthetical groups, only the final numeric dosage
        # group is strength. Earlier groups remain part of the ingredient name.
        if name.endswith(")") and not _has_numeric_strength_with_unit(strength):
            parsed_all = False
        ingredients.append({"name": name, "strength": strength})

    parsed_all = parsed_all and len(ingredients) == len(candidates)
    return repair_ingredients(ingredients), parsed_all


def parse_ingredients_text(text: str) -> list[dict[str, str | None]]:
    """Parse ``Name (strength)`` entries after the final ``Thành phần:``."""
    ingredients, _ = _parse_ingredients_with_status(text)
    return ingredients


def _has_numeric_value(value: Any) -> bool:
    return bool(re.search(r"\d", str(value or "")))


def _has_numeric_strength_with_unit(value: Any) -> bool:
    """Check accepted dosage units case-insensitively without changing value."""
    return bool(_STRENGTH_WITH_UNIT.search(str(value or "")))


def _is_invalid_ingredient_name(value: Any) -> bool:
    return normalize_text(value or "").casefold() in _INVALID_INGREDIENT_NAMES


def repair_ingredients(
    ingredients: list[dict[str, str | None]],
) -> list[dict[str, str | None]]:
    """Repair adjacent ``A (alias), None (strength)`` parser artifacts."""
    repaired: list[dict[str, str | None]] = []
    index = 0
    while index < len(ingredients):
        current = dict(ingredients[index])
        if index + 1 < len(ingredients):
            following = ingredients[index + 1]
            alias = current.get("strength")
            if (
                alias is not None
                and not _has_numeric_value(alias)
                and _is_invalid_ingredient_name(following.get("name"))
                and _has_numeric_strength_with_unit(following.get("strength"))
            ):
                current_name = normalize_text(current.get("name") or "")
                current["name"] = f"{current_name} ({normalize_text(alias)})"
                current["strength"] = following.get("strength")
                repaired.append(current)
                index += 2
                continue
        repaired.append(current)
        index += 1
    return repaired


def validate_ingredients(
    ingredients: list[dict[str, str | None]],
) -> list[str]:
    """Return stable, deduplicated quality warnings without dropping data."""
    reasons: list[str] = []
    seen_reasons: set[str] = set()
    seen_ingredients: set[tuple[str, str]] = set()

    def add_reason(reason: str) -> None:
        if reason not in seen_reasons:
            seen_reasons.add(reason)
            reasons.append(reason)

    for ingredient in ingredients:
        raw_name = ingredient.get("name")
        name = normalize_text(raw_name or "")
        strength = ingredient.get("strength")

        if _is_invalid_ingredient_name(raw_name):
            display_name = name or str(raw_name)
            add_reason(f"Suspicious ingredient parsed: name={display_name}")
        elif not any(character.isalnum() for character in name):
            add_reason("Suspicious ingredient name")

        if strength is not None and not _has_numeric_value(strength):
            add_reason("Suspicious strength without numeric value")

        normalized_pair = (
            name.casefold(),
            normalize_text(strength or "").casefold(),
        )
        if normalized_pair in seen_ingredients:
            add_reason(f"Duplicate ingredient parsed: name={name}")
        else:
            seen_ingredients.add(normalized_pair)

    return reasons


def _join_review_reasons(
    existing: str | None,
    additional: Iterable[str],
) -> str | None:
    reasons = [part.strip() for part in (existing or "").split(";") if part.strip()]
    for reason in additional:
        if reason and reason not in reasons:
            reasons.append(reason)
    return "; ".join(reasons) or None


def parse_ingredients_from_description(
    text: str,
) -> list[dict[str, str | None]]:
    """Extract ingredient names from a descriptive sentence."""
    match = _DESCRIPTION_MARKER.search(text or "")
    if not match:
        return []

    content = text[match.end() :]
    content = re.split(r"[.!?\n]", content, maxsplit=1)[0]
    content = re.split(
        r"\s+(?:được|dùng|sử dụng|là thuốc|có tác dụng)\s+",
        content,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    content = re.sub(r"\s+và\s+", ", ", content, flags=re.IGNORECASE)

    ingredients = []
    for candidate in _split_outside_parentheses(content, {",", ";"}):
        name = normalize_text(candidate).strip(" .:-")
        if name:
            ingredients.append({"name": name, "strength": None})
    return ingredients


def extract_brand_name(full_name: str) -> tuple[str, bool]:
    """Return ``(brand_name, used_fallback)`` without raising on odd names."""
    normalized = normalize_text(full_name)
    if not normalized:
        return "Unknown", True

    candidate = normalized
    removed_prefix = False
    lowered = candidate.casefold()
    for prefix in _FORM_PREFIXES:
        prefix_folded = prefix.casefold()
        if lowered == prefix_folded or lowered.startswith(prefix_folded + " "):
            candidate = candidate[len(prefix) :].strip(" :-")
            removed_prefix = True
            break

    tokens = candidate.split()
    if removed_prefix and tokens:
        first = tokens[0].strip(" ,.;:()[]")
        if (
            first
            and not _DOSAGE_OR_STRENGTH.match(first)
            and first.casefold() not in _GENERIC_LEADING_WORDS
        ):
            return first, False

    # Safe approximation: retain the meaningful leading portion, never fail.
    fallback_tokens: list[str] = []
    for token in tokens:
        clean_token = token.strip(" ,.;:()[]")
        if not clean_token:
            continue
        if _DOSAGE_OR_STRENGTH.match(clean_token):
            break
        if fallback_tokens and clean_token.casefold() in _ACTION_WORDS:
            break
        fallback_tokens.append(clean_token)
        if len(fallback_tokens) == 4:
            break

    fallback = " ".join(fallback_tokens) or normalized
    return fallback, True


def _first_metadata_value(chunks: list[dict[str, Any]], field: str) -> Any:
    for chunk in chunks:
        metadata = chunk.get("metadata")
        if isinstance(metadata, dict) and metadata.get(field) not in (None, ""):
            return metadata[field]
    return None


def _has_inconsistent_metadata(chunks: list[dict[str, Any]]) -> bool:
    for field in _METADATA_FIELDS:
        values = {
            str(chunk.get("metadata", {}).get(field))
            for chunk in chunks
            if isinstance(chunk.get("metadata"), dict)
            and chunk["metadata"].get(field) not in (None, "")
        }
        if len(values) > 1:
            return True
    return False


def build_product(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one normalized product from all chunks in a source file."""
    if not chunks:
        raise ValueError("Product chunk list must not be empty")
    if not all(isinstance(chunk, dict) for chunk in chunks):
        raise ValueError("Every product chunk must be a JSON object")

    full_name = normalize_text(_first_metadata_value(chunks, "name") or "")
    brand_name, brand_used_fallback = extract_brand_name(full_name)
    metadata_inconsistent = _has_inconsistent_metadata(chunks)

    ingredient_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("metadata", {}).get("field") == "ingredients"
    ]
    ingredients: list[dict[str, str | None]] = []
    ingredient_source: str | None = None
    needs_review = False
    review_reason: str | None = None

    if ingredient_chunks:
        ingredient_text = "\n".join(
            str(chunk.get("text", ""))
            for chunk in sorted(
                ingredient_chunks,
                key=lambda item: item.get("metadata", {}).get("chunk_index", 0),
            )
        )
        ingredients, parsed_all = _parse_ingredients_with_status(ingredient_text)
        if ingredients:
            ingredient_source = "ingredients"
            if not parsed_all:
                needs_review = True
                review_reason = PARTIAL_INGREDIENTS_REVIEW_REASON

    if not ingredients:
        describe_chunks = [
            chunk
            for chunk in chunks
            if chunk.get("metadata", {}).get("field") == "describe"
        ]
        description = "\n".join(
            str(chunk.get("text", ""))
            for chunk in sorted(
                describe_chunks,
                key=lambda item: item.get("metadata", {}).get("chunk_index", 0),
            )
        )
        ingredients = parse_ingredients_from_description(description)
        if ingredients:
            ingredient_source = "describe"
            needs_review = True
            review_reason = DESCRIPTION_REVIEW_REASON
        else:
            ingredient_source = None
            needs_review = True
            review_reason = NO_INGREDIENTS_REVIEW_REASON

    # Preserve ingredient reasons. Metadata is used when no ingredient issue
    # exists, and validation warnings are always appended rather than replacing
    # an existing reason.
    if not needs_review and metadata_inconsistent:
        needs_review = True
        review_reason = METADATA_REVIEW_REASON

    validation_reasons = validate_ingredients(ingredients)
    if validation_reasons:
        needs_review = True
        review_reason = _join_review_reasons(review_reason, validation_reasons)

    # Brand fallback is the lowest-priority review reason.
    if not needs_review and brand_used_fallback:
        needs_review = True
        review_reason = BRAND_REVIEW_REASON

    product_id = _first_metadata_value(chunks, "id")
    return {
        "product_id": str(product_id) if product_id is not None else "",
        "full_name": full_name,
        "brand_name": brand_name,
        "source_url": _first_metadata_value(chunks, "url"),
        "category": _first_metadata_value(chunks, "category"),
        "product_type": _first_metadata_value(chunks, "type"),
        "source": _first_metadata_value(chunks, "source"),
        "ingredients": ingredients,
        "ingredient_source": ingredient_source,
        "needs_review": needs_review,
        "review_reason": review_reason,
    }


def build_drug_products(input_dir: Path) -> list[dict[str, Any]]:
    """Read every product JSON recursively and return stable ordered output."""
    products: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.json"), key=lambda item: str(item).casefold()):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Failed to read product JSON {path}: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Product JSON root must be a list: {path}")
        try:
            products.append(build_product(payload))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Failed to build product from {path}: {exc}") from exc
    return products


def write_drug_products(
    products: Iterable[dict[str, Any]],
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    """Atomically replace the output JSON after complete serialization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(list(products), ensure_ascii=False, indent=2) + "\n"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(serialized)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_drug_mapping(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Backward-compatible simple mapping from brand name to product ID."""
    return {
        str(record["brand_name"]): str(record["product_id"])
        for record in records
        if record.get("brand_name") and record.get("product_id")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Long Châu drug products")
    parser.add_argument("--input", type=Path, help="Override chunked input directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_dir = args.input or resolve_input_directory()
    products = build_drug_products(input_dir)
    write_drug_products(products, args.output)
    print(f"Wrote {len(products)} products to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
