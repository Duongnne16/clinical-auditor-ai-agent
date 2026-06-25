"""Deterministic parser for one prescription medication line."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


ORDER_RE = re.compile(r"^\s*(?P<index>\d+)\s*(?:[.)-])\s*")
QUANTITY_RE = re.compile(
    r"\s+[x×]\s*(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>[^\d\s]+)?\s*$",
    re.IGNORECASE,
)
PAREN_RE = re.compile(r"\(([^()]*)\)")
STRENGTH_START_RE = re.compile(
    r"(?P<strength>"
    r"\(\s*\d+(?:[.,]\d+)?[^)]*\)\s*(?:mg/g|g/ml|%|/[\d.,]*\s*ml)"
    r"|"
    r"\d+(?:[.,]\d+)?\s*(?:mui|iu|ui|mcg|µg|mg|g|ml|%)"
    r"(?:\s*\+\s*\d+(?:[.,]\d+)?\s*(?:mui|iu|ui|mcg|µg|mg|g|ml|%))*"
    r")\s*$",
    re.IGNORECASE,
)
SIMPLE_STRENGTH_RE = re.compile(
    r"^\s*(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>mui|iu|ui|mcg|µg|mg|g|ml|%)"
    r"(?P<denominator>/.*)?\s*$",
    re.IGNORECASE,
)


def _normalize_space(text: str) -> str:
    value = unicodedata.normalize("NFC", str(text or ""))
    return re.sub(r"\s+", " ", value).strip()


def _number(value: str) -> int | float:
    parsed = float(value.replace(",", "."))
    return int(parsed) if parsed.is_integer() else parsed


def _looks_like_dosage_parenthesis(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not re.search(r"\d", compact):
        return False
    return bool(
        re.fullmatch(
            r"[\d.,+/%\s]*(?:mui|iu|ui|mcg|µg|mg|g|ml)?",
            compact,
            flags=re.IGNORECASE,
        )
    )


def _parse_strength(raw: str) -> dict[str, Any] | None:
    match = SIMPLE_STRENGTH_RE.match(raw)
    if not match:
        return None
    unit = match.group("unit")
    normalized_unit = (
        "MUI"
        if unit.lower() == "mui"
        else "IU"
        if unit.lower() in {"iu", "ui"}
        else unit.lower()
    )
    return {
        "strength_raw": _normalize_space(raw),
        "strength_value": float(match.group("value").replace(",", ".")),
        "strength_unit": normalized_unit,
    }


def _strength_parts(strength_text: str) -> list[dict[str, Any]]:
    text = _normalize_space(strength_text)
    complex_match = re.fullmatch(
        r"\((?P<inside>[^)]+)\)(?P<suffix>.*)", text
    )
    raw_parts: list[str]
    if complex_match and "+" in complex_match.group("inside"):
        inside = complex_match.group("inside")
        suffix = complex_match.group("suffix").replace(" ", "")
        parts = [part.strip() for part in inside.split("+")]
        if suffix.lower() in {"mg/g", "g/ml"}:
            raw_parts = [f"{part}{suffix}" for part in parts]
        elif suffix.startswith("/"):
            raw_parts = [f"{part}{suffix}" for part in parts]
        else:
            raw_parts = [f"{part}{suffix}" for part in parts]
    else:
        raw_parts = [
            part.strip() for part in re.split(r"\s*\+\s*", text)
        ]

    parsed: list[dict[str, Any]] = []
    for raw_part in raw_parts:
        item = _parse_strength(raw_part)
        if item is None:
            return []
        parsed.append(item)
    return parsed


class MedicationLineParser:
    """Parse one structured medication line without OCR or LLM calls."""

    def parse_line(self, line: str) -> dict[str, Any]:
        if not isinstance(line, str):
            raise TypeError("line must be a string")

        raw_line = line
        working = _normalize_space(line)
        warnings: list[str] = []
        order_index: int | None = None

        order_match = ORDER_RE.match(working)
        if order_match:
            order_index = int(order_match.group("index"))
            working = working[order_match.end():].strip()

        quantity: dict[str, Any] | None = None
        quantity_match = QUANTITY_RE.search(working)
        if quantity_match:
            quantity = {
                "value": _number(quantity_match.group("value")),
                "unit": quantity_match.group("unit") or "",
            }
            working = working[:quantity_match.start()].strip()

        brand_text: str | None = None
        brand_span: tuple[int, int] | None = None
        for match in PAREN_RE.finditer(working):
            content = _normalize_space(match.group(1))
            if content and not _looks_like_dosage_parenthesis(content):
                brand_text = content
                brand_span = match.span()
                break

        without_brand = working
        if brand_span:
            without_brand = (
                working[:brand_span[0]] + " " + working[brand_span[1]:]
            )
            without_brand = _normalize_space(without_brand)

        strength_match = STRENGTH_START_RE.search(without_brand)
        if strength_match:
            strength_text = _normalize_space(
                strength_match.group("strength")
            )
            generic_text = _normalize_space(
                without_brand[:strength_match.start()]
            )
        else:
            strength_text = None
            generic_text = _normalize_space(without_brand)

        ingredient_names = [
            _normalize_space(part)
            for part in re.split(r"\s*\+\s*", generic_text)
            if _normalize_space(part)
        ]
        parsed_strengths = (
            _strength_parts(strength_text) if strength_text else []
        )
        aligned = (
            bool(ingredient_names)
            and bool(parsed_strengths)
            and len(ingredient_names) == len(parsed_strengths)
        )
        if strength_text and ingredient_names and not aligned:
            warnings.append("strength_ingredient_alignment_uncertain")

        ingredients: list[dict[str, Any]] = []
        for index, name in enumerate(ingredient_names):
            strength = parsed_strengths[index] if aligned else {}
            ingredients.append(
                {
                    "name": name,
                    "strength_raw": strength.get("strength_raw"),
                    "strength_value": strength.get("strength_value"),
                    "strength_unit": strength.get("strength_unit"),
                }
            )

        parse_status = "parsed" if ingredient_names else "unparsed"
        if parse_status == "unparsed":
            warnings.append("generic_text_not_found")

        return {
            "raw_line": raw_line,
            "order_index": order_index,
            "generic_text": generic_text or None,
            "brand_text": brand_text,
            "strength_text": strength_text,
            "quantity": quantity,
            "ingredients": ingredients,
            "is_combination": len(ingredient_names) > 1,
            "parse_status": parse_status,
            "warnings": warnings,
        }

    def parse_many(self, lines: list[str]) -> list[dict[str, Any]]:
        """Parse medication lines while preserving input order."""
        return [self.parse_line(line) for line in lines]
