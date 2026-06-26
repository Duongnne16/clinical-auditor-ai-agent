from __future__ import annotations

import re
import unicodedata
from typing import Any


NUMBERED_MEDICATION_RE = re.compile(r"^\s*\d+\s*[.)-]\s+\S+")


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", str(value or "").casefold())
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    without_marks = without_marks.replace("đ", "d")
    without_marks = re.sub(r"\s+", " ", without_marks)
    return without_marks.strip()


def _label_key(label: str) -> str:
    folded = _fold_text(label)
    folded = folded.replace("/", " ")
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    return folded.strip()


def _split_key_value(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    label, value = line.split(":", 1)
    label = _normalize_space(label)
    value = _normalize_space(value)
    if not label:
        return None
    return label, value


def _parse_age(value: str) -> int | None:
    match = re.search(r"\d+", value or "")
    if not match:
        return None
    return int(match.group(0))


def _parse_sex(value: str) -> str:
    folded = _fold_text(value)
    if folded in {"nam", "male", "m"}:
        return "male"
    if folded in {"nu", "female", "f"}:
        return "female"
    return "unknown"


def _split_diagnoses(value: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"[/,;\n]+", value or "")
        if item.strip()
    ]


def _is_structured_medication_heading(line: str) -> bool:
    folded = _fold_text(line).rstrip(":")
    return bool(
        re.fullmatch(r"(?:iii\s*\.\s*)?chi dinh dung thuoc", folded)
    )


def _is_section_heading(line: str) -> bool:
    folded = _fold_text(line).rstrip(":")
    return bool(
        re.fullmatch(
            r"(?:i|ii|iii|iv|v)\s*\.\s*[a-z0-9 ].*",
            folded,
        )
    )


def _is_footer_or_request_line(line: str) -> bool:
    folded = _fold_text(line).strip(" .")
    if not folded:
        return False
    compact = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    if re.fullmatch(r"ngay(?: \d+)? thang(?: \d+)? nam(?: \d+)?", compact):
        return True
    if compact.startswith("bac si"):
        return True
    if compact in {"ky ten", "chu ky"}:
        return True
    request_phrases = {
        "hay kiem tra don thuoc nay",
        "kiem tra don thuoc nay",
        "nho kiem tra don thuoc nay",
    }
    return folded in request_phrases


def _fallback_marker_info(line: str) -> tuple[bool, bool, str]:
    folded = _fold_text(line)
    colon_match = re.match(r"^\s*(don thuoc|toa thuoc)\s*:\s*(.*)$", folded)
    if colon_match:
        original_remainder = line.split(":", 1)[1].strip()
        return True, True, original_remainder
    if re.fullmatch(r"\s*(don thuoc|toa thuoc)\s*", folded):
        return True, False, ""
    return False, False, ""


def _has_numbered_line_soon(lines: list[str], start_index: int, limit: int = 3) -> bool:
    checked = 0
    for line in lines[start_index + 1 :]:
        if not line.strip():
            continue
        checked += 1
        if NUMBERED_MEDICATION_RE.match(line):
            return True
        if checked >= limit:
            return False
    return False


class PrescriptionDocumentParser:
    """Rule-based parser for canonical structured Vietnamese prescriptions."""

    FIELD_MAP = {
        "benh vien": "hospital",
        "khoa": "department",
        "ho va ten": "patient_name",
        "ho ten": "patient_name",
        "ten benh nhan": "patient_name",
        "benh nhan": "patient_name",
        "tuoi": "age",
        "nam nu": "sex",
        "gioi tinh": "sex",
        "can nang": "weight",
        "dia chi": "address",
        "chan doan": "diagnoses",
        "di ung thuoc": "allergies",
        "di ung": "allergies",
        "benh nen": "comorbidities",
        "chuc nang gan": "hepatic_function",
        "chuc nang than": "renal_function",
        "thai ky cho con bu": "pregnancy_lactation",
        "thuoc khac dang dung": "current_medications",
        "thuoc dang dung truoc do": "current_medications",
    }

    def parse(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return self._not_applied()

        lines = text.splitlines()
        section_index, first_medication_line = self._medication_section_start(lines)
        if section_index is None:
            return self._not_applied()

        patient_context = self._parse_patient_context(lines[:section_index])
        prescription_text = self._extract_prescription_text(
            lines,
            section_index,
            first_medication_line=first_medication_line,
        )
        warnings: list[str] = []
        if not prescription_text:
            warnings.append("prescription_document_parser_no_medication_lines")

        return {
            "applied": True,
            "patient_context": patient_context,
            "prescription_text": prescription_text,
            "warnings": warnings,
        }

    @staticmethod
    def _not_applied() -> dict[str, Any]:
        return {
            "applied": False,
            "patient_context": {},
            "prescription_text": "",
            "warnings": [],
        }

    def _medication_section_start(
        self, lines: list[str]
    ) -> tuple[int | None, str | None]:
        for index, line in enumerate(lines):
            if _is_structured_medication_heading(line):
                return index, None

        for index, line in enumerate(lines):
            has_marker, has_colon, remainder = _fallback_marker_info(line)
            if has_marker and has_colon:
                return index, remainder or None
            if has_marker and _has_numbered_line_soon(lines, index):
                return index, None

        return None, None

    def _parse_patient_context(self, lines: list[str]) -> dict[str, Any]:
        context: dict[str, Any] = {}
        for line in lines:
            split = _split_key_value(line)
            if split is None:
                continue
            label, value = split
            field = self.FIELD_MAP.get(_label_key(label))
            if field is None:
                continue
            if field == "age":
                context[field] = _parse_age(value)
            elif field == "sex":
                context[field] = _parse_sex(value)
            elif field == "diagnoses":
                context[field] = _split_diagnoses(value)
            else:
                context[field] = value
        return context

    def _extract_prescription_text(
        self,
        lines: list[str],
        section_index: int,
        *,
        first_medication_line: str | None = None,
    ) -> str:
        output: list[str] = []
        seen_numbered_medication = False

        if first_medication_line:
            output.append(first_medication_line)
            seen_numbered_medication = bool(
                NUMBERED_MEDICATION_RE.match(first_medication_line)
            )

        for line in lines[section_index + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            if _is_footer_or_request_line(stripped):
                break
            if _is_section_heading(stripped) and seen_numbered_medication:
                break

            if NUMBERED_MEDICATION_RE.match(stripped):
                output.append(stripped)
                seen_numbered_medication = True
                continue

            if seen_numbered_medication:
                output.append(line.rstrip())

        if not seen_numbered_medication:
            return ""
        return "\n".join(output).strip()
