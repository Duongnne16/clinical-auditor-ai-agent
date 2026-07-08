"""Resolve product ingredient names to Trung Tâm Thuốc evidence records."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:  # pragma: no cover - tested via monkeypatch.
    rapidfuzz_fuzz = None


DEFAULT_CATALOG_PATH = Path(
    "data/processed/evidence_ingredients_v2/evidence_ingredient_catalog.jsonl"
)
MANUAL_ALIASES: dict[str, tuple[str, ...]] = {
    "acyclovir": ("aciclovir",),
    "amlodipine": ("amlodipin",),
    "amiodarone": ("amiodaron",),
    "epinephrine": ("adrenalin",),
    "adrenaline": ("adrenalin",),
    "hydrochlorothiazide": ("hydrochlorothiazid", "hydroclorothiazid"),
    "metronidazole": ("metronidazol",),
    "cetirizine": ("cetirizin", "cetirizine"),
    "caffeine": ("cafein",),
    "clavulanic acid": ("acid clavulanic",),
    "folic acid": ("acid folic",),
    "ascorbic acid": ("acid ascorbic",),
    "sodium chloride": ("natri clorid",),
    "potassium chloride": ("kali clorid",),
    "sodium bicarbonate": ("natri bicarbonat",),
    "potassium bicarbonate": ("kali bicarbonat",),
}
PROTECTED_COMPOUNDS = {
    "natri clorid",
    "sodium chloride",
    "kali clorid",
    "potassium chloride",
    "natri bicarbonat",
    "sodium bicarbonate",
    "kali bicarbonat",
    "potassium bicarbonate",
}
SALT_PREFIX_TOKENS = {"natri", "sodium", "kali", "potassium"}
SALT_SUFFIX_TOKENS = {
    "hcl",
    "hydrochloride",
    "hydroclorid",
    "dihydrochloride",
    "dihydroclorid",
    "sodium",
    "natri",
    "potassium",
    "kali",
    "calcium",
    "calci",
    "magnesium",
    "magnesi",
    "besylate",
    "besilate",
    "maleate",
    "fumarate",
    "citrate",
    "phosphate",
    "phosphat",
    "sulfate",
    "sulphate",
    "sulfat",
    "bisulfate",
    "succinate",
    "tartrate",
    "mesylate",
    "tosylate",
    "acetate",
    "propionate",
    "dipropionate",
    "valerate",
    "palmitate",
    "stearate",
    "nitrate",
    "monohydrate",
    "hydrate",
    "hemihydrate",
}
IMPORTANT_SECTIONS = {
    "chong_chi_dinh",
    "than_trong",
    "tuong_tac_thuoc",
    "thai_ky_cho_con_bu",
    "tac_dung_khong_mong_muon",
    "qua_lieu_va_xu_tri",
    "lieu_luong_va_cach_dung",
}
GENERIC_FUZZY_TOKENS = {
    "acid",
    "sodium",
    "natri",
    "potassium",
    "kali",
    "calcium",
    "calci",
    "magnesium",
    "magnesi",
    "vitamin",
    *SALT_SUFFIX_TOKENS,
}
MATCH_PRIORITY = {
    "exact": 5,
    "manual_alias": 4,
    "salt_stripped_exact": 3,
    "prefix_stripped_exact": 2,
    "fuzzy": 1,
}
STEREO_PREFIX_RE = re.compile(r"^(?:dl|l|d|n)-\s*", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize text conservatively for ingredient matching."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("®", "").replace("™", "")
    value = re.sub(r"[\u2010-\u2015\u2212]", "-", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def strip_vietnamese_diacritics(text: str) -> str:
    """Remove combining marks and normalize Vietnamese đ."""
    value = unicodedata.normalize("NFD", normalize_text(text))
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Mn"
    )
    return value.replace("đ", "d")


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _variants(text: str) -> list[str]:
    normalized = normalize_text(text)
    return _deduplicate(
        (normalized, strip_vietnamese_diacritics(normalized))
    )


def strip_salt_or_form(name: str) -> list[str]:
    """Return safe textual salt/form candidates without catalog assumptions."""
    normalized = normalize_text(name)
    if normalized in {
        normalize_text(value) for value in PROTECTED_COMPOUNDS
    }:
        return []
    tokens = normalized.split()
    candidates: list[str] = []
    if len(tokens) > 1 and tokens[0] in SALT_PREFIX_TOKENS:
        candidate = " ".join(tokens[1:]).strip()
        if len(candidate) >= 4:
            candidates.append(candidate)
    while len(tokens) > 1 and tokens[-1] in SALT_SUFFIX_TOKENS:
        tokens = tokens[:-1]
        candidate = " ".join(tokens).strip()
        if len(candidate) >= 4:
            candidates.append(candidate)
    return _deduplicate(candidates)


class IngredientEvidenceResolver:
    """Load an evidence catalog once and resolve ingredient names."""

    def __init__(
        self,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        enable_fuzzy: bool = True,
        fuzzy_threshold: int = 88,
    ) -> None:
        if not 0 <= fuzzy_threshold <= 100:
            raise ValueError("fuzzy_threshold must be between 0 and 100")
        self.catalog_path = Path(catalog_path)
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_backend = (
            "rapidfuzz" if rapidfuzz_fuzz is not None else "difflib"
        )
        self.records: list[dict[str, Any]] = []
        self.invalid_line_count = 0
        self.invalid_record_count = 0
        self.invalid_samples: list[dict[str, Any]] = []
        self.exact_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.no_diacritics_index: dict[
            str, list[dict[str, Any]]
        ] = defaultdict(list)
        self.fuzzy_choices: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.manual_alias_index: dict[
            str, list[dict[str, Any]]
        ] = defaultdict(list)
        self.manual_alias_count = 0
        self._load()
        self._build_manual_alias_index()

    def _record_invalid(
        self, line_number: int, reason: str, detail: str = ""
    ) -> None:
        if len(self.invalid_samples) < 20:
            self.invalid_samples.append(
                {
                    "line_number": line_number,
                    "reason": reason,
                    "detail": detail,
                }
            )

    def _load(self) -> None:
        if not self.catalog_path.is_file():
            raise FileNotFoundError(
                f"Evidence catalog does not exist: {self.catalog_path}"
            )
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.invalid_line_count += 1
                    self._record_invalid(
                        line_number, "invalid_json", str(exc)
                    )
                    continue
                if not isinstance(payload, dict):
                    self.invalid_record_count += 1
                    self._record_invalid(
                        line_number, "non_object_record"
                    )
                    continue
                record = self._sanitize_record(payload)
                if record is None:
                    self.invalid_record_count += 1
                    self._record_invalid(
                        line_number, "missing_catalog_identity"
                    )
                    continue
                self.records.append(record)
                self._index_record(record)

    def _sanitize_record(
        self, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        catalog_id = str(payload.get("catalog_id") or "")
        slug = normalize_text(str(payload.get("slug") or "")).replace(
            " ", "-"
        )
        entity_name = str(payload.get("entity_name") or "")
        aliases = payload.get("aliases")
        aliases = aliases if isinstance(aliases, list) else []
        if not catalog_id or not slug or (
            not entity_name and not aliases
        ):
            return None
        sections = payload.get("sections")
        warnings = payload.get("warnings")
        chunk_count = payload.get("chunk_count")
        try:
            chunk_count = int(chunk_count or 0)
        except (TypeError, ValueError):
            chunk_count = 0
        return {
            **payload,
            "catalog_id": catalog_id,
            "slug": slug,
            "entity_name": entity_name or slug.replace("-", " "),
            "aliases": [
                str(alias) for alias in aliases if str(alias).strip()
            ],
            "sections": [
                str(section)
                for section in sections
                if str(section).strip()
            ] if isinstance(sections, list) else [],
            "chunk_count": max(chunk_count, 0),
            "warnings": [
                str(warning) for warning in warnings
            ] if isinstance(warnings, list) else [],
        }

    @staticmethod
    def _append_index(
        index: dict[str, list[dict[str, Any]]],
        key: str,
        record: dict[str, Any],
    ) -> None:
        if key and all(
            existing["catalog_id"] != record["catalog_id"]
            for existing in index[key]
        ):
            index[key].append(record)

    def _index_record(self, record: dict[str, Any]) -> None:
        keys = [
            record["entity_name"],
            str(record.get("normalized_name") or ""),
            str(record.get("normalized_name_no_diacritics") or ""),
            record["slug"],
            record["slug"].replace("-", " "),
            *record["aliases"],
        ]
        for key in keys:
            normalized = normalize_text(key)
            no_diacritics = strip_vietnamese_diacritics(key)
            self._append_index(self.exact_index, normalized, record)
            self._append_index(
                self.no_diacritics_index, no_diacritics, record
            )
            self._append_index(self.fuzzy_choices, normalized, record)
            self._append_index(
                self.fuzzy_choices, no_diacritics, record
            )

    def _exact_records(self, name: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for variant in _variants(name):
            records.extend(self.exact_index.get(variant, []))
            records.extend(self.no_diacritics_index.get(variant, []))
        return self._unique_records(records)

    def _build_manual_alias_index(self) -> None:
        for source, targets in MANUAL_ALIASES.items():
            target_records: list[dict[str, Any]] = []
            for target in targets:
                target_records.extend(self._exact_records(target))
            target_records = [
                record
                for record in self._unique_records(target_records)
                if record["chunk_count"] > 0
            ]
            if not target_records:
                continue
            for source_variant in _variants(source):
                for record in target_records:
                    self._append_index(
                        self.manual_alias_index,
                        source_variant,
                        record,
                    )
            self.manual_alias_count += 1

    @staticmethod
    def _unique_records(
        records: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        unique: dict[str, dict[str, Any]] = {}
        for record in records:
            unique[record["catalog_id"]] = record
        return list(unique.values())

    def _manual_records(self, name: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for variant in _variants(name):
            records.extend(self.manual_alias_index.get(variant, []))
        return self._unique_records(records)

    def _exact_or_manual_records(
        self, name: str
    ) -> list[dict[str, Any]]:
        records = self._exact_records(name)
        return records or self._manual_records(name)

    def _salt_stripped_records(
        self, name: str
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for candidate in strip_salt_or_form(name):
            records.extend(self._exact_or_manual_records(candidate))
        return [
            record
            for record in self._unique_records(records)
            if record["chunk_count"] > 0
        ]

    def _prefix_stripped_records(
        self, name: str
    ) -> list[dict[str, Any]]:
        normalized = normalize_text(name)
        stripped = STEREO_PREFIX_RE.sub("", normalized).strip()
        if not stripped or stripped == normalized:
            return []
        return [
            record
            for record in self._exact_or_manual_records(stripped)
            if record["chunk_count"] > 0
        ]

    def _fuzzy_score(self, query: str, choice: str) -> float:
        if rapidfuzz_fuzz is not None:
            return float(rapidfuzz_fuzz.WRatio(query, choice)) / 100.0
        return SequenceMatcher(None, query, choice).ratio()

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9µ]+", strip_vietnamese_diacritics(text))
            if len(token) >= 3 and token not in GENERIC_FUZZY_TOKENS
        }

    @classmethod
    def _fuzzy_pair_allowed(cls, query: str, choice: str) -> bool:
        normalized_query = strip_vietnamese_diacritics(query)
        if len(normalized_query.replace(" ", "")) < 4:
            return False
        query_tokens = cls._meaningful_tokens(query)
        choice_tokens = cls._meaningful_tokens(choice)
        if not query_tokens:
            return False
        if len(query_tokens) == 1:
            query_token = next(iter(query_tokens))
            return any(
                SequenceMatcher(None, query_token, token).ratio() >= 0.65
                for token in choice_tokens
            )
        return bool(query_tokens & choice_tokens)

    def _fuzzy_records(
        self, name: str
    ) -> list[tuple[dict[str, Any], float]]:
        threshold = self.fuzzy_threshold / 100.0
        best: dict[str, tuple[dict[str, Any], float]] = {}
        for query in _variants(name):
            for choice, records in self.fuzzy_choices.items():
                if not self._fuzzy_pair_allowed(query, choice):
                    continue
                score = self._fuzzy_score(query, choice)
                if score < threshold:
                    continue
                for record in records:
                    current = best.get(record["catalog_id"])
                    if current is None or score > current[1]:
                        best[record["catalog_id"]] = (record, score)
        return list(best.values())

    def _candidate(
        self,
        record: dict[str, Any],
        match_type: str,
        score: float,
    ) -> dict[str, Any]:
        confidence = (
            "medium"
            if match_type == "fuzzy" and score >= 0.93
            else "low"
            if match_type == "fuzzy"
            else "high"
        )
        return {
            "catalog_id": record["catalog_id"],
            "evidence_name": record["entity_name"],
            "evidence_slug": record["slug"],
            "match_type": match_type,
            "score": round(float(score), 4),
            "confidence": confidence,
            "url": record.get("url"),
            "sections": list(record["sections"]),
            "chunk_count": record["chunk_count"],
            "warnings": list(record["warnings"]),
        }

    @staticmethod
    def _sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        important_count = len(
            set(candidate["sections"]) & IMPORTANT_SECTIONS
        )
        slug = candidate["evidence_slug"]
        noisy_slug = int(bool(re.search(r"-\d+$", slug)))
        return (
            -MATCH_PRIORITY[candidate["match_type"]],
            -candidate["score"],
            -candidate["chunk_count"],
            -important_count,
            noisy_slug,
            len(slug),
            candidate["catalog_id"],
        )

    def resolve(
        self, ingredient_name: str, top_k: int = 5
    ) -> dict[str, Any]:
        """Resolve one ingredient name to ranked evidence candidates."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        input_name = str(ingredient_name or "")
        normalized_input = strip_vietnamese_diacritics(input_name)
        if not normalize_text(input_name):
            return self._unresolved(
                input_name,
                normalized_input,
                ["empty_input", "no_evidence_ingredient_found"],
            )

        candidate_records: list[tuple[dict[str, Any], str, float]] = []
        manual = self._manual_records(input_name)
        if manual:
            candidate_records = [
                (record, "manual_alias", 0.99)
                for record in manual
            ]
        else:
            exact = self._exact_records(input_name)
            if exact:
                candidate_records = [
                    (record, "exact", 1.0) for record in exact
                ]
        if not candidate_records:
            salt = self._salt_stripped_records(input_name)
            if salt:
                candidate_records = [
                    (record, "salt_stripped_exact", 0.97)
                    for record in salt
                ]
            else:
                prefix = self._prefix_stripped_records(input_name)
                if prefix:
                    candidate_records = [
                        (record, "prefix_stripped_exact", 0.94)
                        for record in prefix
                    ]
                elif self.enable_fuzzy:
                    candidate_records = [
                        (record, "fuzzy", score)
                        for record, score in self._fuzzy_records(
                            input_name
                        )
                    ]

        if not candidate_records:
            return self._unresolved(
                input_name,
                normalized_input,
                ["no_evidence_ingredient_found"],
            )
        candidates = [
            self._candidate(record, match_type, score)
            for record, match_type, score in candidate_records
        ]
        candidates.sort(key=self._sort_key)
        candidates = candidates[:top_k]
        best_match = candidates[0]
        requires_review = (
            best_match["match_type"] in {
                "fuzzy", "prefix_stripped_exact"
            }
            or best_match["confidence"] == "low"
            or bool(best_match["warnings"])
        )
        warnings = (
            ["evidence_resolution_requires_review"]
            if requires_review
            else []
        )
        return {
            "input_ingredient": input_name,
            "normalized_input": normalized_input,
            "status": "resolved",
            "best_match": best_match,
            "candidates": candidates,
            "requires_review": requires_review,
            "warnings": warnings,
        }

    @staticmethod
    def _unresolved(
        input_name: str,
        normalized_input: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        return {
            "input_ingredient": input_name,
            "normalized_input": normalized_input,
            "status": "unresolved",
            "best_match": None,
            "candidates": [],
            "requires_review": True,
            "warnings": warnings,
        }

    def resolve_many(
        self, ingredient_names: list[str], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Resolve multiple ingredients while preserving input order."""
        return [
            self.resolve(name, top_k=top_k) for name in ingredient_names
        ]

    def get_stats(self) -> dict[str, Any]:
        """Return catalog load and index statistics."""
        return {
            "catalog_path": str(self.catalog_path),
            "records_loaded": len(self.records),
            "invalid_line_count": self.invalid_line_count,
            "invalid_record_count": self.invalid_record_count,
            "invalid_samples": list(self.invalid_samples),
            "exact_index_count": len(self.exact_index),
            "alias_count": len(
                set(self.exact_index) | set(self.no_diacritics_index)
            ),
            "fuzzy_choice_count": len(self.fuzzy_choices),
            "enable_fuzzy": self.enable_fuzzy,
            "fuzzy_threshold": self.fuzzy_threshold,
            "fuzzy_backend": self.fuzzy_backend,
            "manual_alias_count": self.manual_alias_count,
        }
