"""In-memory lookup service for Long Châu drug-product mappings."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:  # pragma: no cover - exercised through backend override test.
    rapidfuzz_fuzz = None


DEFAULT_MAPPING_PATH = Path(
    "data/processed/longchau_mapping/drug_mapping.jsonl"
)
MATCH_PRIORITY = {
    "exact_brand": 5,
    "exact_alias": 4,
    "ingredient_name": 3,
    "fuzzy_alias": 2,
    "fuzzy_brand": 1,
}
CONFIDENCE_PRIORITY = {"high": 3, "medium": 2, "low": 1}
FORM_PREFIXES = (
    "dung dịch tiêm truyền tĩnh mạch",
    "dung dịch truyền tĩnh mạch",
    "dung dịch tiêm truyền",
    "thuốc nhỏ mắt",
    "thuốc nhỏ tai",
    "viên nén",
    "viên sủi",
    "dung dịch",
    "thuốc",
    "viên",
    "siro",
    "gel",
    "kem",
)
FUZZY_STOP_TOKENS = {
    "thuoc",
    "thuốc",
    "vien",
    "viên",
    "dung",
    "dịch",
    "siro",
    "gel",
    "kem",
    "cao",
}


def normalize_text(text: str) -> str:
    """Normalize text conservatively for deterministic lookup."""
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


def _strip_form_prefix(text: str) -> str:
    normalized = normalize_text(text)
    for prefix in FORM_PREFIXES:
        if normalized == prefix:
            return ""
        if normalized.startswith(prefix + " "):
            return normalized[len(prefix):].strip()
    return normalized


def _lookup_variants(text: str) -> list[str]:
    normalized = normalize_text(text)
    stripped = _strip_form_prefix(normalized)
    variants = [
        normalized,
        strip_vietnamese_diacritics(normalized),
        stripped,
        strip_vietnamese_diacritics(stripped),
    ]
    return _deduplicate(value for value in variants if value)


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


class DrugMappingService:
    """Load product mappings once and resolve raw medication names."""

    def __init__(
        self,
        mapping_path: str | Path = DEFAULT_MAPPING_PATH,
        enable_fuzzy: bool = True,
        fuzzy_threshold: int = 85,
    ) -> None:
        if not 0 <= fuzzy_threshold <= 100:
            raise ValueError("fuzzy_threshold must be between 0 and 100")
        self.mapping_path = Path(mapping_path)
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_backend = (
            "rapidfuzz" if rapidfuzz_fuzz is not None else "difflib"
        )
        self.records: list[dict[str, Any]] = []
        self.invalid_line_count = 0
        self.invalid_line_samples: list[dict[str, Any]] = []

        self.brand_exact_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.brand_no_diacritics_index: dict[
            str, list[dict[str, Any]]
        ] = defaultdict(list)
        self.alias_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.alias_no_diacritics_index: dict[
            str, list[dict[str, Any]]
        ] = defaultdict(list)
        self.ingredient_index: dict[
            str, list[dict[str, Any]]
        ] = defaultdict(list)
        self.ingredient_no_diacritics_index: dict[
            str, list[dict[str, Any]]
        ] = defaultdict(list)
        self.fuzzy_alias_choices: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.fuzzy_brand_choices: dict[str, list[dict[str, Any]]] = defaultdict(list)

        self._load()

    def _load(self) -> None:
        if not self.mapping_path.is_file():
            raise FileNotFoundError(
                f"Drug mapping file does not exist: {self.mapping_path}"
            )
        with self.mapping_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._record_invalid(line_number, "invalid_json", str(exc))
                    continue
                if not isinstance(payload, dict):
                    self._record_invalid(
                        line_number, "invalid_schema", "Root must be an object"
                    )
                    continue
                record = self._sanitize_record(payload, line_number)
                if record is None:
                    continue
                self.records.append(record)
                self._index_record(record)

    def _record_invalid(
        self, line_number: int, reason: str, detail: str
    ) -> None:
        self.invalid_line_count += 1
        if len(self.invalid_line_samples) < 20:
            self.invalid_line_samples.append(
                {
                    "line_number": line_number,
                    "reason": reason,
                    "detail": detail,
                }
            )

    def _sanitize_record(
        self, payload: dict[str, Any], line_number: int
    ) -> dict[str, Any] | None:
        mapping_id = str(payload.get("mapping_id") or "")
        brand_name = str(payload.get("brand_name") or "")
        if not mapping_id or not brand_name:
            self._record_invalid(
                line_number,
                "invalid_schema",
                "mapping_id and brand_name are required",
            )
            return None
        aliases = payload.get("brand_aliases")
        aliases = aliases if isinstance(aliases, list) else []
        ingredients = payload.get("active_ingredients")
        ingredients = ingredients if isinstance(ingredients, list) else []
        warnings = payload.get("warnings")
        warnings = warnings if isinstance(warnings, list) else []
        confidence = str(payload.get("confidence") or "low").lower()
        if confidence not in CONFIDENCE_PRIORITY:
            confidence = "low"
        return {
            **payload,
            "mapping_id": mapping_id,
            "brand_name": brand_name,
            "normalized_brand_name": str(
                payload.get("normalized_brand_name")
                or normalize_text(brand_name)
            ),
            "normalized_brand_name_no_diacritics": str(
                payload.get("normalized_brand_name_no_diacritics")
                or strip_vietnamese_diacritics(brand_name)
            ),
            "brand_aliases": [
                str(alias) for alias in aliases if str(alias).strip()
            ],
            "active_ingredients": [
                ingredient
                for ingredient in ingredients
                if isinstance(ingredient, dict)
            ],
            "warnings": [str(warning) for warning in warnings],
            "confidence": confidence,
        }

    @staticmethod
    def _append_index(
        index: dict[str, list[dict[str, Any]]],
        key: str,
        record: dict[str, Any],
    ) -> None:
        if key and all(
            existing["mapping_id"] != record["mapping_id"]
            for existing in index[key]
        ):
            index[key].append(record)

    def _index_record(self, record: dict[str, Any]) -> None:
        brand = normalize_text(record["normalized_brand_name"])
        brand_no_diacritics = strip_vietnamese_diacritics(
            record["normalized_brand_name_no_diacritics"]
        )
        for variant in _lookup_variants(brand):
            target = (
                self.brand_no_diacritics_index
                if variant == strip_vietnamese_diacritics(variant)
                else self.brand_exact_index
            )
            self._append_index(target, variant, record)
            self._append_index(self.fuzzy_brand_choices, variant, record)
        self._append_index(
            self.brand_no_diacritics_index, brand_no_diacritics, record
        )
        self._append_index(
            self.fuzzy_brand_choices, brand_no_diacritics, record
        )

        for alias in record["brand_aliases"]:
            normalized_alias = normalize_text(alias)
            alias_no_diacritics = strip_vietnamese_diacritics(alias)
            self._append_index(self.alias_index, normalized_alias, record)
            self._append_index(
                self.alias_no_diacritics_index,
                alias_no_diacritics,
                record,
            )
            self._append_index(
                self.fuzzy_alias_choices, normalized_alias, record
            )
            self._append_index(
                self.fuzzy_alias_choices, alias_no_diacritics, record
            )

        for ingredient in record["active_ingredients"]:
            name = str(ingredient.get("normalized_name") or ingredient.get("name") or "")
            if not name:
                continue
            normalized_name = normalize_text(name)
            no_diacritics = str(
                ingredient.get("normalized_name_no_diacritics")
                or strip_vietnamese_diacritics(name)
            )
            no_diacritics = strip_vietnamese_diacritics(no_diacritics)
            self._append_index(
                self.ingredient_index, normalized_name, record
            )
            self._append_index(
                self.ingredient_no_diacritics_index,
                no_diacritics,
                record,
            )

    def _exact_candidates(
        self, variants: list[str]
    ) -> list[dict[str, Any]]:
        layers = (
            ("exact_brand", self.brand_exact_index),
            ("exact_brand", self.brand_no_diacritics_index),
            ("exact_alias", self.alias_index),
            ("exact_alias", self.alias_no_diacritics_index),
            ("ingredient_name", self.ingredient_index),
            ("ingredient_name", self.ingredient_no_diacritics_index),
        )
        for match_type, index in layers:
            records: list[dict[str, Any]] = []
            for variant in variants:
                records.extend(index.get(variant, []))
            if records:
                score = 0.95 if match_type == "ingredient_name" else 1.0
                return self._candidate_records(records, match_type, score)
        return []

    def _fuzzy_score(self, query: str, choice: str) -> float:
        if rapidfuzz_fuzz is not None:
            return float(rapidfuzz_fuzz.WRatio(query, choice)) / 100.0
        return SequenceMatcher(None, query, choice).ratio()

    @staticmethod
    def _fuzzy_pair_allowed(query: str, choice: str) -> bool:
        """Reject WRatio substring false positives for unrelated phrases."""
        query_tokens = {
            token
            for token in query.split()
            if token not in FUZZY_STOP_TOKENS and len(token) >= 3
        }
        choice_tokens = {
            token
            for token in choice.split()
            if token not in FUZZY_STOP_TOKENS and len(token) >= 3
        }
        if len(query_tokens) <= 1:
            return True
        overlap = query_tokens & choice_tokens
        if len(overlap) >= 2:
            return True
        first_query_token = next(
            (
                token
                for token in query.split()
                if token in query_tokens
            ),
            "",
        )
        if first_query_token in choice_tokens:
            return True
        return bool(overlap) and any(
            SequenceMatcher(None, first_query_token, token).ratio() >= 0.8
            for token in choice_tokens
        )

    def _fuzzy_candidates(
        self, variants: list[str]
    ) -> list[dict[str, Any]]:
        threshold = self.fuzzy_threshold / 100.0
        for match_type, choices in (
            ("fuzzy_alias", self.fuzzy_alias_choices),
            ("fuzzy_brand", self.fuzzy_brand_choices),
        ):
            best_by_mapping: dict[str, tuple[dict[str, Any], float]] = {}
            for query in variants:
                for choice, records in choices.items():
                    if not self._fuzzy_pair_allowed(query, choice):
                        continue
                    score = self._fuzzy_score(query, choice)
                    if score < threshold:
                        continue
                    for record in records:
                        current = best_by_mapping.get(record["mapping_id"])
                        if current is None or score > current[1]:
                            best_by_mapping[record["mapping_id"]] = (
                                record,
                                score,
                            )
            if best_by_mapping:
                return [
                    self._candidate(record, match_type, score)
                    for record, score in best_by_mapping.values()
                ]
        return []

    def _candidate_records(
        self,
        records: list[dict[str, Any]],
        match_type: str,
        score: float,
    ) -> list[dict[str, Any]]:
        unique: dict[str, dict[str, Any]] = {}
        for record in records:
            unique[record["mapping_id"]] = record
        return [
            self._candidate(record, match_type, score)
            for record in unique.values()
        ]

    @staticmethod
    def _public_ingredient(ingredient: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": ingredient.get("name"),
            "normalized_name": ingredient.get("normalized_name"),
            "strength_raw": ingredient.get("strength_raw"),
            "strength_value": ingredient.get("strength_value"),
            "strength_unit": ingredient.get("strength_unit"),
        }

    def _candidate(
        self,
        record: dict[str, Any],
        match_type: str,
        score: float,
    ) -> dict[str, Any]:
        return {
            "mapping_id": record["mapping_id"],
            "matched_brand": record["brand_name"],
            "match_type": match_type,
            "score": round(float(score), 4),
            "confidence": record["confidence"],
            "category": record.get("category"),
            "active_ingredients": [
                self._public_ingredient(ingredient)
                for ingredient in record["active_ingredients"]
            ],
            "url": record.get("url"),
            "warnings": list(record["warnings"]),
        }

    @staticmethod
    def _sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        ingredient_count = len(candidate["active_ingredients"])
        single_ingredient_priority = (
            1 if candidate["match_type"] == "ingredient_name"
            and ingredient_count == 1 else 0
        )
        return (
            -MATCH_PRIORITY[candidate["match_type"]],
            -candidate["score"],
            -CONFIDENCE_PRIORITY.get(candidate["confidence"], 0),
            -int(bool(ingredient_count)),
            -single_ingredient_priority,
            len(candidate["warnings"]),
            len(candidate["matched_brand"]),
            candidate["mapping_id"],
        )

    def lookup(self, raw_name: str, top_k: int = 5) -> dict[str, Any]:
        """Resolve one raw medication name to ranked mapping candidates."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        input_name = str(raw_name or "")
        normalized_input = strip_vietnamese_diacritics(input_name)
        variants = _lookup_variants(input_name)
        if not variants:
            return self._unmatched(
                input_name,
                normalized_input,
                ["empty_input", "no_mapping_found"],
            )

        candidates = self._exact_candidates(variants)
        if not candidates and self.enable_fuzzy:
            candidates = self._fuzzy_candidates(variants)
        if not candidates:
            return self._unmatched(
                input_name, normalized_input, ["no_mapping_found"]
            )

        candidates.sort(key=self._sort_key)
        candidates = candidates[:top_k]
        best_match = candidates[0]
        requires_review = (
            best_match["confidence"] != "high"
            or bool(best_match["warnings"])
            or best_match["match_type"].startswith("fuzzy_")
        )
        warnings: list[str] = []
        if requires_review:
            warnings.append("mapping_requires_review")
        return {
            "input_name": input_name,
            "normalized_input": normalized_input,
            "status": "matched",
            "best_match": best_match,
            "candidates": candidates,
            "requires_review": requires_review,
            "warnings": warnings,
        }

    @staticmethod
    def _unmatched(
        input_name: str,
        normalized_input: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        return {
            "input_name": input_name,
            "normalized_input": normalized_input,
            "status": "unmatched",
            "best_match": None,
            "candidates": [],
            "requires_review": True,
            "warnings": warnings,
        }

    def lookup_many(
        self, raw_names: list[str], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Resolve multiple names while preserving input order."""
        return [self.lookup(name, top_k=top_k) for name in raw_names]

    def get_stats(self) -> dict[str, Any]:
        """Return immutable load/index statistics."""
        confidence_counts = Counter(
            record["confidence"] for record in self.records
        )
        return {
            "mapping_path": str(self.mapping_path),
            "records_loaded": len(self.records),
            "invalid_line_count": self.invalid_line_count,
            "invalid_line_samples": list(self.invalid_line_samples),
            "records_with_active_ingredients": sum(
                bool(record["active_ingredients"]) for record in self.records
            ),
            "confidence_counts": {
                key: confidence_counts.get(key, 0)
                for key in ("high", "medium", "low")
            },
            "alias_count": len(
                set(self.alias_index) | set(self.alias_no_diacritics_index)
            ),
            "ingredient_index_count": len(
                set(self.ingredient_index)
                | set(self.ingredient_no_diacritics_index)
            ),
            "enable_fuzzy": self.enable_fuzzy,
            "fuzzy_threshold": self.fuzzy_threshold,
            "fuzzy_backend": self.fuzzy_backend,
        }
