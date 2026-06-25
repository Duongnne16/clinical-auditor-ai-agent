"""Compose drug-product mapping with ingredient evidence resolution."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from backend.app.services.drug_mapping_service import DrugMappingService
from backend.app.services.ingredient_evidence_resolver import (
    IngredientEvidenceResolver,
)
from backend.app.services.medication_line_parser import MedicationLineParser


STANDARD_FIELDS = (
    "dose",
    "frequency",
    "route",
    "duration",
    "note",
)
NAME_FIELDS = (
    "raw_name",
    "name",
    "drug_name",
    "medication_name",
)
PARSED_FIELDS = (
    "raw_line",
    "order_index",
    "generic_text",
    "brand_text",
    "strength_text",
    "quantity",
    "ingredients",
    "is_combination",
    "parse_status",
    "instruction",
    "warnings",
)
DANGEROUS_INGREDIENT_FUZZY_PAIRS = {
    ("omeprazole", "esomeprazole"),
    ("esomeprazole", "omeprazole"),
    ("levofloxacin", "ofloxacin"),
    ("ofloxacin", "levofloxacin"),
    ("cefuroxime", "cefixime"),
    ("cefixime", "cefuroxime"),
    ("cefazolin", "cefuroxime"),
    ("cefuroxime", "cefazolin"),
}
STRENGTH_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<value>\d+(?:[.,]\d+)?(?:\s*/\s*\d+(?:[.,]\d+)?)?)"
    r"\s*(?P<unit>mcg|µg|mg|g|ml|iu|ui)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
PARENTHETICAL_RE = re.compile(r"\(([^()]*)\)")


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _normalize_candidate(text: str) -> str:
    value = unicodedata.normalize("NFC", str(text or ""))
    value = re.sub(r"[\u2010-\u2015\u2212]", "-", value)
    return re.sub(r"\s+", " ", value).strip(" ,;:-").lower()


def extract_strengths_and_clean_name(raw_name: str) -> dict[str, Any]:
    """Extract strength tokens without changing the original medication name."""
    text = unicodedata.normalize("NFC", str(raw_name or ""))
    strengths: list[dict[str, Any]] = []
    has_combination_strength = False
    for match in STRENGTH_RE.finditer(text):
        raw = match.group(0).strip()
        raw_value = re.sub(r"\s+", "", match.group("value"))
        unit = match.group("unit")
        normalized_unit = "IU" if unit.lower() in {"iu", "ui"} else unit.lower()
        is_combination = "/" in raw_value
        has_combination_strength = (
            has_combination_strength or is_combination
        )
        strength_value: float | None = None
        if not is_combination:
            try:
                strength_value = float(raw_value.replace(",", "."))
            except ValueError:
                strength_value = None
        strengths.append(
            {
                "strength_raw": raw,
                "strength_value": strength_value,
                "strength_unit": normalized_unit,
                "is_combination": is_combination,
            }
        )

    clean_name = STRENGTH_RE.sub(" ", text)
    clean_name = re.sub(r"\s+", " ", clean_name).strip(" ,;:-")
    return {
        "clean_name": clean_name,
        "strengths": strengths,
        "has_combination_strength": has_combination_strength,
    }


class NormalizeDrugsService:
    """Normalize medications and attach evidence identities."""

    def __init__(
        self,
        drug_mapping_service: DrugMappingService | None = None,
        ingredient_resolver: IngredientEvidenceResolver | None = None,
        medication_line_parser: MedicationLineParser | None = None,
    ) -> None:
        self.drug_mapping_service = (
            drug_mapping_service or DrugMappingService()
        )
        self.ingredient_resolver = (
            ingredient_resolver or IngredientEvidenceResolver()
        )
        self.medication_line_parser = (
            medication_line_parser or MedicationLineParser()
        )

    def _prepare_input(
        self,
        medication: dict[str, Any] | str,
    ) -> tuple[dict[str, Any], str | None]:
        if isinstance(medication, str):
            return {"raw_name": medication}, None
        if not isinstance(medication, dict):
            return {"raw_name": ""}, "invalid_medication_input"

        source = dict(medication)
        raw_line = source.get("raw_line")
        if (
            isinstance(raw_line, str)
            and raw_line.strip()
            and not source.get("generic_text")
            and not source.get("ingredients")
        ):
            parsed = self.medication_line_parser.parse_line(raw_line)
            source.update(parsed)

        raw_name = ""
        for field in NAME_FIELDS:
            value = source.get(field)
            if value is not None and str(value).strip():
                raw_name = str(value)
                break
        if not raw_name and source.get("raw_line"):
            raw_name = str(source["raw_line"])

        prepared: dict[str, Any] = {"raw_name": raw_name}
        for field in STANDARD_FIELDS:
            if field in source:
                prepared[field] = source[field]
        for field in PARSED_FIELDS:
            if field in source:
                prepared[field] = source[field]

        excluded = (
            set(NAME_FIELDS) | set(STANDARD_FIELDS) | set(PARSED_FIELDS)
        )
        original_fields = {
            key: value
            for key, value in source.items()
            if key not in excluded
        }
        if original_fields:
            prepared["original_fields"] = original_fields
        return prepared, None

    @staticmethod
    def _empty_mapping_fields(output: dict[str, Any]) -> None:
        output.update(
            {
                "mapping_status": "unmatched",
                "matched_brand": None,
                "mapping_id": None,
                "mapping_match_type": None,
                "mapping_score": None,
                "mapping_confidence": None,
                "source_url": None,
                "active_ingredients": [],
                "mapping_candidates": [],
            }
        )

    @staticmethod
    def _mapping_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        ingredients = candidate.get("active_ingredients")
        ingredients = ingredients if isinstance(ingredients, list) else []
        return {
            "mapping_id": candidate.get("mapping_id"),
            "matched_brand": candidate.get("matched_brand"),
            "match_type": candidate.get("match_type"),
            "score": candidate.get("score"),
            "confidence": candidate.get("confidence"),
            "active_ingredient_names": [
                ingredient.get("name")
                for ingredient in ingredients
                if isinstance(ingredient, dict) and ingredient.get("name")
            ],
            "url": candidate.get("url"),
        }

    @staticmethod
    def _ingredient_output(
        ingredient: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        best_match = evidence.get("best_match")
        best_match = best_match if isinstance(best_match, dict) else {}
        status = evidence.get("status", "unresolved")
        return {
            "name": ingredient.get("name"),
            "normalized_name": ingredient.get("normalized_name"),
            "strength_raw": ingredient.get("strength_raw"),
            "strength_value": ingredient.get("strength_value"),
            "strength_unit": ingredient.get("strength_unit"),
            "evidence_status": status,
            "evidence_name": best_match.get("evidence_name"),
            "evidence_slug": best_match.get("evidence_slug"),
            "evidence_match_type": best_match.get("match_type"),
            "evidence_score": best_match.get("score"),
            "evidence_requires_review": bool(
                evidence.get("requires_review", status != "resolved")
            ),
            "evidence_warnings": [
                str(warning)
                for warning in evidence.get("warnings", [])
            ],
        }

    @staticmethod
    def _fallback_candidates(raw_name: str) -> list[str]:
        candidates: list[str] = []
        for parenthetical in PARENTHETICAL_RE.findall(raw_name):
            parsed = extract_strengths_and_clean_name(parenthetical)
            candidate = str(parsed["clean_name"])
            if not parsed["has_combination_strength"]:
                candidates.append(candidate)

        outside_parentheses = PARENTHETICAL_RE.sub(" ", raw_name)
        parsed_outside = extract_strengths_and_clean_name(
            outside_parentheses
        )
        candidates.append(str(parsed_outside["clean_name"]))

        result: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = _normalize_candidate(candidate)
            if (
                len(normalized) < 4
                or not any(character.isalpha() for character in normalized)
                or normalized in seen
            ):
                continue
            seen.add(normalized)
            result.append(candidate.strip())
        return result

    @classmethod
    def _dangerous_fuzzy_product_match(
        cls,
        raw_name: str,
        best_match: dict[str, Any],
    ) -> bool:
        match_type = str(best_match.get("match_type") or "")
        if not match_type.startswith("fuzzy_"):
            return False
        input_candidates = {
            _normalize_candidate(candidate)
            for candidate in cls._fallback_candidates(raw_name)
        }
        ingredients = best_match.get("active_ingredients")
        ingredients = ingredients if isinstance(ingredients, list) else []
        matched_ingredients = {
            _normalize_candidate(
                str(
                    ingredient.get("normalized_name")
                    or ingredient.get("name")
                    or ""
                )
            )
            for ingredient in ingredients
            if isinstance(ingredient, dict)
        }
        return any(
            (input_candidate, matched_ingredient)
            in DANGEROUS_INGREDIENT_FUZZY_PAIRS
            for input_candidate in input_candidates
            for matched_ingredient in matched_ingredients
        )

    @staticmethod
    def _ingredient_strength(
        parsed: dict[str, Any],
    ) -> dict[str, Any]:
        strengths = parsed["strengths"]
        if len(strengths) != 1:
            return {
                "strength_raw": None,
                "strength_value": None,
                "strength_unit": None,
            }
        strength = strengths[0]
        return {
            "strength_raw": strength["strength_raw"],
            "strength_value": strength["strength_value"],
            "strength_unit": strength["strength_unit"],
        }

    def _ingredient_only_fallback(
        self,
        output: dict[str, Any],
        raw_name: str,
        top_k: int,
        *,
        unsafe_product_rejected: bool = False,
    ) -> dict[str, Any] | None:
        parsed = extract_strengths_and_clean_name(raw_name)
        if parsed["has_combination_strength"]:
            return None

        strength = self._ingredient_strength(parsed)
        for candidate in self._fallback_candidates(raw_name):
            evidence = self.ingredient_resolver.resolve(
                candidate, top_k=top_k
            )
            if evidence.get("status") != "resolved":
                continue
            ingredient = {
                "name": candidate,
                "normalized_name": _normalize_candidate(candidate),
                **strength,
            }
            enriched = self._ingredient_output(ingredient, evidence)
            warnings = [
                "drug_product_mapping_not_found",
                "ingredient_resolved_without_product_mapping",
            ]
            if unsafe_product_rejected:
                warnings.append("unsafe_product_fuzzy_match_rejected")
            output.update(
                {
                    "mapping_status": "ingredient_only",
                    "matched_brand": None,
                    "mapping_id": None,
                    "mapping_match_type": "ingredient_fallback",
                    "mapping_score": None,
                    "mapping_confidence": "medium",
                    "source_url": None,
                    "active_ingredients": [enriched],
                    "requires_review": True,
                    "warnings": warnings,
                    "mapping_candidates": [],
                }
            )
            return output
        return None

    @staticmethod
    def _is_dangerous_evidence_match(
        ingredient_name: str,
        evidence: dict[str, Any],
    ) -> bool:
        best_match = evidence.get("best_match")
        if not isinstance(best_match, dict):
            return False
        if best_match.get("match_type") != "fuzzy":
            return False
        source = _normalize_candidate(ingredient_name).replace("-", " ")
        target = _normalize_candidate(
            str(
                best_match.get("evidence_slug")
                or best_match.get("evidence_name")
                or ""
            )
        ).replace("-", " ")
        return (source, target) in DANGEROUS_INGREDIENT_FUZZY_PAIRS

    def _resolve_public_ingredient(
        self,
        ingredient: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        name = str(ingredient.get("name") or "")
        evidence = self.ingredient_resolver.resolve(name, top_k=top_k)
        if self._is_dangerous_evidence_match(name, evidence):
            evidence = {
                "status": "unresolved",
                "best_match": None,
                "requires_review": True,
                "warnings": ["dangerous_ingredient_fuzzy_match_rejected"],
            }
        normalized_ingredient = {
            "name": name,
            "normalized_name": (
                ingredient.get("normalized_name")
                or _normalize_candidate(name)
            ),
            "strength_raw": ingredient.get("strength_raw"),
            "strength_value": ingredient.get("strength_value"),
            "strength_unit": ingredient.get("strength_unit"),
        }
        return self._ingredient_output(normalized_ingredient, evidence)

    def _brand_verification(
        self,
        brand_text: str | None,
        parsed_slugs: set[str],
        top_k: int,
    ) -> tuple[str, list[str]]:
        if not brand_text:
            return "not_checked", []
        mapping = self.drug_mapping_service.lookup(
            brand_text, top_k=top_k
        )
        best_match = mapping.get("best_match")
        if (
            mapping.get("status") != "matched"
            or not isinstance(best_match, dict)
            or best_match.get("match_type")
            not in {"exact_brand", "exact_alias"}
        ):
            return "not_found", []

        product_slugs: set[str] = set()
        ingredients = best_match.get("active_ingredients")
        ingredients = ingredients if isinstance(ingredients, list) else []
        for ingredient in ingredients:
            if not isinstance(ingredient, dict):
                continue
            evidence = self.ingredient_resolver.resolve(
                str(ingredient.get("name") or ""), top_k=top_k
            )
            evidence_best = evidence.get("best_match")
            if (
                evidence.get("status") == "resolved"
                and isinstance(evidence_best, dict)
                and evidence_best.get("evidence_slug")
            ):
                product_slugs.add(str(evidence_best["evidence_slug"]))

        if parsed_slugs and product_slugs:
            if parsed_slugs == product_slugs:
                return "verified", []
            return "conflict", ["ingredient_brand_conflict"]
        return "not_found", []

    def _normalize_generic_first(
        self,
        output: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any] | None:
        raw_ingredients = output.get("ingredients")
        raw_ingredients = (
            raw_ingredients if isinstance(raw_ingredients, list) else []
        )
        if not raw_ingredients and output.get("generic_text"):
            raw_ingredients = [
                {
                    "name": output["generic_text"],
                    "strength_raw": None,
                    "strength_value": None,
                    "strength_unit": None,
                }
            ]
        raw_ingredients = [
            ingredient
            for ingredient in raw_ingredients
            if isinstance(ingredient, dict)
            and str(ingredient.get("name") or "").strip()
        ]
        if not raw_ingredients:
            return None

        enriched = [
            self._resolve_public_ingredient(ingredient, top_k)
            for ingredient in raw_ingredients
        ]
        if not any(
            ingredient["evidence_status"] == "resolved"
            for ingredient in enriched
        ):
            return None
        parsed_slugs = {
            str(ingredient["evidence_slug"])
            for ingredient in enriched
            if ingredient.get("evidence_status") == "resolved"
            and ingredient.get("evidence_slug")
        }
        brand_text = (
            str(output.get("brand_text") or "").strip() or None
        )
        verification_status, verification_warnings = (
            self._brand_verification(
                brand_text, parsed_slugs, top_k
            )
        )

        parser_warnings = [
            str(warning)
            for warning in output.get("warnings", [])
        ]
        warnings = [*parser_warnings, *verification_warnings]
        unresolved = any(
            ingredient["evidence_status"] != "resolved"
            for ingredient in enriched
        )
        evidence_review = any(
            ingredient["evidence_requires_review"]
            for ingredient in enriched
        )
        if unresolved:
            warnings.append("ingredient_evidence_unresolved")
        if evidence_review:
            warnings.append("ingredient_evidence_requires_review")
        if any(
            "dangerous_ingredient_fuzzy_match_rejected"
            in ingredient["evidence_warnings"]
            for ingredient in enriched
        ):
            warnings.append("dangerous_ingredient_fuzzy_match_rejected")

        quality_parser_warnings = {
            "strength_ingredient_alignment_uncertain",
            "generic_text_not_found",
        }
        requires_review = (
            unresolved
            or evidence_review
            or verification_status == "conflict"
            or bool(set(parser_warnings) & quality_parser_warnings)
        )
        output.update(
            {
                "mapping_status": (
                    "ingredient_with_brand"
                    if brand_text
                    else "ingredient_only"
                ),
                "brand_verification_status": verification_status,
                "matched_brand": None,
                "mapping_id": None,
                "mapping_match_type": "generic_first",
                "mapping_score": None,
                "mapping_confidence": "high",
                "source_url": None,
                "active_ingredients": enriched,
                "requires_review": requires_review,
                "warnings": _deduplicate(warnings),
                "mapping_candidates": [],
            }
        )
        return output

    def normalize_medication(
        self,
        medication: dict[str, Any] | str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Normalize one medication and resolve its active ingredients."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        output, input_warning = self._prepare_input(medication)
        raw_name = str(output["raw_name"] or "")
        warnings: list[str] = []
        if input_warning:
            warnings.append(input_warning)
        if not raw_name.strip():
            warnings.append("empty_drug_name")
            self._empty_mapping_fields(output)
            output.update(
                {
                    "requires_review": True,
                    "warnings": _deduplicate(warnings),
                }
            )
            return output

        if output.get("ingredients") or output.get("generic_text"):
            generic_result = self._normalize_generic_first(
                output, top_k
            )
            if generic_result is not None:
                return generic_result

        mapping = self.drug_mapping_service.lookup(raw_name, top_k=top_k)
        mapping_warnings = [
            str(warning) for warning in mapping.get("warnings", [])
        ]
        warnings.extend(mapping_warnings)
        candidates = mapping.get("candidates")
        candidates = candidates if isinstance(candidates, list) else []
        projected_candidates = [
            self._mapping_candidate(candidate)
            for candidate in candidates[:top_k]
            if isinstance(candidate, dict)
        ]

        parsed_input = extract_strengths_and_clean_name(raw_name)
        best_match = mapping.get("best_match")
        best_match = best_match if isinstance(best_match, dict) else {}
        unsafe_product_rejected = self._dangerous_fuzzy_product_match(
            raw_name, best_match
        )
        has_strength = bool(parsed_input["strengths"])
        match_type = str(best_match.get("match_type") or "")
        generic_strength_product_rejected = (
            mapping.get("status") == "matched"
            and has_strength
            and match_type not in {"exact_brand", "exact_alias"}
        )
        product_match_rejected = (
            unsafe_product_rejected
            or generic_strength_product_rejected
        )

        if mapping.get("status") != "matched" or product_match_rejected:
            fallback = self._ingredient_only_fallback(
                output,
                raw_name,
                top_k,
                unsafe_product_rejected=unsafe_product_rejected,
            )
            if fallback is not None:
                return fallback
            warnings.append("drug_mapping_not_found")
            warnings.append("drug_or_ingredient_not_found")
            self._empty_mapping_fields(output)
            output["mapping_candidates"] = (
                [] if product_match_rejected else projected_candidates
            )
            output.update(
                {
                    "requires_review": True,
                    "warnings": _deduplicate(warnings),
                }
            )
            return output

        best_warnings = [
            str(warning) for warning in best_match.get("warnings", [])
        ]
        warnings.extend(best_warnings)
        mapping_requires_review = bool(mapping.get("requires_review"))
        mapping_confidence = str(
            best_match.get("confidence") or "low"
        ).lower()
        if mapping_requires_review:
            warnings.append("mapping_requires_review")

        enriched_ingredients: list[dict[str, Any]] = []
        ingredient_requires_review = False
        unresolved_found = False
        raw_ingredients = best_match.get("active_ingredients")
        raw_ingredients = (
            raw_ingredients if isinstance(raw_ingredients, list) else []
        )
        for ingredient in raw_ingredients:
            if not isinstance(ingredient, dict):
                continue
            ingredient_name = str(ingredient.get("name") or "")
            evidence = self.ingredient_resolver.resolve(
                ingredient_name, top_k=top_k
            )
            enriched = self._ingredient_output(ingredient, evidence)
            enriched_ingredients.append(enriched)
            if enriched["evidence_status"] != "resolved":
                unresolved_found = True
            if enriched["evidence_requires_review"]:
                ingredient_requires_review = True

        if unresolved_found:
            warnings.append("ingredient_evidence_unresolved")

        requires_review = (
            mapping_requires_review
            or mapping_confidence != "high"
            or bool(best_warnings)
            or unresolved_found
            or ingredient_requires_review
        )
        output.update(
            {
                "mapping_status": "matched",
                "matched_brand": best_match.get("matched_brand"),
                "mapping_id": best_match.get("mapping_id"),
                "mapping_match_type": best_match.get("match_type"),
                "mapping_score": best_match.get("score"),
                "mapping_confidence": best_match.get("confidence"),
                "source_url": best_match.get("url"),
                "active_ingredients": enriched_ingredients,
                "requires_review": requires_review,
                "warnings": _deduplicate(warnings),
                "mapping_candidates": projected_candidates,
            }
        )
        return output

    def normalize_many(
        self,
        medications: list[dict[str, Any] | str],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Normalize multiple medications and aggregate evidence coverage."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        normalized = [
            self.normalize_medication(medication, top_k=top_k)
            for medication in medications
        ]
        unmapped_medications: list[dict[str, Any]] = []
        ingredient_only_medications: list[dict[str, Any]] = []
        ingredient_with_brand_medications: list[dict[str, Any]] = []
        unresolved_ingredients: list[dict[str, Any]] = []
        resolved_evidence_slugs: list[str] = []
        total_active_ingredients = 0
        resolved_ingredients = 0

        for medication_index, medication in enumerate(normalized):
            if medication["mapping_status"] == "unmatched":
                reason = (
                    "empty_drug_name"
                    if "empty_drug_name" in medication["warnings"]
                    else "invalid_medication_input"
                    if "invalid_medication_input" in medication["warnings"]
                    else "drug_mapping_not_found"
                )
                unmapped_medications.append(
                    {
                        "raw_name": medication["raw_name"],
                        "reason": reason,
                    }
                )
            elif medication["mapping_status"] == "ingredient_only":
                ingredient_only_medications.append(
                    {
                        "medication_index": medication_index,
                        "raw_name": medication["raw_name"],
                        "evidence_slugs": [
                            ingredient["evidence_slug"]
                            for ingredient in medication[
                                "active_ingredients"
                            ]
                            if ingredient.get("evidence_slug")
                        ],
                    }
                )
            elif medication["mapping_status"] == "ingredient_with_brand":
                ingredient_with_brand_medications.append(
                    {
                        "medication_index": medication_index,
                        "raw_name": medication["raw_name"],
                        "brand_text": medication.get("brand_text"),
                        "brand_verification_status": medication.get(
                            "brand_verification_status"
                        ),
                        "evidence_slugs": [
                            ingredient["evidence_slug"]
                            for ingredient in medication[
                                "active_ingredients"
                            ]
                            if ingredient.get("evidence_slug")
                        ],
                    }
                )
            for ingredient in medication["active_ingredients"]:
                total_active_ingredients += 1
                if ingredient["evidence_status"] == "resolved":
                    resolved_ingredients += 1
                    slug = ingredient.get("evidence_slug")
                    if slug:
                        resolved_evidence_slugs.append(str(slug))
                else:
                    unresolved_ingredients.append(
                        {
                            "medication_index": medication_index,
                            "raw_name": medication["raw_name"],
                            "ingredient_name": ingredient.get("name"),
                            "evidence_warnings": list(
                                ingredient["evidence_warnings"]
                            ),
                        }
                    )

        matched_medications = sum(
            medication["mapping_status"] == "matched"
            for medication in normalized
        )
        ingredient_only_count = sum(
            medication["mapping_status"] == "ingredient_only"
            for medication in normalized
        )
        ingredient_with_brand_count = sum(
            medication["mapping_status"] == "ingredient_with_brand"
            for medication in normalized
        )
        unmatched_medications = sum(
            medication["mapping_status"] == "unmatched"
            for medication in normalized
        )
        requires_review = any(
            medication["requires_review"] for medication in normalized
        )
        usable_medications = (
            matched_medications
            + ingredient_with_brand_count
            + ingredient_only_count
        )
        return {
            "medications": normalized,
            "summary": {
                "total_medications": len(normalized),
                "matched_medications": matched_medications,
                "product_matched_medications": matched_medications,
                "ingredient_with_brand_medications": (
                    ingredient_with_brand_count
                ),
                "ingredient_only_medications": ingredient_only_count,
                "unmatched_medications": unmatched_medications,
                "usable_medications": usable_medications,
                "total_active_ingredients": total_active_ingredients,
                "resolved_ingredients": resolved_ingredients,
                "unresolved_ingredients": len(unresolved_ingredients),
                "requires_review": requires_review,
            },
            "unmapped_medications": unmapped_medications,
            "ingredient_only_medications": ingredient_only_medications,
            "ingredient_with_brand_medications": (
                ingredient_with_brand_medications
            ),
            "unresolved_ingredients": unresolved_ingredients,
            "resolved_evidence_slugs": resolved_evidence_slugs,
            "unique_evidence_slugs": sorted(
                set(resolved_evidence_slugs)
            ),
            "requires_review": requires_review,
            "warnings": [],
        }

    def get_stats(self) -> dict[str, Any]:
        """Return stats from both underlying services."""
        return {
            "service": "NormalizeDrugsService",
            "drug_mapping": self.drug_mapping_service.get_stats(),
            "ingredient_resolver": self.ingredient_resolver.get_stats(),
        }
