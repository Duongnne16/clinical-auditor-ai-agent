from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.risk_analyzer_service import RiskAnalyzerService


def _normalized_result() -> dict[str, Any]:
    return {
        "medications": [
            {
                "raw_name": "Omeprazol 20mg",
                "raw_line": "1. Omeprazol (Kagascdine) 20mg",
                "generic_text": "Omeprazol",
                "brand_text": "Kagascdine",
                "mapping_status": "ingredient_with_brand",
                "requires_review": False,
                "warnings": [],
                "active_ingredients": [
                    {
                        "name": "Omeprazole",
                        "evidence_slug": "omeprazole",
                        "strength_raw": "20mg",
                        "strength_value": 20,
                        "strength_unit": "mg",
                    }
                ],
            },
            {
                "raw_name": "Metformin 750mg",
                "raw_line": "2. Metformin (Panfor SR) 750mg",
                "generic_text": "Metformin",
                "brand_text": "Panfor SR",
                "mapping_status": "ingredient_with_brand",
                "requires_review": False,
                "warnings": ["brand_not_found"],
                "active_ingredients": [
                    {
                        "name": "Metformin",
                        "evidence_slug": "metformin",
                        "strength_raw": "750mg",
                        "strength_value": 750,
                        "strength_unit": "mg",
                    }
                ],
            },
            {
                "raw_name": "Paracetamol 500mg",
                "raw_line": "3. Paracetamol (Hapacol) 500mg",
                "generic_text": "Paracetamol",
                "brand_text": "Hapacol",
                "mapping_status": "ingredient_with_brand",
                "requires_review": False,
                "warnings": [],
                "active_ingredients": [
                    {
                        "name": "Paracetamol",
                        "evidence_slug": "paracetamol",
                        "strength_raw": "500mg",
                        "strength_value": 500,
                        "strength_unit": "mg",
                    }
                ],
            },
        ]
    }


def _chunk(
    chunk_id: str,
    slug: str,
    section: str,
    score: float = 1.0,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "slug": slug,
        "section": section,
        "source": "trungtamthuoc",
        "url": f"https://example.test/{slug}",
        "text": f"Evidence text for {slug} in {section}. " * 80,
        "rerank_score": score,
    }


def _evidence_bundle() -> dict[str, Any]:
    return {
        "query_results": {
            "interaction": {
                "chunks": [
                    _chunk("interaction-1", "omeprazole", "tuong_tac_thuoc"),
                    _chunk("interaction-2", "metformin", "tuong_tac_thuoc"),
                    _chunk("interaction-3", "paracetamol", "tuong_tac_thuoc"),
                ]
            },
            "renal_hepatic": {
                "chunks": [
                    _chunk("renal-1", "metformin", "suy_than"),
                    _chunk("renal-2", "metformin", "duoc_dong_hoc"),
                ]
            },
        },
        "unique_chunks": [
            _chunk("interaction-1", "omeprazole", "tuong_tac_thuoc"),
            _chunk("interaction-2", "metformin", "tuong_tac_thuoc"),
            _chunk("interaction-3", "paracetamol", "tuong_tac_thuoc"),
            _chunk("renal-1", "metformin", "suy_than"),
            _chunk("renal-2", "metformin", "duoc_dong_hoc"),
        ],
    }


class MethodLLM:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def analyze_risks(self, evidence_context: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(evidence_context)
        return self.result


class CallableLLM:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(self, evidence_context: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(evidence_context)
        return self.result


class RaisingLLM:
    def analyze_risks(self, evidence_context: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("LLM failed")


def test_no_evidence_bundle_returns_insufficient_evidence() -> None:
    result = RiskAnalyzerService().analyze(_normalized_result(), None)

    assert result["status"] == "insufficient_evidence"
    assert result["overall_risk_level"] == "unknown"
    assert result["risk_items"] == []
    assert result["evidence_context"] is None
    assert result["warnings"] == ["no_evidence_available_for_analysis"]


def test_evidence_without_llm_returns_context_ready_and_no_risks() -> None:
    result = RiskAnalyzerService().analyze(
        _normalized_result(), _evidence_bundle()
    )

    assert result["status"] == "analysis_context_ready"
    assert result["overall_risk_level"] == "unknown"
    assert result["risk_items"] == []
    assert result["warnings"] == ["analysis_not_run_without_llm"]
    assert result["evidence_context"]["valid_evidence_refs"]


def test_context_groups_chunks_and_limits_per_query_type() -> None:
    context = RiskAnalyzerService(max_chunks_per_query_type=2).build_evidence_context(
        _normalized_result(), _evidence_bundle()
    )

    assert list(context["evidence_by_query_type"]) == [
        "interaction",
        "renal_hepatic",
    ]
    assert len(context["evidence_by_query_type"]["interaction"]) == 2
    assert context["valid_evidence_refs"] == [
        "interaction-1",
        "interaction-2",
        "renal-1",
        "renal-2",
    ]
    assert len(context["evidence_by_query_type"]["interaction"][0]["snippet"]) == 700


def test_context_includes_concise_medications_and_interaction_candidates() -> None:
    context = RiskAnalyzerService().build_evidence_context(
        _normalized_result(), _evidence_bundle()
    )

    first_medication = context["medications"][0]
    assert first_medication == {
        "raw_name": "Omeprazol 20mg",
        "raw_line": "1. Omeprazol (Kagascdine) 20mg",
        "generic_text": "Omeprazol",
        "brand_text": "Kagascdine",
        "mapping_status": "ingredient_with_brand",
        "requires_review": False,
        "warnings": [],
        "active_ingredients": [
            {
                "name": "Omeprazole",
                "evidence_slug": "omeprazole",
                "strength_raw": "20mg",
                "strength_value": 20,
                "strength_unit": "mg",
            }
        ],
    }
    assert context["interaction_candidates"] == [
        {"slug_a": "omeprazole", "slug_b": "metformin"},
        {"slug_a": "omeprazole", "slug_b": "paracetamol"},
        {"slug_a": "metformin", "slug_b": "paracetamol"},
    ]


def test_missing_information_is_reported_but_not_failure() -> None:
    result = RiskAnalyzerService().analyze(
        _normalized_result(),
        _evidence_bundle(),
        patient_context={"age": 60, "sex": "male"},
    )

    assert result["status"] == "analysis_context_ready"
    assert "age" not in result["missing_information"]
    assert "sex" not in result["missing_information"]
    assert "renal_function" in result["missing_information"]
    assert "allergies" in result["missing_information"]


def test_valid_fake_llm_output_is_preserved() -> None:
    llm = MethodLLM(
        {
            "overall_risk_level": "moderate",
            "risk_items": [
                {
                    "risk_type": "interaction",
                    "severity": "moderate",
                    "title": "Potential interaction to review",
                    "explanation": "Evidence-supported explanation.",
                    "affected_slugs": ["omeprazole", "metformin"],
                    "evidence_refs": ["interaction-1", "interaction-2"],
                    "recommendation": "Review with clinician.",
                }
            ],
            "missing_information": ["renal_function"],
        }
    )

    result = RiskAnalyzerService(llm_client=llm).analyze(
        _normalized_result(), _evidence_bundle()
    )

    assert result["status"] == "analysis_ready"
    assert result["overall_risk_level"] == "moderate"
    assert result["risk_items"][0]["recommendation"] == "Review with clinician."
    assert result["risk_items"][0]["evidence_refs"] == [
        "interaction-1",
        "interaction-2",
    ]
    assert llm.calls


def test_unsupported_evidence_refs_are_removed() -> None:
    result = RiskAnalyzerService().validate_llm_analysis(
        {
            "overall_risk_level": "low",
            "risk_items": [
                {
                    "risk_type": "interaction",
                    "severity": "low",
                    "title": "title",
                    "explanation": "explanation",
                    "affected_slugs": ["omeprazole"],
                    "evidence_refs": ["interaction-1", "missing-ref"],
                    "recommendation": "Use cited evidence only.",
                }
            ],
        },
        {"interaction-1"},
    )

    assert result["risk_items"][0]["evidence_refs"] == ["interaction-1"]
    assert "unsupported_evidence_ref_removed" in result["warnings"]


def test_risk_item_without_valid_refs_is_removed() -> None:
    result = RiskAnalyzerService().validate_llm_analysis(
        {
            "overall_risk_level": "high",
            "risk_items": [
                {
                    "risk_type": "interaction",
                    "severity": "high",
                    "evidence_refs": ["missing-ref"],
                }
            ],
        },
        {"interaction-1"},
    )

    assert result["risk_items"] == []
    assert "unsupported_evidence_ref_removed" in result["warnings"]
    assert "risk_item_removed_due_to_missing_evidence" in result["warnings"]


def test_overall_risk_level_resets_when_no_supported_risk_items_remain() -> None:
    result = RiskAnalyzerService().validate_llm_analysis(
        {
            "overall_risk_level": "high",
            "risk_items": [
                {
                    "risk_type": "interaction",
                    "severity": "high",
                    "evidence_refs": ["missing-ref"],
                }
            ],
        },
        {"interaction-1"},
    )

    assert result["risk_items"] == []
    assert result["overall_risk_level"] == "unknown"
    assert "unsupported_evidence_ref_removed" in result["warnings"]
    assert "risk_item_removed_due_to_missing_evidence" in result["warnings"]
    assert (
        "overall_risk_level_reset_due_to_no_supported_risk_items"
        in result["warnings"]
    )


def test_invalid_risk_type_severity_and_overall_are_normalized() -> None:
    result = RiskAnalyzerService().validate_llm_analysis(
        {
            "overall_risk_level": "critical",
            "risk_items": [
                {
                    "risk_type": "diagnosis",
                    "severity": "severe",
                    "evidence_refs": ["interaction-1"],
                }
            ],
        },
        {"interaction-1"},
    )

    assert result["overall_risk_level"] == "unknown"
    assert result["risk_items"][0]["risk_type"] == "general"
    assert result["risk_items"][0]["severity"] == "unknown"
    assert result["warnings"] == [
        "invalid_overall_risk_level_normalized",
        "invalid_risk_type_normalized",
        "invalid_severity_normalized",
    ]


def test_callable_llm_is_supported() -> None:
    llm = CallableLLM(
        {
            "overall_risk_level": "low",
            "risk_items": [
                {
                    "risk_type": "general",
                    "severity": "low",
                    "evidence_refs": ["interaction-1"],
                }
            ],
        }
    )

    result = RiskAnalyzerService(llm_client=llm).analyze(
        _normalized_result(), _evidence_bundle()
    )

    assert result["status"] == "analysis_ready"
    assert llm.calls


def test_llm_exception_returns_analysis_failed() -> None:
    result = RiskAnalyzerService(llm_client=RaisingLLM()).analyze(
        _normalized_result(), _evidence_bundle()
    )

    assert result["status"] == "analysis_failed"
    assert result["errors"] == ["risk_analysis_failed"]
    assert result["risk_items"] == []


def test_invalid_max_chunks_per_query_type_raises() -> None:
    with pytest.raises(ValueError):
        RiskAnalyzerService(max_chunks_per_query_type=0)


def test_get_stats_returns_service_info() -> None:
    stats = RiskAnalyzerService(
        llm_client=object(), max_chunks_per_query_type=3
    ).get_stats()

    assert stats["service"] == "RiskAnalyzerService"
    assert stats["llm_enabled"] is True
    assert stats["max_chunks_per_query_type"] == 3
    assert "interaction" in stats["valid_risk_types"]
    assert "moderate" in stats["valid_risk_levels"]
