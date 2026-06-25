from __future__ import annotations

from copy import deepcopy
from typing import Any

from backend.app.services.safety_layer_service import (
    CONTRAINDICATION_SAFE_RECOMMENDATION,
    INTERACTION_SAFE_RECOMMENDATION,
    SafetyLayerService,
)


def _bundle(*chunk_ids: str) -> dict[str, Any]:
    return {
        "unique_chunks": [
            {"chunk_id": chunk_id, "text": f"Evidence {chunk_id}"}
            for chunk_id in chunk_ids
        ]
    }


def _risk_item(
    *,
    evidence_refs: list[str] | None = None,
    severity: str = "moderate",
    risk_type: str = "general",
    title: str = "Risk title",
    explanation: str = "Risk explanation",
    recommendation: str = "Review clinically.",
    affected_slugs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "risk_type": risk_type,
        "severity": severity,
        "title": title,
        "explanation": explanation,
        "affected_slugs": affected_slugs if affected_slugs is not None else ["metformin"],
        "evidence_refs": evidence_refs if evidence_refs is not None else ["chunk-1"],
        "recommendation": recommendation,
    }


def _risk_analysis(items: list[dict[str, Any]], level: str = "moderate") -> dict[str, Any]:
    return {
        "status": "analysis_ready",
        "overall_risk_level": level,
        "risk_items": items,
        "evidence_context": {"valid_evidence_refs": ["chunk-1", "chunk-2"]},
        "missing_information": [],
        "warnings": [],
        "errors": [],
    }


def test_removes_risk_item_without_evidence_refs() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis([_risk_item(evidence_refs=[])]),
        normalized_result={},
        evidence_bundle=_bundle("chunk-1"),
    )

    assert result["risk_items"] == []
    assert "safety_removed_risk_item_without_evidence" in result["warnings"]
    assert result["overall_risk_level"] == "unknown"


def test_removes_unsupported_evidence_refs() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis([_risk_item(evidence_refs=["missing"])]),
        normalized_result={},
        evidence_bundle=_bundle("chunk-1"),
    )

    assert result["risk_items"] == []
    assert "safety_removed_risk_item_without_evidence" in result["warnings"]


def test_keeps_valid_evidence_ref_and_recomputes_high() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis([_risk_item(evidence_refs=["chunk-1"], severity="high")], level="low"),
        normalized_result={},
        evidence_bundle=_bundle("chunk-1"),
    )

    assert len(result["risk_items"]) == 1
    assert result["risk_items"][0]["evidence_refs"] == ["chunk-1"]
    assert result["overall_risk_level"] == "high"
    assert "safety_overall_risk_recomputed" in result["warnings"]


def test_rewrites_unsafe_recommendation_for_contraindication() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis(
            [
                _risk_item(
                    risk_type="contraindication",
                    recommendation="Cần ngừng thuốc và thay thế bằng thuốc khác.",
                )
            ]
        ),
        normalized_result={},
        evidence_bundle=_bundle("chunk-1"),
    )

    recommendation = result["risk_items"][0]["recommendation"]
    assert recommendation == CONTRAINDICATION_SAFE_RECOMMENDATION
    assert "ngừng thuốc" not in recommendation.casefold()
    assert "thay thế bằng" not in recommendation.casefold()
    assert "safety_rewrote_unsafe_recommendation" in result["warnings"]


def test_rewrites_unsafe_recommendation_for_interaction() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis(
            [
                _risk_item(
                    risk_type="interaction",
                    recommendation="Switch to another medication.",
                )
            ]
        ),
        normalized_result={},
        evidence_bundle=_bundle("chunk-1"),
    )

    recommendation = result["risk_items"][0]["recommendation"]
    assert recommendation == INTERACTION_SAFE_RECOMMENDATION
    assert "switch to" not in recommendation.casefold()
    assert "safety_rewrote_unsafe_recommendation" in result["warnings"]


def test_rewrites_recommendation_with_drug_name_between_thay_the_and_bang() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis(
            [
                _risk_item(
                    risk_type="interaction",
                    recommendation=(
                        "Cân nhắc thay thế Omeprazole bằng một thuốc ức chế acid khác."
                    ),
                )
            ]
        ),
        normalized_result={},
        evidence_bundle=_bundle("chunk-1"),
    )

    recommendation = result["risk_items"][0]["recommendation"]
    assert recommendation == INTERACTION_SAFE_RECOMMENDATION
    assert "thay thế" not in recommendation.casefold()
    assert "safety_rewrote_unsafe_recommendation" in result["warnings"]


def test_rewrites_tam_ngung_recommendation() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis(
            [
                _risk_item(
                    risk_type="contraindication",
                    recommendation="Cần tạm ngừng metformin và đánh giá lại.",
                )
            ]
        ),
        normalized_result={},
        evidence_bundle=_bundle("chunk-1"),
    )

    recommendation = result["risk_items"][0]["recommendation"]
    assert recommendation == CONTRAINDICATION_SAFE_RECOMMENDATION
    assert "tạm ngừng" not in recommendation.casefold()
    assert "safety_rewrote_unsafe_recommendation" in result["warnings"]


def test_rewrites_replace_x_with_y_recommendation() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis(
            [
                _risk_item(
                    risk_type="interaction",
                    recommendation=(
                        "Replace Omeprazole with another acid-suppressing drug."
                    ),
                )
            ]
        ),
        normalized_result={},
        evidence_bundle=_bundle("chunk-1"),
    )

    recommendation = result["risk_items"][0]["recommendation"]
    assert recommendation == INTERACTION_SAFE_RECOMMENDATION
    assert "replace" not in recommendation.casefold()
    assert "safety_rewrote_unsafe_recommendation" in result["warnings"]


def test_removes_unsupported_mild_renal_inference() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis(
            [
                _risk_item(
                    severity="low",
                    title="eGFR 70 được coi là suy thận nhẹ",
                    explanation="Bệnh nhân suy thận nhẹ dựa trên eGFR 70.",
                )
            ],
            level="low",
        ),
        normalized_result={},
        evidence_bundle=_bundle("chunk-1"),
        patient_context={"renal_function": "eGFR 70 ml/min/1.73m2", "diagnoses": []},
    )

    assert result["risk_items"] == []
    assert result["overall_risk_level"] == "unknown"
    assert "safety_removed_unsupported_diagnosis_inference" in result["warnings"]


def test_preserves_metformin_egfr_below_30_contraindication_without_ckd_diagnosis() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis(
            [
                _risk_item(
                    risk_type="contraindication",
                    severity="high",
                    title="Chống chỉ định metformin khi eGFR < 30",
                    explanation="Metformin contraindication is cited for eGFR < 30.",
                )
            ],
            level="high",
        ),
        normalized_result={},
        evidence_bundle=_bundle("chunk-1"),
        patient_context={"renal_function": "eGFR 25 ml/min/1.73m2", "diagnoses": []},
    )

    assert len(result["risk_items"]) == 1
    assert result["overall_risk_level"] == "high"
    assert "safety_removed_unsupported_diagnosis_inference" not in result["warnings"]


def test_preserves_metformin_egfr_below_30_title_with_renal_wording() -> None:
    evidence_ref = "trungtamthuoc:ingredient:metformin:chong_chi_dinh:0001"
    title = "Chống chỉ định Metformin do bệnh nhân có chức năng thận giảm"
    explanation = (
        "Bệnh nhân có chức năng thận giảm (eGFR 25 ml/min/1.73m2) là "
        "chống chỉ định với Metformin, vì eGFR < 30 ml/phút/1,73m2."
    )
    result = SafetyLayerService().apply(
        _risk_analysis(
            [
                _risk_item(
                    risk_type="contraindication",
                    severity="high",
                    title=title,
                    explanation=explanation,
                    affected_slugs=["metformin"],
                    evidence_refs=[evidence_ref],
                )
            ],
            level="high",
        )
        | {"evidence_context": {"valid_evidence_refs": [evidence_ref]}},
        normalized_result={},
        evidence_bundle=_bundle(evidence_ref),
        patient_context={"renal_function": "eGFR 25 ml/min/1.73m2", "diagnoses": []},
    )

    assert len(result["risk_items"]) == 1
    assert result["risk_items"][0]["title"] == title
    assert result["risk_items"][0]["explanation"] == explanation
    assert result["overall_risk_level"] == "high"
    assert "safety_rewrote_unsupported_diagnosis_wording" not in result["warnings"]
    assert "safety_removed_unsupported_diagnosis_inference" not in result["warnings"]


def test_preserves_metformin_egfr_below_30_but_rewrites_unsafe_recommendation() -> None:
    evidence_ref = "trungtamthuoc:ingredient:metformin:chong_chi_dinh:0001"
    title = "Chống chỉ định Metformin do bệnh nhân có chức năng thận giảm"
    result = SafetyLayerService().apply(
        _risk_analysis(
            [
                _risk_item(
                    risk_type="contraindication",
                    severity="high",
                    title=title,
                    explanation="Metformin contraindication applies when eGFR < 30.",
                    recommendation="Cần ngừng thuốc Metformin.",
                    affected_slugs=["metformin"],
                    evidence_refs=[evidence_ref],
                )
            ],
            level="high",
        )
        | {"evidence_context": {"valid_evidence_refs": [evidence_ref]}},
        normalized_result={},
        evidence_bundle=_bundle(evidence_ref),
        patient_context={"renal_function": "eGFR < 30 ml/min/1.73m2", "diagnoses": []},
    )

    assert len(result["risk_items"]) == 1
    assert result["risk_items"][0]["title"] == title
    assert result["risk_items"][0]["recommendation"] == CONTRAINDICATION_SAFE_RECOMMENDATION
    assert "safety_rewrote_unsafe_recommendation" in result["warnings"]
    assert "safety_rewrote_unsupported_diagnosis_wording" not in result["warnings"]


def test_preserves_supported_renal_disease_case_from_diagnoses() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis(
            [
                _risk_item(
                    severity="low",
                    title="Bệnh nhân suy thận cần theo dõi",
                    explanation="Patient has renal impairment.",
                )
            ],
            level="low",
        ),
        normalized_result={},
        evidence_bundle=_bundle("chunk-1"),
        patient_context={"renal_function": "eGFR 70", "diagnoses": ["chronic kidney disease"]},
    )

    assert len(result["risk_items"]) == 1
    assert result["overall_risk_level"] == "low"


def test_no_evidence_sets_insufficient_evidence() -> None:
    risk_analysis = _risk_analysis([_risk_item()], level="high")
    risk_analysis["evidence_context"] = {"valid_evidence_refs": []}
    result = SafetyLayerService().apply(
        risk_analysis,
        normalized_result={},
        evidence_bundle={"unique_chunks": []},
    )

    assert result["status"] == "insufficient_evidence"
    assert result["risk_items"] == []
    assert result["overall_risk_level"] == "unknown"
    assert "safety_no_valid_evidence" in result["warnings"]


def test_no_evidence_does_not_override_error_status() -> None:
    risk_analysis = _risk_analysis([_risk_item()], level="high")
    risk_analysis["status"] = "analysis_failed"
    risk_analysis["evidence_context"] = {"valid_evidence_refs": []}
    result = SafetyLayerService().apply(
        risk_analysis,
        normalized_result={},
        evidence_bundle={"unique_chunks": []},
    )

    assert result["status"] == "analysis_failed"
    assert result["overall_risk_level"] == "unknown"


def test_unresolved_medications_adds_warning() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis([_risk_item()]),
        normalized_result={"unmapped_medications": [{"raw_name": "ABC"}]},
        evidence_bundle=_bundle("chunk-1"),
    )

    assert "safety_unresolved_medications" in result["warnings"]


def test_mapping_requires_review_adds_warning() -> None:
    result = SafetyLayerService().apply(
        _risk_analysis([_risk_item()]),
        normalized_result={"medications": [{"requires_review": True}]},
        evidence_bundle=_bundle("chunk-1"),
    )

    assert "safety_mapping_requires_review" in result["warnings"]


def test_warnings_deduplicated() -> None:
    risk_analysis = _risk_analysis([_risk_item(evidence_refs=[])])
    risk_analysis["warnings"] = [
        "same",
        "same",
        "safety_removed_risk_item_without_evidence",
    ]
    result = SafetyLayerService().apply(
        risk_analysis,
        normalized_result={"requires_review": True},
        evidence_bundle=_bundle("chunk-1"),
    )

    assert result["warnings"].count("same") == 1
    assert result["warnings"].count("safety_removed_risk_item_without_evidence") == 1


def test_apply_does_not_mutate_inputs() -> None:
    risk_analysis = _risk_analysis(
        [
            _risk_item(
                risk_type="interaction",
                recommendation="stop taking this medication",
            )
        ]
    )
    normalized_result = {"medications": [{"requires_review": True}]}
    evidence_bundle = _bundle("chunk-1")
    original_risk = deepcopy(risk_analysis)
    original_normalized = deepcopy(normalized_result)
    original_bundle = deepcopy(evidence_bundle)

    SafetyLayerService().apply(risk_analysis, normalized_result, evidence_bundle)

    assert risk_analysis == original_risk
    assert normalized_result == original_normalized
    assert evidence_bundle == original_bundle
