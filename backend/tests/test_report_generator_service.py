from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.report_generator_service import (
    CONTEXT_READY_SUMMARY,
    NO_SUPPORTED_RISK_SUMMARY,
    ReportGeneratorService,
)


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
                "raw_name": "Thuốc ABC không có thật",
                "raw_line": "2. Thuốc ABC không có thật",
                "generic_text": None,
                "brand_text": None,
                "mapping_status": "unmatched",
                "requires_review": True,
                "warnings": ["drug_or_ingredient_not_found"],
                "active_ingredients": [],
            },
        ]
    }


def _chunk(chunk_id: str = "interaction-1") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "slug": "omeprazole",
        "section": "tuong_tac_thuoc",
        "section_title": "Tương tác thuốc",
        "source": "trungtamthuoc",
        "source_type": "duoc_thu",
        "url": "https://example.test/omeprazole",
        "text": "  Đây là   đoạn bằng chứng về tương tác thuốc. " * 40,
        "rerank_score": 1.23,
    }


def _evidence_bundle() -> dict[str, Any]:
    return {
        "unique_chunks": [_chunk()],
        "query_results": {"interaction": {"chunks": [_chunk()]}},
    }


def _risk_item(evidence_refs: list[str] | None = None) -> dict[str, Any]:
    return {
        "risk_type": "interaction",
        "severity": "moderate",
        "title": "Cần theo dõi tương tác",
        "explanation": "Có bằng chứng cần rà soát.",
        "affected_slugs": ["omeprazole"],
        "evidence_refs": evidence_refs or ["interaction-1"],
        "recommendation": "Bác sĩ/dược sĩ cần rà soát theo bối cảnh lâm sàng.",
    }


def test_invalid_risk_analysis_returns_cannot_generate_report() -> None:
    report = ReportGeneratorService().generate_report(
        _normalized_result(), _evidence_bundle(), risk_analysis=None
    )

    assert report["status"] == "cannot_generate_report"
    assert report["overall_risk_level"] == "unknown"
    assert report["risk_items"] == []
    assert report["warnings"] == []
    assert report["errors"] == ["invalid_risk_analysis"]


def test_analysis_context_ready_does_not_claim_no_warnings_and_keeps_sources() -> None:
    report = ReportGeneratorService().generate_report(
        _normalized_result(),
        _evidence_bundle(),
        {
            "status": "analysis_context_ready",
            "overall_risk_level": "unknown",
            "risk_items": [],
            "missing_information": ["renal_function"],
            "warnings": ["analysis_not_run_without_llm"],
            "errors": [],
        },
    )

    assert report["status"] == "report_context_ready"
    assert report["overall_risk_level"] == "unknown"
    assert report["risk_items"] == []
    assert report["source_count"] == 1
    assert CONTEXT_READY_SUMMARY in report["summary"]
    assert NO_SUPPORTED_RISK_SUMMARY not in report["summary"]
    assert CONTEXT_READY_SUMMARY in report["markdown_report"]
    assert NO_SUPPORTED_RISK_SUMMARY not in report["markdown_report"]


def test_insufficient_evidence_status() -> None:
    report = ReportGeneratorService().generate_report(
        _normalized_result(),
        None,
        {
            "status": "insufficient_evidence",
            "overall_risk_level": "unknown",
            "risk_items": [],
            "missing_information": [],
            "warnings": ["no_evidence_available_for_analysis"],
            "errors": [],
        },
    )

    assert report["status"] == "report_insufficient_evidence"
    assert report["overall_risk_level"] == "unknown"
    assert "Chưa đủ bằng chứng" in report["summary"]
    assert "Chưa đủ bằng chứng" in report["markdown_report"]
    assert NO_SUPPORTED_RISK_SUMMARY not in report["markdown_report"]


def test_analysis_failed_status_preserves_errors() -> None:
    report = ReportGeneratorService().generate_report(
        _normalized_result(),
        _evidence_bundle(),
        {
            "status": "analysis_failed",
            "overall_risk_level": "unknown",
            "risk_items": [],
            "missing_information": [],
            "warnings": [],
            "errors": ["risk_analysis_failed"],
        },
    )

    assert report["status"] == "report_analysis_failed"
    assert report["errors"] == ["risk_analysis_failed"]
    assert "không thành công" in report["summary"]
    assert "không thành công" in report["markdown_report"]
    assert NO_SUPPORTED_RISK_SUMMARY not in report["markdown_report"]


def test_analysis_ready_attaches_evidence_to_risk_item() -> None:
    report = ReportGeneratorService().generate_report(
        _normalized_result(),
        _evidence_bundle(),
        {
            "status": "analysis_ready",
            "overall_risk_level": "moderate",
            "risk_items": [_risk_item()],
            "missing_information": [],
            "warnings": [],
            "errors": [],
        },
    )

    assert report["status"] == "report_ready"
    assert report["overall_risk_level"] == "moderate"
    assert report["risk_items"][0]["evidence"][0]["chunk_id"] == "interaction-1"
    assert report["risk_items"][0]["evidence"][0]["url"] == "https://example.test/omeprazole"
    assert report["source_count"] == 1


def test_analysis_ready_without_risk_items_uses_no_supported_risk_summary() -> None:
    report = ReportGeneratorService().generate_report(
        _normalized_result(),
        _evidence_bundle(),
        {
            "status": "analysis_ready",
            "overall_risk_level": "unknown",
            "risk_items": [],
            "missing_information": [],
            "warnings": [],
            "errors": [],
        },
    )

    assert report["status"] == "report_ready"
    assert NO_SUPPORTED_RISK_SUMMARY in report["summary"]
    assert NO_SUPPORTED_RISK_SUMMARY in report["markdown_report"]


def test_missing_evidence_ref_adds_warning_without_fake_evidence() -> None:
    report = ReportGeneratorService().generate_report(
        _normalized_result(),
        _evidence_bundle(),
        {
            "status": "analysis_ready",
            "overall_risk_level": "high",
            "risk_items": [_risk_item(["missing-ref"])],
            "missing_information": [],
            "warnings": [],
            "errors": [],
        },
    )

    assert report["risk_items"][0]["evidence_refs"] == ["missing-ref"]
    assert report["risk_items"][0]["evidence"] == []
    assert "risk_item_evidence_ref_missing_in_bundle" in report["warnings"]


def test_medication_summary_and_medications_requiring_review() -> None:
    service = ReportGeneratorService()
    medication_summary = service.build_medication_summary(_normalized_result())

    assert medication_summary[0]["active_ingredients"] == [
        {
            "name": "Omeprazole",
            "evidence_slug": "omeprazole",
            "strength_raw": "20mg",
            "strength_value": 20,
            "strength_unit": "mg",
        }
    ]

    report = service.generate_report(
        _normalized_result(),
        _evidence_bundle(),
        {
            "status": "analysis_ready",
            "overall_risk_level": "unknown",
            "risk_items": [],
            "missing_information": [],
            "warnings": [],
            "errors": [],
        },
    )
    assert [item["raw_name"] for item in report["medications_requiring_review"]] == [
        "Thuốc ABC không có thật"
    ]


def test_medication_summary_preserves_instruction() -> None:
    normalized = _normalized_result()
    normalized["medications"][0]["instruction"] = "Ngày uống 1 lần, mỗi lần 1 viên"

    medication_summary = ReportGeneratorService().build_medication_summary(normalized)

    assert medication_summary[0]["instruction"] == "Ngày uống 1 lần, mỗi lần 1 viên"


def test_missing_information_is_preserved() -> None:
    report = ReportGeneratorService().generate_report(
        _normalized_result(),
        _evidence_bundle(),
        {
            "status": "analysis_ready",
            "overall_risk_level": "unknown",
            "risk_items": [],
            "missing_information": ["age", "renal_function"],
            "warnings": [],
            "errors": [],
        },
    )

    assert report["missing_information"] == ["age", "renal_function"]
    assert "age, renal_function" in report["summary"]


def test_markdown_contains_expected_sections_url_and_disclaimer() -> None:
    report = ReportGeneratorService().generate_report(
        _normalized_result(),
        _evidence_bundle(),
        {
            "status": "analysis_ready",
            "overall_risk_level": "moderate",
            "risk_items": [_risk_item()],
            "missing_information": [],
            "warnings": [],
            "errors": [],
        },
    )

    markdown = report["markdown_report"]
    assert "# Báo cáo kiểm tra đơn thuốc" in markdown
    assert "## Kết luận tổng quan" in markdown
    assert "## Danh sách thuốc" in markdown
    assert "## Cảnh báo cần xem xét" in markdown
    assert "https://example.test/omeprazole" in markdown
    assert "không thay thế quyết định chuyên môn" in markdown


def test_report_uses_doctor_facing_warning_and_avoids_decisive_wording() -> None:
    risk_item = _risk_item()
    risk_item["recommendation"] = "Cần ngừng thuốc và đổi thuốc."
    report = ReportGeneratorService().generate_report(
        _normalized_result(),
        _evidence_bundle(),
        {
            "status": "analysis_ready",
            "overall_risk_level": "moderate",
            "risk_items": [risk_item],
            "missing_information": [],
            "warnings": ["drug_mapping_not_found"],
            "errors": [],
        },
    )
    joined = str(report).casefold()

    assert report["doctor_facing_warnings"] == [
        "Một số dòng thuốc chưa được hệ thống nhận diện chắc chắn, cần rà soát lại."
    ]
    for phrase in [
        "đơn thuốc an toàn",
        "đơn thuốc không an toàn",
        "dùng được",
        "không dùng được",
        "ngừng thuốc",
        "đổi thuốc",
        "tăng liều",
        "giảm liều",
    ]:
        assert phrase not in joined


def test_evidence_snippet_length_limit() -> None:
    service = ReportGeneratorService(max_evidence_snippet_length=40)
    index = service.build_evidence_index(_evidence_bundle())

    assert len(index["interaction-1"]["snippet"]) <= 40


def test_unknown_risk_status_returns_cannot_generate_report() -> None:
    report = ReportGeneratorService().generate_report(
        _normalized_result(),
        _evidence_bundle(),
        {"status": "surprising_status", "warnings": [], "errors": []},
    )

    assert report["status"] == "cannot_generate_report"
    assert report["errors"] == ["unsupported_risk_analysis_status"]


def test_invalid_max_evidence_snippet_length_raises() -> None:
    with pytest.raises(ValueError):
        ReportGeneratorService(max_evidence_snippet_length=0)


def test_get_stats_returns_metadata() -> None:
    stats = ReportGeneratorService(max_evidence_snippet_length=123).get_stats()

    assert stats == {
        "service": "ReportGeneratorService",
        "report_type": "prescription_audit",
        "audience": "doctor_or_pharmacist",
        "max_evidence_snippet_length": 123,
    }
