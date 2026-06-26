from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from backend.app.core.config import get_settings
from backend.app.services.doctor_report_composer_service import (
    GEMINI_COMPOSER_FAILED_WARNING,
    GEMINI_COMPOSER_SAFETY_WARNING,
    DoctorReportComposerService,
    build_composer_payload,
)
from backend.app.services.doctor_report_text_safety import (
    has_unsafe_doctor_report_text,
    sanitize_doctor_report_text,
)
from backend.app.services.gemini_doctor_report_composer_client import (
    GeminiDoctorReportComposerClient,
)


DISCLAIMER = (
    "Báo cáo này chỉ nhằm hỗ trợ bác sĩ/dược sĩ rà soát đơn thuốc và không "
    "thay thế quyết định chuyên môn."
)


def _report() -> dict[str, Any]:
    return {
        "status": "report_ready",
        "overall_risk_level": "high",
        "summary": "Mức nguy cơ tổng quan: high. Cần rà soát theo bối cảnh lâm sàng.",
        "risk_items": [
            {
                "severity": "high",
                "title": "Metformin và suy thận",
                "explanation": "eGFR thấp nên cần rà soát sử dụng metformin.",
                "recommendation": "Bác sĩ/dược sĩ cần rà soát trước khi quyết định.",
                "evidence_refs": ["chunk-1"],
                "evidence": [
                    {
                        "chunk_id": "chunk-1",
                        "source": "trungtamthuoc",
                        "slug": "metformin",
                        "section": "than_trong",
                        "url": "https://example.test/metformin",
                    }
                ],
            }
        ],
        "evidence_sources": [{"chunk_id": "chunk-1", "url": "https://example.test"}],
        "medication_summary": [
            {"raw_line": "1. Metformin 750mg", "mapping_status": "mapped"}
        ],
        "medications_requiring_review": [
            {
                "raw_line": "1. Metformin 750mg",
                "instruction": "Ngày uống 1 lần",
                "mapping_status": "unmatched",
                "requires_review": True,
                "warnings": ["drug_mapping_not_found"],
            }
        ],
        "missing_information": ["pregnancy_status", "renal_function"],
        "doctor_facing_warnings": [
            "Một số dòng thuốc chưa được hệ thống nhận diện chắc chắn, cần rà soát lại."
        ],
        "warnings": ["drug_mapping_not_found"],
        "errors": ["technical_error"],
        "safety_disclaimer": DISCLAIMER,
    }


class FakeGeminiClient:
    def __init__(self, response: str | None = None, raises: bool = False) -> None:
        self.response = response
        self.raises = raises
        self.payloads: list[dict[str, Any]] = []

    def compose(self, payload: dict[str, Any]) -> dict[str, str]:
        self.payloads.append(payload)
        if self.raises:
            raise RuntimeError("Gemini failed")
        return {"doctor_facing_response": self.response or ""}


def test_disabled_composer_generates_response_without_gemini_warning() -> None:
    report = _report()
    result = DoctorReportComposerService(enabled=False).compose(report)

    assert result["doctor_facing_response"].startswith("Kết quả kiểm tra đơn thuốc")
    assert "Mức nguy cơ tổng quan" not in result["doctor_facing_response"]
    assert " high" not in result["doctor_facing_response"].casefold()
    assert " moderate" not in result["doctor_facing_response"].casefold()
    assert " low" not in result["doctor_facing_response"].casefold()
    assert GEMINI_COMPOSER_FAILED_WARNING not in result["warnings"]
    assert GEMINI_COMPOSER_SAFETY_WARNING not in result["warnings"]


def test_gemini_report_enable_env_boolean_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("GEMINI_REPORT_ENABLE", "true")
    assert get_settings().gemini_report_enable is True

    get_settings.cache_clear()
    monkeypatch.setenv("GEMINI_REPORT_ENABLE", "false")
    assert get_settings().gemini_report_enable is False

    get_settings.cache_clear()


def test_composer_does_not_mutate_input_or_structured_fields() -> None:
    report = _report()
    original = copy.deepcopy(report)

    result = DoctorReportComposerService(enabled=False).compose(report)

    assert report == original
    for field in (
        "risk_items",
        "evidence_sources",
        "medication_summary",
        "medications_requiring_review",
        "missing_information",
    ):
        assert result[field] == original[field]


def test_build_composer_payload_excludes_technical_internals() -> None:
    payload = build_composer_payload(_report())
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["review_priority"] == "Cao"
    assert payload["risk_items"][0]["severity"] == "Cao"
    assert "chunk-1" not in serialized
    assert "drug_mapping_not_found" not in serialized
    assert "technical_error" not in serialized
    assert "mapping_status" not in serialized
    assert (
        "Trung Tâm Thuốc / Dược thư Quốc gia Việt Nam — metformin — Thận trọng"
        in serialized
    )
    assert "Tình trạng thai kỳ/cho con bú chưa được ghi nhận." in serialized


def test_deterministic_fallback_uses_natural_opening_without_raw_summary() -> None:
    result = DoctorReportComposerService(enabled=False).compose(_report())
    text = result["doctor_facing_response"]

    assert text.startswith("Kết quả kiểm tra đơn thuốc")
    assert (
        "Hệ thống đã rà soát đơn thuốc dựa trên dữ liệu hiện có và ghi nhận "
        "một số điểm cần bác sĩ/dược sĩ xem xét thêm."
    ) in text
    assert "Mức ưu tiên rà soát: Cao." in text
    assert "Mức nguy cơ tổng quan" not in text


def test_enabled_composer_uses_safe_gemini_response() -> None:
    response = (
        "Kết quả kiểm tra đơn thuốc\n\n"
        "Mức ưu tiên rà soát: cao.\n\n"
        "1. Metformin và suy thận\n"
        "- Bác sĩ/dược sĩ cần rà soát theo bối cảnh lâm sàng.\n\n"
        f"{DISCLAIMER}"
    )
    client = FakeGeminiClient(response=response)
    result = DoctorReportComposerService(
        enabled=True, gemini_client_factory=lambda: client
    ).compose(_report())

    assert result["doctor_facing_response"] == response
    assert client.payloads
    assert GEMINI_COMPOSER_FAILED_WARNING not in result["warnings"]


def test_enabled_composer_failure_falls_back_with_warning() -> None:
    result = DoctorReportComposerService(
        enabled=True, gemini_client_factory=lambda: FakeGeminiClient(raises=True)
    ).compose(_report())

    assert result["doctor_facing_response"].startswith("Kết quả kiểm tra đơn thuốc")
    assert GEMINI_COMPOSER_FAILED_WARNING in result["warnings"]


def test_enabled_composer_safety_violation_falls_back_with_warning() -> None:
    unsafe_response = (
        "Kết quả kiểm tra đơn thuốc\n\n"
        "Đơn thuốc an toàn, dùng được.\n\n"
        f"{DISCLAIMER}"
    )
    result = DoctorReportComposerService(
        enabled=True,
        gemini_client_factory=lambda: FakeGeminiClient(response=unsafe_response),
    ).compose(_report())

    assert result["doctor_facing_response"].startswith("Kết quả kiểm tra đơn thuốc")
    assert GEMINI_COMPOSER_SAFETY_WARNING in result["warnings"]
    assert "đơn thuốc an toàn" not in result["doctor_facing_response"].casefold()
    assert "dùng được" not in result["doctor_facing_response"].casefold()


def test_final_text_safety_handles_tu_y_patterns() -> None:
    assert not has_unsafe_doctor_report_text("Người bệnh không tự ý ngừng thuốc.")
    assert has_unsafe_doctor_report_text("Người bệnh tự ý ngừng thuốc.")

    sanitized = sanitize_doctor_report_text("Có thể tự ý tăng liều.")

    assert "tự ý tăng" not in sanitized.casefold()


def test_gemini_client_prompt_and_parse_response() -> None:
    client = GeminiDoctorReportComposerClient(api_key="test", model_name="gemini-test")
    prompt = client.build_prompt({"risk_items": [], "disclaimer": DISCLAIMER})

    assert "Không tạo phát hiện mới" in prompt
    assert "Không dùng Markdown table" in prompt
    assert "doctor_facing_response" in prompt

    parsed = client.parse_response(
        '```json\n{"doctor_facing_response": "Kết quả kiểm tra đơn thuốc"}\n```'
    )

    assert parsed == {"doctor_facing_response": "Kết quả kiểm tra đơn thuốc"}


@pytest.mark.parametrize("response", ["not-json", "{}", "[]"])
def test_gemini_client_rejects_invalid_response(response: str) -> None:
    client = GeminiDoctorReportComposerClient(api_key="test")

    with pytest.raises(ValueError):
        client.parse_response(response)


def test_deterministic_fallback_avoids_form_labels_and_long_source_repetition() -> None:
    result = DoctorReportComposerService(enabled=False).compose(_report())
    text = result["doctor_facing_response"]

    assert "Mức độ cần chú ý:" not in text
    assert "Nội dung đánh giá:" not in text
    assert "Gợi ý rà soát:" not in text
    assert text.count("Trung Tâm Thuốc / Dược thư Quốc gia Việt Nam") <= 1
    assert "Có nguồn tham khảo kèm theo bên dưới." in text


def test_missing_information_variants_are_mapped_deduplicated_and_filtered() -> None:
    report = _report()
    report["missing_information"] = [
        "pregnancy_status",
        "pregnancy_lactation",
        "Tình trạng thai kỳ/cho con bú của bệnh nhân",
        "Cần xác nhận tình trạng thai kỳ/cho con bú của bệnh nhân.",
        "hepatic_function",
        "Tình trạng chức năng gan của bệnh nhân",
        "current_medications",
        "Cần xác nhận các thuốc bệnh nhân đang sử dụng ngoài đơn thuốc này.",
        "Thông tin chi tiết về Sucralfate chưa được xác định rõ ràng trong cơ sở dữ liệu để phân tích toàn diện.",
    ]

    result = DoctorReportComposerService(enabled=False).compose(report)
    text = result["doctor_facing_response"]

    assert text.count("Tình trạng thai kỳ/cho con bú chưa được ghi nhận.") == 1
    assert text.count("Chức năng gan chưa được ghi nhận hoặc cần xác nhận.") == 1
    assert (
        text.count(
            "Các thuốc bệnh nhân đang dùng ngoài đơn thuốc chưa được ghi nhận."
        )
        == 1
    )
    assert "pregnancy_status" not in text
    assert "pregnancy_lactation" not in text
    assert "hepatic_function" not in text
    assert "current_medications" not in text
    assert "Thông tin chi tiết về Sucralfate" not in text
    assert "cơ sở dữ liệu" not in text
    assert (
        "Một số tên thuốc/biệt dược cần được hệ thống rà soát thêm về mặt nhận diện dữ liệu."
        not in text
    )


def test_patient_context_known_negative_values_filter_missing_information() -> None:
    report = _report()
    report["patient_context"] = {
        "allergies": "Chưa ghi nhận",
        "diagnoses": ["Viêm phế quản", "loét dạ dày tá tràng"],
        "comorbidities": "Không ghi nhận",
        "pregnancy_lactation": "Không",
        "current_medications": "Không",
        "hepatic_function": "Chưa có thông tin",
        "renal_function": "Bình thường",
    }
    report["missing_information"] = [
        "allergies",
        "diagnoses",
        "current_medications",
        "pregnancy_status",
        "pregnancy_lactation",
        "hepatic_function",
        "Tiền sử dị ứng",
        "Các bệnh đồng mắc khác",
    ]

    result = DoctorReportComposerService(enabled=False).compose(report)
    text = result["doctor_facing_response"]

    assert "Chức năng gan chưa được ghi nhận hoặc cần xác nhận." in text
    assert text.count("Chức năng gan chưa được ghi nhận hoặc cần xác nhận.") == 1
    assert "Tiền sử dị ứng" not in text
    assert "Các bệnh đồng mắc khác" not in text
    assert "Bệnh nền/chẩn đoán chưa được ghi nhận đầy đủ" not in text
    assert "Tình trạng thai kỳ/cho con bú chưa được ghi nhận" not in text
    assert "Các thuốc bệnh nhân đang dùng ngoài đơn thuốc chưa được ghi nhận" not in text
    assert "pregnancy_status" not in text
    assert "pregnancy_lactation" not in text
    assert "current_medications" not in text


def test_patient_context_lactation_value_filters_pregnancy_missing_information() -> None:
    report = _report()
    report["patient_context"] = {"pregnancy_lactation": "Cho con bú"}
    report["missing_information"] = [
        "pregnancy_status",
        "pregnancy_lactation",
    ]

    result = DoctorReportComposerService(enabled=False).compose(report)
    text = result["doctor_facing_response"]

    assert "Tình trạng thai kỳ/cho con bú chưa được ghi nhận" not in text
    assert "pregnancy_status" not in text
    assert "pregnancy_lactation" not in text


def test_deterministic_fallback_hides_mapping_database_uncertainty_from_main_text() -> None:
    result = DoctorReportComposerService(enabled=False).compose(_report())
    text = result["doctor_facing_response"]

    assert "Thuốc/dòng cần rà soát lại" not in text
    assert "Cần rà soát nhận diện/chuẩn hóa thuốc" not in text
    assert "cơ sở dữ liệu" not in text
    assert "mapping" not in text.casefold()
    assert "database" not in text.casefold()
    assert "Một số dòng thuốc chưa được hệ thống nhận diện chắc chắn" not in text


def test_directive_wording_is_softened_in_doctor_facing_response() -> None:
    report = _report()
    report["risk_items"][0]["recommendation"] = (
        "Khuyên bệnh nhân uống Sucralfate cách Levofloxacin ít nhất 2 giờ."
    )
    report["risk_items"].append(
        {
            "severity": "moderate",
            "title": "Thận trọng khi dùng Omeprazole",
            "explanation": "Cân nhắc thực hiện các xét nghiệm cần thiết để loại trừ nguy cơ.",
            "recommendation": "Hướng dẫn bệnh nhân uống thuốc theo hướng dẫn. Để tránh tương tác, cần uống cách xa thuốc khác.",
            "evidence_refs": [],
            "evidence": [],
        }
    )

    result = DoctorReportComposerService(enabled=False).compose(report)
    text = result["doctor_facing_response"]

    assert "Khuyên bệnh nhân" not in text
    assert "Cân nhắc thực hiện các xét nghiệm" not in text
    assert "Hướng dẫn bệnh nhân" not in text
    assert "cần uống" not in text.casefold()
    assert "Bác sĩ/dược sĩ nên rà soát thời điểm dùng thuốc" in text
    assert "Bác sĩ/dược sĩ nên đối chiếu" in text


def test_omeprazole_directive_wording_is_softened() -> None:
    report = _report()
    report["risk_items"] = [
        {
            "severity": "moderate",
            "title": "Thận trọng khi dùng Omeprazole cho bệnh nhân loét dạ dày",
            "explanation": (
                "Cần thăm khám và thực hiện các xét nghiệm cần thiết để loại trừ "
                "khả năng u ác tính dạ dày trước khi tiếp tục điều trị bằng Omeprazole."
            ),
            "recommendation": (
                "Cần thực hiện các xét nghiệm cần thiết trước khi tiếp tục điều trị bằng Omeprazole."
            ),
            "evidence_refs": [],
            "evidence": [],
        }
    ]

    result = DoctorReportComposerService(enabled=False).compose(report)
    text = result["doctor_facing_response"]

    assert "Cần thăm khám" not in text
    assert "Cần thực hiện các xét nghiệm" not in text
    assert "thực hiện các xét nghiệm cần thiết" not in text
    assert "tiếp tục điều trị bằng" not in text
    assert (
        "Bác sĩ/dược sĩ nên đối chiếu với triệu chứng, chẩn đoán và kế hoạch theo dõi"
        in text
    )
    assert "cân nhắc đánh giá thêm nếu phù hợp" in text


def test_gemini_prompt_requests_natural_style_and_no_debug_missing_info() -> None:
    client = GeminiDoctorReportComposerClient(api_key="test", model_name="gemini-test")
    prompt = client.build_prompt({"risk_items": [], "disclaimer": DISCLAIMER})

    assert "Không lặp lại các nhãn" in prompt
    assert "pregnancy_status" in prompt
    assert "mapping/database uncertainty" in prompt
    assert "Không lặp lại nhãn nguồn dài" in prompt
    assert "Không viết như chỉ dẫn trực tiếp cho bệnh nhân" in prompt
    assert "Không ghi nhận hoặc Chưa ghi nhận" in prompt
    assert "cảnh báo Omeprazole/PPI" in prompt
