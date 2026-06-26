from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.prescription_audit_service import PrescriptionAuditService


def _prescription_check(
    status: str = "evidence_ready",
    normalized_result: dict[str, Any] | None = None,
    evidence_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "normalized_result": normalized_result
        if normalized_result is not None
        else {"medications": [], "unique_evidence_slugs": ["omeprazole"]},
        "evidence_bundle": evidence_bundle
        if evidence_bundle is not None
        else {"unique_chunks": [{"chunk_id": "chunk-1", "text": "Evidence"}]},
        "warnings": ["check_warning"],
        "errors": [],
    }


def _risk_analysis(status: str = "analysis_context_ready") -> dict[str, Any]:
    return {
        "status": status,
        "overall_risk_level": "unknown",
        "risk_items": [],
        "missing_information": [],
        "warnings": ["risk_warning"],
        "errors": [],
    }


def _report(status: str = "report_context_ready") -> dict[str, Any]:
    return {
        "status": status,
        "overall_risk_level": "unknown",
        "warnings": ["report_warning"],
        "errors": [],
    }


class FakePrescriptionCheckService:
    def __init__(self, result: dict[str, Any] | None = None, raises: bool = False) -> None:
        self.result = result or _prescription_check()
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def check_text(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError("check failed")
        return self.result

    def get_stats(self) -> dict[str, Any]:
        return {"service": "FakePrescriptionCheckService"}


class FakeRiskAnalyzer:
    def __init__(self, result: dict[str, Any] | None = None, raises: bool = False) -> None:
        self.result = result or _risk_analysis()
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def analyze(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError("analysis failed")
        return self.result


class FakeRiskAnalyzerFactory:
    def __init__(self, analyzer: FakeRiskAnalyzer) -> None:
        self.analyzer = analyzer
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeRiskAnalyzer:
        self.calls.append(kwargs)
        return self.analyzer


class FakeReportGeneratorService:
    def __init__(self, result: dict[str, Any] | None = None, raises: bool = False) -> None:
        self.result = result or _report()
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def generate_report(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError("report failed")
        return self.result

    def get_stats(self) -> dict[str, Any]:
        return {"service": "FakeReportGeneratorService"}


class FakeDoctorReportComposerService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def compose(self, report: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(report)
        return report | {"doctor_facing_response": "Kết quả kiểm tra đơn thuốc"}

    def get_stats(self) -> dict[str, Any]:
        return {"service": "FakeDoctorReportComposerService"}


class FakeSafetyLayerService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        risk_analysis = dict(kwargs["risk_analysis"])
        risk_analysis["warnings"] = [
            *(risk_analysis.get("warnings") or []),
            "safety_applied",
        ]
        risk_analysis["safety_marker"] = True
        return risk_analysis

    def get_stats(self) -> dict[str, Any]:
        return {"service": "FakeSafetyLayerService"}


class FakeDoctorMemoryService:
    def __init__(
        self,
        result: dict[str, Any] | None = None,
        raises: bool = False,
    ) -> None:
        self.result = result if result is not None else {"matched_notes": []}
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def retrieve_for_audit_context(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError("memory failed")
        return self.result

    def get_stats(self) -> dict[str, Any]:
        return {"service": "FakeDoctorMemoryService"}


class FakeGeminiFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        return object()


STRUCTURED_DOCUMENT = """ĐƠN NGOẠI TRÚ 1
Bệnh viện: Bệnh viện A
Khoa: Tiêu hóa
Đơn thuốc

I.THÔNG TIN BỆNH NHÂN
Họ và tên: Hoàng Thị P.
Tuổi: 28
Nam/Nữ: Nữ
Cân nặng: 60kg
Địa chỉ: Nga Sơn, Thanh Hóa

II.THÔNG TIN LÂM SÀNG
Chẩn đoán: Viêm phế quản/loét dạ dày tá tràng
Dị ứng thuốc: Chưa ghi nhận
Bệnh nền: Không ghi nhận
Chức năng gan: Chưa có thông tin
Chức năng thận: Bình thường
Thai kỳ/ cho con bú: Chưa có thông tin
Thuốc khác đang dùng: Chưa có thông tin

III.CHỈ ĐỊNH DÙNG THUỐC
1. Omeprazole (Losec) 20mg x 15 viên
   Ngày uống 1 lần, mỗi lần 1 viên
2. Sucralfate (Sucrate Gel) 1g/5mL x 15 gói
   Ngày uống 3 lần, mỗi lần 1 gói

Ngày, tháng, năm
Bác sĩ khám bệnh
"""


STRUCTURED_MEDICATION_TEXT = """1. Omeprazole (Losec) 20mg x 15 viên
   Ngày uống 1 lần, mỗi lần 1 viên
2. Sucralfate (Sucrate Gel) 1g/5mL x 15 gói
   Ngày uống 3 lần, mỗi lần 1 gói"""


def test_use_gemini_false_context_ready_returns_partial_success() -> None:
    checker = FakePrescriptionCheckService()
    analyzer = FakeRiskAnalyzer(_risk_analysis("analysis_context_ready"))
    analyzer_factory = FakeRiskAnalyzerFactory(analyzer)
    reporter = FakeReportGeneratorService(_report("report_context_ready"))
    safety = FakeSafetyLayerService()
    service = PrescriptionAuditService(
        prescription_check_service=checker,
        report_generator_service=reporter,
        doctor_report_composer_service=FakeDoctorReportComposerService(),
        safety_layer_service=safety,
        risk_analyzer_service_factory=analyzer_factory,
    )

    result = service.audit_text("1. Omeprazol 20mg", doctor_id="doctor-1")

    assert result["status"] == "partial_success"
    assert result["prescription_check"]["status"] == "evidence_ready"
    assert result["risk_analysis"]["status"] == "analysis_context_ready"
    assert result["report"]["status"] == "report_context_ready"
    assert analyzer_factory.calls == [{}]
    assert checker.calls[0]["doctor_id"] == "doctor-1"
    assert safety.calls[0]["risk_analysis"]["status"] == "analysis_context_ready"
    assert reporter.calls[0]["risk_analysis"]["safety_marker"] is True
    assert reporter.calls[0]["normalized_result"] == checker.result["normalized_result"]
    assert result["report"]["doctor_facing_response"].startswith(
        "Kết quả kiểm tra đơn thuốc"
    )


def test_use_gemini_true_uses_gemini_factory_and_success() -> None:
    analyzer = FakeRiskAnalyzer(_risk_analysis("analysis_ready"))
    analyzer_factory = FakeRiskAnalyzerFactory(analyzer)
    reporter = FakeReportGeneratorService(_report("report_ready"))
    gemini_factory = FakeGeminiFactory()
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(),
        report_generator_service=reporter,
        doctor_report_composer_service=FakeDoctorReportComposerService(),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=analyzer_factory,
        gemini_client_factory=gemini_factory,
    )

    result = service.audit_text("1. Omeprazol 20mg", use_gemini=True)

    assert result["status"] == "success"
    assert gemini_factory.calls == 1
    assert "llm_client" in analyzer_factory.calls[0]


@pytest.mark.parametrize("status", ["invalid_input", "error"])
def test_invalid_or_error_prescription_check_returns_failed_early(status: str) -> None:
    checker = FakePrescriptionCheckService(
        _prescription_check(status=status, normalized_result={"medications": []})
    )
    analyzer = FakeRiskAnalyzer()
    reporter = FakeReportGeneratorService()
    service = PrescriptionAuditService(
        prescription_check_service=checker,
        report_generator_service=reporter,
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(analyzer),
    )

    result = service.audit_text("bad input")

    assert result["status"] == "failed"
    assert analyzer.calls == []
    assert reporter.calls == []


def test_insufficient_information_with_normalized_result_continues() -> None:
    checker = FakePrescriptionCheckService(
        _prescription_check(status="insufficient_information")
    )
    analyzer = FakeRiskAnalyzer(_risk_analysis("insufficient_evidence"))
    reporter = FakeReportGeneratorService(_report("report_insufficient_evidence"))
    safety = FakeSafetyLayerService()
    service = PrescriptionAuditService(
        prescription_check_service=checker,
        report_generator_service=reporter,
        safety_layer_service=safety,
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(analyzer),
    )

    result = service.audit_text("1. Omeprazol 20mg")

    assert result["status"] == "partial_success"
    assert analyzer.calls
    assert safety.calls
    assert reporter.calls
    assert result["report"]["status"] == "report_insufficient_evidence"


def test_missing_normalized_result_returns_failed() -> None:
    checker = FakePrescriptionCheckService(
        _prescription_check(normalized_result=None) | {"normalized_result": None}
    )
    analyzer = FakeRiskAnalyzer()
    service = PrescriptionAuditService(
        prescription_check_service=checker,
        report_generator_service=FakeReportGeneratorService(),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(analyzer),
    )

    result = service.audit_text("1. Omeprazol 20mg")

    assert result["status"] == "failed"
    assert "normalized_result_missing" in result["errors"]
    assert analyzer.calls == []


def test_top_k_per_type_must_be_positive() -> None:
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(),
        report_generator_service=FakeReportGeneratorService(),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(FakeRiskAnalyzer()),
    )

    with pytest.raises(ValueError):
        service.audit_text("1. Omeprazol 20mg", top_k_per_type=0)


def test_warnings_and_errors_are_deduplicated() -> None:
    checker_result = _prescription_check()
    checker_result["warnings"] = ["same", "same"]
    risk_result = _risk_analysis()
    risk_result["warnings"] = ["same", "risk"]
    report_result = _report("report_context_ready")
    report_result["warnings"] = ["risk", "report"]
    report_result["errors"] = ["e1", "e1"]
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(checker_result),
        report_generator_service=FakeReportGeneratorService(report_result),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(FakeRiskAnalyzer(risk_result)),
    )

    result = service.audit_text("1. Omeprazol 20mg")

    assert result["warnings"] == ["same", "risk", "safety_applied", "report"]
    assert result["errors"] == ["e1"]


def test_prescription_check_exception_returns_failed() -> None:
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(raises=True),
        report_generator_service=FakeReportGeneratorService(),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(FakeRiskAnalyzer()),
    )

    result = service.audit_text("1. Omeprazol 20mg")

    assert result["status"] == "failed"
    assert result["errors"] == ["prescription_check_failed"]


def test_analyzer_exception_flows_to_report_analysis_failed() -> None:
    reporter = FakeReportGeneratorService(_report("report_analysis_failed"))
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(),
        report_generator_service=reporter,
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(FakeRiskAnalyzer(raises=True)),
    )

    result = service.audit_text("1. Omeprazol 20mg")

    assert result["status"] == "partial_success"
    assert result["risk_analysis"]["status"] == "analysis_failed"
    assert "risk_analysis_failed" in result["errors"]
    assert "safety_applied" in result["warnings"]


def test_report_exception_returns_failed() -> None:
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(),
        report_generator_service=FakeReportGeneratorService(raises=True),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(FakeRiskAnalyzer()),
    )

    result = service.audit_text("1. Omeprazol 20mg")

    assert result["status"] == "failed"
    assert "report_generation_failed" in result["errors"]


def test_get_stats_returns_metadata() -> None:
    composer = FakeDoctorReportComposerService()
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(),
        report_generator_service=FakeReportGeneratorService(),
        doctor_report_composer_service=composer,
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(FakeRiskAnalyzer()),
    )

    stats = service.get_stats()

    assert stats["service"] == "PrescriptionAuditService"
    assert stats["prescription_check_service"] == {
        "service": "FakePrescriptionCheckService"
    }
    assert stats["report_generator_service"] == {
        "service": "FakeReportGeneratorService"
    }
    assert stats["doctor_report_composer_service"] == {
        "service": "FakeDoctorReportComposerService"
    }
    assert stats["safety_layer_service"] == {
        "service": "FakeSafetyLayerService"
    }


def test_audit_service_calls_doctor_report_composer_after_report_generation() -> None:
    reporter = FakeReportGeneratorService(_report("report_ready"))
    composer = FakeDoctorReportComposerService()
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(),
        report_generator_service=reporter,
        doctor_report_composer_service=composer,
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(
            FakeRiskAnalyzer(_risk_analysis("analysis_ready"))
        ),
    )

    result = service.audit_text("1. Omeprazol 20mg")

    assert composer.calls == [reporter.result]
    assert result["report"]["doctor_facing_response"] == "Kết quả kiểm tra đơn thuốc"


def test_structured_document_parser_sends_only_medications_to_checker() -> None:
    checker = FakePrescriptionCheckService()
    service = PrescriptionAuditService(
        prescription_check_service=checker,
        report_generator_service=FakeReportGeneratorService(),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(FakeRiskAnalyzer()),
    )

    result = service.audit_text(
        STRUCTURED_DOCUMENT,
        patient_context={
            "age": None,
            "sex": "unknown",
            "renal_function": "not provided",
            "diagnoses": [],
            "allergies": "",
            "current_medications": "not provided",
        },
    )

    assert checker.calls[0]["prescription_text"] == STRUCTURED_MEDICATION_TEXT
    assert checker.calls[0]["patient_context"]["age"] == 28
    assert checker.calls[0]["patient_context"]["sex"] == "female"
    assert checker.calls[0]["patient_context"]["renal_function"] == "Bình thường"
    assert checker.calls[0]["patient_context"]["diagnoses"] == [
        "Viêm phế quản",
        "loét dạ dày tá tràng",
    ]
    assert checker.calls[0]["patient_context"]["allergies"] == "Chưa ghi nhận"
    assert "prescription_document_parser_applied" in result["warnings"]


def test_structured_document_meaningful_incoming_context_overrides_parsed() -> None:
    checker = FakePrescriptionCheckService()
    service = PrescriptionAuditService(
        prescription_check_service=checker,
        report_generator_service=FakeReportGeneratorService(),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(FakeRiskAnalyzer()),
    )

    service.audit_text(
        STRUCTURED_DOCUMENT,
        patient_context={
            "age": 99,
            "sex": "male",
            "renal_function": "eGFR 25 ml/min/1.73m2",
            "allergies": "Không rõ",
            "extra_context": {"source": "manual"},
        },
    )

    context = checker.calls[0]["patient_context"]
    assert context["age"] == 99
    assert context["sex"] == "male"
    assert context["renal_function"] == "eGFR 25 ml/min/1.73m2"
    assert context["allergies"] == "Không rõ"
    assert context["patient_name"] == "Hoàng Thị P."
    assert context["extra_context"] == {"source": "manual"}


def test_structured_document_without_medications_returns_early() -> None:
    checker = FakePrescriptionCheckService()
    analyzer = FakeRiskAnalyzer()
    reporter = FakeReportGeneratorService()
    safety = FakeSafetyLayerService()
    service = PrescriptionAuditService(
        prescription_check_service=checker,
        report_generator_service=reporter,
        safety_layer_service=safety,
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(analyzer),
    )

    result = service.audit_text(
        """I. THÔNG TIN BỆNH NHÂN
Họ tên: Nguyễn Văn A
III. CHỈ ĐỊNH DÙNG THUỐC
Ngày, tháng, năm
Bác sĩ khám bệnh
"""
    )

    assert result["status"] == "failed"
    assert result["warnings"] == [
        "prescription_document_parser_applied",
        "prescription_document_parser_no_medication_lines",
    ]
    assert checker.calls == []
    assert analyzer.calls == []
    assert reporter.calls == []
    assert safety.calls == []


def test_non_structured_text_remains_unchanged() -> None:
    checker = FakePrescriptionCheckService()
    service = PrescriptionAuditService(
        prescription_check_service=checker,
        report_generator_service=FakeReportGeneratorService(),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(FakeRiskAnalyzer()),
    )
    text = "Aspirin có tương tác Warfarin không?"

    result = service.audit_text(text, patient_context={"sex": "unknown"})

    assert checker.calls[0]["prescription_text"] == text
    assert checker.calls[0]["patient_context"] == {"sex": "unknown"}
    assert "prescription_document_parser_applied" not in result["warnings"]


def test_audit_includes_matching_doctor_memory_section() -> None:
    normalized_result = {
        "medications": [
            {"active_ingredients": [{"evidence_slug": "levofloxacin"}]},
            {"active_ingredients": [{"evidence_slug": "sucralfate"}]},
        ],
        "unique_evidence_slugs": ["levofloxacin", "sucralfate"],
    }
    memory = FakeDoctorMemoryService(
        {
            "matched_notes": [
                {
                    "note_id": "n1",
                    "title": "Levofloxacin + Sucralfate",
                    "note_text": "Rà soát thời điểm dùng.",
                    "drug_pair_keys": ["levofloxacin|sucralfate"],
                    "score": 9.5,
                }
            ]
        }
    )
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(
            _prescription_check(normalized_result=normalized_result)
        ),
        report_generator_service=FakeReportGeneratorService(_report("report_ready")),
        doctor_report_composer_service=FakeDoctorReportComposerService(),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(
            FakeRiskAnalyzer(_risk_analysis("analysis_ready"))
        ),
        doctor_memory_service=memory,
    )

    result = service.audit_text("1. Levofloxacin\n2. Sucralfate", doctor_id="doctor-1")

    assert result["doctor_memory"]["matched_notes"][0]["note_id"] == "n1"
    assert result["report"]["doctor_memory"] == result["doctor_memory"]
    assert "Ghi chú riêng của bác sĩ" in result["report"]["doctor_facing_response"]
    assert "Rà soát thời điểm dùng" in result["report"]["doctor_facing_response"]
    assert memory.calls[0]["doctor_id"] == "doctor-1"
    assert "evidence_sources" not in result["report"] or not any(
        "Rà soát thời điểm dùng" in str(source)
        for source in result["report"].get("evidence_sources", [])
    )


def test_audit_no_memory_does_not_alter_doctor_facing_response() -> None:
    memory = FakeDoctorMemoryService({"matched_notes": []})
    composer = FakeDoctorReportComposerService()
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(),
        report_generator_service=FakeReportGeneratorService(_report("report_ready")),
        doctor_report_composer_service=composer,
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(
            FakeRiskAnalyzer(_risk_analysis("analysis_ready"))
        ),
        doctor_memory_service=memory,
    )

    result = service.audit_text("1. Omeprazol 20mg", doctor_id="doctor-1")

    assert result["doctor_memory"] == {"matched_notes": []}
    assert result["report"]["doctor_facing_response"] == composer.compose({})[
        "doctor_facing_response"
    ]
    assert "Ghi chú riêng của bác sĩ" not in result["report"]["doctor_facing_response"]


def test_audit_memory_failure_does_not_fail_audit() -> None:
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(),
        report_generator_service=FakeReportGeneratorService(_report("report_ready")),
        doctor_report_composer_service=FakeDoctorReportComposerService(),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(
            FakeRiskAnalyzer(_risk_analysis("analysis_ready"))
        ),
        doctor_memory_service=FakeDoctorMemoryService(raises=True),
    )

    result = service.audit_text("1. Omeprazol 20mg", doctor_id="doctor-1")

    assert result["status"] == "success"
    assert result["doctor_memory"] == {"matched_notes": []}
    assert "doctor_memory_retrieval_failed" in result["warnings"]


def test_doctor_memory_notes_are_sanitized_and_not_evidence_sources() -> None:
    memory = FakeDoctorMemoryService(
        {
            "matched_notes": [
                {
                    "note_id": "n1",
                    "title": "Private note",
                    "note_text": "Tăng liều hoặc đổi thuốc.",
                    "score": 8,
                }
            ]
        }
    )
    report = _report("report_ready") | {
        "evidence_sources": [{"chunk_id": "c1", "snippet": "medical evidence"}],
        "markdown_report": "## Nguồn tham khảo\n- c1",
    }
    service = PrescriptionAuditService(
        prescription_check_service=FakePrescriptionCheckService(),
        report_generator_service=FakeReportGeneratorService(report),
        doctor_report_composer_service=FakeDoctorReportComposerService(),
        safety_layer_service=FakeSafetyLayerService(),
        risk_analyzer_service_factory=FakeRiskAnalyzerFactory(
            FakeRiskAnalyzer(_risk_analysis("analysis_ready"))
        ),
        doctor_memory_service=memory,
    )

    result = service.audit_text("1. Omeprazol 20mg", doctor_id="doctor-1")
    doctor_text = result["report"]["doctor_facing_response"]

    assert "tăng liều" not in doctor_text.casefold()
    assert "đổi thuốc" not in doctor_text.casefold()
    assert "Ghi chú riêng của bác sĩ" in doctor_text
    assert result["report"]["evidence_sources"] == [
        {"chunk_id": "c1", "snippet": "medical evidence"}
    ]
    assert "Private note" not in result["report"]["markdown_report"]
