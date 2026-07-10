from __future__ import annotations

import copy
import unicodedata
from typing import Any, Callable, Iterable

from backend.app.core.config import get_settings
from backend.app.services.doctor_report_text_safety import (
    has_unsafe_doctor_report_text,
    sanitize_doctor_report_text,
)
from backend.app.services.gemini_doctor_report_composer_client import (
    GeminiDoctorReportComposerClient,
)
from backend.app.services.report_generator_service import SAFETY_DISCLAIMER


GEMINI_COMPOSER_FAILED_WARNING = "gemini_doctor_report_composer_failed"
GEMINI_COMPOSER_SAFETY_WARNING = "gemini_doctor_report_composer_safety_fallback"
DOCTOR_FACING_SECTION_ORDER = (
    "prescription_check",
    "interaction_check",
    "doctor_memory",
    "safety_note",
)
FULL_DOCTOR_FACING_SECTION_KEYS = set(DOCTOR_FACING_SECTION_ORDER)
NO_INTERACTION_WARNING = (
    "Dựa trên bằng chứng truy xuất hiện có, chưa ghi nhận tương tác thuốc-thuốc "
    "cần cảnh báo giữa các thuốc trong đơn."
)

SEVERITY_LABELS = {
    "high": "Cao",
    "moderate": "Trung bình",
    "low": "Thấp",
    "unknown": "Cần bổ sung thông tin",
}

SOURCE_LABELS = {
    "trungtamthuoc": "Trung Tâm Thuốc / Dược thư Quốc gia Việt Nam",
    "levofloxacin": "Levofloxacin",
    "omeprazole": "Omeprazole",
    "sucralfate": "Sucralfate",
    "tuong_tac_thuoc": "Tương tác thuốc",
    "than_trong": "Thận trọng",
    "chong_chi_dinh": "Chống chỉ định",
    "thai_ky_cho_con_bu": "Thai kỳ/cho con bú",
    "lieu_luong_va_cach_dung": "Liều lượng và cách dùng",
    "tac_dung_khong_mong_muon": "Tác dụng không mong muốn",
}

MISSING_INFORMATION_LABELS = {
    "pregnancy_status": "Tình trạng thai kỳ/cho con bú chưa được ghi nhận.",
    "pregnancy_lactation": "Tình trạng thai kỳ/cho con bú chưa được ghi nhận.",
    "tình trạng thai kỳ/cho con bú của bệnh nhân": "Tình trạng thai kỳ/cho con bú chưa được ghi nhận.",
    "cần xác nhận tình trạng thai kỳ/cho con bú của bệnh nhân": "Tình trạng thai kỳ/cho con bú chưa được ghi nhận.",
    "tình trạng thai kỳ/cho con bú chưa được ghi nhận": "Tình trạng thai kỳ/cho con bú chưa được ghi nhận.",
    "hepatic_function": "Chức năng gan chưa được ghi nhận hoặc cần xác nhận.",
    "tình trạng chức năng gan của bệnh nhân": "Chức năng gan chưa được ghi nhận hoặc cần xác nhận.",
    "cần xác nhận chức năng gan của bệnh nhân": "Chức năng gan chưa được ghi nhận hoặc cần xác nhận.",
    "renal_function": "Chức năng thận chưa được ghi nhận hoặc cần xác nhận.",
    "tình trạng chức năng thận của bệnh nhân": "Chức năng thận chưa được ghi nhận hoặc cần xác nhận.",
    "current_medications": "Các thuốc bệnh nhân đang dùng ngoài đơn thuốc chưa được ghi nhận.",
    "các thuốc đang sử dụng khác của bệnh nhân": "Các thuốc bệnh nhân đang dùng ngoài đơn thuốc chưa được ghi nhận.",
    "cần xác nhận các thuốc bệnh nhân đang sử dụng ngoài đơn thuốc này": "Các thuốc bệnh nhân đang dùng ngoài đơn thuốc chưa được ghi nhận.",
    "allergies": "Tiền sử dị ứng thuốc chưa được ghi nhận.",
    "tiền sử dị ứng": "Tiền sử dị ứng thuốc chưa được ghi nhận.",
    "tiền sử dị ứng thuốc của bệnh nhân": "Tiền sử dị ứng thuốc chưa được ghi nhận.",
    "diagnoses": "Bệnh nền/chẩn đoán chưa được ghi nhận đầy đủ.",
    "comorbidities": "Bệnh nền/chẩn đoán chưa được ghi nhận đầy đủ.",
    "bệnh nền của bệnh nhân": "Bệnh nền/chẩn đoán chưa được ghi nhận đầy đủ.",
    "các bệnh đồng mắc khác": "Bệnh nền/chẩn đoán chưa được ghi nhận đầy đủ.",
}

MISSING_INFORMATION_CATEGORIES = {
    "pregnancy_status": "pregnancy_lactation",
    "pregnancy_lactation": "pregnancy_lactation",
    "tình trạng thai kỳ/cho con bú của bệnh nhân": "pregnancy_lactation",
    "cần xác nhận tình trạng thai kỳ/cho con bú của bệnh nhân": "pregnancy_lactation",
    "tình trạng thai kỳ/cho con bú chưa được ghi nhận": "pregnancy_lactation",
    "hepatic_function": "hepatic_function",
    "tình trạng chức năng gan của bệnh nhân": "hepatic_function",
    "cần xác nhận chức năng gan của bệnh nhân": "hepatic_function",
    "renal_function": "renal_function",
    "tình trạng chức năng thận của bệnh nhân": "renal_function",
    "current_medications": "current_medications",
    "các thuốc đang sử dụng khác của bệnh nhân": "current_medications",
    "cần xác nhận các thuốc bệnh nhân đang sử dụng ngoài đơn thuốc này": "current_medications",
    "allergies": "allergies",
    "tiền sử dị ứng": "allergies",
    "tiền sử dị ứng thuốc của bệnh nhân": "allergies",
    "diagnoses": "diagnoses",
    "comorbidities": "diagnoses",
    "bệnh nền của bệnh nhân": "diagnoses",
    "các bệnh đồng mắc khác": "diagnoses",
}

KNOWN_NEGATIVE_VALUES = {
    "khong",
    "khong ghi nhan",
    "chua ghi nhan",
    "khong co",
    "khong mang thai",
    "khong cho con bu",
    "not pregnant",
    "not breastfeeding",
    "none",
    "no",
    "not recorded",
    "not noted",
    "not_applicable",
}

UNKNOWN_OR_MISSING_VALUES = {
    "",
    "unknown",
    "not provided",
    "chua co thong tin",
}

DATA_UNCERTAINTY_NOTE = (
    "Một số tên thuốc/biệt dược cần được hệ thống rà soát thêm về mặt nhận diện dữ liệu."
)

DATA_UNCERTAINTY_TERMS = (
    "sucralfate",
    "mapping",
    "database",
    "cơ sở dữ liệu",
    "không được nhận diện",
    "chưa được nhận diện",
    "drug not recognized",
    "not recognized",
    "ingredient evidence requires review",
    "không thể phân tích toàn diện",
    "chưa được xác định rõ ràng trong cơ sở dữ liệu",
    "thông tin chi tiết về levofloxacine",
    "thong tin chi tiet ve levofloxacine",
    "cơ sở dữ liệu để phân tích",
    "co so du lieu de phan tich",
)


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _fold_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFD", _text(value).casefold())
    return "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


def _normalized_missing_key(value: Any) -> str:
    return _text(value).strip(" .:;").casefold()


def _normalized_missing_category(value: Any) -> str:
    text = _normalized_missing_key(value)
    category = MISSING_INFORMATION_CATEGORIES.get(text)
    if category:
        return category
    folded = _fold_text(text)
    if "thai" in folded or "pregnancy" in folded or "breastfeeding" in folded:
        return "pregnancy_lactation"
    if "di ung" in folded or "allerg" in folded:
        return "allergies"
    if "benh dong mac" in folded or "benh nen" in folded or folded in {
        "diagnoses",
        "comorbidities",
    }:
        return "diagnoses"
    if "thuoc dang su dung" in folded or "thuoc khac" in folded:
        return "current_medications"
    if "chuc nang gan" in folded or folded == "hepatic_function":
        return "hepatic_function"
    if "chuc nang than" in folded or folded == "renal_function":
        return "renal_function"
    return ""


def _value_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_fold_text(item) for item in value if _text(item)]
    if isinstance(value, dict):
        return [_fold_text(item) for item in value.values() if _text(item)]
    return [_fold_text(value)] if _text(value) else []


def _is_known_negative(value: Any) -> bool:
    tokens = _value_tokens(value)
    return bool(tokens) and all(token in KNOWN_NEGATIVE_VALUES for token in tokens)


def _is_unknown_or_missing(value: Any) -> bool:
    tokens = _value_tokens(value)
    return not tokens or all(token in UNKNOWN_OR_MISSING_VALUES for token in tokens)


def _has_meaningful_value(value: Any) -> bool:
    tokens = _value_tokens(value)
    return bool(tokens) and not _is_known_negative(value) and not _is_unknown_or_missing(value)


def _patient_context_suppresses_missing(
    patient_context: dict[str, Any], category: str
) -> bool:
    if not category or not isinstance(patient_context, dict):
        return False
    if category == "allergies":
        return _is_known_negative(patient_context.get("allergies")) or _has_meaningful_value(
            patient_context.get("allergies")
        )
    if category == "diagnoses":
        return any(
            _is_known_negative(patient_context.get(key))
            or _has_meaningful_value(patient_context.get(key))
            for key in ("diagnoses", "comorbidities")
        )
    if category == "current_medications":
        return _is_known_negative(
            patient_context.get("current_medications")
        ) or _has_meaningful_value(patient_context.get("current_medications"))
    if category == "pregnancy_lactation":
        values = [
            patient_context.get("pregnancy_lactation"),
            patient_context.get("pregnancy_status"),
        ]
        return any(
            _is_known_negative(value) or _has_meaningful_value(value)
            for value in values
        )
    if category == "hepatic_function":
        return _is_known_negative(
            patient_context.get("hepatic_function")
        ) or _has_meaningful_value(patient_context.get("hepatic_function"))
    if category == "renal_function":
        return _is_known_negative(
            patient_context.get("renal_function")
        ) or _has_meaningful_value(patient_context.get("renal_function"))
    return False


def _is_data_uncertainty(value: str) -> bool:
    normalized = value.casefold()
    return any(term in normalized for term in DATA_UNCERTAINTY_TERMS)


def _severity_label(value: Any) -> str:
    normalized = _text(value).casefold()
    return SEVERITY_LABELS.get(normalized, "Cần bổ sung thông tin")


def _source_label(source: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("source", "slug", "section_title", "section"):
        value = _text(source.get(key))
        if not value:
            continue
        parts.append(SOURCE_LABELS.get(value.casefold(), value))
    return " — ".join(_deduplicate(parts)) or "Nguồn tham khảo"


def _map_missing_information(values: Any, patient_context: dict[str, Any] | None = None) -> list[str]:
    if not isinstance(values, list):
        return []
    mapped: list[str] = []
    for value in values:
        text = _text(value)
        if not text:
            continue
        normalized = _normalized_missing_key(text)
        if _is_data_uncertainty(normalized):
            continue
        category = _normalized_missing_category(text)
        if isinstance(patient_context, dict) and _patient_context_suppresses_missing(
            patient_context, category
        ):
            continue
        mapped.append(MISSING_INFORMATION_LABELS.get(normalized, text))
    return _deduplicate(mapped)


def _medication_display_name(medication: dict[str, Any]) -> str:
    return (
        _text(medication.get("raw_line"))
        or _text(medication.get("raw_name"))
        or _text(medication.get("generic_text"))
        or _text(medication.get("brand_text"))
        or "Không rõ tên thuốc"
    )


def build_composer_payload(report: dict[str, Any]) -> dict[str, Any]:
    risk_items: list[dict[str, Any]] = []
    for item in _as_dict_list(report.get("risk_items")):
        evidence_labels = _deduplicate(
            _source_label(source) for source in _as_dict_list(item.get("evidence"))
        )
        risk_items.append(
            {
                "risk_type": _text(item.get("risk_type")) or "general",
                "severity": _severity_label(item.get("severity")),
                "title": _text(item.get("title")) or "Điểm cần lưu ý",
                "explanation": _text(item.get("explanation")),
                "recommendation": _text(item.get("recommendation")),
                "evidence_sources": evidence_labels,
            }
        )

    medications_requiring_review: list[dict[str, Any]] = []
    for medication in _as_dict_list(report.get("medications_requiring_review")):
        medications_requiring_review.append(
            {
                "name": _medication_display_name(medication),
                "instruction": _text(medication.get("instruction")),
                "reason": "Cần rà soát nhận diện/chuẩn hóa thuốc."
                if medication.get("requires_review") is True
                else "",
            }
        )

    return {
        "heading": "Kết quả kiểm tra đơn thuốc",
        "review_priority_label": "Mức ưu tiên rà soát",
        "review_priority": _severity_label(report.get("overall_risk_level")),
        "summary": _text(report.get("summary")),
        "doctor_facing_warnings": [
            str(warning)
            for warning in report.get("doctor_facing_warnings", [])
            if warning and not _is_data_uncertainty(str(warning))
        ]
        if isinstance(report.get("doctor_facing_warnings"), list)
        else [],
        "risk_items": risk_items,
        "missing_information": _map_missing_information(
            report.get("missing_information"),
            report.get("patient_context") if isinstance(report.get("patient_context"), dict) else None,
        ),
        "medications_requiring_review": medications_requiring_review,
        "checked_query_types": [
            str(query_type)
            for query_type in report.get("checked_query_types", [])
            if query_type
        ]
        if isinstance(report.get("checked_query_types"), list)
        else [],
        "disclaimer": _text(report.get("safety_disclaimer")) or SAFETY_DISCLAIMER,
    }


def _combine_finding_text(item: dict[str, Any]) -> str:
    parts = [
        _text(item.get("explanation")),
    ]
    text = " ".join(part.rstrip(".") for part in parts if part).strip()
    if not text:
        return ""
    if not text.endswith("."):
        text = f"{text}."
    return text


def _section_item(item: dict[str, Any]) -> dict[str, str]:
    return {
        "title": sanitize_doctor_report_text(
            _text(item.get("title")) or "Điểm cần lưu ý"
        ),
        "severity": sanitize_doctor_report_text(_text(item.get("severity"))),
        "content": sanitize_doctor_report_text(_combine_finding_text(item)),
    }


def _risk_section_items(
    risk_items: list[dict[str, Any]],
    *,
    interaction: bool,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for item in risk_items:
        is_interaction = _text(item.get("risk_type")) == "interaction"
        if is_interaction != interaction:
            continue
        output.append(_section_item(item))
    return output


def build_doctor_facing_sections(
    payload: dict[str, Any],
    existing_sections: dict[str, Any] | None = None,
) -> dict[str, Any]:
    risk_items = _as_dict_list(payload.get("risk_items"))
    checked_query_types = {
        str(query_type) for query_type in payload.get("checked_query_types", [])
    }
    prescription_items = _risk_section_items(risk_items, interaction=False)
    interaction_items = _risk_section_items(risk_items, interaction=True)

    if prescription_items:
        prescription_summary = (
            f"Hệ thống ghi nhận {len(prescription_items)} điểm cần lưu ý."
        )
    else:
        prescription_summary = (
            "Trong phạm vi dữ liệu đã truy xuất, hệ thống chưa ghi nhận điểm cần "
            "bác sĩ/dược sĩ rà soát ngoài nhóm tương tác thuốc-thuốc."
        )
    missing_information = [
        sanitize_doctor_report_text(str(item))
        for item in payload.get("missing_information", [])
        if item
    ]
    if missing_information:
        prescription_summary = (
            f"{prescription_summary} Thông tin cần bổ sung: "
            f"{'; '.join(_deduplicate(missing_information))}."
        )

    if interaction_items:
        interaction_summary = (
            f"Hệ thống ghi nhận {len(interaction_items)} điểm cần lưu ý liên quan "
            "tương tác thuốc-thuốc."
        )
    elif "interaction" in checked_query_types:
        interaction_summary = NO_INTERACTION_WARNING
    else:
        interaction_summary = (
            "Chưa có tín hiệu kiểm tra tương tác thuốc-thuốc trong lần rà soát này."
        )

    sections: dict[str, Any] = {
        "prescription_check": {
            "title": "KẾT QUẢ KIỂM TRA ĐƠN THUỐC",
            "summary": sanitize_doctor_report_text(prescription_summary),
            "items": prescription_items,
        },
        "interaction_check": {
            "title": "KIỂM TRA TƯƠNG TÁC GIỮA CÁC THUỐC TRONG ĐƠN",
            "summary": sanitize_doctor_report_text(interaction_summary),
            "items": interaction_items,
        },
        "doctor_memory": {
            "title": "GHI CHÚ RIÊNG CỦA BÁC SĨ",
            "summary": "Chưa có ghi chú liên quan.",
            "items": [],
        },
        "safety_note": {
            "title": "LƯU Ý AN TOÀN",
            "content": sanitize_doctor_report_text(
                _text(payload.get("disclaimer")) or SAFETY_DISCLAIMER
            ),
        },
    }

    if isinstance(existing_sections, dict):
        for key, section in existing_sections.items():
            if key not in sections:
                sections[key] = section
        existing_memory = existing_sections.get("doctor_memory")
        if isinstance(existing_memory, dict) and existing_memory.get("items"):
            sections["doctor_memory"] = existing_memory
    return sections


def has_full_doctor_facing_sections(sections: Any) -> bool:
    return isinstance(sections, dict) and FULL_DOCTOR_FACING_SECTION_KEYS <= set(
        sections
    )


def render_doctor_facing_response_from_sections(sections: dict[str, Any]) -> str:
    lines: list[str] = []
    for section_key in DOCTOR_FACING_SECTION_ORDER:
        section = sections.get(section_key)
        if not isinstance(section, dict):
            continue
        title = _text(section.get("title"))
        summary = _text(section.get("summary"))
        content = _text(section.get("content"))
        items = _as_dict_list(section.get("items"))

        if title:
            lines.extend([title, ""])
        if summary:
            lines.extend([summary, ""])
        if content:
            lines.extend([content, ""])
        for index, item in enumerate(items, start=1):
            item_title = _text(item.get("title")) or "Điểm cần lưu ý"
            severity = _text(item.get("severity"))
            item_content = _text(item.get("content"))
            heading = f"{index}. {item_title}"
            if severity:
                heading = f"{heading} ({severity})"
            lines.append(heading)
            if item_content:
                lines.append(f"   {item_content}")
            lines.append("")

    return sanitize_doctor_report_text("\n".join(lines).strip())


class DoctorReportComposerService:
    """Compose the doctor-facing report text from a sanitized structured report."""

    def __init__(
        self,
        enabled: bool | None = None,
        gemini_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        settings = get_settings()
        self.enabled = settings.gemini_report_enable if enabled is None else enabled
        self.gemini_client_factory = (
            gemini_client_factory or GeminiDoctorReportComposerClient
        )

    @staticmethod
    def _with_warning(report: dict[str, Any], warning: str) -> dict[str, Any]:
        warnings = report.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
        report["warnings"] = _deduplicate([*warnings, warning])
        return report

    def compose(self, report: dict[str, Any]) -> dict[str, Any]:
        output = copy.deepcopy(report) if isinstance(report, dict) else {}
        payload = build_composer_payload(output)
        existing_sections = output.get("doctor_facing_sections")
        sections = build_doctor_facing_sections(
            payload,
            existing_sections if isinstance(existing_sections, dict) else None,
        )
        output["doctor_facing_sections"] = sections
        deterministic_response = render_doctor_facing_response_from_sections(sections)

        if not self.enabled:
            output["doctor_facing_response"] = deterministic_response
            return output

        try:
            client = self.gemini_client_factory()
            response = client.compose(payload)
            doctor_text = str(response["doctor_facing_response"]).strip()
        except Exception:
            output["doctor_facing_response"] = deterministic_response
            return self._with_warning(output, GEMINI_COMPOSER_FAILED_WARNING)

        if (
            has_unsafe_doctor_report_text(doctor_text)
            or not doctor_text.startswith("Kết quả kiểm tra đơn thuốc")
            or _text(payload.get("disclaimer")) not in doctor_text
        ):
            output["doctor_facing_response"] = deterministic_response
            return self._with_warning(output, GEMINI_COMPOSER_SAFETY_WARNING)

        output["doctor_facing_response"] = deterministic_response
        return output

    def get_stats(self) -> dict[str, Any]:
        return {
            "service": "DoctorReportComposerService",
            "gemini_report_enable": self.enabled,
        }
