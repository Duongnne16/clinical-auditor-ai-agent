from __future__ import annotations

from backend.app.services.prescription_document_parser import (
    PrescriptionDocumentParser,
)


CANONICAL_DOCUMENT = """ĐƠN NGOẠI TRÚ 1
Bệnh viện: Bệnh viện A
Khoa: Tiêu hóa
Đơn thuốc

I. THÔNG TIN BỆNH NHÂN
Họ và tên: Hoàng Thị P.
Tuổi: 28
Nam/Nữ: Nữ
Cân nặng: 60kg
Địa chỉ: Nga Sơn, Thanh Hóa

II. THÔNG TIN LÂM SÀNG
Chẩn đoán: Viêm phế quản/loét dạ dày tá tràng
Dị ứng thuốc: Chưa ghi nhận
Bệnh nền: Không ghi nhận
Chức năng gan: Chưa có thông tin
Chức năng thận: Bình thường
Thai kỳ/ cho con bú: Chưa có thông tin
Thuốc khác đang dùng: Chưa có thông tin

III. CHỈ ĐỊNH DÙNG THUỐC

1. Omeprazole (Losec) 20mg x 15 viên
   Ngày uống 1 lần, mỗi lần 1 viên
2. Sucralfate (Sucrate Gel) 1g/5mL x 15 gói
   Ngày uống 3 lần, mỗi lần 1 gói
3. Levofloxacine 500mg x 7 viên
   Ngày uống 1 viên

Ngày, tháng, năm
Bác sĩ khám bệnh
"""


EXPECTED_PRESCRIPTION_TEXT = """1. Omeprazole (Losec) 20mg x 15 viên
   Ngày uống 1 lần, mỗi lần 1 viên
2. Sucralfate (Sucrate Gel) 1g/5mL x 15 gói
   Ngày uống 3 lần, mỗi lần 1 gói
3. Levofloxacine 500mg x 7 viên
   Ngày uống 1 viên"""


def test_parses_canonical_outpatient_prescription() -> None:
    result = PrescriptionDocumentParser().parse(CANONICAL_DOCUMENT)

    assert result["applied"] is True
    assert result["warnings"] == []
    assert result["prescription_text"] == EXPECTED_PRESCRIPTION_TEXT
    assert result["patient_context"] == {
        "hospital": "Bệnh viện A",
        "department": "Tiêu hóa",
        "patient_name": "Hoàng Thị P.",
        "age": 28,
        "sex": "female",
        "weight": "60kg",
        "address": "Nga Sơn, Thanh Hóa",
        "diagnoses": ["Viêm phế quản", "loét dạ dày tá tràng"],
        "allergies": "Chưa ghi nhận",
        "comorbidities": "Không ghi nhận",
        "hepatic_function": "Chưa có thông tin",
        "renal_function": "Bình thường",
        "pregnancy_lactation": "Chưa có thông tin",
        "current_medications": "Chưa có thông tin",
    }


def test_parses_utf8_outpatient_prescription_with_breastfeeding_context() -> None:
    text = """ĐƠN NGOẠI TRÚ 1
Bệnh viện: Bệnh viện A
Khoa: Tiêu hóa
Đơn thuốc
I.THÔNG TIN BỆNH NHÂN
Họ và tên: Hoàng Thị P.
Tuổi: 28
Nam/Nữ: Nữ
Cân nặng: 60kg
Địa chỉ: Nga Sơn, Thanh Hóa
II. THÔNG TIN LÂM SÀNG
Chẩn đoán: Viêm phế quản/loét dạ dày tá tràng
Dị ứng thuốc: Không
Bệnh nền: Không ghi nhận
Chức năng gan: Bình thường
Chức năng thận: Bình thường
Thai kỳ/ cho con bú: Cho con bú
Thuốc khác đang dùng: Không
III. CHỈ ĐỊNH DÙNG THUỐC
1.	Omeprazole (Losec) 20mg			x	15 viên
Ngày uống 1 lần, mỗi lần 1 viên
2.	Sucralfate (Sucrate Gel) 1g/5mL		x	15 gói
Ngày uống 3 lần, mỗi lần 1 gói
3.	Levofloxacine 500mg 			x	7 viên
Ngày uống 1 viên

					Ngày , tháng, năm
					Bác sĩ khám bệnh
"""

    result = PrescriptionDocumentParser().parse(text)

    assert result["applied"] is True
    assert result["warnings"] == []
    assert result["patient_context"]["patient_name"] == "Hoàng Thị P."
    assert result["patient_context"]["sex"] == "female"
    assert result["patient_context"]["diagnoses"] == [
        "Viêm phế quản",
        "loét dạ dày tá tràng",
    ]
    assert result["patient_context"]["hepatic_function"] == "Bình thường"
    assert result["patient_context"]["renal_function"] == "Bình thường"
    assert result["patient_context"]["pregnancy_lactation"] == "Cho con bú"
    assert result["prescription_text"] == (
        "1.\tOmeprazole (Losec) 20mg\t\t\tx\t15 viên\n"
        "Ngày uống 1 lần, mỗi lần 1 viên\n"
        "2.\tSucralfate (Sucrate Gel) 1g/5mL\t\tx\t15 gói\n"
        "Ngày uống 3 lần, mỗi lần 1 gói\n"
        "3.\tLevofloxacine 500mg \t\t\tx\t7 viên\n"
        "Ngày uống 1 viên"
    )


def test_top_standalone_don_thuoc_title_is_not_medication_boundary() -> None:
    result = PrescriptionDocumentParser().parse(CANONICAL_DOCUMENT)

    assert result["prescription_text"].startswith(
        "1. Omeprazole (Losec) 20mg x 15 viên"
    )
    assert "I. THÔNG TIN BỆNH NHÂN" not in result["prescription_text"]
    assert "Họ và tên" not in result["prescription_text"]


def test_fallback_colon_marker_applies_without_structured_heading() -> None:
    text = """Bệnh nhân: Nguyễn Văn A
Chẩn đoán: Viêm dạ dày
Đơn thuốc:
1. Omeprazole 20mg x 14 viên
   Ngày uống 1 viên
"""

    result = PrescriptionDocumentParser().parse(text)

    assert result["applied"] is True
    assert result["patient_context"]["patient_name"] == "Nguyễn Văn A"
    assert result["prescription_text"] == (
        "1. Omeprazole 20mg x 14 viên\n"
        "   Ngày uống 1 viên"
    )


def test_fallback_marker_without_colon_requires_numbered_lines_soon() -> None:
    parser = PrescriptionDocumentParser()
    not_applied = parser.parse(
        """Đơn thuốc
Bệnh viện: Bệnh viện A
Khoa: Tiêu hóa
I. THÔNG TIN BỆNH NHÂN
Họ tên: Nguyễn Văn A
"""
    )
    applied = parser.parse(
        """Toa thuốc

1. Paracetamol 500mg x 10 viên
   Ngày uống 2 lần
"""
    )

    assert not_applied["applied"] is False
    assert applied["applied"] is True
    assert applied["prescription_text"] == (
        "1. Paracetamol 500mg x 10 viên\n"
        "   Ngày uống 2 lần"
    )


def test_section_headings_without_space_after_roman_numeral_are_supported() -> None:
    text = CANONICAL_DOCUMENT.replace(
        "I. THÔNG TIN BỆNH NHÂN", "I.THÔNG TIN BỆNH NHÂN"
    ).replace(
        "II. THÔNG TIN LÂM SÀNG", "II.THÔNG TIN LÂM SÀNG"
    ).replace(
        "III. CHỈ ĐỊNH DÙNG THUỐC", "III.CHỈ ĐỊNH DÙNG THUỐC"
    )

    result = PrescriptionDocumentParser().parse(text)

    assert result["applied"] is True
    assert result["patient_context"]["patient_name"] == "Hoàng Thị P."
    assert result["prescription_text"] == EXPECTED_PRESCRIPTION_TEXT


def test_usage_instruction_lines_are_preserved() -> None:
    result = PrescriptionDocumentParser().parse(CANONICAL_DOCUMENT)

    assert "   Ngày uống 1 lần, mỗi lần 1 viên" in result["prescription_text"]
    assert "   Ngày uống 3 lần, mỗi lần 1 gói" in result["prescription_text"]


def test_footer_signature_and_chat_request_lines_are_removed() -> None:
    text = CANONICAL_DOCUMENT.replace(
        "Ngày, tháng, năm\nBác sĩ khám bệnh",
        "Ngày , tháng, năm\nHãy kiểm tra đơn thuốc này\nBác sĩ khám bệnh",
    )

    result = PrescriptionDocumentParser().parse(text)

    assert "Ngày, tháng, năm" not in result["prescription_text"]
    assert "Ngày , tháng, năm" not in result["prescription_text"]
    assert "Bác sĩ khám bệnh" not in result["prescription_text"]
    assert "Hãy kiểm tra đơn thuốc này" not in result["prescription_text"]


def test_returns_not_applied_when_no_structured_prescription_marker_exists() -> None:
    result = PrescriptionDocumentParser().parse(
        "Aspirin có tương tác Warfarin không?"
    )

    assert result == {
        "applied": False,
        "patient_context": {},
        "prescription_text": "",
        "warnings": [],
    }


def test_applied_with_warning_when_no_medication_lines_found() -> None:
    result = PrescriptionDocumentParser().parse(
        """I. THÔNG TIN BỆNH NHÂN
Họ tên: Nguyễn Văn A
III. CHỈ ĐỊNH DÙNG THUỐC
Ngày, tháng, năm
Bác sĩ khám bệnh
"""
    )

    assert result["applied"] is True
    assert result["prescription_text"] == ""
    assert result["warnings"] == [
        "prescription_document_parser_no_medication_lines"
    ]
