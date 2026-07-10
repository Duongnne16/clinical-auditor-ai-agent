from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
SUITE_PATH = TESTS_DIR / "clinical_workflow_50_prescription_suite.json"
RESULT_PATH = TESTS_DIR / "clinical_workflow_50_prescription_results.json"

BASE_URL = "http://127.0.0.1:8000"
LOGIN_PATH = "/api/v1/auth/login"
RUN_PATH = "/api/v1/clinical-workflow/run"

EMAIL = "abc@gmail.com"
PASSWORD = "12345678"
EXPECTED_DOCTOR_ID = "doctor-66b9b2c0a92246e0ab5b64b2dfd6e11d"


def _fold(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower() if value is not None else ""


def _contains_any(blob: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in blob for keyword in keywords)


def build_prescription_text(
    *,
    patient_name: str,
    age: int,
    diagnosis: str,
    medications: list[tuple[str, str]],
    allergy: str = "Khong",
    comorbidity: str = "Khong",
    hepatic: str = "Binh thuong",
    renal: str = "Binh thuong",
    pregnancy: str = "Khong",
    other_meds: str = "Khong",
) -> str:
    lines = [
        "DON NGOAI TRU",
        "I. THONG TIN BENH NHAN",
        f"- Ho va ten: {patient_name}",
        f"- Tuoi: {age}",
        "II. THONG TIN LAM SANG",
        f"- Chan doan: {diagnosis}",
        f"- Di ung thuoc: {allergy}",
        f"- Benh nen: {comorbidity}",
        f"- Chuc nang gan: {hepatic}",
        f"- Chuc nang than: {renal}",
        f"- Thai ky/ cho con bu: {pregnancy}",
        f"- Thuoc khac dang dung: {other_meds}",
        "III. CHI DINH DUNG THUOC",
    ]
    for index, (name, instruction) in enumerate(medications, start=1):
        lines.append(f"{index}. {name}")
        lines.append(instruction)
    return "\n".join(lines)


@dataclass
class Expected:
    result_type: str
    must_contain_all: list[str]
    answer_keywords_any: list[str]
    severity_keywords_any: list[str]
    expect_sources: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "result_type": self.result_type,
            "must_contain_all": self.must_contain_all,
            "answer_keywords_any": self.answer_keywords_any,
            "severity_keywords_any": self.severity_keywords_any,
            "expect_sources": self.expect_sources,
        }


def case(
    *,
    case_id: str,
    category: str,
    patient_name: str,
    age: int,
    diagnosis: str,
    medications: list[tuple[str, str]],
    expected: Expected,
    patient_context: dict[str, Any] | None = None,
    allergy: str = "Khong",
    comorbidity: str = "Khong",
    hepatic: str = "Binh thuong",
    renal: str = "Binh thuong",
    pregnancy: str = "Khong",
    other_meds: str = "Khong",
) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "input_text": build_prescription_text(
            patient_name=patient_name,
            age=age,
            diagnosis=diagnosis,
            medications=medications,
            allergy=allergy,
            comorbidity=comorbidity,
            hepatic=hepatic,
            renal=renal,
            pregnancy=pregnancy,
            other_meds=other_meds,
        ),
        "patient_context": patient_context or {},
        "expected": expected.as_dict(),
    }


def build_suite() -> list[dict[str, Any]]:
    high = ["high", "cao", "nghiem trong", "major", "serious", "contraind", "chong chi dinh"]
    monitor = ["theo doi", "can duoc ra soat", "monitor", "moderate", "trung binh", "than trong"]
    bleeding = ["bleed", "xuat huyet", "chay mau", "hemorrhag"]
    muscle = ["co", "tieu co van", "rhabdom", "cyp3a4"]
    hyperk = ["kal", "tang kali", "hyperkal"]
    brady = ["nhip cham", "brady", "ha huyet ap", "block"]
    serotonin = ["serotonin", "kich dong", "run", "sot"]
    teratogen = ["thai", "pregnan", "di tat", "terat"]
    kidney = ["than", "egfr", "suy than", "creatinin"]
    liver = ["gan", "men gan", "hepat", "doc gan"]
    insufficient = ["review", "khong du bang chung", "chua tim thay du bang chung", "requires review"]

    suite = [
        case(
            case_id="rx_01_simvastatin_clarithromycin",
            category="severe_interaction",
            patient_name="Nguyen Van A",
            age=65,
            diagnosis="Roi loan lipid mau",
            medications=[
                ("Simvastatin 20 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien buoi toi."),
                ("Clarithromycin 500 mg x 14 vien", "Ngay uong 2 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["simvastatin", "clarithromycin"], ["tuong tac", *muscle], high, True),
        ),
        case(
            case_id="rx_02_warfarin_aspirin",
            category="severe_interaction",
            patient_name="Tran Van B",
            age=70,
            diagnosis="Rung nhi",
            medications=[
                ("Warfarin 5 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Aspirin 81 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["warfarin", "aspirin"], ["tuong tac", *bleeding], high, True),
        ),
        case(
            case_id="rx_03_sildenafil_nitroglycerin",
            category="severe_interaction",
            patient_name="Le Van C",
            age=62,
            diagnosis="Dau that nguc",
            medications=[
                ("Sildenafil 50 mg x 4 vien", "Khi can, moi lan 1 vien."),
                ("Nitroglycerin 0.5 mg x 20 vien", "Khi dau nguc, ngam 1 vien duoi luoi."),
            ],
            expected=Expected("audit", ["sildenafil", "nitroglycerin"], ["ha huyet ap", "tuong tac", "huyet ap"], high, True),
        ),
        case(
            case_id="rx_04_digoxin_clarithromycin",
            category="severe_interaction",
            patient_name="Pham Van D",
            age=74,
            diagnosis="Suy tim",
            medications=[
                ("Digoxin 0.25 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Clarithromycin 500 mg x 14 vien", "Ngay uong 2 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["digoxin", "clarithromycin"], ["doc tinh", "nhip tim", "tuong tac"], high, True),
        ),
        case(
            case_id="rx_05_spironolactone_potassium",
            category="severe_interaction",
            patient_name="Hoang Van E",
            age=68,
            diagnosis="Suy tim",
            medications=[
                ("Spironolactone 25 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Potassium chloride 600 mg x 20 vien", "Ngay uong 2 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["spironolactone", "potassium"], ["tuong tac", *hyperk], high, True),
        ),
        case(
            case_id="rx_06_allopurinol_azathioprine",
            category="severe_interaction",
            patient_name="Bui Van F",
            age=55,
            diagnosis="Gout va benh tu mien",
            medications=[
                ("Allopurinol 300 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Azathioprine 50 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["allopurinol", "azathioprine"], ["uy xuong", "doc tinh", "tuong tac"], high, True),
        ),
        case(
            case_id="rx_07_methotrexate_cotrimoxazole",
            category="severe_interaction",
            patient_name="Do Thi G",
            age=48,
            diagnosis="Viem khop dang thap",
            medications=[
                ("Methotrexate 10 mg x 4 vien", "Moi tuan uong 1 lan, moi lan 1 vien."),
                ("Co-trimoxazole 960 mg x 10 vien", "Ngay uong 2 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["methotrexate", "trimox"], ["uy xuong", "doc tinh", "tuong tac"], high, True),
        ),
        case(
            case_id="rx_08_linezolid_sertraline",
            category="severe_interaction",
            patient_name="Vu Van H",
            age=41,
            diagnosis="Nhiem khuan da",
            medications=[
                ("Linezolid 600 mg x 14 vien", "Ngay uong 2 lan, moi lan 1 vien."),
                ("Sertraline 50 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["linezolid", "sertraline"], ["tuong tac", *serotonin], high, True),
        ),
        case(
            case_id="rx_09_tramadol_fluoxetine",
            category="severe_interaction",
            patient_name="Dang Thi I",
            age=36,
            diagnosis="Dau lung",
            medications=[
                ("Tramadol 50 mg x 20 vien", "Ngay uong 2 lan, moi lan 1 vien khi dau."),
                ("Fluoxetine 20 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["tramadol", "fluoxetine"], ["tuong tac", *serotonin], high, True),
        ),
        case(
            case_id="rx_10_verapamil_metoprolol",
            category="severe_interaction",
            patient_name="Nguyen Thi K",
            age=67,
            diagnosis="Tang huyet ap",
            medications=[
                ("Verapamil 80 mg x 30 vien", "Ngay uong 2 lan, moi lan 1 vien."),
                ("Metoprolol 50 mg x 30 vien", "Ngay uong 2 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["verapamil", "metoprolol"], ["tuong tac", *brady], high, True),
        ),
        case(
            case_id="rx_11_lisinopril_spironolactone",
            category="monitor_interaction",
            patient_name="Tran Thi L",
            age=64,
            diagnosis="Tang huyet ap",
            medications=[
                ("Lisinopril 10 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Spironolactone 25 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["lisinopril", "spironolactone"], ["tuong tac", *hyperk], monitor, True),
        ),
        case(
            case_id="rx_12_losartan_ibuprofen",
            category="monitor_interaction",
            patient_name="Le Thi M",
            age=61,
            diagnosis="Thoai hoa khop va tang huyet ap",
            medications=[
                ("Losartan 50 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Ibuprofen 400 mg x 15 vien", "Ngay uong 2 lan, moi lan 1 vien sau an."),
            ],
            expected=Expected("audit", ["losartan", "ibuprofen"], ["tuong tac", *kidney], monitor, True),
        ),
        case(
            case_id="rx_13_furosemide_gentamicin",
            category="monitor_interaction",
            patient_name="Pham Thi N",
            age=58,
            diagnosis="Nhiem khuan tiet nieu",
            medications=[
                ("Furosemide 40 mg x 20 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Gentamicin 80 mg x 10 ong", "Ngay tiem 2 lan, moi lan 1 ong."),
            ],
            expected=Expected("audit", ["furosemide", "gentamicin"], ["doc tai", "doc than", "tuong tac"], monitor, True),
        ),
        case(
            case_id="rx_14_levothyroxine_calcium",
            category="monitor_interaction",
            patient_name="Do Van O",
            age=45,
            diagnosis="Suy giap",
            medications=[
                ("Levothyroxine 100 mcg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien truoc an sang."),
                ("Calcium carbonate 500 mg x 30 vien", "Ngay uong 2 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["levothyroxine", "calcium"], ["hap thu", "tuong tac", "tach thoi diem"], monitor, True),
        ),
        case(
            case_id="rx_15_alendronate_calcium",
            category="monitor_interaction",
            patient_name="Bui Thi P",
            age=69,
            diagnosis="Loang xuong",
            medications=[
                ("Alendronate 70 mg x 4 vien", "Moi tuan uong 1 lan, moi lan 1 vien buoi sang."),
                ("Calcium carbonate 500 mg x 60 vien", "Ngay uong 2 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["alendronate", "calcium"], ["hap thu", "tuong tac", "tach thoi diem"], monitor, True),
        ),
        case(
            case_id="rx_16_ciprofloxacin_antacid",
            category="monitor_interaction",
            patient_name="Vu Thi Q",
            age=34,
            diagnosis="Tieu chay nhiem khuan",
            medications=[
                ("Ciprofloxacin 500 mg x 10 vien", "Ngay uong 2 lan, moi lan 1 vien."),
                ("Magnesium hydroxide 400 mg x 10 goi", "Ngay uong 2 lan, moi lan 1 goi."),
            ],
            expected=Expected("audit", ["ciprofloxacin", "magnesium"], ["hap thu", "tuong tac", "tach thoi diem"], monitor, True),
        ),
        case(
            case_id="rx_17_doxycycline_iron",
            category="monitor_interaction",
            patient_name="Nguyen Van R",
            age=27,
            diagnosis="Mun viem",
            medications=[
                ("Doxycycline 100 mg x 20 vien", "Ngay uong 2 lan, moi lan 1 vien."),
                ("Ferrous sulfate 325 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["doxycycline", "ferrous"], ["hap thu", "tuong tac", "tach thoi diem"], monitor, True),
        ),
        case(
            case_id="rx_18_amlodipine_simvastatin",
            category="monitor_interaction",
            patient_name="Tran Van S",
            age=66,
            diagnosis="Tang huyet ap va roi loan lipid mau",
            medications=[
                ("Amlodipine 5 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Simvastatin 40 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien buoi toi."),
            ],
            expected=Expected("audit", ["amlodipine", "simvastatin"], ["tuong tac", *muscle], monitor, True),
        ),
        case(
            case_id="rx_19_digoxin_furosemide",
            category="monitor_interaction",
            patient_name="Le Van T",
            age=72,
            diagnosis="Suy tim",
            medications=[
                ("Digoxin 0.25 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Furosemide 40 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["digoxin", "furosemide"], ["kali", "nhip tim", "tuong tac"], monitor, True),
        ),
        case(
            case_id="rx_20_insulin_propranolol",
            category="monitor_interaction",
            patient_name="Pham Van U",
            age=59,
            diagnosis="Dai thao duong type 2",
            medications=[
                ("Insulin glargine 100 IU/ml x 1 but", "Ngay tiem 1 lan, moi lan 10 don vi."),
                ("Propranolol 40 mg x 30 vien", "Ngay uong 2 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["insulin", "propranolol"], ["ha duong huyet", "che lap", "tuong tac"], monitor, True),
        ),
        case(
            case_id="rx_21_paracetamol_cetirizine",
            category="low_risk",
            patient_name="Ho Thi V",
            age=29,
            diagnosis="Viem mui di ung",
            medications=[
                ("Paracetamol 500 mg x 10 vien", "Ngay uong 3 lan, moi lan 1 vien."),
                ("Cetirizine 10 mg x 10 vien", "Ngay uong 1 lan, moi lan 1 vien buoi toi."),
            ],
            expected=Expected("audit", ["paracetamol", "cetirizine"], ["theo doi", "khong ghi nhan", "ra soat"], monitor, True),
        ),
        case(
            case_id="rx_22_amoxicillin_paracetamol",
            category="low_risk",
            patient_name="Do Thi W",
            age=25,
            diagnosis="Viem hong",
            medications=[
                ("Amoxicillin 500 mg x 15 vien", "Ngay uong 3 lan, moi lan 1 vien."),
                ("Paracetamol 500 mg x 10 vien", "Ngay uong 3 lan, moi lan 1 vien khi sot."),
            ],
            expected=Expected("audit", ["amoxicillin", "paracetamol"], ["theo doi", "khong ghi nhan", "ra soat"], monitor, True),
        ),
        case(
            case_id="rx_23_omeprazole_domperidone",
            category="low_risk",
            patient_name="Bui Van X",
            age=43,
            diagnosis="Trao nguoc da day",
            medications=[
                ("Omeprazole 20 mg x 14 vien", "Ngay uong 2 lan, moi lan 1 vien."),
                ("Domperidone 10 mg x 14 vien", "Ngay uong 3 lan, moi lan 1 vien truoc an."),
            ],
            expected=Expected("audit", ["omeprazole", "domperidone"], ["theo doi", "can duoc ra soat", "than trong"], monitor, True),
        ),
        case(
            case_id="rx_24_rosuvastatin_ezetimibe",
            category="low_risk",
            patient_name="Tran Thi Y",
            age=63,
            diagnosis="Roi loan lipid mau",
            medications=[
                ("Rosuvastatin 10 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Ezetimibe 10 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["rosuvastatin", "ezetimibe"], ["theo doi", "can duoc ra soat", "than trong"], monitor, True),
        ),
        case(
            case_id="rx_25_cefixime_paracetamol",
            category="low_risk",
            patient_name="Nguyen Van Z",
            age=19,
            diagnosis="Viem hong",
            medications=[
                ("Cefixime 200 mg x 10 vien", "Ngay uong 2 lan, moi lan 1 vien."),
                ("Paracetamol 500 mg x 10 vien", "Ngay uong 3 lan, moi lan 1 vien khi sot."),
            ],
            expected=Expected("audit", ["cefixime", "paracetamol"], ["theo doi", "khong ghi nhan", "ra soat"], monitor, True),
        ),
        case(
            case_id="rx_26_isotretinoin_pregnancy",
            category="pregnancy_contraindication",
            patient_name="Nguyen Thi AA",
            age=26,
            diagnosis="Mun trung ca nang",
            medications=[("Isotretinoin 10 mg x 30 vien", "Ngay uong 2 lan, moi lan 1 vien.")],
            pregnancy="Co",
            patient_context={"pregnancy_status": "pregnant"},
            expected=Expected("audit", ["isotretinoin"], ["thai", *teratogen], high, True),
        ),
        case(
            case_id="rx_27_enalapril_pregnancy",
            category="pregnancy_contraindication",
            patient_name="Le Thi AB",
            age=31,
            diagnosis="Tang huyet ap",
            medications=[("Enalapril 5 mg x 30 vien", "Ngay uong 2 lan, moi lan 1 vien.")],
            pregnancy="Co",
            patient_context={"pregnancy_status": "pregnant"},
            expected=Expected("audit", ["enalapril"], ["thai", "fet", "pregnan"], high, True),
        ),
        case(
            case_id="rx_28_valproate_pregnancy",
            category="pregnancy_contraindication",
            patient_name="Pham Thi AC",
            age=28,
            diagnosis="Dong kinh",
            medications=[("Valproate 500 mg x 30 vien", "Ngay uong 2 lan, moi lan 1 vien.")],
            pregnancy="Co",
            patient_context={"pregnancy_status": "pregnant"},
            expected=Expected("audit", ["valproate"], ["thai", *teratogen], high, True),
        ),
        case(
            case_id="rx_29_warfarin_pregnancy",
            category="pregnancy_contraindication",
            patient_name="Do Thi AD",
            age=33,
            diagnosis="Huyet khoi tinh mach",
            medications=[("Warfarin 5 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien.")],
            pregnancy="Co",
            patient_context={"pregnancy_status": "pregnant"},
            expected=Expected("audit", ["warfarin"], ["thai", "pregnan", "xuat huyet"], high, True),
        ),
        case(
            case_id="rx_30_ibuprofen_third_trimester",
            category="pregnancy_contraindication",
            patient_name="Tran Thi AE",
            age=30,
            diagnosis="Dau rang",
            medications=[("Ibuprofen 400 mg x 12 vien", "Ngay uong 3 lan, moi lan 1 vien sau an.")],
            pregnancy="Co",
            patient_context={"pregnancy_status": "pregnant", "trimester": 3},
            expected=Expected("audit", ["ibuprofen"], ["thai", "pregnan", "thai ky"], high, True),
        ),
        case(
            case_id="rx_31_metformin_egfr_low",
            category="renal_risk",
            patient_name="Nguyen Van AF",
            age=73,
            diagnosis="Dai thao duong type 2",
            medications=[("Metformin 850 mg x 60 vien", "Ngay uong 2 lan, moi lan 1 vien.")],
            renal="Suy than do 4",
            patient_context={"renal_function": "eGFR 22 ml/min/1.73m2"},
            expected=Expected("audit", ["metformin"], ["than", "egfr", "nhiem toan"], high, True),
        ),
        case(
            case_id="rx_32_nitrofurantoin_egfr_low",
            category="renal_risk",
            patient_name="Le Thi AG",
            age=69,
            diagnosis="Nhiem khuan tiet nieu",
            medications=[("Nitrofurantoin 100 mg x 10 vien", "Ngay uong 2 lan, moi lan 1 vien.")],
            renal="Suy than man",
            patient_context={"renal_function": "eGFR 25 ml/min/1.73m2"},
            expected=Expected("audit", ["nitrofurantoin"], ["than", "egfr", "hieu qua"], high, True),
        ),
        case(
            case_id="rx_33_ibuprofen_ckd",
            category="renal_risk",
            patient_name="Pham Van AH",
            age=71,
            diagnosis="Thoai hoa khop",
            medications=[("Ibuprofen 400 mg x 15 vien", "Ngay uong 3 lan, moi lan 1 vien sau an.")],
            renal="Suy than man",
            patient_context={"renal_function": "eGFR 28 ml/min/1.73m2"},
            expected=Expected("audit", ["ibuprofen"], ["than", "egfr", "suy than"], high, True),
        ),
        case(
            case_id="rx_34_spironolactone_ckd",
            category="renal_risk",
            patient_name="Do Thi AI",
            age=76,
            diagnosis="Suy tim",
            medications=[("Spironolactone 25 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien.")],
            renal="Suy than man",
            patient_context={"renal_function": "eGFR 24 ml/min/1.73m2"},
            expected=Expected("audit", ["spironolactone"], ["than", *hyperk], high, True),
        ),
        case(
            case_id="rx_35_gabapentin_ckd",
            category="renal_risk",
            patient_name="Tran Van AJ",
            age=67,
            diagnosis="Dau than kinh",
            medications=[("Gabapentin 300 mg x 30 vien", "Ngay uong 3 lan, moi lan 1 vien.")],
            renal="Suy than man",
            patient_context={"renal_function": "eGFR 18 ml/min/1.73m2"},
            expected=Expected("audit", ["gabapentin"], ["than", "egfr", "hieu chinh lieu"], high, True),
        ),
        case(
            case_id="rx_36_valproate_liver_disease",
            category="hepatic_risk",
            patient_name="Nguyen Thi AK",
            age=40,
            diagnosis="Dong kinh",
            medications=[("Valproate 500 mg x 30 vien", "Ngay uong 2 lan, moi lan 1 vien.")],
            hepatic="Xo gan",
            patient_context={"hepatic_function": "Child-Pugh C"},
            expected=Expected("audit", ["valproate"], ["gan", *liver], high, True),
        ),
        case(
            case_id="rx_37_paracetamol_liver_disease",
            category="hepatic_risk",
            patient_name="Le Van AL",
            age=57,
            diagnosis="Dau xuong khop",
            medications=[("Paracetamol 500 mg x 20 vien", "Ngay uong 4 lan, moi lan 2 vien.")],
            hepatic="Xo gan",
            patient_context={"hepatic_function": "Child-Pugh B"},
            expected=Expected("audit", ["paracetamol"], ["gan", *liver], high, True),
        ),
        case(
            case_id="rx_38_fluconazole_liver_disease",
            category="hepatic_risk",
            patient_name="Pham Thi AM",
            age=52,
            diagnosis="Nhiem nam",
            medications=[("Fluconazole 150 mg x 3 vien", "Ngay uong 1 lan, moi lan 1 vien.")],
            hepatic="Viem gan man",
            patient_context={"hepatic_function": "men gan tang"},
            expected=Expected("audit", ["fluconazole"], ["gan", *liver], high, True),
        ),
        case(
            case_id="rx_39_atorvastatin_liver_disease",
            category="hepatic_risk",
            patient_name="Tran Thi AN",
            age=61,
            diagnosis="Roi loan lipid mau",
            medications=[("Atorvastatin 40 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien buoi toi.")],
            hepatic="Men gan tang",
            patient_context={"hepatic_function": "AST ALT tang gap 3 lan gioi han tren"},
            expected=Expected("audit", ["atorvastatin"], ["gan", *liver], high, True),
        ),
        case(
            case_id="rx_40_haloperidol_parkinson",
            category="disease_contraindication",
            patient_name="Nguyen Van AO",
            age=72,
            diagnosis="Kich dong",
            medications=[("Haloperidol 1.5 mg x 20 vien", "Ngay uong 2 lan, moi lan 1 vien.")],
            comorbidity="Parkinson",
            patient_context={"comorbidities": ["Parkinson"]},
            expected=Expected("audit", ["haloperidol"], ["parkinson", "ngoai thap", "than trong"], high, True),
        ),
        case(
            case_id="rx_41_propranolol_asthma",
            category="disease_contraindication",
            patient_name="Le Thi AP",
            age=50,
            diagnosis="Run tay",
            medications=[("Propranolol 40 mg x 30 vien", "Ngay uong 2 lan, moi lan 1 vien.")],
            comorbidity="Hen phe quan",
            patient_context={"comorbidities": ["asthma"]},
            expected=Expected("audit", ["propranolol"], ["hen", "co that phe quan", "than trong"], high, True),
        ),
        case(
            case_id="rx_42_nonselective_nsaid_ulcer",
            category="disease_contraindication",
            patient_name="Pham Van AQ",
            age=60,
            diagnosis="Dau khop",
            medications=[("Diclofenac 50 mg x 20 vien", "Ngay uong 2 lan, moi lan 1 vien sau an.")],
            comorbidity="Loet da day tai phat",
            patient_context={"comorbidities": ["peptic ulcer"]},
            expected=Expected("audit", ["diclofenac"], ["loet", "xuat huyet", "da day"], high, True),
        ),
        case(
            case_id="rx_43_unknown_drug_pair_1",
            category="insufficient_evidence",
            patient_name="Tran Thi AR",
            age=44,
            diagnosis="Dau dau",
            medications=[
                ("Xylorafin 250 mg x 10 vien", "Ngay uong 2 lan, moi lan 1 vien."),
                ("Noverex 5 mg x 10 vien", "Ngay uong 1 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["xylorafin", "noverex"], insufficient, monitor, False),
        ),
        case(
            case_id="rx_44_unknown_drug_pair_2",
            category="insufficient_evidence",
            patient_name="Bui Van AS",
            age=39,
            diagnosis="Mat ngu",
            medications=[
                ("Cardioprex 10 mg x 10 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Neuromedix 25 mg x 10 vien", "Ngay uong 1 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["cardioprex", "neuromedix"], insufficient, monitor, False),
        ),
        case(
            case_id="rx_45_unknown_drug_single",
            category="insufficient_evidence",
            patient_name="Do Thi AT",
            age=46,
            diagnosis="Met moi",
            medications=[("Hepalorin 150 mg x 20 vien", "Ngay uong 2 lan, moi lan 1 vien.")],
            expected=Expected("audit", ["hepalorin"], insufficient, monitor, False),
        ),
        case(
            case_id="rx_46_unknown_drug_pair_3",
            category="insufficient_evidence",
            patient_name="Hoang Van AU",
            age=53,
            diagnosis="Dau cot song",
            medications=[
                ("Osteorix 200 mg x 20 vien", "Ngay uong 2 lan, moi lan 1 vien."),
                ("Myocalmex 50 mg x 20 vien", "Ngay uong 2 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["osteorix", "myocalmex"], insufficient, monitor, False),
        ),
        case(
            case_id="rx_47_unknown_drug_pair_4",
            category="insufficient_evidence",
            patient_name="Pham Thi AV",
            age=32,
            diagnosis="Cam cum",
            medications=[
                ("Viracold 500 mg x 10 vien", "Ngay uong 2 lan, moi lan 1 vien."),
                ("Pulmocarex 250 mg x 10 vien", "Ngay uong 2 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["viracold", "pulmocarex"], insufficient, monitor, False),
        ),
        case(
            case_id="rx_48_amoxicillin_penicillin_allergy",
            category="allergy_risk",
            patient_name="Nguyen Van AW",
            age=35,
            diagnosis="Viem hong",
            medications=[("Amoxicillin 500 mg x 15 vien", "Ngay uong 3 lan, moi lan 1 vien.")],
            allergy="Penicillin",
            patient_context={"drug_allergies": ["penicillin"]},
            expected=Expected("audit", ["amoxicillin"], ["di ung", "penicillin", "phan ve"], high, True),
        ),
        case(
            case_id="rx_49_ibuprofen_nsaid_allergy",
            category="allergy_risk",
            patient_name="Le Thi AX",
            age=42,
            diagnosis="Dau rang",
            medications=[("Ibuprofen 400 mg x 12 vien", "Ngay uong 3 lan, moi lan 1 vien sau an.")],
            allergy="NSAID",
            patient_context={"drug_allergies": ["NSAID"]},
            expected=Expected("audit", ["ibuprofen"], ["di ung", "nsaid", "phan ung"], high, True),
        ),
        case(
            case_id="rx_50_blank_context_common_rx",
            category="general_prescription_check",
            patient_name="Tran Van AY",
            age=54,
            diagnosis="Tang huyet ap va dai thao duong",
            medications=[
                ("Amlodipine 5 mg x 30 vien", "Ngay uong 1 lan, moi lan 1 vien."),
                ("Metformin 500 mg x 60 vien", "Ngay uong 2 lan, moi lan 1 vien."),
            ],
            expected=Expected("audit", ["amlodipine", "metformin"], ["theo doi", "ra soat", "than trong"], monitor, True),
        ),
    ]
    return suite


def evaluate_case(case_data: dict[str, Any], response_body: Any, status_code: int | None) -> tuple[float, list[dict[str, Any]], str]:
    expected = case_data["expected"]
    blob = _fold(response_body)
    criteria: list[dict[str, Any]] = []

    actual_result_type = response_body.get("result_type") if isinstance(response_body, dict) else None
    criteria.append(
        {
            "name": "http_status_is_200",
            "passed": status_code == 200,
            "weight": 15,
            "detail": f"actual_status={status_code}",
        }
    )
    criteria.append(
        {
            "name": "result_type_matches",
            "passed": actual_result_type == expected["result_type"],
            "weight": 20,
            "detail": f"expected={expected['result_type']} actual={actual_result_type}",
        }
    )

    expected_terms = expected.get("must_contain_all") or []
    matched_count = sum(1 for term in expected_terms if term.lower() in blob)
    criteria.append(
        {
            "name": "expected_drug_terms_present",
            "passed": matched_count == len(expected_terms),
            "weight": 20,
            "detail": f"matched={matched_count}/{len(expected_terms)} expected={expected_terms}",
        }
    )

    answer_keywords = expected.get("answer_keywords_any") or []
    criteria.append(
        {
            "name": "expected_answer_signal_present",
            "passed": _contains_any(blob, answer_keywords),
            "weight": 20,
            "detail": f"keywords_any={answer_keywords}",
        }
    )

    severity_keywords = expected.get("severity_keywords_any") or []
    criteria.append(
        {
            "name": "expected_severity_signal_present",
            "passed": _contains_any(blob, severity_keywords),
            "weight": 15,
            "detail": f"keywords_any={severity_keywords}",
        }
    )

    sources = response_body.get("sources") if isinstance(response_body, dict) else []
    has_sources = isinstance(sources, list) and len(sources) > 0
    criteria.append(
        {
            "name": "source_presence_matches_expectation",
            "passed": has_sources == expected["expect_sources"],
            "weight": 10,
            "detail": f"expected={expected['expect_sources']} actual={has_sources}",
        }
    )

    total_weight = sum(item["weight"] for item in criteria)
    earned_weight = sum(item["weight"] for item in criteria if item["passed"])
    accuracy = round(earned_weight * 100 / total_weight, 2)
    failed = [item for item in criteria if not item["passed"]]
    reason = "; ".join(f"{item['name']}: {item['detail']}" for item in failed)
    return accuracy, criteria, reason


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    suite = build_suite()
    write_json(
        SUITE_PATH,
        {
            "description": "50 prescription-oriented clinical workflow test cases.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cases": suite,
        },
    )

    session = requests.Session()
    login_response = session.post(
        BASE_URL + LOGIN_PATH,
        json={"email": EMAIL, "password": PASSWORD},
        timeout=30,
    )
    login_body: dict[str, Any] | None = None
    token = None
    if login_response.status_code == 200:
        login_body = login_response.json()
        token = login_body.get("access_token")

    results: list[dict[str, Any]] = []
    result_types: Counter[str] = Counter()

    for index, case_data in enumerate(suite, start=1):
        payload = {
            "input_text": case_data["input_text"],
            "patient_context": case_data.get("patient_context") or {},
            "use_gemini": False,
            "top_k_per_type": 8,
        }
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        start = time.perf_counter()
        status_code: int | None = None
        body: Any
        error: str | None = None
        try:
            response = session.post(
                BASE_URL + RUN_PATH,
                json=payload,
                headers=headers,
                timeout=180,
            )
            elapsed = time.perf_counter() - start
            status_code = response.status_code
            try:
                body = response.json()
            except Exception:
                body = {"raw_text": response.text}
        except Exception as exc:
            elapsed = time.perf_counter() - start
            body = None
            error = repr(exc)

        accuracy, criteria, reason = evaluate_case(case_data, body, status_code)
        if isinstance(body, dict):
            result_types[str(body.get("result_type") or "unknown")] += 1
        else:
            result_types["non_json_or_error"] += 1

        results.append(
            {
                "index": index,
                "id": case_data["id"],
                "category": case_data["category"],
                "endpoint": f"POST {RUN_PATH}",
                "input": payload,
                "output": {
                    "status_code": status_code,
                    "body": body,
                    "request_error": error,
                },
                "runtime": {
                    "seconds": round(elapsed, 6),
                    "milliseconds": round(elapsed * 1000, 3),
                },
                "accuracy_evaluation": {
                    "accuracy_percent": accuracy,
                    "criteria": criteria,
                    "reason_not_full_score": reason,
                },
            }
        )

        snapshot = {
            "suite_file": str(SUITE_PATH),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "server_base_url": BASE_URL,
            "auth": {
                "email": EMAIL,
                "doctor_id_expected": EXPECTED_DOCTOR_ID,
                "login_status_code": login_response.status_code,
                "doctor_id_from_login": login_body.get("doctor_id") if login_body else None,
            },
            "summary": {
                "completed_cases": len(results),
                "total_cases": len(suite),
                "average_accuracy_percent": round(
                    sum(item["accuracy_evaluation"]["accuracy_percent"] for item in results) / len(results),
                    2,
                ),
                "average_runtime_seconds": round(
                    sum(item["runtime"]["seconds"] for item in results) / len(results),
                    6,
                ),
                "result_type_counts": dict(result_types),
            },
            "cases": results,
        }
        write_json(RESULT_PATH, snapshot)
        print(f"[{index:02d}/{len(suite)}] {case_data['id']} -> status={status_code} accuracy={accuracy}% time={elapsed:.3f}s")

    final_data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    final_data["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(RESULT_PATH, final_data)
    print(str(RESULT_PATH))


if __name__ == "__main__":
    main()
