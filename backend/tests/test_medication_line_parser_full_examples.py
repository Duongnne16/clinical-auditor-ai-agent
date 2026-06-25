import pytest

from backend.app.services.medication_line_parser import (
    MedicationLineParser,
)


EXAMPLES = [
    ("Omeprazol (Kagascdine) 20mg x 56 Viên", 1),
    ("Metformin (Panfor SR) 750mg x 112 Viên", 1),
    ("Paracetamol (Hapacol Caplet) 500mg x 10 Viên", 1),
    ("Bisoprolol (Concor) 5mg x 56 Viên", 1),
    (
        "Spiramycin + metronidazol (Spirastad Plus) "
        "0,75MUI + 125mg x 20 Viên",
        2,
    ),
    (
        "Betamethason + Acid salicylic (Asosalic) "
        "(0,5+30)mg/g x 30g",
        2,
    ),
    (
        "Tobramycin + dexamethason (Tobradex) "
        "(0,3%+0,1%)/5ml x 1 Lọ",
        2,
    ),
    (
        "Glucosamine sulfate + Chondroitin sulfate (Nuflam) "
        "500mg + 250mg x 40 Viên",
        2,
    ),
    (
        "Calcium carbonate + Vitamin D (Bonclum) "
        "500mg + 250IU x 56 Viên",
        2,
    ),
    ("Levofloxacine 500mg", 1),
]


@pytest.mark.parametrize(("line", "ingredient_count"), EXAMPLES)
def test_full_prescription_examples(
    line: str, ingredient_count: int
) -> None:
    result = MedicationLineParser().parse_line(line)

    assert result["parse_status"] == "parsed"
    assert len(result["ingredients"]) == ingredient_count
    assert result["generic_text"]
    assert all(item["name"] for item in result["ingredients"])
