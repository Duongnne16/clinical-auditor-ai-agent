from backend.app.services.intent_router import (
    DRUG_INTERACTION_QUERY,
    OUT_OF_SCOPE,
    SINGLE_DRUG_QUERY,
    IntentRouter,
)


def test_interaction_intents() -> None:
    router = IntentRouter()

    cases = [
        "Omeprazole có tương tác với Clopidogrel không?",
        "Levofloxacin dùng cùng Sucralfate có sao không?",
        "Aspirin và Warfarin có tương tác không?",
    ]

    for message in cases:
        result = router.classify(message)
        assert result.intent == DRUG_INTERACTION_QUERY
        assert len(result.drug_mentions) == 2


def test_single_drug_intents() -> None:
    router = IntentRouter()

    cases = {
        "Paracetamol có tác dụng phụ gì?": "adverse_effect",
        "Omeprazole cần thận trọng gì?": "caution",
        "Levofloxacin chống chỉ định khi nào?": "contraindication",
    }

    for message, topic in cases.items():
        result = router.classify(message)
        assert result.intent == SINGLE_DRUG_QUERY
        assert result.topic == topic
        assert result.drug_mentions


def test_out_of_scope_intents() -> None:
    router = IntentRouter()

    cases = [
        "Viết giúp tôi bài văn",
        "Hôm nay thời tiết thế nào?",
        "Dịch câu này sang tiếng Nhật",
    ]

    for message in cases:
        result = router.classify(message)
        assert result.intent == OUT_OF_SCOPE


def test_dich_medical_question_is_not_translation_out_of_scope() -> None:
    result = IntentRouter().classify("Dịch truyền Paracetamol có tác dụng phụ gì?")

    assert result.intent == SINGLE_DRUG_QUERY
    assert result.drug_mentions == ["Paracetamol"]

