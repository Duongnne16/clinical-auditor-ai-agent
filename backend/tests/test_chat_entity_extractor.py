from backend.app.services.chat_entity_extractor import ChatEntityExtractor


def test_extracts_interaction_drug_mentions() -> None:
    extractor = ChatEntityExtractor()

    assert extractor.extract("Aspirin và Warfarin có tương tác không?") == [
        "Aspirin",
        "Warfarin",
    ]
    assert extractor.extract(
        "Levofloxacin dùng cùng Sucralfate có sao không?"
    ) == ["Levofloxacin", "Sucralfate"]
    assert extractor.extract(
        "Omeprazole có tương tác với Clopidogrel không?"
    ) == ["Omeprazole", "Clopidogrel"]


def test_extracts_single_drug_mentions() -> None:
    extractor = ChatEntityExtractor()

    assert extractor.extract("Paracetamol có tác dụng phụ gì?") == [
        "Paracetamol"
    ]
    assert extractor.extract("Dịch truyền Paracetamol có tác dụng phụ gì?") == [
        "Paracetamol"
    ]

