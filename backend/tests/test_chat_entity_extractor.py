from backend.app.services.chat_entity_extractor import ChatEntityExtractor


def test_extracts_vietnamese_interaction_pair_patterns() -> None:
    extractor = ChatEntityExtractor()

    cases = [
        (
            "Amlodipin và Alfuzosin có tương tác với nhau không?",
            ["Amlodipin", "Alfuzosin"],
        ),
        (
            "Amlodipin với Alfuzosin có tương tác không?",
            ["Amlodipin", "Alfuzosin"],
        ),
        (
            "Amlodipin có tương tác với Alfuzosin không?",
            ["Amlodipin", "Alfuzosin"],
        ),
        (
            "Aspirin và Warfarin dùng chung có sao không?",
            ["Aspirin", "Warfarin"],
        ),
    ]

    for message, expected in cases:
        mentions = extractor.extract(message)

        assert mentions == expected
        assert "nhau" not in mentions


def test_extracts_interaction_drug_mentions() -> None:
    extractor = ChatEntityExtractor()

    assert extractor.extract("Aspirin và Warfarin có tương tác không?") == [
        "Aspirin",
        "Warfarin",
    ]


def test_extracts_utf8_demo_chat_drug_mentions() -> None:
    extractor = ChatEntityExtractor()

    assert extractor.extract("Omeprazole có tương tác với Clopidogrel không?") == [
        "Omeprazole",
        "Clopidogrel",
    ]
    assert extractor.extract("Paracetamol có tác dụng phụ gì?") == ["Paracetamol"]
    assert extractor.extract("Levofloxacin dùng cần lưu ý gì?") == ["Levofloxacin"]
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


def test_single_drug_query_does_not_extract_generic_words() -> None:
    extractor = ChatEntityExtractor()

    mentions = extractor.extract("Paracetamol có tác dụng phụ gì?")

    assert "Paracetamol" in mentions
    assert "gì" not in mentions
    assert "tác dụng phụ" not in mentions


def test_extracts_single_drug_interaction_lookup_mentions() -> None:
    extractor = ChatEntityExtractor()

    cases = [
        "Paracetamol có những tương tác thuốc nào?",
        "Paracetamol tương tác với thuốc nào?",
        "Paracetamol có tương tác gì?",
    ]

    for message in cases:
        mentions = extractor.extract(message)

        assert mentions == ["Paracetamol"]
        assert "Paracetamol tương tác" not in mentions
        assert "thuốc nào" not in mentions
        assert "nào" not in mentions
