from __future__ import annotations

from typing import Any

from backend.app.api.routes.chat import get_chat_query_service
from backend.app.schemas.chat import ChatRequest
from backend.app.services.chat_query_service import (
    GEMINI_CHAT_ANSWER_FAILED,
    GEMINI_CHAT_FAILURE_ANSWER,
    OUT_OF_SCOPE_REFUSAL,
    ChatQueryService,
)


FORBIDDEN_PHRASES = [
    "an toàn",
    "không an toàn",
    "dùng được",
    "không dùng được",
    "ngừng dùng thuốc",
    "ngừng thuốc",
    "dừng thuốc",
    "không nên dùng",
    "không được dùng",
    "không được trộn",
    "đổi thuốc",
    "thay thuốc",
    "tăng liều",
    "giảm liều",
    "kê thêm",
]


class FakeNormalizer:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result

    def normalize_many(self, medications: list[str], top_k: int = 5) -> dict[str, Any]:
        self.calls.append({"medications": medications, "top_k": top_k})
        if self.result is not None:
            return self.result
        return {
            "medications": [
                {
                    "raw_name": medication,
                    "mapping_status": "ingredient_only",
                    "active_ingredients": [
                        {"name": medication, "evidence_slug": medication.lower()}
                    ],
                }
                for medication in medications
            ],
            "unique_evidence_slugs": [
                medication.lower() for medication in medications
            ],
        }


class FakeRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.chunks = chunks if chunks is not None else [
            {
                "chunk_id": "hidden-id",
                "rank": 1,
                "slug": "omeprazole",
                "entity_name": "Omeprazole",
                "section": "tuong_tac_thuoc",
                "section_title": "Tương tác thuốc",
                "source": "trungtamthuoc",
                "title": "Omeprazole",
                "url": "https://example.test/omeprazole",
                "text": "Bằng chứng mô tả thông tin cần rà soát khi phối hợp.",
            }
        ]
        self.warnings = warnings or []
        self.calls: list[dict[str, Any]] = []

    def retrieve_for_normalized_result(
        self, normalized_result: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        kwargs["normalized_result"] = normalized_result
        self.calls.append(kwargs)
        return {"chunks": self.chunks, "warnings": self.warnings}


class FakeIngredientResolver:
    def resolve(self, ingredient_name: str, top_k: int = 5) -> dict[str, Any]:
        slug = str(ingredient_name).lower()
        return {
            "input_ingredient": ingredient_name,
            "normalized_input": slug,
            "status": "resolved",
            "best_match": {
                "evidence_name": ingredient_name,
                "evidence_slug": slug,
                "match_type": "exact",
                "score": 1.0,
            },
            "requires_review": False,
            "warnings": [],
        }


class FakeAnswerClient:
    def __init__(
        self,
        response: str = "Bác sĩ/dược sĩ nên đối chiếu bằng chứng đã truy xuất.",
        warnings: list[str] | None = None,
        raises: bool = False,
    ) -> None:
        self.response = response
        self.warnings = warnings or []
        self.raises = raises
        self.payloads: list[dict[str, Any]] = []

    def answer(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        if self.raises:
            raise RuntimeError("Gemini failed")
        return {"answer": self.response, "warnings": self.warnings}


class FakeRawAnswerClient:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.payloads: list[dict[str, Any]] = []

    def answer(self, payload: dict[str, Any]) -> Any:
        self.payloads.append(payload)
        return self.result


def _service(
    normalizer: FakeNormalizer | None = None,
    retriever: FakeRetriever | None = None,
    answer_client: Any | None = None,
) -> ChatQueryService:
    return ChatQueryService(
        normalizer=normalizer or FakeNormalizer(),  # type: ignore[arg-type]
        retriever=retriever or FakeRetriever(),  # type: ignore[arg-type]
        ingredient_resolver=FakeIngredientResolver(),  # type: ignore[arg-type]
        answer_client=answer_client
        or FakeAnswerClient(
            response="Bác sĩ/dược sĩ nên đối chiếu bằng chứng đã truy xuất."
        ),  # type: ignore[arg-type]
    )


def test_out_of_scope_returns_fixed_refusal() -> None:
    result = _service().answer(ChatRequest(message="Viết giúp tôi bài văn"))

    assert result["intent"] == "out_of_scope"
    assert result["message"] == OUT_OF_SCOPE_REFUSAL
    assert result["answer"] == OUT_OF_SCOPE_REFUSAL
    assert result["normalized_drugs"] == []


def test_interaction_query_normalizes_retrieves_and_answers() -> None:
    normalizer = FakeNormalizer()
    retriever = FakeRetriever()
    result = _service(normalizer, retriever).answer(
        ChatRequest(message="Omeprazole có tương tác với Clopidogrel không?")
    )

    assert result["intent"] == "drug_interaction_query"
    assert result["answer"]
    assert result["message"] == result["answer"]
    assert normalizer.calls == [
        {"medications": ["Omeprazole", "Clopidogrel"], "top_k": 5}
    ]
    assert retriever.calls[0]["query_type"] == "interaction"
    assert retriever.calls[0]["query_text"] == "Omeprazole Clopidogrel tương tác thuốc"
    assert "hidden-id" not in result["answer"]
    assert "chunk_id" not in result["sources"][0]


def test_answer_client_called_with_grounded_payload_when_evidence_exists() -> None:
    normalizer = FakeNormalizer()
    retriever = FakeRetriever()
    retriever.chunks[0]["vector_score"] = 0.91
    retriever.chunks[0]["rerank_score"] = 1.12
    retriever.chunks[0]["source_type"] = "internal"
    answer_client = FakeAnswerClient(
        response="Gemini grounded interaction answer.",
        warnings=["gemini_low_confidence"],
    )
    question = "Omeprazole c\u00f3 t\u01b0\u01a1ng t\u00e1c v\u1edbi Clopidogrel kh\u00f4ng?"

    result = _service(normalizer, retriever, answer_client).answer(
        ChatRequest(message=question)
    )

    assert result["answer"] == "Gemini grounded interaction answer."
    assert result["intent"] == "drug_interaction_query"
    assert "gemini_low_confidence" in result["warnings"]
    payload = answer_client.payloads[0]
    assert payload["question"] == question
    assert payload["intent"] == "drug_interaction_query"
    assert payload["topic"] == "interaction"
    assert payload["query_type"] == "interaction"
    assert payload["drug_mentions"] == ["Omeprazole", "Clopidogrel"]
    assert [drug["raw_name"] for drug in payload["normalized_drugs"]] == [
        "Omeprazole",
        "Clopidogrel",
    ]
    evidence_chunk = payload["evidence_chunks"][0]
    assert evidence_chunk["rank"] == 1
    assert evidence_chunk["slug"] == "omeprazole"
    assert evidence_chunk["text"] == retriever.chunks[0]["text"]
    for internal_field in (
        "chunk_id",
        "vector_score",
        "rerank_score",
        "source_type",
    ):
        assert internal_field not in evidence_chunk
    assert payload["sources"][0]["slug"] == "omeprazole"
    assert "chunk_id" not in payload["sources"][0]
    assert payload["safety_rules"] == {
        "no_diagnosis": True,
        "no_prescribing": True,
        "use_only_evidence": True,
    }


def test_single_drug_payload_uses_original_question_not_retriever_query_text() -> None:
    answer_client = FakeAnswerClient(response="Gemini grounded single drug answer.")
    question = "Paracetamol c\u00f3 t\u00e1c d\u1ee5ng ph\u1ee5 g\u00ec?"

    _service(answer_client=answer_client).answer(ChatRequest(message=question))

    payload = answer_client.payloads[0]
    assert payload["question"] == question
    assert payload["intent"] == "single_drug_query"
    assert payload["query_type"] == "adverse_effect"
    assert payload["drug_mentions"] == ["Paracetamol"]


def test_interaction_query_ignores_generic_nhau_mention() -> None:
    normalizer = FakeNormalizer()
    retriever = FakeRetriever()
    result = _service(normalizer, retriever).answer(
        ChatRequest(message="Amlodipin và Alfuzosin có tương tác với nhau không?")
    )

    assert result["intent"] == "drug_interaction_query"
    assert normalizer.calls == [
        {"medications": ["Amlodipin", "Alfuzosin"], "top_k": 5}
    ]
    assert [drug["raw_name"] for drug in result["normalized_drugs"]] == [
        "Amlodipin",
        "Alfuzosin",
    ]
    assert "nhau" not in [
        str(drug.get("raw_name", "")).casefold()
        for drug in result["normalized_drugs"]
    ]


def test_single_drug_query_uses_topic_query_type() -> None:
    normalizer = FakeNormalizer()
    retriever = FakeRetriever()
    result = _service(normalizer, retriever).answer(
        ChatRequest(message="Paracetamol có tác dụng phụ gì?")
    )

    assert result["intent"] == "single_drug_query"
    assert result["answer"]
    assert normalizer.calls == [
        {"medications": ["Paracetamol"], "top_k": 5}
    ]
    assert retriever.calls[0]["query_type"] == "adverse_effect"
    assert "Paracetamol tác dụng không mong muốn tác dụng phụ" == retriever.calls[0][
        "query_text"
    ]


def test_single_drug_interaction_lookup_uses_interaction_query_type() -> None:
    for message in [
        "Paracetamol có những tương tác thuốc nào?",
        "Paracetamol tương tác với thuốc nào?",
    ]:
        normalizer = FakeNormalizer()
        retriever = FakeRetriever()
        result = _service(normalizer, retriever).answer(ChatRequest(message=message))

        assert result["intent"] == "single_drug_query"
        assert normalizer.calls == [
            {"medications": ["Paracetamol"], "top_k": 5}
        ]
        assert retriever.calls[0]["query_type"] == "interaction"
        assert retriever.calls[0]["query_text"] == "Paracetamol tương tác thuốc"
        assert "interaction_query_requires_two_drugs" not in result["warnings"]
        assert "missing_interaction_drug_mentions" not in result["warnings"]


def test_missing_drug_mentions_return_warning_answer() -> None:
    result = _service().answer(ChatRequest(message="Thuốc này có tương tác không?"))

    assert result["intent"] == "drug_interaction_query"
    assert "Vui lòng nhập rõ" in result["answer"]
    assert "missing_interaction_drug_mentions" in result["warnings"]


def test_insufficient_evidence_returns_no_guess_answer() -> None:
    result = _service(retriever=FakeRetriever(chunks=[])).answer(
        ChatRequest(message="Paracetamol có tác dụng phụ gì?")
    )

    assert result["intent"] == "single_drug_query"
    assert "chưa tìm thấy đủ bằng chứng" in result["answer"]
    assert "insufficient_evidence" in result["warnings"]


def test_insufficient_evidence_does_not_call_answer_client() -> None:
    answer_client = FakeAnswerClient()

    result = _service(
        retriever=FakeRetriever(chunks=[]),
        answer_client=answer_client,
    ).answer(ChatRequest(message="Paracetamol cÃ³ tÃ¡c dá»¥ng phá»¥ gÃ¬?"))

    assert result["intent"] == "single_drug_query"
    assert "insufficient_evidence" in result["warnings"]
    assert answer_client.payloads == []


def test_answer_client_failure_returns_safe_answer_without_deterministic_fallback() -> None:
    answer_client = FakeAnswerClient(raises=True)

    result = _service(answer_client=answer_client).answer(
        ChatRequest(message="Paracetamol cÃ³ tÃ¡c dá»¥ng phá»¥ gÃ¬?")
    )

    assert result["answer"] == GEMINI_CHAT_FAILURE_ANSWER
    assert GEMINI_CHAT_ANSWER_FAILED in result["warnings"]
    assert result["sources"]
    assert result["normalized_drugs"]
    assert "Paracetamol" not in result["answer"]
    assert "Vá»›i dá»¯ liá»‡u" not in result["answer"]


def test_invalid_answer_client_outputs_return_safe_failure() -> None:
    invalid_outputs = [
        {},
        {"answer": ""},
        {"answer": 123, "warnings": []},
        {"answer": "Valid answer.", "warnings": "bad"},
        {"answer": "Valid answer.", "warnings": ["ok", 123]},
        "not-a-dict",
        None,
    ]

    for invalid_output in invalid_outputs:
        answer_client = FakeRawAnswerClient(invalid_output)
        result = _service(answer_client=answer_client).answer(
            ChatRequest(message="Paracetamol có tác dụng phụ gì?")
        )

        assert result["answer"] == GEMINI_CHAT_FAILURE_ANSWER
        assert GEMINI_CHAT_ANSWER_FAILED in result["warnings"]
        assert answer_client.payloads


def test_gemini_warnings_are_merged_after_existing_warnings_and_deduplicated() -> None:
    retriever = FakeRetriever(warnings=["retriever_warning", "shared_warning"])
    answer_client = FakeAnswerClient(
        warnings=["shared_warning", "gemini_warning"]
    )

    result = _service(retriever=retriever, answer_client=answer_client).answer(
        ChatRequest(message="Paracetamol cÃ³ tÃ¡c dá»¥ng phá»¥ gÃ¬?")
    )

    assert result["warnings"] == [
        "retriever_warning",
        "shared_warning",
        "gemini_warning",
    ]


def test_chat_query_service_inner_response_shape_excludes_public_route_fields() -> None:
    result = _service().answer(
        ChatRequest(message="Paracetamol cÃ³ tÃ¡c dá»¥ng phá»¥ gÃ¬?")
    )

    assert set(result) == {
        "message",
        "answer",
        "intent",
        "normalized_drugs",
        "sources",
        "warnings",
    }
    assert "doctor_memory" not in result
    assert "doctor_id" not in result
    assert "disclaimer" not in result


def test_generated_answer_filters_forbidden_phrases() -> None:
    answer_client = FakeAnswerClient(
        response="Không an toàn, ngừng thuốc, tăng liều, giảm liều, đổi thuốc."
    )
    result = _service(
        retriever=FakeRetriever(
            chunks=[
                {
                    "rank": 1,
                    "slug": "test",
                    "text": (
                        "Không an toàn, không dùng được, ngừng thuốc, đổi thuốc, "
                        "tăng liều, giảm liều hoặc kê thêm."
                    ),
                }
            ]
        ),
        answer_client=answer_client,
    ).answer(ChatRequest(message="Paracetamol có tác dụng phụ gì?"))

    folded = result["answer"].casefold()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in folded


def test_paracetamol_adverse_effect_directive_is_rewritten_for_review() -> None:
    result = _service(
        retriever=FakeRetriever(
            chunks=[
                {
                    "rank": 1,
                    "slug": "paracetamol",
                    "entity_name": "Paracetamol",
                    "section": "tac_dung_khong_mong_muon",
                    "section_title": "Tác dụng không mong muốn",
                    "source": "trungtamthuoc",
                    "title": "Paracetamol",
                    "url": "https://example.test/paracetamol",
                    "text": (
                        "Nếu thấy xuất hiện ban hoặc các biểu hiện khác về da, "
                        "phải ngừng dùng thuốc và thăm khám thầy thuốc."
                    ),
                }
            ]
        )
    ).answer(ChatRequest(message="Paracetamol có tác dụng phụ gì?"))

    folded = result["answer"].casefold()
    assert "phải ngừng" not in folded
    assert "ngừng dùng thuốc" not in folded
    assert "thăm khám thầy thuốc" not in folded
    for phrase in [
        "không nên dùng",
        "không được dùng",
        "không được trộn",
        "tăng liều",
        "giảm liều",
        "đổi thuốc",
        "thay thuốc",
    ]:
        assert phrase not in folded
    assert result["answer"]
    assert result["sources"]
    assert result["sources"][0]["slug"] == "paracetamol"


def test_evidence_snippet_cleanup_avoids_fragments_and_bad_punctuation() -> None:
    result = _service(
        retriever=FakeRetriever(
            chunks=[
                {
                    "rank": 1,
                    "slug": "paracetamol",
                    "text": (
                        "iảm toàn thể huyết cầu khi dùng paracetamol.. "
                        "Phản ứng da nghiêm trọng hiếm gặp, bao gồm Stevens-Johnson , "
                        "được ghi nhận với paracetamol.."
                    ),
                }
            ]
        )
    ).answer(ChatRequest(message="Paracetamol có tác dụng phụ gì?"))

    assert "iảm toàn thể" not in result["answer"]
    assert ".." not in result["answer"]
    assert " ," not in result["answer"]


def test_answer_strips_raw_evidence_metadata() -> None:
    metadata_text = "\n".join(
        [
            "Hoạt chất: Omeprazole",
            "Slug: omeprazole",
            "Mục: Tương tác thuốc",
            "Nguồn: trungtamthuoc",
            "URL: https://example.test/omeprazole",
            "Nội dung: Nội dung lâm sàng cần giữ lại. Câu thứ hai cần giữ.",
        ]
    )
    result = _service(retriever=FakeRetriever(chunks=[{"text": metadata_text}])).answer(
        ChatRequest(message="Omeprazole có tương tác với Clopidogrel không?")
    )

    assert result["answer"]
    for leaked in ["Slug:", "Mục:", "Nguồn:", "URL:", "Nội dung:", "chunk_id"]:
        assert leaked not in result["answer"]
    assert "https://example.test" not in result["answer"]


def test_duplicate_sources_are_deduplicated_and_reranked() -> None:
    duplicate = {
        "rank": 9,
        "slug": "omeprazole",
        "section": "tuong_tac_thuoc",
        "source": "trungtamthuoc",
        "url": "https://example.test/omeprazole",
        "text": "Bằng chứng cần rà soát.",
    }
    result = _service(
        retriever=FakeRetriever(
            chunks=[
                duplicate,
                {**duplicate, "rank": 10, "text": "Bằng chứng lặp."},
                {
                    "rank": 11,
                    "slug": "clopidogrel",
                    "section": "tuong_tac_thuoc",
                    "source": "trungtamthuoc",
                    "url": "https://example.test/clopidogrel",
                    "text": "Bằng chứng khác.",
                },
            ]
        )
    ).answer(ChatRequest(message="Omeprazole có tương tác với Clopidogrel không?"))

    assert len(result["sources"]) == 2
    assert [source["rank"] for source in result["sources"]] == [1, 2]
    assert result["sources"][0]["slug"] == "omeprazole"
    assert result["sources"][1]["slug"] == "clopidogrel"


def test_direct_ingredient_mention_preferred_over_unresolved_product_mapping() -> None:
    normalizer = FakeNormalizer(
        result={
            "medications": [
                {
                    "raw_name": "Paracetamol",
                    "mapping_status": "matched",
                    "active_ingredients": [
                        {
                            "name": "Acetaminophen",
                            "normalized_name": "acetaminophen",
                            "evidence_status": "unresolved",
                            "evidence_slug": None,
                        }
                    ],
                    "warnings": ["ingredient_evidence_unresolved"],
                }
            ],
            "unique_evidence_slugs": [],
            "resolved_evidence_slugs": [],
        }
    )
    retriever = FakeRetriever(
        chunks=[
            {
                "rank": 1,
                "slug": "paracetamol",
                "text": "Paracetamol có bằng chứng phù hợp.",
            }
        ]
    )

    result = _service(normalizer=normalizer, retriever=retriever).answer(
        ChatRequest(message="Paracetamol có tác dụng phụ gì?")
    )

    ingredient = result["normalized_drugs"][0]["active_ingredients"][0]
    assert ingredient["normalized_name"] == "paracetamol"
    assert ingredient["evidence_status"] == "resolved"
    assert ingredient["evidence_slug"] == "paracetamol"
    assert retriever.calls[0]["normalized_result"]["unique_evidence_slugs"] == [
        "paracetamol"
    ]
    assert "insufficient_evidence" not in result["warnings"]


def test_get_chat_query_service_is_cached() -> None:
    get_chat_query_service.cache_clear()
    try:
        service_1 = get_chat_query_service()
        service_2 = get_chat_query_service()
    finally:
        get_chat_query_service.cache_clear()

    assert service_1 is service_2


def test_utf8_demo_chat_questions_render_expected_intents() -> None:
    service = _service()

    interaction = service.answer(
        ChatRequest(message="Omeprazole có tương tác với Clopidogrel không?")
    )
    adverse_effect = service.answer(
        ChatRequest(message="Paracetamol có tác dụng phụ gì?")
    )
    caution = service.answer(
        ChatRequest(message="Levofloxacin dùng cần lưu ý gì?")
    )
    out_of_scope = service.answer(ChatRequest(message="Viết giúp tôi bài văn"))

    assert interaction["intent"] == "drug_interaction_query"
    assert interaction["answer"]
    assert interaction["sources"]
    assert interaction["normalized_drugs"][0]["raw_name"] == "Omeprazole"
    assert interaction["normalized_drugs"][1]["raw_name"] == "Clopidogrel"

    assert adverse_effect["intent"] == "single_drug_query"
    assert adverse_effect["answer"]
    assert adverse_effect["sources"]
    assert adverse_effect["normalized_drugs"][0]["raw_name"] == "Paracetamol"

    assert caution["intent"] == "single_drug_query"
    assert caution["answer"]
    assert caution["sources"]
    assert caution["normalized_drugs"][0]["raw_name"] == "Levofloxacin"

    assert out_of_scope["intent"] == "out_of_scope"
    assert out_of_scope["answer"] == OUT_OF_SCOPE_REFUSAL
    assert out_of_scope["sources"] == []
