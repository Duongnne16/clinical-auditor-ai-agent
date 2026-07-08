from __future__ import annotations

from typing import Any

from backend.app.api.routes.chat import get_chat_query_service
from backend.app.schemas.chat import ChatRequest
from backend.app.services.chat_query_service import (
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
    def __init__(self, chunks: list[dict[str, Any]] | None = None) -> None:
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
        self.calls: list[dict[str, Any]] = []

    def retrieve_for_normalized_result(
        self, normalized_result: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        kwargs["normalized_result"] = normalized_result
        self.calls.append(kwargs)
        return {"chunks": self.chunks, "warnings": []}


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


def _service(
    normalizer: FakeNormalizer | None = None,
    retriever: FakeRetriever | None = None,
) -> ChatQueryService:
    return ChatQueryService(
        normalizer=normalizer or FakeNormalizer(),  # type: ignore[arg-type]
        retriever=retriever or FakeRetriever(),  # type: ignore[arg-type]
        ingredient_resolver=FakeIngredientResolver(),  # type: ignore[arg-type]
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


def test_generated_answer_filters_forbidden_phrases() -> None:
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
        )
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
    assert "Bác sĩ/dược sĩ" in result["answer"]
    assert "rà soát" in result["answer"] or "đối chiếu" in result["answer"]
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

    assert "Nội dung lâm sàng cần giữ lại" in result["answer"]
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
