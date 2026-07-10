from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.gemini_chat_answer_client import GeminiChatAnswerClient


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModels:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_content(
        self,
        model: str,
        contents: str,
        config: Any | None = None,
    ) -> Any:
        self.calls.append(
            {"model": model, "contents": contents, "config": config}
        )
        return self.response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.models = FakeModels(response)


def _payload() -> dict[str, Any]:
    return {
        "question": "Paracetamol adverse effects?",
        "intent": "single_drug_query",
        "topic": "adverse_effect",
        "query_type": "adverse_effect",
        "drug_mentions": ["Paracetamol"],
        "normalized_drugs": [{"raw_name": "Paracetamol"}],
        "evidence_chunks": [
            {
                "chunk_id": "hidden-id",
                "slug": "paracetamol",
                "text": "Grounding evidence.",
            }
        ],
        "sources": [{"rank": 1, "slug": "paracetamol"}],
        "safety_rules": {
            "no_diagnosis": True,
            "no_prescribing": True,
            "use_only_evidence": True,
        },
    }


def test_constructor_is_safe_without_api_key() -> None:
    client = GeminiChatAnswerClient(api_key="")

    assert client.get_stats()["service"] == "GeminiChatAnswerClient"


def test_build_prompt_contains_grounding_safety_and_json_instructions() -> None:
    prompt = GeminiChatAnswerClient(client=FakeClient({})).build_prompt(_payload())

    assert "Chi su dung evidence_chunks" in prompt
    assert "Khong dung kien thuc ben ngoai" in prompt
    assert "Khong chan doan benh" in prompt
    assert "Khong ke don" in prompt
    assert "tu ngung thuoc" in prompt
    assert "tu tang lieu" in prompt
    assert "tu giam lieu" in prompt
    assert "tu thay" in prompt
    assert "Khong dung Markdown table" in prompt
    assert "Khong dua raw chunk_id" in prompt
    assert "JSON schema" in prompt
    assert '"answer"' in prompt
    assert '"warnings"' in prompt


def test_parse_response_accepts_dict() -> None:
    parsed = GeminiChatAnswerClient().parse_response(
        {"answer": "Grounded answer.", "warnings": ["review"]}
    )

    assert parsed == {"answer": "Grounded answer.", "warnings": ["review"]}


def test_parse_response_accepts_text_json() -> None:
    parsed = GeminiChatAnswerClient().parse_response(
        FakeResponse('{"answer": "Grounded answer.", "warnings": []}')
    )

    assert parsed == {"answer": "Grounded answer.", "warnings": []}


def test_parse_response_removes_json_code_fence() -> None:
    parsed = GeminiChatAnswerClient().parse_response(
        FakeResponse('```json\n{"answer": "Grounded answer.", "warnings": []}\n```')
    )

    assert parsed["answer"] == "Grounded answer."


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        "{}",
        "[]",
        '{"answer": "", "warnings": []}',
        '{"answer": "ok", "warnings": "bad"}',
    ],
)
def test_parse_response_rejects_invalid_response(response: str) -> None:
    with pytest.raises(ValueError):
        GeminiChatAnswerClient().parse_response(FakeResponse(response))


def test_answer_calls_injected_fake_client_and_returns_parsed_result() -> None:
    fake_client = FakeClient(
        {"answer": "Grounded answer.", "warnings": ["gemini_low_confidence"]}
    )
    client = GeminiChatAnswerClient(
        client=fake_client,
        model_name="gemini-test",
        temperature=0.0,
    )

    result = client.answer(_payload())

    assert result == {
        "answer": "Grounded answer.",
        "warnings": ["gemini_low_confidence"],
    }
    call = fake_client.models.calls[0]
    assert call["model"] == "gemini-test"
    assert "chat_answer_payload" in call["contents"]
    assert call["config"] is not None


def test_missing_api_key_raises_only_when_answer_attempts_real_call() -> None:
    client = GeminiChatAnswerClient(api_key="", client=None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is required"):
        client.answer(_payload())
