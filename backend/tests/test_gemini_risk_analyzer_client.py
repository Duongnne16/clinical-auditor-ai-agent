from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.gemini_risk_analyzer_client import (
    GeminiRiskAnalyzerClient,
)
from backend.app.services.risk_analyzer_service import RiskAnalyzerService


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


def _evidence_context() -> dict[str, Any]:
    return {
        "patient_context": {"age": 60},
        "medications": [
            {
                "raw_name": "Omeprazol 20mg",
                "active_ingredients": [
                    {"name": "Omeprazole", "evidence_slug": "omeprazole"}
                ],
            }
        ],
        "interaction_candidates": [
            {"slug_a": "omeprazole", "slug_b": "metformin"}
        ],
        "evidence_by_query_type": {
            "interaction": [
                {
                    "chunk_id": "chunk-1",
                    "slug": "omeprazole",
                    "section": "tuong_tac_thuoc",
                    "snippet": "Evidence text",
                }
            ]
        },
        "valid_evidence_refs": ["chunk-1"],
        "missing_information": ["renal_function"],
    }


def _valid_llm_result(ref: str = "chunk-1") -> dict[str, Any]:
    return {
        "overall_risk_level": "moderate",
        "risk_items": [
            {
                "risk_type": "interaction",
                "severity": "moderate",
                "title": "Interaction to review",
                "explanation": "Grounded explanation.",
                "affected_slugs": ["omeprazole", "metformin"],
                "evidence_refs": [ref],
                "recommendation": "Review with clinician.",
            }
        ],
        "missing_information": [],
    }


def test_constructor_is_safe_without_api_key() -> None:
    client = GeminiRiskAnalyzerClient(api_key="")

    assert client.get_stats()["service"] == "GeminiRiskAnalyzerClient"


def test_build_prompt_contains_required_safety_and_json_instructions() -> None:
    prompt = GeminiRiskAnalyzerClient(client=FakeClient({})).build_prompt(
        _evidence_context()
    )

    assert "kiểm tra đơn thuốc" in prompt
    assert "tương tác thuốc" in prompt
    assert "valid_evidence_refs" in prompt
    assert "chunk-1" in prompt
    assert "interaction_candidates" in prompt
    assert "Chỉ trả về JSON hợp lệ" in prompt
    assert "Không dùng kiến thức nội tại ngoài evidence_context" in prompt
    assert "không khuyên bệnh nhân tự ngừng thuốc" in prompt
    assert "tự tăng liều" in prompt
    assert "tự giảm liều" in prompt
    assert "tự thay thuốc" in prompt
    assert "Nếu không có evidence trực tiếp" in prompt


def test_parse_response_accepts_dict() -> None:
    response = _valid_llm_result()

    assert GeminiRiskAnalyzerClient().parse_response(response) is response


def test_parse_response_accepts_text_json() -> None:
    parsed = GeminiRiskAnalyzerClient().parse_response(
        FakeResponse('{"overall_risk_level": "low", "risk_items": []}')
    )

    assert parsed["overall_risk_level"] == "low"


def test_parse_response_removes_json_code_fence() -> None:
    parsed = GeminiRiskAnalyzerClient().parse_response(
        FakeResponse(
            '```json\n{"overall_risk_level": "unknown", "risk_items": []}\n```'
        )
    )

    assert parsed["overall_risk_level"] == "unknown"


def test_parse_response_extracts_json_object_from_surrounding_text() -> None:
    parsed = GeminiRiskAnalyzerClient().parse_response(
        FakeResponse(
            'Here is JSON: {"overall_risk_level": "high", "risk_items": []} done.'
        )
    )

    assert parsed["overall_risk_level"] == "high"


def test_parse_response_invalid_json_raises() -> None:
    with pytest.raises(ValueError, match="Gemini response is not valid JSON"):
        GeminiRiskAnalyzerClient().parse_response(FakeResponse("not json"))


def test_analyze_risks_calls_injected_fake_client_and_returns_parsed_dict() -> None:
    fake_client = FakeClient(_valid_llm_result())
    client = GeminiRiskAnalyzerClient(
        client=fake_client,
        model_name="gemini-test",
        temperature=0.0,
    )

    result = client.analyze_risks(_evidence_context())

    assert result["overall_risk_level"] == "moderate"
    assert result["risk_items"][0]["evidence_refs"] == ["chunk-1"]
    call = fake_client.models.calls[0]
    assert call["model"] == "gemini-test"
    assert "evidence_context" in call["contents"]
    assert call["config"] is not None


def test_analyze_risks_does_not_validate_or_remove_unsupported_refs() -> None:
    fake_client = FakeClient(_valid_llm_result(ref="unsupported-ref"))
    client = GeminiRiskAnalyzerClient(client=fake_client)

    result = client.analyze_risks(_evidence_context())

    assert result["risk_items"][0]["evidence_refs"] == ["unsupported-ref"]


def test_missing_api_key_raises_only_when_real_call_is_attempted() -> None:
    client = GeminiRiskAnalyzerClient(api_key="", client=None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is required"):
        client.analyze_risks(_evidence_context())


def test_get_stats() -> None:
    stats = GeminiRiskAnalyzerClient(
        client=FakeClient({}),
        model_name="gemini-test",
        temperature=0.2,
        timeout_seconds=30,
    ).get_stats()

    assert stats == {
        "service": "GeminiRiskAnalyzerClient",
        "provider": "gemini",
        "model_name": "gemini-test",
        "temperature": 0.2,
        "timeout_seconds": 30,
    }


def test_risk_analyzer_service_accepts_gemini_client_with_valid_refs() -> None:
    fake_client = FakeClient(_valid_llm_result())
    gemini_client = GeminiRiskAnalyzerClient(client=fake_client)
    analyzer = RiskAnalyzerService(llm_client=gemini_client)

    result = analyzer.analyze(
        normalized_result={"medications": []},
        evidence_bundle={
            "query_results": {
                "interaction": {
                    "chunks": [
                        {
                            "chunk_id": "chunk-1",
                            "slug": "omeprazole",
                            "section": "tuong_tac_thuoc",
                            "text": "Evidence text",
                        }
                    ]
                }
            },
            "unique_chunks": [{"chunk_id": "chunk-1"}],
        },
    )

    assert result["status"] == "analysis_ready"
    assert result["risk_items"][0]["evidence_refs"] == ["chunk-1"]


def test_risk_analyzer_service_removes_unsupported_refs_from_gemini_client() -> None:
    fake_client = FakeClient(_valid_llm_result(ref="unsupported-ref"))
    analyzer = RiskAnalyzerService(
        llm_client=GeminiRiskAnalyzerClient(client=fake_client)
    )

    result = analyzer.analyze(
        normalized_result={"medications": []},
        evidence_bundle={
            "query_results": {
                "interaction": {
                    "chunks": [
                        {
                            "chunk_id": "chunk-1",
                            "slug": "omeprazole",
                            "section": "tuong_tac_thuoc",
                            "text": "Evidence text",
                        }
                    ]
                }
            },
            "unique_chunks": [{"chunk_id": "chunk-1"}],
        },
    )

    assert result["risk_items"] == []
    assert "risk_item_removed_due_to_missing_evidence" in result["warnings"]
