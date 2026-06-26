from __future__ import annotations

import json
import re
from typing import Any

from backend.app.core.config import get_settings


class GeminiRiskAnalyzerClient:
    """Gemini adapter that returns raw JSON-like risk analysis for validation."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        timeout_seconds: int | None = None,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self.temperature = (
            temperature if temperature is not None else settings.gemini_temperature
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.gemini_timeout_seconds
        )
        self.client = client

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required to call GeminiRiskAnalyzerClient"
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for GeminiRiskAnalyzerClient"
            ) from exc
        self.client = genai.Client(api_key=self.api_key)
        return self.client

    def _generation_config(self) -> Any | None:
        try:
            from google.genai import types

            return types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=self.temperature,
            )
        except Exception:
            return {
                "response_mime_type": "application/json",
                "temperature": self.temperature,
            }

    def build_prompt(self, evidence_context: dict[str, Any]) -> str:
        context_json = json.dumps(
            evidence_context, ensure_ascii=False, indent=2
        )
        return (
            "Bạn là AI hỗ trợ bác sĩ/dược sĩ kiểm tra đơn thuốc và phân tích "
            "tương tác thuốc.\n"
            "Bạn không chẩn đoán bệnh. Bạn không kê đơn.\n"
            "Bạn không khuyên bệnh nhân tự ngừng thuốc, tự tăng liều, tự giảm "
            "liều, hoặc tự thay thuốc.\n"
            "Chỉ phân tích dựa trên evidence_context được cung cấp. Không dùng "
            "kiến thức nội tại ngoài evidence_context.\n"
            "Nếu không có evidence trực tiếp trong evidence_context thì không "
            "tạo risk_item.\n"
            "Mọi risk_item bắt buộc phải có evidence_refs là chunk_id nằm trong "
            "valid_evidence_refs.\n"
            "Nếu thiếu thông tin bệnh nhân, ghi vào missing_information.\n"
            "Ưu tiên kiểm tra: drug-drug interaction/tương tác thuốc-thuốc, "
            "contraindication/chống chỉ định, precaution/thận trọng, bối cảnh "
            "suy thận/suy gan, pregnancy/lactation nếu liên quan, adverse effect "
            "nếu evidence rõ, overdose chỉ khi có thông tin liều/hàm lượng đủ "
            "để nghi ngờ.\n"
            "Có evidence overdose không có nghĩa là bệnh nhân đang quá liều. "
            "Có evidence pregnancy không có nghĩa là bệnh nhân đang mang thai.\n"
            "Nếu tình trạng thai kỳ/cho con bú là unknown, not provided hoặc chưa có thông tin, "
            "hãy ghi vào missing_information và diễn đạt là cần xác nhận tình trạng thai kỳ/cho con bú, "
            "không được viết như chống chỉ định đã xác nhận cho bệnh nhân.\n"
            "Chỉ trả về JSON hợp lệ, không Markdown, không giải thích ngoài JSON, "
            "không code fence.\n"
            "JSON schema cần trả về:\n"
            "{\n"
            '  "overall_risk_level": "low|moderate|high|unknown",\n'
            '  "risk_items": [\n'
            "    {\n"
            '      "risk_type": "interaction|contraindication|precaution|'
            'pregnancy_lactation|renal_hepatic|adverse_effect|overdose|general",\n'
            '      "severity": "low|moderate|high|unknown",\n'
            '      "title": "...",\n'
            '      "explanation": "...",\n'
            '      "affected_slugs": ["..."],\n'
            '      "evidence_refs": ["chunk_id_1"],\n'
            '      "recommendation": "..."\n'
            "    }\n"
            "  ],\n"
            '  "missing_information": ["..."]\n'
            "}\n\n"
            "evidence_context:\n"
            f"{context_json}"
        )

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return stripped

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return text
        return text[start : end + 1]

    def parse_response(self, response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if hasattr(response, "text"):
            text = str(response.text)
        else:
            text = str(response)
        text = self._strip_code_fence(text)
        text = self._extract_json_object(text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini response is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Gemini response is not valid JSON")
        return parsed

    def analyze_risks(self, evidence_context: dict[str, Any]) -> dict[str, Any]:
        prompt = self.build_prompt(evidence_context)
        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self._generation_config(),
            )
        except TypeError:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
        return self.parse_response(response)

    def get_stats(self) -> dict[str, Any]:
        return {
            "service": "GeminiRiskAnalyzerClient",
            "provider": "gemini",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
        }
