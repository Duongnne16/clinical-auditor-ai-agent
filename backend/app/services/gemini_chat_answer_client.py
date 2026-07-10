from __future__ import annotations

import json
import re
from typing import Any

from backend.app.core.config import get_settings


class GeminiChatAnswerClient:
    """Gemini adapter that synthesizes a grounded chat answer from retrieved evidence."""

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
            raise RuntimeError("GEMINI_API_KEY is required to call GeminiChatAnswerClient")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is required for GeminiChatAnswerClient") from exc
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

    def build_prompt(self, payload: dict[str, Any]) -> str:
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        return (
            "Ban la AI ho tro bac si/duoc si tra loi cau hoi ve thuoc bang tieng Viet.\n"
            "Chi su dung evidence_chunks trong chat_answer_payload. Khong dung kien thuc ben ngoai.\n"
            "Khong chan doan benh. Khong ke don. Khong bao benh nhan tu ngung thuoc, "
            "tu tang lieu, tu giam lieu, tu thay the hoac tu thay doi thuoc.\n"
            "Neu bang chung duoc truy xuat khong du de tra loi, hay noi ro bang chung "
            "duoc truy xuat chua du.\n"
            "Khong dung Markdown table. Khong dua raw chunk_id vao cau tra loi. "
            "Khong tao nguon moi, khong tu tao URL, khong noi rang co nguon neu payload "
            "khong co nguon.\n"
            "Giu cau tra loi phu hop de bac si/duoc si ra soat, khong viet nhu chi dan "
            "truc tiep cho benh nhan.\n"
            "Chi tra ve JSON hop le, khong code fence, khong giai thich ngoai JSON.\n"
            'JSON schema: {"answer": "...", "warnings": ["..."]}\n\n'
            "chat_answer_payload:\n"
            f"{payload_json}"
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
            parsed = response
        else:
            text = str(response.text) if hasattr(response, "text") else str(response)
            text = self._strip_code_fence(text)
            text = self._extract_json_object(text)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Gemini chat answer response is not valid JSON") from exc

        if not isinstance(parsed, dict):
            raise ValueError("Gemini chat answer response is not valid JSON")
        answer = parsed.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Gemini chat answer response is missing answer")
        warnings = parsed.get("warnings", [])
        if warnings is None:
            warnings = []
        if not isinstance(warnings, list) or not all(
            isinstance(warning, str) for warning in warnings
        ):
            raise ValueError("Gemini chat answer response has invalid warnings")
        return {"answer": answer.strip(), "warnings": warnings}

    def answer(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = self.build_prompt(payload)
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
            "service": "GeminiChatAnswerClient",
            "provider": "gemini",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
        }
