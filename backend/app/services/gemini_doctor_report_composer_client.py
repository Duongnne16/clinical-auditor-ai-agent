from __future__ import annotations

import json
import re
from typing import Any

from backend.app.core.config import get_settings


class GeminiDoctorReportComposerClient:
    """Gemini adapter that turns a sanitized report payload into doctor-facing text."""

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
        self.model_name = model_name or settings.gemini_report_model or settings.gemini_model
        self.temperature = (
            temperature if temperature is not None else settings.gemini_temperature
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.gemini_report_timeout_seconds
            if settings.gemini_report_timeout_seconds is not None
            else settings.gemini_timeout_seconds
        )
        self.client = client

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required to call GeminiDoctorReportComposerClient"
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for GeminiDoctorReportComposerClient"
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

    def build_prompt(self, composer_payload: dict[str, Any]) -> str:
        payload_json = json.dumps(composer_payload, ensure_ascii=False, indent=2)
        return (
            "Bạn là AI hỗ trợ bác sĩ/dược sĩ diễn đạt lại kết quả kiểm tra đơn thuốc.\n"
            "Chỉ sử dụng composer_payload được cung cấp. Không dùng kiến thức bên ngoài.\n"
            "Không tạo phát hiện mới, không thêm thuốc mới, không thêm nguy cơ mới, "
            "không thêm nguồn mới, không tự tạo URL.\n"
            "Không dùng Markdown table. Viết plain Vietnamese chat-style text với các đoạn ngắn, "
            "các phát hiện đánh số và bullet nếu cần.\n"
            "Không lặp lại các nhãn 'Mức độ cần chú ý', 'Nội dung đánh giá', "
            "'Gợi ý rà soát' cho từng phát hiện. Hãy viết mỗi phát hiện thành "
            "một đoạn ngắn tự nhiên.\n"
            "Không hiển thị raw missing-information keys như pregnancy_status, pregnancy_lactation, "
            "hepatic_function, renal_function, current_medications. Không đưa mapping/database uncertainty "
            "vào phần thông tin bệnh nhân cần xác nhận.\n"
            "Nếu dữ liệu đã được lọc cho thấy allergies, comorbidities, current medications hoặc "
            "pregnancy/lactation là Không, Không ghi nhận hoặc Chưa ghi nhận, không xem các trường đó "
            "là thông tin còn thiếu. Chỉ viết những thông tin thật sự còn cần xác nhận trong payload.\n"
            "Không lặp lại nhãn nguồn dài trong mỗi phát hiện; frontend sẽ hiển thị link nguồn "
            "riêng ở cuối phần trả lời. Nếu cần, chỉ nhắc ngắn gọn rằng có nguồn tham khảo kèm theo.\n"
            "Dùng cách diễn đạt mềm cho bác sĩ/dược sĩ: nên rà soát, đối chiếu, cân nhắc "
            "nếu phù hợp. Với cảnh báo Omeprazole/PPI, không viết như hệ thống đang ra lệnh "
            "thăm khám, làm xét nghiệm hoặc tiếp tục điều trị; hãy dùng wording bác sĩ/dược sĩ "
            "nên đối chiếu với triệu chứng, chẩn đoán và kế hoạch theo dõi để cân nhắc đánh giá "
            "thêm nếu phù hợp. Không viết như chỉ dẫn trực tiếp cho bệnh nhân.\n"
            "Phần trả lời phải bắt đầu chính xác bằng: Kết quả kiểm tra đơn thuốc\n"
            "Dùng nhãn 'Mức ưu tiên rà soát' hoặc 'Mức độ cần chú ý'. "
            "Không dùng 'overall risk' hoặc 'mức nguy cơ tổng quan'.\n"
            "Không dùng các cụm quyết đoán hoặc chỉ đạo điều trị như: đơn thuốc an toàn, "
            "đơn thuốc không an toàn, dùng được, không dùng được, ngừng thuốc, dừng thuốc, "
            "đổi thuốc, thay thuốc, tăng liều, giảm liều, kê thêm, tự ý ngừng, tự ý dừng, "
            "tự ý đổi, tự ý thay, tự ý tăng, tự ý giảm.\n"
            "Bắt buộc giữ nguyên disclaimer trong payload.\n"
            "Chỉ trả về JSON hợp lệ, không code fence, không giải thích ngoài JSON.\n"
            'JSON schema: {"doctor_facing_response": "..."}\n\n'
            "composer_payload:\n"
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

    def parse_response(self, response: Any) -> dict[str, str]:
        if isinstance(response, dict):
            parsed = response
        else:
            text = str(response.text) if hasattr(response, "text") else str(response)
            text = self._strip_code_fence(text)
            text = self._extract_json_object(text)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Gemini doctor report response is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Gemini doctor report response is not valid JSON")
        doctor_facing_response = parsed.get("doctor_facing_response")
        if not isinstance(doctor_facing_response, str) or not doctor_facing_response.strip():
            raise ValueError("Gemini doctor report response is missing doctor_facing_response")
        return {"doctor_facing_response": doctor_facing_response.strip()}

    def compose(self, composer_payload: dict[str, Any]) -> dict[str, str]:
        prompt = self.build_prompt(composer_payload)
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
            "service": "GeminiDoctorReportComposerClient",
            "provider": "gemini",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
        }
