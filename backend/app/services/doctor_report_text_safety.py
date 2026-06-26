from __future__ import annotations

import re


FORBIDDEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"đơn\s+thuốc\s+không\s+an\s+toàn", re.IGNORECASE),
        "đơn thuốc cần được bác sĩ/dược sĩ rà soát",
    ),
    (
        re.compile(r"đơn\s+thuốc\s+an\s+toàn", re.IGNORECASE),
        "đơn thuốc cần được bác sĩ/dược sĩ rà soát",
    ),
    (
        re.compile(r"không\s+dùng\s+được", re.IGNORECASE),
        "cần rà soát trước khi sử dụng",
    ),
    (
        re.compile(r"dùng\s+được", re.IGNORECASE),
        "cần được đánh giá theo bối cảnh lâm sàng",
    ),
    (re.compile(r"(?<!không tự ý )ngừng\s+thuốc", re.IGNORECASE), "rà soát thuốc"),
    (re.compile(r"(?<!không tự ý )dừng\s+thuốc", re.IGNORECASE), "rà soát thuốc"),
    (
        re.compile(r"(?<!không tự ý )đổi\s+thuốc", re.IGNORECASE),
        "cân nhắc phương án xử trí phù hợp",
    ),
    (
        re.compile(r"(?<!không tự ý )thay\s+thuốc", re.IGNORECASE),
        "cân nhắc phương án xử trí phù hợp",
    ),
    (re.compile(r"(?<!không tự ý )tăng\s+liều", re.IGNORECASE), "rà soát liều"),
    (re.compile(r"(?<!không tự ý )giảm\s+liều", re.IGNORECASE), "rà soát liều"),
    (
        re.compile(r"kê\s+thêm", re.IGNORECASE),
        "cân nhắc phương án điều trị phù hợp",
    ),
    (
        re.compile(r"khuyên\s+bệnh\s+nhân[^.\n]*", re.IGNORECASE),
        "Bác sĩ/dược sĩ nên rà soát thời điểm dùng thuốc và cân nhắc hướng dẫn dùng cách xa nếu phù hợp",
    ),
    (
        re.compile(r"hướng\s+dẫn\s+bệnh\s+nhân\s+(uống|dùng)[^.\n]*", re.IGNORECASE),
        "Bác sĩ/dược sĩ nên rà soát thời điểm dùng thuốc và cân nhắc hướng dẫn dùng cách xa nếu phù hợp",
    ),
    (
        re.compile(r"để\s+tránh[^.\n]*cần\s+uống[^.\n]*", re.IGNORECASE),
        "Bác sĩ/dược sĩ nên rà soát thời điểm dùng thuốc và cân nhắc hướng dẫn dùng cách xa nếu phù hợp",
    ),
    (
        re.compile(r"cân\s+nhắc\s+thực\s+hiện\s+các\s+xét\s+nghiệm[^.\n]*", re.IGNORECASE),
        "Bác sĩ/dược sĩ nên đối chiếu với triệu chứng, chẩn đoán và kế hoạch theo dõi để cân nhắc đánh giá thêm nếu phù hợp",
    ),
    (
        re.compile(r"cần\s+thăm\s+khám\s+và\s+thực\s+hiện\s+các\s+xét\s+nghiệm[^.\n]*", re.IGNORECASE),
        "Bác sĩ/dược sĩ nên đối chiếu với triệu chứng, chẩn đoán và kế hoạch theo dõi của bệnh nhân để cân nhắc đánh giá thêm nếu phù hợp",
    ),
    (
        re.compile(r"cần\s+thực\s+hiện\s+các\s+xét\s+nghiệm[^.\n]*", re.IGNORECASE),
        "Bác sĩ/dược sĩ nên đối chiếu với triệu chứng, chẩn đoán và kế hoạch theo dõi để cân nhắc đánh giá thêm nếu phù hợp",
    ),
    (
        re.compile(r"thực\s+hiện\s+các\s+xét\s+nghiệm\s+cần\s+thiết[^.\n]*", re.IGNORECASE),
        "Bác sĩ/dược sĩ nên đối chiếu với triệu chứng, chẩn đoán và kế hoạch theo dõi để cân nhắc đánh giá thêm nếu phù hợp",
    ),
    (
        re.compile(r"trước\s+khi\s+tiếp\s+tục\s+điều\s+trị\s+bằng[^.\n]*", re.IGNORECASE),
        "trước khi cân nhắc phương án theo dõi hoặc xử trí phù hợp",
    ),
    (
        re.compile(r"tiếp\s+tục\s+điều\s+trị\s+bằng[^.\n]*", re.IGNORECASE),
        "cân nhắc phương án theo dõi hoặc xử trí phù hợp",
    ),
    (
        re.compile(r"cần\s+uống\s+", re.IGNORECASE),
        "Cần rà soát thời điểm dùng ",
    ),
    (
        re.compile(r"overall\s+risk", re.IGNORECASE),
        "Mức ưu tiên rà soát",
    ),
    (
        re.compile(r"mức\s+nguy\s+cơ\s+tổng\s+quan", re.IGNORECASE),
        "Mức ưu tiên rà soát",
    ),
    (
        re.compile(
            r"(?<!không\s)tự\s+ý\s+(ngừng|dừng|đổi|thay|tăng|giảm)",
            re.IGNORECASE,
        ),
        "cần trao đổi với bác sĩ/dược sĩ trước khi điều chỉnh",
    ),
)


def sanitize_doctor_report_text(text: str) -> str:
    safe_text = str(text or "")
    for pattern, replacement in FORBIDDEN_PATTERNS:
        safe_text = pattern.sub(replacement, safe_text)
    return safe_text


def has_unsafe_doctor_report_text(text: str) -> bool:
    value = str(text or "")
    return any(pattern.search(value) for pattern, _ in FORBIDDEN_PATTERNS)
