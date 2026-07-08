from __future__ import annotations

from pathlib import Path


ROUTES_DIR = Path(__file__).resolve().parents[1] / "app" / "api" / "routes"


def test_api_routes_do_not_use_legacy_fake_doctor_dependency() -> None:
    forbidden_tokens = [
        "get_doctor_id",
        "fake_doctor_id",
        "dev-doctor-001",
    ]

    offenders: list[str] = []
    for route_file in ROUTES_DIR.glob("*.py"):
        source = route_file.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in source:
                offenders.append(f"{route_file.name}: {token}")

    assert offenders == []
