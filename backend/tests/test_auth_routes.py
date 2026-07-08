from __future__ import annotations

from collections.abc import Generator
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.security import create_access_token, hash_password
from backend.app.db.models import Base, User
from backend.app.db.session import get_db
from backend.app.main import app


@pytest.fixture()
def auth_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "auth.sqlite3"
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    testing_session = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False
    )
    Base.metadata.create_all(bind=engine)

    def override_get_db() -> Generator[Session, None, None]:
        with testing_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _register_payload(email: str = "doctor@example.test") -> dict[str, str]:
    return {
        "email": email,
        "password": "secure-password-123",
        "full_name": "Doctor Example",
    }


def _create_user(
    _client: TestClient,
    *,
    email: str = "inactive@example.test",
    is_active: bool = True,
) -> User:
    override = app.dependency_overrides[get_db]
    session_generator = override()
    session = next(session_generator)
    try:
        user = User(
            doctor_id=f"doctor-{email.split('@')[0]}",
            email=email,
            full_name="Stored Doctor",
            hashed_password=hash_password("secure-password-123"),
            is_active=is_active,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    finally:
        session.close()
        session_generator.close()


def test_register_success_returns_safe_token_response(
    auth_client: TestClient,
) -> None:
    response = auth_client.post(
        "/api/v1/auth/register",
        json=_register_payload(),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["doctor_id"].startswith("doctor-")
    assert body["email"] == "doctor@example.test"
    assert body["full_name"] == "Doctor Example"
    assert "hashed_password" not in body
    assert "password" not in body


def test_register_duplicate_email_returns_400(
    auth_client: TestClient,
) -> None:
    payload = _register_payload("duplicate@example.test")

    first = auth_client.post("/api/v1/auth/register", json=payload)
    second = auth_client.post("/api/v1/auth/register", json=payload)

    assert first.status_code == 201
    assert second.status_code == 400


def test_login_success_returns_token(auth_client: TestClient) -> None:
    payload = _register_payload("login@example.test")
    auth_client.post("/api/v1/auth/register", json=payload)

    response = auth_client.post(
        "/api/v1/auth/login",
        json={
            "email": "login@example.test",
            "password": "secure-password-123",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["email"] == "login@example.test"
    assert "hashed_password" not in body


def test_login_wrong_password_returns_401(auth_client: TestClient) -> None:
    payload = _register_payload("wrong@example.test")
    auth_client.post("/api/v1/auth/register", json=payload)

    response = auth_client.post(
        "/api/v1/auth/login",
        json={"email": "wrong@example.test", "password": "bad-password"},
    )

    assert response.status_code == 401


def test_login_inactive_user_returns_401(auth_client: TestClient) -> None:
    _create_user(auth_client, email="inactive@example.test", is_active=False)

    response = auth_client.post(
        "/api/v1/auth/login",
        json={
            "email": "inactive@example.test",
            "password": "secure-password-123",
        },
    )

    assert response.status_code == 401


def test_me_with_valid_token_returns_current_user(
    auth_client: TestClient,
) -> None:
    register_response = auth_client.post(
        "/api/v1/auth/register",
        json=_register_payload("me@example.test"),
    )
    token = register_response.json()["access_token"]

    response = auth_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "me@example.test"
    assert body["doctor_id"].startswith("doctor-")
    assert body["is_active"] is True
    assert "hashed_password" not in body


def test_me_without_token_returns_401(auth_client: TestClient) -> None:
    response = auth_client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_me_without_bearer_scheme_returns_401(
    auth_client: TestClient,
) -> None:
    response = auth_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Token abc"},
    )

    assert response.status_code == 401


def test_me_with_invalid_token_returns_401(auth_client: TestClient) -> None:
    response = auth_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer not-a-token"},
    )

    assert response.status_code == 401


def test_me_with_expired_token_returns_401(auth_client: TestClient) -> None:
    user = _create_user(auth_client, email="expired@example.test")
    token = create_access_token(
        subject=str(user.id),
        doctor_id=user.doctor_id,
        email=user.email,
        expires_delta=timedelta(seconds=-1),
    )

    response = auth_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_me_with_token_for_missing_user_returns_401(
    auth_client: TestClient,
) -> None:
    token = create_access_token(
        subject="999",
        doctor_id="doctor-missing",
        email="missing@example.test",
    )

    response = auth_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
