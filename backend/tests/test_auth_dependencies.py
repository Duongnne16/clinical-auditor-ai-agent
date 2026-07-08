from __future__ import annotations

from collections.abc import Generator
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.dependencies import (
    get_current_doctor_id,
    get_current_user,
    get_doctor_id,
)
from backend.app.core.security import create_access_token, hash_password
from backend.app.db.models import Base, User
from backend.app.db.session import get_db


@pytest.fixture()
def dependency_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "auth_dependencies.sqlite3"
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    testing_session = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False
    )
    Base.metadata.create_all(bind=engine)

    app = FastAPI()

    def override_get_db() -> Generator[Session, None, None]:
        with testing_session() as session:
            yield session

    @app.get("/current-user")
    def current_user(user: User = Depends(get_current_user)) -> dict:
        return {
            "id": user.id,
            "doctor_id": user.doctor_id,
            "email": user.email,
        }

    @app.get("/current-doctor-id")
    def current_doctor_id(
        doctor_id: str = Depends(get_current_doctor_id),
    ) -> dict:
        return {"doctor_id": doctor_id}

    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)
        client.testing_session = testing_session  # type: ignore[attr-defined]
        yield client
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _create_user(
    client: TestClient,
    *,
    email: str = "doctor@example.test",
    is_active: bool = True,
) -> User:
    testing_session = client.testing_session  # type: ignore[attr-defined]
    with testing_session() as session:
        user = User(
            doctor_id=f"doctor-{email.split('@')[0]}",
            email=email,
            full_name="Dependency Doctor",
            hashed_password=hash_password("secure-password-123"),
            is_active=is_active,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def _token_for_user(user: User, expires_delta: timedelta | None = None) -> str:
    return create_access_token(
        subject=str(user.id),
        doctor_id=user.doctor_id,
        email=user.email,
        expires_delta=expires_delta,
    )


def test_get_current_user_with_valid_token_returns_user(
    dependency_client: TestClient,
) -> None:
    user = _create_user(dependency_client)
    token = _token_for_user(user)

    response = dependency_client.get(
        "/current-user",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": user.id,
        "doctor_id": user.doctor_id,
        "email": user.email,
    }


def test_get_current_doctor_id_returns_doctor_id(
    dependency_client: TestClient,
) -> None:
    user = _create_user(dependency_client, email="doctorid@example.test")
    token = _token_for_user(user)

    response = dependency_client.get(
        "/current-doctor-id",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json() == {"doctor_id": user.doctor_id}


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "Token abc",
        "Bearer",
        "Bearer ",
        "Bearer abc def",
    ],
)
def test_missing_or_malformed_authorization_returns_401(
    dependency_client: TestClient,
    authorization: str | None,
) -> None:
    headers = {}
    if authorization is not None:
        headers["Authorization"] = authorization

    response = dependency_client.get("/current-user", headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_invalid_token_returns_401(dependency_client: TestClient) -> None:
    response = dependency_client.get(
        "/current-user",
        headers={"Authorization": "Bearer not-a-valid-token"},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_expired_token_returns_401(dependency_client: TestClient) -> None:
    user = _create_user(dependency_client, email="expired@example.test")
    token = _token_for_user(user, expires_delta=timedelta(seconds=-1))

    response = dependency_client.get(
        "/current-user",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_valid_token_for_missing_user_returns_401(
    dependency_client: TestClient,
) -> None:
    token = create_access_token(
        subject="999",
        doctor_id="doctor-missing",
        email="missing@example.test",
    )

    response = dependency_client.get(
        "/current-user",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_valid_token_for_inactive_user_returns_401(
    dependency_client: TestClient,
) -> None:
    user = _create_user(
        dependency_client,
        email="inactive@example.test",
        is_active=False,
    )
    token = _token_for_user(user)

    response = dependency_client.get(
        "/current-user",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_get_doctor_id_keeps_development_fake_doctor_id() -> None:
    assert get_doctor_id() == "dev-doctor-001"
