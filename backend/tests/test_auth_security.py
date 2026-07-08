from datetime import timedelta

import pytest

from backend.app.core.security import (
    TokenDecodeError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_does_not_store_plaintext_and_verifies() -> None:
    password = "secure-password-123"

    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrong-password", hashed) is False


def test_create_and_decode_access_token_contains_required_claims() -> None:
    token = create_access_token(
        subject="42",
        doctor_id="doctor-42",
        email="doctor@example.test",
        expires_delta=timedelta(minutes=5),
    )

    payload = decode_access_token(token)

    assert payload["sub"] == "42"
    assert payload["doctor_id"] == "doctor-42"
    assert payload["email"] == "doctor@example.test"
    assert payload["exp"]


def test_decode_access_token_rejects_invalid_token() -> None:
    with pytest.raises(TokenDecodeError, match="Invalid or expired"):
        decode_access_token("not-a-valid-token")


def test_decode_access_token_rejects_expired_token() -> None:
    token = create_access_token(
        subject="42",
        doctor_id="doctor-42",
        email="doctor@example.test",
        expires_delta=timedelta(seconds=-1),
    )

    with pytest.raises(TokenDecodeError, match="Invalid or expired"):
        decode_access_token(token)
