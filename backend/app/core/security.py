from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.app.core.config import get_settings


password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenDecodeError(ValueError):
    """Raised when an access token cannot be decoded or validated."""


def hash_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_context.verify(plain_password, hashed_password)


def create_access_token(
    subject: str,
    doctor_id: str,
    email: str,
    expires_delta: timedelta | None = None,
) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        "doctor_id": str(doctor_id),
        "email": str(email),
        "exp": expire,
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as exc:
        raise TokenDecodeError("Invalid or expired access token") from exc

    required_fields = ("sub", "doctor_id", "email", "exp")
    missing = [field for field in required_fields if not payload.get(field)]
    if missing:
        raise TokenDecodeError(
            "Access token missing required fields: " + ", ".join(missing)
        )
    return payload
