from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.security import TokenDecodeError, decode_access_token
from backend.app.db.models import User
from backend.app.db.session import get_db


def get_doctor_id() -> str:
    """Legacy development fallback; protected routes must use JWT dependencies."""
    return get_settings().fake_doctor_id


def _auth_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise _auth_error()
    scheme, separator, token = authorization.partition(" ")
    if (
        not separator
        or scheme.casefold() != "bearer"
        or not token.strip()
        or " " in token.strip()
    ):
        raise _auth_error()
    return token.strip()


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    token = _extract_bearer_token(authorization)
    try:
        payload = decode_access_token(token)
    except TokenDecodeError as exc:
        raise _auth_error() from exc

    try:
        user_id = int(str(payload.get("sub") or ""))
    except ValueError as exc:
        raise _auth_error() from exc

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise _auth_error()
    return user


def get_current_doctor_id(
    current_user: User = Depends(get_current_user),
) -> str:
    return current_user.doctor_id
