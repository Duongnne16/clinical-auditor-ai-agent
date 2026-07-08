from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.security import (
    TokenDecodeError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.app.db.models import User
from backend.app.db.session import get_db
from backend.app.schemas.auth import (
    CurrentUserResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)


router = APIRouter(prefix="/auth", tags=["auth"])


def _auth_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _token_response(user: User) -> TokenResponse:
    token = create_access_token(
        subject=str(user.id),
        doctor_id=user.doctor_id,
        email=user.email,
    )
    return TokenResponse(
        access_token=token,
        doctor_id=user.doctor_id,
        email=user.email,
        full_name=user.full_name,
    )


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise _auth_error()
    scheme, _, token = authorization.partition(" ")
    if scheme.casefold() != "bearer" or not token.strip():
        raise _auth_error()
    return token.strip()


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    user = User(
        doctor_id=f"doctor-{uuid4().hex}",
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        ) from exc
    db.refresh(user)
    return _token_response(user)


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    user = db.query(User).filter(User.email == payload.email).first()
    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password, user.hashed_password)
    ):
        raise _auth_error()
    return _token_response(user)


@router.get("/me", response_model=CurrentUserResponse)
def me(
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
