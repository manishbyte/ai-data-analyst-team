
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from auth.dependencies import (
    ClaimsDep,
    CurrentUserDep,
    SessionDep,
    unauthorized,
)
from auth.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from database.models import RevokedToken, User
from schemas.auth import (
    LoginRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/signup",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def signup(
    request: SignupRequest,
    session: SessionDep,
) -> UserResponse:
    existing_user = await session.scalar(
        select(User.id).where(User.email == request.email)
    )
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        name=request.name,
        email=request.email,
        password_hash=hash_password(request.password),
    )
    session.add(user)

    try:
        await session.commit()
        await session.refresh(user)
    except IntegrityError:
        # Handles concurrent signup attempts for the same email.
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from None

    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    session: SessionDep,
) -> TokenResponse:
    user = await session.scalar(
        select(User).where(User.email == request.email)
    )

    # Use the same public error for an unknown email and wrong password.
    if (
        user is None
        or not verify_password(request.password, user.password_hash)
        or not user.is_active
    ):
        raise unauthorized()

    settings = get_settings()
    if settings.jwt_secret_key is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is not configured.",
        )

    expiry = settings.access_token_expire_minutes
    token = create_access_token(
        user.id,
        secret_key=settings.jwt_secret_key.get_secret_value(),
        expires_minutes=expiry,
    )

    return TokenResponse(
        access_token=token,
        expires_in=expiry * 60,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: CurrentUserDep) -> UserResponse:
    return UserResponse.model_validate(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    claims: ClaimsDep,
    session: SessionDep,
) -> None:
    # Revoke this token until it expires. Repeating logout for the
    # same token is safe and does not create duplicate records.
    existing = await session.get(RevokedToken, claims.jti)
    if existing is None:
        session.add(
            RevokedToken(
                jti=claims.jti,
                expires_at=claims.expires_at,
            )
        )
        await session.commit()