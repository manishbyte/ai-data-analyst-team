
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from auth.security import (
    AccessTokenClaims,
    InvalidTokenError,
    decode_access_token_details,
)
from database.app_session import get_db_session
from database.models import RevokedToken, User


bearer_scheme = HTTPBearer(auto_error=False)

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_token_claims(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    session: SessionDep,
) -> AccessTokenClaims:
    if credentials is None:
        raise unauthorized()

    settings = get_settings()
    if settings.jwt_secret_key is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is not configured.",
        )

    try:
        claims = decode_access_token_details(
            credentials.credentials,
            secret_key=settings.jwt_secret_key.get_secret_value(),
        )
    except InvalidTokenError:
        raise unauthorized() from None

    revoked = await session.scalar(
        select(RevokedToken.jti).where(
            RevokedToken.jti == claims.jti
        )
    )
    if revoked is not None:
        raise unauthorized()

    return claims


ClaimsDep = Annotated[AccessTokenClaims, Depends(get_token_claims)]


async def get_current_user(
    claims: ClaimsDep,
    session: SessionDep,
) -> User:
    user = await session.scalar(
        select(User).where(User.id == claims.user_id)
    )

    if user is None or not user.is_active:
        raise unauthorized()

    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]