
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import jwt
from pwdlib import PasswordHash


_password_hash = PasswordHash.recommended()
JWT_ALGORITHM = "HS256"


class InvalidTokenError(ValueError):
    """Raised when an access token is invalid or expired."""


@dataclass(frozen=True)
class AccessTokenClaims:
    user_id: UUID
    jti: str
    expires_at: datetime


def hash_password(password: str) -> str:
    """Hash a password using Argon2."""
    if not password or len(password) > 128:
        raise ValueError("Password length is invalid.")

    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its stored hash."""
    try:
        return _password_hash.verify(password, password_hash)
    except (ValueError, TypeError):
        return False


def create_access_token(
    user_id: UUID,
    *,
    secret_key: str,
    expires_minutes: int = 30,
) -> str:
    if len(secret_key) < 32:
        raise ValueError("JWT secret must contain at least 32 characters.")

    if not 1 <= expires_minutes <= 1440:
        raise ValueError("Token expiry must be between 1 and 1440 minutes.")

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=expires_minutes)

    payload = {
        "sub": str(user_id),
        "jti": str(uuid4()),
        "iat": now,
        "exp": expires_at,
        "type": "access",
    }

    return jwt.encode(payload, secret_key, algorithm=JWT_ALGORITHM)


def decode_access_token_details(
    token: str,
    *,
    secret_key: str,
) -> AccessTokenClaims:
    """Validate a JWT and extract its identity and revocation ID."""
    try:
        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "jti", "iat", "exp", "type"]},
        )

        if payload.get("type") != "access":
            raise InvalidTokenError("Invalid access token.")

        user_id = UUID(payload["sub"])
        jti = str(UUID(payload["jti"]))
        expires_timestamp = payload["exp"]

        if not isinstance(expires_timestamp, (int, float)):
            raise InvalidTokenError("Invalid token expiration.")

        expires_at = datetime.fromtimestamp(
            expires_timestamp,
            tz=timezone.utc,
        )

        return AccessTokenClaims(
            user_id=user_id,
            jti=jti,
            expires_at=expires_at,
        )

    except (jwt.InvalidTokenError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise InvalidTokenError("Invalid or expired access token.") from exc


def decode_access_token(
    token: str,
    *,
    secret_key: str,
) -> UUID:
    """Validate a JWT and return its subject as a UUID."""
    return decode_access_token_details(
        token,
        secret_key=secret_key,
    ).user_id