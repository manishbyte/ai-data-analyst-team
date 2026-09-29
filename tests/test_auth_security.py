
from uuid import uuid4

import pytest
from pydantic import ValidationError

from auth.security import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from schemas.auth import LoginRequest, SignupRequest


TEST_SECRET = "test-only-secret-key-at-least-32-characters-long"


def test_signup_normalizes_email_and_name():
    request = SignupRequest(
        name="  Manish  ",
        email="  MANISH@example.com  ",
        password="strong-test-password",
    )

    assert request.name == "Manish"
    assert request.email == "manish@example.com"


def test_signup_rejects_short_password():
    with pytest.raises(ValidationError):
        SignupRequest(
            name="Manish",
            email="manish@example.com",
            password="short",
        )


def test_signup_rejects_empty_name():
    with pytest.raises(ValidationError):
        SignupRequest(
            name="   ",
            email="manish@example.com",
            password="strong-test-password",
        )


def test_password_hashing_and_verification():
    password = "strong-test-password"
    password_hash = hash_password(password)

    assert password_hash != password
    assert verify_password(password, password_hash)
    assert not verify_password("wrong-password", password_hash)


def test_access_token_round_trip():
    user_id = uuid4()

    token = create_access_token(
        user_id,
        secret_key=TEST_SECRET,
        expires_minutes=30,
    )

    assert decode_access_token(
        token,
        secret_key=TEST_SECRET,
    ) == user_id


def test_access_token_rejects_wrong_secret():
    token = create_access_token(
        uuid4(),
        secret_key=TEST_SECRET,
    )

    with pytest.raises(InvalidTokenError):
        decode_access_token(
            token,
            secret_key="another-test-secret-at-least-32-characters-long",
        )


def test_access_token_rejects_malformed_token():
    with pytest.raises(InvalidTokenError):
        decode_access_token(
            "not-a-valid-jwt",
            secret_key=TEST_SECRET,
        )


def test_login_normalizes_email():
    request = LoginRequest(
        email=" MANISH@example.com ",
        password="strong-test-password",
    )

    assert request.email == "manish@example.com"