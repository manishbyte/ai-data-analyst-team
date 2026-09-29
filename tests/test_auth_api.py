
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from app import main
from auth import dependencies, router
from auth.security import create_access_token, hash_password
from database.app_session import get_db_session
from database.models import User


TEST_SECRET = "test-only-secret-key-at-least-32-characters-long"


@pytest.fixture
def api_client(monkeypatch):
    """Create a test client with a mocked database session."""
    session = AsyncMock(spec=AsyncSession)

    async def override_db_session():
        yield session

    test_settings = SimpleNamespace(
        jwt_secret_key=SecretStr(TEST_SECRET),
        access_token_expire_minutes=30,
        app_name="AI Data Analyst Test",
    )

    monkeypatch.setattr(router, "get_settings", lambda: test_settings)
    monkeypatch.setattr(
        dependencies, "get_settings", lambda: test_settings
    )

    main.app.dependency_overrides[get_db_session] = override_db_session

    # Do not use TestClient as a context manager here:
    # that would run the application's database startup lifespan.
    client = TestClient(main.app)

    yield client, session

    main.app.dependency_overrides.clear()


@pytest.fixture
def test_user():
    return User(
        id=uuid4(),
        name="Test User",
        email="testuser@example.com",
        password_hash=hash_password("TestPassword123!"),
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


def auth_headers(user_id):
    token = create_access_token(
        user_id,
        secret_key=TEST_SECRET,
        expires_minutes=30,
    )
    return {"Authorization": f"Bearer {token}"}


def test_signup_success(api_client):
    client, session = api_client
    session.scalar.return_value = None

    async def refresh_user(user):
      if user.id is None:
          user.id = uuid4()

      if user.created_at is None:
          user.created_at = datetime.now(timezone.utc)

      if user.is_active is None:
          user.is_active = True

    session.refresh.side_effect = refresh_user

    response = client.post(
        "/auth/signup",
        json={
            "name": "New User",
            "email": "newuser@example.com",
            "password": "TestPassword123!",
        },
    )

    assert response.status_code == 201
    assert response.json()["email"] == "newuser@example.com"
    assert "password" not in response.json()
    session.commit.assert_awaited_once()


def test_signup_duplicate_email(api_client, test_user):
    client, session = api_client
    session.scalar.return_value = test_user.id

    response = client.post(
        "/auth/signup",
        json={
            "name": "Test User",
            "email": "testuser@example.com",
            "password": "TestPassword123!",
        },
    )

    assert response.status_code == 409
    session.commit.assert_not_awaited()


def test_login_success(api_client, test_user):
    client, session = api_client
    session.scalar.return_value = test_user

    response = client.post(
        "/auth/login",
        json={
            "email": "testuser@example.com",
            "password": "TestPassword123!",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 1800
    assert isinstance(body["access_token"], str)
    assert len(body["access_token"]) > 20


def test_login_wrong_password(api_client, test_user):
    client, session = api_client
    session.scalar.return_value = test_user

    response = client.post(
        "/auth/login",
        json={
            "email": "testuser@example.com",
            "password": "WrongPassword123!",
        },
    )

    assert response.status_code == 401


def test_me_requires_authentication(api_client):
    client, _ = api_client

    response = client.get("/auth/me")

    assert response.status_code == 401


def test_me_returns_authenticated_user(api_client, test_user):
    client, session = api_client
    # First query checks token revocation; second loads the user.
    session.scalar.side_effect = [None, test_user]

    response = client.get(
        "/auth/me",
        headers=auth_headers(test_user.id),
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(test_user.id)
    assert response.json()["email"] == test_user.email


def test_logout_revokes_token(api_client, test_user):
    client, session = api_client
    session.scalar.return_value = None
    session.get.return_value = None

    token = create_access_token(
        test_user.id,
        secret_key=TEST_SECRET,
        expires_minutes=30,
    )
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post("/auth/logout", headers=headers)

    assert response.status_code == 204
    session.add.assert_called_once()
    session.commit.assert_awaited_once()