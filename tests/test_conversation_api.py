
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from api import conversations as conversations_api
from app.main import app
from auth import dependencies
from database.app_session import get_db_session
from database.models import Conversation, User


TEST_DATABASE_URL = "postgresql://test:test@localhost/testdb"


def make_conversation(user_id):
    now = datetime.now(timezone.utc)

    return Conversation(
        id=uuid4(),
        user_id=user_id,
        title="Test conversation",
        created_at=now,
        updated_at=now,
        messages=[],
    )


def make_analysis_result():
    """Return a predictable result without connecting to a real database."""
    return {
        "status": "completed",
        "question": "Show monthly sales",
        "answer": "The query found 42 sales.",
        "sql": "SELECT COUNT(*) FROM sales",
        "query_result": {
            "columns": ["count"],
            "rows": [{"count": 42}],
            "row_count": 1,
        },
        "result_analysis": "The query returned 42 sales.",
        "database": {
            "schema": "public",
            "table_count": 1,
            "tables": ["sales"],
        },
    }


@pytest.fixture
def api_client(monkeypatch):
    """Provide a test client with mocked authentication, DB, and graph."""
    session = AsyncMock(spec=AsyncSession)

    user = User(
        id=uuid4(),
        name="Test User",
        email="testuser@example.com",
        password_hash="test-hash",
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )

    async def override_db_session():
        yield session

    async def override_current_user():
        return user

    mock_graph_service = SimpleNamespace(
        analyze=AsyncMock(return_value=make_analysis_result())
    )

    monkeypatch.setattr(
        conversations_api,
        "GraphAnalysisService",
        lambda: mock_graph_service,
    )

    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[
        dependencies.get_current_user
    ] = override_current_user

    with TestClient(app) as client:
        yield client, session, user, mock_graph_service

    app.dependency_overrides.clear()


def test_create_conversation(api_client):
    client, session, user, _ = api_client

    async def refresh_conversation(conversation):
        if conversation.id is None:
            conversation.id = uuid4()

        if conversation.created_at is None:
            conversation.created_at = datetime.now(timezone.utc)

        if conversation.updated_at is None:
            conversation.updated_at = datetime.now(timezone.utc)

    session.refresh.side_effect = refresh_conversation

    response = client.post(
        "/conversations",
        json={"title": "Sales analysis"},
    )

    assert response.status_code == 201
    body = response.json()

    assert body["title"] == "Sales analysis"
    assert body["user_id"] == str(user.id)

    session.add.assert_called_once()
    session.commit.assert_awaited_once()


def test_list_conversations(api_client):
    client, session, user, _ = api_client
    conversation = make_conversation(user.id)

    session.scalar.return_value = 1
    session.scalars.return_value = SimpleNamespace(
        all=lambda: [conversation]
    )

    response = client.get("/conversations")

    assert response.status_code == 200

    body = response.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["title"] == "Test conversation"
    assert body["limit"] == 20
    assert body["offset"] == 0


def test_list_conversations_pagination(api_client):
    client, session, _, _ = api_client

    session.scalar.return_value = 0
    session.scalars.return_value = SimpleNamespace(
        all=lambda: []
    )

    response = client.get("/conversations?limit=10&offset=20")

    assert response.status_code == 200
    assert response.json()["limit"] == 10
    assert response.json()["offset"] == 20


def test_list_conversations_rejects_invalid_limit(api_client):
    client, _, _, _ = api_client

    response = client.get("/conversations?limit=101")

    assert response.status_code == 422


def test_get_conversation(api_client):
    client, session, user, _ = api_client
    conversation = make_conversation(user.id)

    session.scalar.return_value = conversation

    response = client.get(f"/conversations/{conversation.id}")

    assert response.status_code == 200
    assert response.json()["id"] == str(conversation.id)
    assert response.json()["messages"] == []


def test_get_missing_or_unowned_conversation_returns_404(api_client):
    client, session, _, _ = api_client
    session.scalar.return_value = None

    response = client.get(f"/conversations/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_rename_conversation(api_client):
    client, session, user, _ = api_client
    conversation = make_conversation(user.id)

    session.scalar.return_value = conversation

    response = client.patch(
        f"/conversations/{conversation.id}",
        json={"title": "Updated title"},
    )

    assert response.status_code == 200
    assert response.json()["title"] == "Updated title"
    session.commit.assert_awaited_once()


def test_rename_unowned_conversation_returns_404(api_client):
    client, session, _, _ = api_client
    session.scalar.return_value = None

    response = client.patch(
        f"/conversations/{uuid4()}",
        json={"title": "Unauthorized rename"},
    )

    assert response.status_code == 404
    session.commit.assert_not_awaited()


def test_delete_conversation(api_client):
    client, session, user, _ = api_client
    conversation = make_conversation(user.id)

    session.scalar.return_value = conversation

    response = client.delete(
        f"/conversations/{conversation.id}"
    )

    assert response.status_code == 204
    session.delete.assert_awaited_once_with(conversation)
    session.commit.assert_awaited_once()


def test_delete_unowned_conversation_returns_404(api_client):
    client, session, _, _ = api_client
    session.scalar.return_value = None

    response = client.delete(f"/conversations/{uuid4()}")

    assert response.status_code == 404
    session.delete.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_create_message(api_client):
    client, session, user, graph_service = api_client
    conversation = make_conversation(user.id)

    # First scalar call: ownership check in create_message().
    # Second scalar call: conversation lookup in the endpoint.
    session.scalar.side_effect = [conversation, conversation]

    async def refresh_message(message):
        if message.id is None:
            message.id = uuid4()

        if message.created_at is None:
            message.created_at = datetime.now(timezone.utc)

    session.refresh.side_effect = refresh_message

    response = client.post(
        f"/conversations/{conversation.id}/messages",
        json={
            "content": "Show monthly sales",
            "database_url": TEST_DATABASE_URL,
        },
    )

    assert response.status_code == 201

    body = response.json()
    assert body["conversation_id"] == str(conversation.id)
    assert body["role"] == "assistant"
    assert body["content"] == "The query found 42 sales."
    assert body["metadata"]["sql"] == "SELECT COUNT(*) FROM sales"
    assert body["metadata"]["query_result"]["rows"] == [{"count": 42}]
    assert "database_url" not in body["metadata"]

    graph_service.analyze.assert_awaited_once_with(
        database_url=TEST_DATABASE_URL,
        question="Show monthly sales",
        schema_name="public",
        max_rows=100,
    )

    # One user message, then one assistant message.
    assert session.add.call_count == 2
    assert session.commit.await_count == 2


def test_create_message_for_unowned_conversation_returns_404(api_client):
    client, session, _, graph_service = api_client

    session.scalar.return_value = None

    response = client.post(
        f"/conversations/{uuid4()}/messages",
        json={
            "content": "Show monthly sales",
            "database_url": TEST_DATABASE_URL,
        },
    )

    assert response.status_code == 404
    session.add.assert_not_called()
    session.commit.assert_not_awaited()
    graph_service.analyze.assert_not_awaited()


def test_analysis_failure_returns_502(api_client, monkeypatch):
    client, session, user, _ = api_client
    conversation = make_conversation(user.id)

    session.scalar.return_value = conversation

    failing_service = SimpleNamespace(
        analyze=AsyncMock(
            side_effect=RuntimeError("Database connection failed")
        )
    )

    monkeypatch.setattr(
        conversations_api,
        "GraphAnalysisService",
        lambda: failing_service,
    )

    response = client.post(
        f"/conversations/{conversation.id}/messages",
        json={
            "content": "Show monthly sales",
            "database_url": TEST_DATABASE_URL,
        },
    )

    assert response.status_code == 502
    assert "Database connection failed" not in response.text
    session.rollback.assert_awaited_once()
    failing_service.analyze.assert_awaited_once()


def test_create_conversation_rejects_blank_title(api_client):
    client, _, _, _ = api_client

    response = client.post(
        "/conversations",
        json={"title": "   "},
    )

    assert response.status_code == 422


def test_create_message_requires_database_url(api_client):
    client, _, _, graph_service = api_client

    response = client.post(
        f"/conversations/{uuid4()}/messages",
        json={"content": "Show monthly sales"},
    )

    assert response.status_code == 422
    graph_service.analyze.assert_not_awaited()