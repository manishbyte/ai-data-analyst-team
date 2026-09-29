
from sqlalchemy.dialects.postgresql import JSONB

from database.base import Base
from database.models import Conversation, Message, User


def test_all_application_tables_are_registered():
    assert {"users", "conversations", "messages"} <= set(
        Base.metadata.tables
    )


def test_user_model_has_required_columns():
    columns = User.__table__.columns

    assert "id" in columns
    assert "name" in columns
    assert "email" in columns
    assert "password_hash" in columns
    assert "is_active" in columns
    assert "created_at" in columns

    assert columns["email"].unique is True
    assert columns["password_hash"].nullable is False


def test_conversation_belongs_to_user():
    columns = Conversation.__table__.columns

    assert "user_id" in columns
    assert "title" in columns

    foreign_keys = list(columns["user_id"].foreign_keys)
    assert len(foreign_keys) == 1
    assert foreign_keys[0].target_fullname == "users.id"
    assert foreign_keys[0].ondelete == "CASCADE"


def test_message_has_jsonb_metadata():
    columns = Message.__table__.columns

    assert "conversation_id" in columns
    assert "role" in columns
    assert "content" in columns
    assert "metadata" in columns

    assert isinstance(columns["metadata"].type, JSONB)


def test_message_role_constraint_exists():
    constraints = Message.__table__.constraints

    assert any(
        getattr(constraint, "name", None)
        in {"valid_role", "ck_messages_valid_role"}
        for constraint in constraints
    )


def test_relationships_are_configured():
    assert User.conversations.property.back_populates == "user"
    assert Conversation.user.property.back_populates == "conversations"
    assert Conversation.messages.property.back_populates == "conversation"
    assert Message.conversation.property.back_populates == "messages"