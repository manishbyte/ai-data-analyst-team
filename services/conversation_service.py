
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import Conversation, Message


async def create_conversation(
    session: AsyncSession,
    *,
    user_id: UUID,
    title: str,
) -> Conversation:
    conversation = Conversation(
        user_id=user_id,
        title=title.strip(),
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def list_conversations(
    session: AsyncSession,
    *,
    user_id: UUID,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[Conversation], int]:
    total = await session.scalar(
        select(func.count())
        .select_from(Conversation)
        .where(Conversation.user_id == user_id)
    )

    result = await session.scalars(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )

    return list(result.all()), int(total or 0)


async def get_conversation(
    session: AsyncSession,
    *,
    user_id: UUID,
    conversation_id: UUID,
) -> Conversation | None:
    result = await session.scalar(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    return result


async def rename_conversation(
    session: AsyncSession,
    *,
    user_id: UUID,
    conversation_id: UUID,
    title: str,
) -> Conversation | None:
    conversation = await session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )

    if conversation is None:
        return None

    conversation.title = title.strip()
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def delete_conversation(
    session: AsyncSession,
    *,
    user_id: UUID,
    conversation_id: UUID,
) -> bool:
    conversation = await session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )

    if conversation is None:
        return False

    await session.delete(conversation)
    await session.commit()
    return True


async def create_message(
    session: AsyncSession,
    *,
    user_id: UUID,
    conversation_id: UUID,
    content: str,
) -> Message | None:
    # Verify ownership before saving a message.
    conversation = await session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )

    if conversation is None:
        return None

    message = Message(
        conversation_id=conversation.id,
        role="user",
        content=content.strip(),
    )
    session.add(message)

    # Keep conversation ordering useful for history.
    conversation.updated_at = func.now()

    await session.commit()
    await session.refresh(message)
    return message