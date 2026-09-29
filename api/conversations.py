
from typing import Annotated
from uuid import UUID
import logging

from sqlalchemy import func

from database.models import Message
from services.graph_analysis_service import GraphAnalysisService
from fastapi import APIRouter, HTTPException, Query, Response, status

from auth.dependencies import CurrentUserDep, SessionDep
from database.models import User
from schemas.conversation import (
    ConversationCreate,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageCreate,
    MessageResponse,
)
from services.conversation_service import (
    create_conversation,
    create_message,
    delete_conversation,
    get_conversation,
    list_conversations,
    rename_conversation,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])

Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation_endpoint(
    request: ConversationCreate,
    user: CurrentUserDep,
    session: SessionDep,
) -> ConversationResponse:
    conversation = await create_conversation(
        session,
        user_id=user.id,
        title=request.title,
    )
    return ConversationResponse.model_validate(conversation)


@router.get("", response_model=ConversationListResponse)
async def list_conversations_endpoint(
    user: CurrentUserDep,
    session: SessionDep,
    limit: Limit = 20,
    offset: Offset = 0,
) -> ConversationListResponse:
    items, total = await list_conversations(
        session,
        user_id=user.id,
        limit=limit,
        offset=offset,
    )
    return ConversationListResponse(
        items=[
            ConversationResponse.model_validate(item)
            for item in items
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
)
async def get_conversation_endpoint(
    conversation_id: UUID,
    user: CurrentUserDep,
    session: SessionDep,
) -> ConversationDetailResponse:
    conversation = await get_conversation(
        session,
        user_id=user.id,
        conversation_id=conversation_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return ConversationDetailResponse.model_validate(conversation)


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
async def rename_conversation_endpoint(
    conversation_id: UUID,
    request: ConversationUpdate,
    user: CurrentUserDep,
    session: SessionDep,
) -> ConversationResponse:
    conversation = await rename_conversation(
        session,
        user_id=user.id,
        conversation_id=conversation_id,
        title=request.title,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return ConversationResponse.model_validate(conversation)


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation_endpoint(
    conversation_id: UUID,
    user: CurrentUserDep,
    session: SessionDep,
) -> Response:
    deleted = await delete_conversation(
        session,
        user_id=user.id,
        conversation_id=conversation_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)



@router.post(
    "/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_message_endpoint(
    conversation_id: UUID,
    request: MessageCreate,
    user: CurrentUserDep,
    session: SessionDep,
) -> MessageResponse:
    # 1. Verify ownership and persist the user's message.
    user_message = await create_message(
        session,
        user_id=user.id,
        conversation_id=conversation_id,
        content=request.content,
    )

    if user_message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    # 2. Run the existing analysis workflow.
    # Never store the analytics database URL in message metadata.
    try:
        result = await GraphAnalysisService().analyze(
            database_url=request.database_url.get_secret_value(),
            question=request.content.strip(),
            schema_name="public",
            max_rows=100,
        )

        answer = result.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Analysis returned no final answer.")

        # 3. Store the answer and its supporting analysis results.
        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer.strip(),
            metadata_json={
                "sql": result.get("sql"),
                "query_result": result.get("query_result"),
                "result_analysis": result.get("result_analysis"),
                "database": result.get("database"),
            },
        )

        session.add(assistant_message)

        conversation = await get_conversation(
            session,
            user_id=user.id,
            conversation_id=conversation_id,
        )

        # Ownership was checked when the user message was created.
        if conversation is None:
            raise RuntimeError("Conversation is no longer available.")

        conversation.updated_at = func.now()

        await session.commit()
        await session.refresh(assistant_message)

        return MessageResponse.model_validate(assistant_message)

    except Exception:
        # Do not expose database URLs, SQL driver details, or raw
        # exception messages to the API caller.
        await session.rollback()

        logger.error(
            "Analysis failed for conversation %s.",
            conversation_id,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Analysis failed. Check the analytics database "
                "connection and try again."
            ),
        ) from None