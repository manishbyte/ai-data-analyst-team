
from typing import Any, Literal
from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncEngine


class AnalystContext(BaseModel):
    """Runtime dependencies for one analyst request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: AsyncEngine


class AnalystState(MessagesState, total=False):
    question: str
    schema_name: str
    allowed_tables: set[str]
    max_rows: int
    next_step: str
    retry_count: int
    schema_context: str
    analysis_plan: dict[str, Any]
    sql: str
    parameters: dict[str, Any]
    query_result: dict[str, Any]
    result_analysis: str
    quality_passed: bool
    quality_feedback: str
    final_answer: str
    error: str | None
    status: Literal["pending", "running", "completed", "failed"]
    sql_draft: dict[str, Any]
    planner_feedback: str
    critic_review: dict[str, Any]
    critic_feedback: str
    revision_count: int
    max_revisions: int
    revision_reason: str