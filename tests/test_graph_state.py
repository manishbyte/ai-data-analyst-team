
from langchain_core.messages import AIMessage, HumanMessage

from graph.state import AnalystState


def test_analyst_state_accepts_initial_request() -> None:
    state: AnalystState = {
        "question": "How many documents are stored?",
        "schema_name": "public",
        "allowed_tables": {"documents"},
        "max_rows": 100,
        "retry_count": 0,
        "status": "pending",
    }

    assert state["question"] == "How many documents are stored?"
    assert state["allowed_tables"] == {"documents"}
    assert state["status"] == "pending"


def test_analyst_state_can_hold_execution_results() -> None:
    state: AnalystState = {
        "question": "Count documents",
        "query_result": {
            "columns": ["document_count"],
            "rows": [{"document_count": 99}],
            "row_count": 1,
        },
        "status": "running",
    }

    assert state["query_result"]["row_count"] == 1


def test_analyst_state_supports_chat_history() -> None:
    state: AnalystState = {
        "messages": [
            HumanMessage(content="How many documents are stored?"),
            AIMessage(content="There are 99 documents."),
        ],
        "question": "How many documents are stored?",
        "status": "completed",
    }

    assert len(state["messages"]) == 2
    assert isinstance(state["messages"][0], HumanMessage)
    assert isinstance(state["messages"][1], AIMessage)