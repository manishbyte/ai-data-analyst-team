
from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Configuration
# ============================================================

DEFAULT_API_URL = "http://127.0.0.1:8000"

API_BASE_URL = (
    os.getenv("API_BASE_URL", DEFAULT_API_URL).strip().rstrip("/")
    or DEFAULT_API_URL
)

REQUEST_TIMEOUT = 180.0

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Session state
# ============================================================

DEFAULTS: dict[str, Any] = {
    "access_token": None,
    "current_user": None,
    "conversations": [],
    "conversations_loaded": False,
    "loaded_user_id": None,
    "active_conversation_id": None,
    "current_conversation": None,
    "database_url": "",
    "analysis_result": None,
    "analysis_question": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# API helpers
# ============================================================

class APIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def api_request(
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    authenticated: bool = True,
) -> Any:
    """Make a request to FastAPI without displaying its URL in the UI."""

    headers: dict[str, str] = {}

    token = st.session_state.get("access_token")

    if authenticated and token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = httpx.request(
            method=method,
            url=f"{API_BASE_URL}{path}",
            headers=headers,
            json=json,
            timeout=REQUEST_TIMEOUT,
        )
    except httpx.TimeoutException as exc:
        raise APIError(
            0,
            "The API request timed out. Please try again.",
        ) from exc
    except httpx.RequestError as exc:
        raise APIError(
            0,
            "Could not connect to the backend. Make sure FastAPI is running.",
        ) from exc

    if response.status_code >= 400:
        try:
            body = response.json()
            detail = body.get("detail", body)
            if isinstance(detail, list):
                message = "; ".join(
                    str(item.get("msg", item))
                    if isinstance(item, dict)
                    else str(item)
                    for item in detail
                )
            else:
                message = str(detail)
        except ValueError:
            message = response.text or "The request failed."

        raise APIError(response.status_code, message)

    if response.status_code == 204 or not response.content:
        return None

    try:
        return response.json()
    except ValueError:
        return response.text


def handle_auth_failure(exc: APIError) -> bool:
    """Clear the session when the backend rejects the access token."""

    if exc.status_code == 401:
        clear_authentication()
        st.warning("Your session has expired. Please log in again.")
        st.rerun()
        return True

    return False


# ============================================================
# Authentication
# ============================================================

def clear_workspace() -> None:
    st.session_state["conversations"] = []
    st.session_state["conversations_loaded"] = False
    st.session_state["loaded_user_id"] = None
    st.session_state["active_conversation_id"] = None
    st.session_state["current_conversation"] = None
    st.session_state["analysis_result"] = None
    st.session_state["analysis_question"] = ""


def clear_authentication() -> None:
    st.session_state["access_token"] = None
    st.session_state["current_user"] = None
    clear_workspace()


def login(email: str, password: str) -> None:
    result = api_request(
        "POST",
        "/auth/login",
        json={
            "email": email.strip(),
            "password": password,
        },
        authenticated=False,
    )

    token = result.get("access_token") if isinstance(result, dict) else None

    if not token:
        raise APIError(500, "The login response did not contain an access token.")

    st.session_state["access_token"] = token

    try:
        user = api_request("GET", "/auth/me")
    except Exception:
        clear_authentication()
        raise

    st.session_state["current_user"] = user
    clear_workspace()


def signup(name: str, email: str, password: str) -> None:
    api_request(
        "POST",
        "/auth/signup",
        json={
            "name": name.strip(),
            "email": email.strip(),
            "password": password,
        },
        authenticated=False,
    )


def logout() -> None:
    try:
        if st.session_state.get("access_token"):
            api_request("POST", "/auth/logout")
    except APIError:
        # Clear the local session even if the server is unreachable.
        pass
    finally:
        clear_authentication()

    st.rerun()


# ============================================================
# Conversation API
# ============================================================

def normalize_list_response(data: Any) -> list[dict[str, Any]]:
    """Support either a plain list or a paginated API response."""

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("items", "conversations", "results"):
            if isinstance(data.get(key), list):
                return data[key]

    return []


def load_conversations() -> list[dict[str, Any]]:
    data = api_request("GET", "/conversations")
    return normalize_list_response(data)


def load_conversation(conversation_id: str) -> dict[str, Any]:
    data = api_request(
        "GET",
        f"/conversations/{conversation_id}",
    )

    if not isinstance(data, dict):
        raise APIError(500, "Unexpected conversation response from the API.")

    return data


def create_conversation(title: str) -> dict[str, Any]:
    data = api_request(
        "POST",
        "/conversations",
        json={"title": title},
    )

    if not isinstance(data, dict) or not data.get("id"):
        raise APIError(500, "The API did not return a conversation ID.")

    return data


def rename_conversation(
    conversation_id: str,
    title: str,
) -> dict[str, Any]:
    return api_request(
        "PATCH",
        f"/conversations/{conversation_id}",
        json={"title": title},
    )


def delete_conversation(conversation_id: str) -> None:
    api_request(
        "DELETE",
        f"/conversations/{conversation_id}",
    )


def refresh_conversations() -> None:
    st.session_state["conversations"] = load_conversations()
    st.session_state["conversations_loaded"] = True


def initialize_user_workspace() -> None:
    """Load history after login, but do not open a conversation automatically."""

    user = st.session_state.get("current_user")

    if not user:
        return

    user_id = str(user.get("id", ""))

    if st.session_state.get("loaded_user_id") != user_id:
        clear_workspace()
        st.session_state["loaded_user_id"] = user_id

    if not st.session_state["conversations_loaded"]:
        refresh_conversations()

    # Intentionally do not select the first conversation.
    # The main chat stays blank until the user selects one.


# ============================================================
# Titles and analytics database URL
# ============================================================

def title_from_question(question: str) -> str:
    """Create a readable conversation title from the first question."""

    cleaned = " ".join(question.strip().split())

    if not cleaned:
        return "New chat"

    words = cleaned.split()
    title = " ".join(words[:8])

    if len(words) > 8:
        title += "..."

    if len(title) > 65:
        title = title[:62].rstrip() + "..."

    return title[0].upper() + title[1:] if title else "New chat"


def validate_database_url(database_url: str) -> str | None:
    """Basic validation before sending a connection URL to the backend."""

    value = database_url.strip()

    if not value:
        return "Enter your analytics PostgreSQL connection URL in the sidebar."

    try:
        parsed = urlparse(value)
    except ValueError:
        return "The database URL is invalid."

    if parsed.scheme not in {
        "postgresql",
        "postgresql+psycopg",
        "postgresql+psycopg2",
    }:
        return "Use a PostgreSQL connection URL."

    if not parsed.hostname:
        return "The database URL must include a hostname."

    return None


# ============================================================
# Query result and metadata rendering
# ============================================================

def get_metadata(message: dict[str, Any]) -> dict[str, Any]:
    metadata = message.get("metadata")

    if metadata is None:
        metadata = message.get("metadata_json")

    return metadata if isinstance(metadata, dict) else {}


def result_to_dataframe(result: Any) -> pd.DataFrame | None:
    """Convert common SQL result structures into a DataFrame."""

    if isinstance(result, list):
        if not result:
            return pd.DataFrame()
        if all(isinstance(row, dict) for row in result):
            return pd.DataFrame(result)
        return pd.DataFrame(result)

    if not isinstance(result, dict):
        return None

    # Common result formats from SQL execution services.
    records = result.get("records")
    if isinstance(records, list):
        return pd.DataFrame(records)

    rows = result.get("rows", result.get("data"))

    if not isinstance(rows, list):
        return None

    columns = result.get("columns")

    if columns and all(isinstance(column, dict) for column in columns):
        columns = [
            column.get("name", str(column))
            for column in columns
        ]

    if columns and all(isinstance(column, str) for column in columns):
        return pd.DataFrame(rows, columns=columns)

    if rows and all(isinstance(row, dict) for row in rows):
        return pd.DataFrame(rows)

    return pd.DataFrame(rows)


def render_chart(
    chart_config: Any,
    dataframe: pd.DataFrame | None = None,
) -> None:
    """Render a chart described by saved metadata."""

    if not isinstance(chart_config, dict):
        return

    chart_type = str(
        chart_config.get("type", chart_config.get("chart_type", ""))
    ).lower()

    x_column = chart_config.get("x")
    y_column = chart_config.get("y")
    title = chart_config.get("title", "Analysis chart")

    chart_data = chart_config.get("data")

    if isinstance(chart_data, list):
        chart_df = pd.DataFrame(chart_data)
    else:
        chart_df = dataframe

    if chart_df is None or chart_df.empty:
        return

    if not x_column or x_column not in chart_df.columns:
        return

    if isinstance(y_column, list):
        y_columns = [
            column for column in y_column
            if column in chart_df.columns
        ]
    elif isinstance(y_column, str) and y_column in chart_df.columns:
        y_columns = [y_column]
    else:
        y_columns = []

    if not y_columns:
        return

    try:
        if chart_type in {"bar", "bar_chart"}:
            fig = px.bar(
                chart_df,
                x=x_column,
                y=y_columns,
                title=title,
            )
        elif chart_type in {"line", "line_chart"}:
            fig = px.line(
                chart_df,
                x=x_column,
                y=y_columns,
                title=title,
            )
        elif chart_type in {"scatter", "scatter_plot"}:
            fig = px.scatter(
                chart_df,
                x=x_column,
                y=y_columns[0],
                title=title,
            )
        elif chart_type in {"pie", "pie_chart"}:
            fig = px.pie(
                chart_df,
                names=x_column,
                values=y_columns[0],
                title=title,
            )
        else:
            return

        st.plotly_chart(fig, width="stretch")

    except (ValueError, TypeError, KeyError) as exc:
        st.caption(f"Could not render the saved chart: {exc}")


def render_result_metadata(metadata: dict[str, Any]) -> None:
    """Render saved SQL, result tables, charts, and CSV downloads."""

    sql = metadata.get("sql") or metadata.get("generated_sql")

    if sql:
        with st.expander("Generated SQL", expanded=False):
            st.code(str(sql), language="sql")

    query_result = (
        metadata.get("query_result")
        or metadata.get("result")
        or metadata.get("data")
    )

    dataframe = result_to_dataframe(query_result)

    if dataframe is not None:
        st.markdown("**Query results**")

        if dataframe.empty:
            st.caption("The query returned no rows.")
        else:
            st.dataframe(dataframe, width="stretch", hide_index=True)

            csv_data = dataframe.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="query_results.csv",
                mime="text/csv",
                key=f"csv_{id(metadata)}",
            )

    chart_config = metadata.get("chart") or metadata.get("chart_config")

    if chart_config:
        render_chart(chart_config, dataframe)


def render_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")

    if role == "system":
        return

    display_role = "user" if role == "user" else "assistant"

    with st.chat_message(display_role):
        content = message.get("content", "")

        if content:
            st.markdown(str(content))

        if display_role == "assistant":
            metadata = get_metadata(message)

            if metadata:
                render_result_metadata(metadata)


# ============================================================
# Send message
# ============================================================

def send_message(question: str) -> None:
    question = question.strip()

    if not question:
        return

    database_url = st.session_state.get("database_url", "").strip()
    validation_error = validate_database_url(database_url)

    if validation_error:
        st.warning(validation_error)
        return

    conversation_id = st.session_state.get("active_conversation_id")

    try:
        # Create the conversation only when the user sends their first question.
        # This allows the chat input to work in a new, blank chat.
        if not conversation_id:
            title = title_from_question(question)
            conversation = create_conversation(title)
            conversation_id = str(conversation["id"])

            st.session_state["active_conversation_id"] = conversation_id
            st.session_state["current_conversation"] = conversation

        with st.spinner("Analysing your data..."):
            api_request(
                "POST",
                f"/conversations/{conversation_id}/messages",
                json={
                    "content": question,
                    "database_url": database_url,
                },
            )

            # Reload persisted messages, including the assistant's response.
            conversation = load_conversation(conversation_id)

            st.session_state["current_conversation"] = conversation
            st.session_state["active_conversation_id"] = conversation_id

            refresh_conversations()

    except APIError as exc:
        if handle_auth_failure(exc):
            return

        st.error(exc.message)
        return

    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
        return

    st.rerun()


# ============================================================
# Sidebar: authentication, database settings, history
# ============================================================

with st.sidebar:
    st.title("📊 AI Data Analyst")

    if not st.session_state.get("access_token"):
        login_tab, signup_tab = st.tabs(["Login", "Sign up"])

        with login_tab:
            with st.form("login_form"):
                login_email = st.text_input(
                    "Email",
                    key="login_email",
                )
                login_password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                )

                login_submitted = st.form_submit_button(
                    "Login",
                    width="stretch",
                )

            if login_submitted:
                try:
                    login(login_email, login_password)
                    st.rerun()
                except APIError as exc:
                    st.error(exc.message)

        with signup_tab:
            with st.form("signup_form"):
                signup_name = st.text_input("Name")
                signup_email = st.text_input("Email")
                signup_password = st.text_input(
                    "Password",
                    type="password",
                    help="Use a password that meets your backend's requirements.",
                )

                signup_submitted = st.form_submit_button(
                    "Create account",
                    width="stretch",
                )

            if signup_submitted:
                if not signup_name.strip() or not signup_email.strip():
                    st.warning("Enter your name and email.")
                elif not signup_password:
                    st.warning("Enter a password.")
                else:
                    try:
                        signup(
                            signup_name,
                            signup_email,
                            signup_password,
                        )
                        st.success("Account created. You can now log in.")
                    except APIError as exc:
                        st.error(exc.message)

    else:
        user = st.session_state.get("current_user") or {}
        user_name = user.get("name") or user.get("email") or "User"

        st.caption(f"Signed in as **{user_name}**")

        if st.button(
            "Log out",
            key="logout_button",
            width="stretch",
        ):
            logout()

        st.divider()

        # Database settings stay in the sidebar; the backend URL is not shown.
        with st.expander("Analytics database", expanded=False):
            st.text_input(
                "PostgreSQL connection URL",
                type="password",
                key="database_url",
                placeholder="postgresql+psycopg://user:password@host:5432/db",
                help=(
                    "This is the database you want to analyse, not the "
                    "application database used for login and chat history."
                ),
            )

        st.divider()

        # ChatGPT-style conversation history.
        st.subheader("Chat history")

        if st.button(
            "＋ New chat",
            key="new_chat_button",
            width="stretch",
        ):
            st.session_state["active_conversation_id"] = None
            st.session_state["current_conversation"] = None
            st.session_state["analysis_result"] = None
            st.session_state["analysis_question"] = ""
            st.rerun()

        try:
            initialize_user_workspace()
        except APIError as exc:
            if not handle_auth_failure(exc):
                st.error(exc.message)

        conversations = st.session_state.get("conversations", [])
        active_id = st.session_state.get("active_conversation_id")

        if not conversations:
            st.caption("Your saved conversations will appear here.")
        else:
            for conversation in conversations:
                conversation_id = str(conversation["id"])
                title = conversation.get("title") or "New chat"

                # Clicking a history item opens its messages.
                is_active = str(active_id) == conversation_id

                if st.button(
                    f"{'● ' if is_active else ''}{title}",
                    key=f"history_{conversation_id}",
                    type="primary" if is_active else "secondary",
                    width="stretch",
                    help=title,
                ):
                    try:
                        st.session_state["current_conversation"] = (
                            load_conversation(conversation_id)
                        )
                        st.session_state["active_conversation_id"] = (
                            conversation_id
                        )
                        st.session_state["analysis_result"] = None
                        st.session_state["analysis_question"] = ""
                        st.rerun()
                    except APIError as exc:
                        if not handle_auth_failure(exc):
                            st.error(exc.message)

                # Rename and delete are tucked away to keep the history compact.
                with st.popover(
                    "⋯",
                    use_container_width=True,
                    key=f"actions_{conversation_id}",
                ):
                    with st.form(f"rename_{conversation_id}"):
                        new_title = st.text_input(
                            "Conversation title",
                            value=title,
                            max_chars=200,
                            key=f"title_{conversation_id}",
                        )

                        rename_submitted = st.form_submit_button(
                            "Rename",
                            width="stretch",
                        )

                    if rename_submitted:
                        if not new_title.strip():
                            st.warning("The title cannot be empty.")
                        else:
                            try:
                                rename_conversation(
                                    conversation_id,
                                    new_title.strip(),
                                )
                                refresh_conversations()

                                if str(
                                    st.session_state.get(
                                        "active_conversation_id"
                                    )
                                ) == conversation_id:
                                    st.session_state[
                                        "current_conversation"
                                    ] = load_conversation(conversation_id)

                                st.rerun()
                            except APIError as exc:
                                if not handle_auth_failure(exc):
                                    st.error(exc.message)

                    if st.button(
                        "Delete conversation",
                        key=f"delete_{conversation_id}",
                        width="stretch",
                    ):
                        try:
                            delete_conversation(conversation_id)

                            if str(
                                st.session_state.get(
                                    "active_conversation_id"
                                )
                            ) == conversation_id:
                                st.session_state[
                                    "active_conversation_id"
                                ] = None
                                st.session_state[
                                    "current_conversation"
                                ] = None

                            refresh_conversations()
                            st.rerun()
                        except APIError as exc:
                            if not handle_auth_failure(exc):
                                st.error(exc.message)


# ============================================================
# Main chat area
# ============================================================

if not st.session_state.get("access_token"):
    st.title("AI Data Analyst")
    st.write(
        "Log in from the sidebar to analyse your PostgreSQL data "
        "and access your saved conversations."
    )

else:
    conversation = st.session_state.get("current_conversation")
    active_id = st.session_state.get("active_conversation_id")

    if conversation and active_id:
        st.title(conversation.get("title") or "Conversation")

        messages = conversation.get("messages", [])

        for message in messages:
            render_message(message)

    else:
        st.title("What would you like to analyse?")
        st.caption(
            "Ask a question about your data, or select a saved conversation "
            "from the sidebar."
        )

    # IMPORTANT:
    # This input is NOT inside the active-conversation conditional.
    # It remains available when the user clicks New chat.
    question = st.chat_input(
        "Ask a question about your data...",
        key="chat_question_input",
    )

    if question:
        send_message(question)