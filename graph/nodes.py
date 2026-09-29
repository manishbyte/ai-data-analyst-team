
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agents.planner import AnalysisPlanner
from agents.result_analyst import ResultAnalyst
from agents.sql_generator import SQLGenerator
from database.schema_inspector import inspect_schema
from graph.state import AnalystContext, AnalystState
from services.analyst_service import AnalystService
from sql.executor import execute_readonly_query
from sql.validator import validate_readonly_sql
from agents.critic import CriticAgent


def create_nodes(
    service: AnalystService | None = None,
) -> dict[str, Any]:
    """Create workflow nodes sharing agent instances."""

    planner = AnalysisPlanner()
    sql_generator = SQLGenerator()
    result_analyst = ResultAnalyst()
    critic = CriticAgent()

    async def supervisor_node(
        state: AnalystState,
    ) -> dict[str, Any]:
        """Validate the request and select the next node."""

        messages = state.get("messages", [])

        latest_question = next(
            (
                message.content
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
                and isinstance(message.content, str)
                and message.content.strip()
            ),
            None,
        )

        question = latest_question or state.get("question", "")

        if not question or not question.strip():
            return {
                "status": "failed",
                "error": "No user question was provided.",
                "next_step": "end",
            }

        if not state.get("allowed_tables"):
            return {
                "status": "failed",
                "error": "An explicit table allowlist is required.",
                "next_step": "end",
            }

        return {
             "question": question.strip(),
            "status": "running",
            "error": None,
            "next_step": "schema_analyst",
            "revision_count": 0,
            "max_revisions": 2,
            "revision_reason": "",
            "critic_review": {},
            "critic_feedback": "",
        }

    async def schema_analyst_node(
    state: AnalystState,
    runtime=None,
    ) -> dict[str, Any]:
        """Inspect and filter schema metadata."""

        try:
            engine = (
                runtime.context.engine
                if (
                    runtime is not None
                    and runtime.context is not None
                )
                else service.engine if service is not None else None
            )

            if engine is None:
                raise ValueError("No database engine was provided.")

            schema_name = state.get("schema_name", "public")
            allowed_tables = state.get("allowed_tables")

            if schema_name != "public":
                raise ValueError("Only the public schema is supported.")

            if not allowed_tables:
                raise ValueError("An explicit table allowlist is required.")

            schema = await inspect_schema(engine, schema=schema_name)

            available_tables = {
                table["table_name"] for table in schema["tables"]
            }
            unknown_tables = allowed_tables - available_tables

            if unknown_tables:
                raise ValueError(
                    "Unknown or unavailable tables: "
                    + ", ".join(sorted(unknown_tables))
                )

            visible_tables = [
                table
                for table in schema["tables"]
                if table["table_name"] in allowed_tables
            ]

            if not visible_tables:
                raise ValueError("No permitted tables are available.")

            filtered_schema = {
                "schema": schema["schema"],
                "table_count": len(visible_tables),
                "tables": visible_tables,
            }

            import json

            return {
                "schema_context": json.dumps(filtered_schema, default=str),
                "next_step": "analysis_planner",
                "error": None,
            }

        except Exception:
            return {
                "status": "failed",
                "next_step": "end",
                "error": "Schema inspection failed.",
            }
    async def analysis_planner_node(
        state: AnalystState,
    ) -> dict[str, Any]:
        """Create a structured analytical plan."""

        try:
            plan = await planner.create_plan(
                question=state["question"],
                schema_context=state["schema_context"],
            )

            allowed_tables = state["allowed_tables"]
            unexpected_tables = (
                set(plan.required_tables) - allowed_tables
            )

            if unexpected_tables:
                raise ValueError(
                    "Planner selected tables outside the allowlist."
                )

            return {
                "analysis_plan": plan.model_dump(),
                "next_step": "sql_generator",
                "error": None,
            }

        except Exception:
            return {
                "status": "failed",
                "error": "Analysis planning failed.",
                "next_step": "end",
            }

    async def sql_generator_node(
        state: AnalystState,
    ) -> dict[str, Any]:
        """Generate SQL, incorporating Critic feedback when revising."""

        try:
            revision_count = state.get("revision_count", 0)

            # Preserve the original call for the first SQL draft.
            if revision_count == 0:
                draft = await sql_generator.generate_sql(
                    question=state["question"],
                    schema_context=state["schema_context"],
                    plan=state["analysis_plan"],
                )
            else:
                draft = await sql_generator.generate_sql(
                    question=state["question"],
                    schema_context=state["schema_context"],
                    plan=state["analysis_plan"],
                    revision_feedback=state.get("critic_feedback"),
                    previous_sql=state.get("sql"),
                )

            return {
                "sql_draft": draft.model_dump(),
                "sql": draft.sql,
                "parameters": draft.parameters,
                "next_step": "analysis",
                "error": None,
            }

        except Exception as exc:
            print(
                f"[SQL GENERATOR ERROR] "
                f"{type(exc).__name__}: {exc}"
            )

            return {
                "status": "failed",
                "error": "SQL generation failed.",
                "next_step": "end",
            }
    async def analysis_node(
    state: AnalystState,
    runtime=None,
    ) -> dict[str, Any]:
        """Validate and execute the generated SQL draft."""

        try:
            engine = (
                runtime.context.engine
                if (
                    runtime is not None
                    and runtime.context is not None
                )
                else service.engine if service is not None else None
            )

            if engine is None:
                raise ValueError("No database engine was provided.")

            max_rows = state.get("max_rows", 100)
            allowed_tables = state["allowed_tables"]
            draft = state["sql_draft"]

            validated = validate_readonly_sql(
                draft["sql"],
                allowed_tables=allowed_tables,
                max_rows=max_rows,
            )

            result = await execute_readonly_query(
                engine,
                validated.sql,
                params=draft.get("parameters", {}),
                allowed_tables=allowed_tables,
                max_rows=max_rows,
            )

            query_result = {
                "columns": result.columns,
                "rows": result.rows,
                "row_count": result.row_count,
                "truncated": result.truncated,
                "referenced_tables": list(result.referenced_tables),
            }

            return {
                "sql": validated.sql,
                "parameters": draft.get("parameters", {}),
                "query_result": query_result,
                "status": "running",
                "next_step": "result_analyst",
                "error": None,
            }

        except Exception as exc:
            print(
                f"[ANALYSIS NODE ERROR] "
                f"{type(exc).__name__}: {exc}"
            )
            return {
                "status": "failed",
                "error": "SQL validation or execution failed.",
                "next_step": "end",
            }

    async def result_analyst_node(
        state: AnalystState,
    ) -> dict[str, Any]:
        """Interpret the actual SQL query results."""

        try:
            query_result = state.get("query_result")

            if query_result is None:
                raise ValueError("Query results are missing.")

            analysis = await result_analyst.analyze_results(
                question=state["question"],
                plan=state["analysis_plan"],
                query_result=query_result,
            )

            analysis_data = analysis.model_dump()

            return {
                "result_analysis": json.dumps(
                    analysis_data,
                    default=str,
                ),
                "next_step": "critic",
                "error": None,
            }

        except Exception as exc:
            print(
                f"[RESULT ANALYST ERROR] "
                f"{type(exc).__name__}: {exc}"
            )

            return {
                "status": "failed",
                "error": "Result analysis failed.",
                "next_step": "end",
            }

    async def critic_node(state: AnalystState) -> dict[str, Any]:
        """Review results and decide whether to revise or finish."""

        try:
            result_analysis = json.loads(state["result_analysis"])

            review = await critic.review(
                question=state["question"],
                plan=state["analysis_plan"],
                query_result=state["query_result"],
                result_analysis=result_analysis,
            )

            review_data = review.model_dump()
            revision_count = state.get("revision_count", 0)
            max_revisions = state.get("max_revisions", 2)

            # A passing review can proceed directly to the final response.
            if review_data["passed"] and not review_data["needs_revision"]:
                return {
                    "critic_review": review_data,
                    "critic_feedback": review_data["feedback"],
                    "revision_reason": "",
                    "next_step": "final_response",
                    "status": "running",
                    "error": None,
                }

            # Stop retrying when the configured limit is reached.
            if revision_count >= max_revisions:
                return {
                    "critic_review": review_data,
                    "critic_feedback": review_data["feedback"],
                    "revision_reason": review_data["feedback"],
                    "next_step": "final_response",
                    "status": "running",
                    "error": None,
                }

            # Otherwise, revise the SQL and rerun the analysis.
            return {
                "critic_review": review_data,
                "critic_feedback": review_data["feedback"],
                "revision_reason": review_data["feedback"],
                "revision_count": revision_count + 1,
                "next_step": "sql_generator",
                "status": "running",
                "error": None,
            }

        except Exception as exc:
            print(
                f"[CRITIC ERROR] {type(exc).__name__}: {exc}"
            )

            return {
                "status": "failed",
                "error": "Quality review failed.",
                "next_step": "end",
            }

    async def final_response_node(
        state: AnalystState,
    ) -> dict[str, Any]:
        """Format a direct answer from actual query results."""

        result = state.get("query_result")

        if result is None:
            answer = "I couldn't produce a query result."

            return {
                "final_answer": answer,
                "messages": [AIMessage(content=answer)],
                "status": "failed",
                "error": state.get("error") or answer,
                "next_step": "end",
            }

        try:
            result_analysis = json.loads(
                state.get("result_analysis", "{}")
            )
        except (json.JSONDecodeError, TypeError):
            result_analysis = {}

        summary = result_analysis.get("summary", "")
        key_findings = result_analysis.get("key_findings", [])
        caveats = result_analysis.get("caveats", [])

        critic_review = state.get("critic_review", {})
        quality_warning = (
            not critic_review.get("passed", False)
            or critic_review.get("needs_revision", False)
        )

        # Lead with the actual result, not the plan's objective.
        if summary and isinstance(summary, str):
            answer_parts = [summary.strip(), ""]
        elif result.get("rows"):
            answer_parts = [
                f"The query returned {result['row_count']} "
                "row(s). The results are shown below.",
                "",
            ]
        else:
            answer_parts = [
                "The query completed successfully and returned "
                f"{result['row_count']} row(s).",
                "",
            ]

        if key_findings:
            answer_parts.append("Key findings:")
            answer_parts.extend(
                f"- {finding}" for finding in key_findings
            )
            answer_parts.append("")

        answer_parts.extend([
            "SQL:",
            state.get("sql", ""),
            "",
            f"Result: {result['row_count']} row(s) returned.",
        ])

        if result.get("rows"):
            answer_parts.append(
                json.dumps(
                    result["rows"],
                    indent=2,
                    default=str,
                )
            )
        else:
            answer_parts.append("No rows returned.")

        if caveats:
            answer_parts.extend(["", "Caveats:"])
            answer_parts.extend(
                f"- {caveat}" for caveat in caveats
            )

        if result.get("truncated"):
            answer_parts.extend([
                "",
                "Note: Results were truncated to respect "
                "the configured row limit.",
            ])

        if quality_warning:
            answer_parts.extend([
                "",
                "Quality warning: The Critic could not approve "
                "the analysis within the configured revision limit.",
            ])

            issues = critic_review.get("issues", [])
            if issues:
                answer_parts.append("Unresolved quality issues:")
                answer_parts.extend(
                    f"- {issue}" for issue in issues
                )

            feedback = critic_review.get("feedback")
            if feedback:
                answer_parts.extend([
                    "",
                    f"Critic feedback: {feedback}",
                ])

        answer = "\n".join(answer_parts)

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "status": "completed",
            "next_step": "end",
            "error": None,
        }

    return {
        "supervisor": supervisor_node,
        "schema_analyst": schema_analyst_node,
        "analysis_planner": analysis_planner_node,
        "sql_generator": sql_generator_node,
        "analysis": analysis_node,
        "result_analyst": result_analyst_node,
        "critic": critic_node,
        "final_response": final_response_node,
    }