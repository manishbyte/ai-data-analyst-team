
from langgraph.graph import END, START, StateGraph

from graph.nodes import create_nodes
from graph.state import AnalystContext, AnalystState
from services.analyst_service import AnalystService


def route_next(state: AnalystState) -> str:
    """Route to the next node selected by the current node."""

    if state.get("status") == "failed":
        return "end"

    next_step = state.get("next_step", "end")

    allowed_steps = {
        "schema_analyst",
        "analysis_planner",
        "sql_generator",
        "analysis",
        "result_analyst",
        "critic",
        "final_response",
        "end",
    }

    if next_step not in allowed_steps:
        return "end"

    return next_step


def build_analyst_graph(service: AnalystService | None = None):
    """Build the graph with request-scoped runtime context."""

    nodes = create_nodes(service)
    workflow = StateGraph(
        AnalystState,
        context_schema=AnalystContext,
    )

    for name, node in nodes.items():
        workflow.add_node(name, node)

    workflow.add_edge(START, "supervisor")

    conditional_routes = {
        "schema_analyst": "schema_analyst",
        "analysis_planner": "analysis_planner",
        "sql_generator": "sql_generator",
        "analysis": "analysis",
        "result_analyst": "result_analyst",
        "critic": "critic",
        "final_response": "final_response",
        "end": END,
    }

    for node_name in (
        "supervisor",
        "schema_analyst",
        "analysis_planner",
        "sql_generator",
        "analysis",
        "result_analyst",
        "critic",
    ):
        workflow.add_conditional_edges(
            node_name,
            route_next,
            conditional_routes,
        )

    workflow.add_edge("final_response", END)

    return workflow.compile()