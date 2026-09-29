
import json

import pytest

from graph.nodes import create_nodes


@pytest.mark.asyncio
async def test_final_response_starts_with_summary():
    nodes = create_nodes()

    state = {
        "question": "Show the top documents by chunk count.",
        "analysis_plan": {
            "objective": (
                "Retrieve the top documents ranked by chunk count."
            ),
        },
        "result_analysis": json.dumps({
            "summary": (
                'The document with the most chunks is '
                '"Business Responsibility Reporting: Disclosures '
                'and Practices (2017-18)", with 1,656 chunks.'
            ),
            "key_findings": [
                "Memorandum & Articles of Association (.pdf): 232 chunks",
                "modification-in-basic-custom-duty.pdf: 226 chunks",
            ],
            "caveats": [],
        }),
        "query_result": {
            "columns": ["title", "chunk_count"],
            "rows": [
                {
                    "title": (
                        "Business Responsibility Reporting: "
                        "Disclosures and Practices (2017-18)"
                    ),
                    "chunk_count": 1656,
                },
            ],
            "row_count": 1,
            "truncated": False,
            "referenced_tables": [
                "public.documents",
                "public.document_chunks",
            ],
        },
        "sql": "SELECT title, COUNT(*) AS chunk_count "
               "FROM public.documents GROUP BY title",
        "critic_review": {
            "passed": True,
            "needs_revision": False,
            "issues": [],
            "feedback": "",
        },
    }

    result = await nodes["final_response"](state)

    assert result["status"] == "completed"
    assert result["final_answer"].startswith(
        "The document with the most chunks is"
    )
    assert not result["final_answer"].startswith("Retrieve the top")
    assert "Key findings:" in result["final_answer"]
    assert "SQL:" in result["final_answer"]
    assert result["messages"][0].content == result["final_answer"]


@pytest.mark.asyncio
async def test_final_response_handles_empty_rows():
    nodes = create_nodes()

    result = await nodes["final_response"]({
        "result_analysis": json.dumps({
            "summary": "No matching records were found.",
            "key_findings": [],
            "caveats": [],
        }),
        "query_result": {
            "columns": ["title"],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "referenced_tables": ["public.documents"],
        },
        "sql": "SELECT title FROM public.documents WHERE id = -1",
        "critic_review": {
            "passed": True,
            "needs_revision": False,
            "issues": [],
            "feedback": "",
        },
    })

    assert result["status"] == "completed"
    assert result["final_answer"].startswith(
        "No matching records were found."
    )
    assert "No rows returned." in result["final_answer"]