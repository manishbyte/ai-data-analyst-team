
import json
from unittest.mock import AsyncMock

import pytest

from agents.result_analyst import ResultAnalyst


@pytest.mark.asyncio
async def test_analyze_results_returns_structured_analysis():
    response = {
        "summary": "The query returned two documents.",
        "key_findings": [
            "Document A has 10 chunks.",
            "Document B has 5 chunks.",
        ],
        "caveats": [],
        "answer_sufficient": True,
    }

    client = AsyncMock()
    client.generate.return_value = json.dumps(response)

    analyst = ResultAnalyst(client=client)

    result = await analyst.analyze_results(
        question="Show the documents with the most chunks.",
        plan={"objective": "Compare document chunk counts"},
        query_result={
            "columns": ["title", "chunk_count"],
            "rows": [
                {"title": "Document A", "chunk_count": 10},
                {"title": "Document B", "chunk_count": 5},
            ],
            "row_count": 2,
            "truncated": False,
            "referenced_tables": [
                "public.documents",
                "public.document_chunks",
            ],
        },
    )

    assert result.summary == response["summary"]
    assert len(result.key_findings) == 2
    assert result.answer_sufficient is True
    client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_results_accepts_markdown_json_fence():
    client = AsyncMock()
    client.generate.return_value = (
        '```json\n'
        '{"summary":"No rows returned.",'
        '"key_findings":[],"caveats":[],'
        '"answer_sufficient":false}\n'
        '```'
    )

    analyst = ResultAnalyst(client=client)

    result = await analyst.analyze_results(
        question="Find matching documents.",
        plan={"objective": "Find documents"},
        query_result={
            "columns": ["title"],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "referenced_tables": ["public.documents"],
        },
    )

    assert result.summary == "No rows returned."
    assert result.answer_sufficient is False


@pytest.mark.asyncio
async def test_analyze_results_rejects_empty_question():
    client = AsyncMock()
    analyst = ResultAnalyst(client=client)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        await analyst.analyze_results(
            question=" ",
            plan={},
            query_result={"columns": [], "rows": []},
        )

    client.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_analyze_results_rejects_invalid_json():
    client = AsyncMock()
    client.generate.return_value = "not valid JSON"

    analyst = ResultAnalyst(client=client)

    with pytest.raises(ValueError, match="invalid JSON"):
        await analyst.analyze_results(
            question="Show document counts.",
            plan={"objective": "Count documents"},
            query_result={
                "columns": ["title", "chunk_count"],
                "rows": [],
            },
        )