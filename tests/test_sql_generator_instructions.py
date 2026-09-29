import json

import pytest

from agents.sql_generator import SQLGenerator


@pytest.mark.asyncio
async def test_sql_generator_returns_document_id_and_groups_per_document():
    """Generated SQL should preserve document identity."""

    expected_sql = """
        SELECT
            d.id AS document_id,
            d.title AS title,
            COUNT(dc.id) AS chunk_count
        FROM public.documents AS d
        JOIN public.document_chunks AS dc
            ON d.id = dc.document_id
        GROUP BY d.id, d.title
        ORDER BY chunk_count DESC
        LIMIT 10
    """

    class FakeGroqClient:
        async def generate(self, **kwargs):
            return json.dumps({
                "explanation": (
                    "Count chunks for each individual document, "
                    "preserving document identity."
                ),
                "sql": expected_sql,
                "parameters": {},
            })

    generator = SQLGenerator(client=FakeGroqClient())

    draft = await generator.generate_sql(
        question=(
            "Show the top 10 documents by chunk count, "
            "including document IDs and titles."
        ),
        schema_context=(
            "public.documents: id, title\n"
            "public.document_chunks: id, document_id"
        ),
        plan={
            "objective": "Count chunks per document",
            "metrics": ["chunk_count"],
            "dimensions": ["document_id", "title"],
            "required_tables": ["documents", "document_chunks"],
            "steps": ["Join documents to chunks", "Count per document"],
            "expected_result": "Document ID, title, chunk count",
        },
    )

    normalized_sql = " ".join(draft.sql.lower().split())

    assert "d.id as document_id" in normalized_sql
    assert "group by d.id, d.title" in normalized_sql
    assert "count(dc.id)" in normalized_sql