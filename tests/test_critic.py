
import json

import pytest

from agents.critic import CriticAgent, CriticReview


class FakeGroqClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_request = None

    async def generate(self, **kwargs) -> str:
        self.last_request = kwargs
        return self.response


@pytest.fixture
def review_context():
    return {
        "question": "Show the top two documents by chunk count.",
        "plan": {
            "objective": "Rank documents by chunk count.",
            "required_tables": ["documents", "document_chunks"],
        },
        "query_result": {
            "columns": ["title", "chunk_count"],
            "rows": [
                {"title": "Document A", "chunk_count": 100},
                {"title": "Document B", "chunk_count": 80},
            ],
            "row_count": 2,
            "truncated": False,
        },
        "result_analysis": {
            "summary": "Document A has more chunks than Document B.",
            "key_findings": [
                "Document A has 100 chunks.",
                "Document B has 80 chunks.",
            ],
            "caveats": [],
            "answer_sufficient": True,
        },
    }


@pytest.mark.asyncio
async def test_critic_accepts_sufficient_analysis(review_context):
    response = {
        "passed": True,
        "issues": [],
        "feedback": "The findings match the supplied query results.",
        "needs_revision": False,
    }

    client = FakeGroqClient(json.dumps(response))
    agent = CriticAgent(client=client)

    review = await agent.review(**review_context)

    assert isinstance(review, CriticReview)
    assert review.passed is True
    assert review.needs_revision is False
    assert review.issues == []


@pytest.mark.asyncio
async def test_critic_identifies_unsupported_finding(review_context):
    response = {
        "passed": False,
        "issues": ["Document B was reported as having 90 chunks, not 80."],
        "feedback": "Correct the finding to match the actual query result.",
        "needs_revision": True,
    }

    client = FakeGroqClient(json.dumps(response))
    agent = CriticAgent(client=client)

    review = await agent.review(**review_context)

    assert review.passed is False
    assert review.needs_revision is True
    assert len(review.issues) == 1


@pytest.mark.asyncio
async def test_critic_accepts_markdown_fenced_json(review_context):
    response = """```json
    {
        "passed": true,
        "issues": [],
        "feedback": "The answer is supported by the results.",
        "needs_revision": false
    }
    ```"""

    agent = CriticAgent(client=FakeGroqClient(response))

    review = await agent.review(**review_context)

    assert review.passed is True


@pytest.mark.asyncio
async def test_critic_rejects_invalid_json(review_context):
    agent = CriticAgent(client=FakeGroqClient("not valid JSON"))

    with pytest.raises(ValueError, match="invalid review"):
        await agent.review(**review_context)


@pytest.mark.asyncio
async def test_critic_rejects_empty_question(review_context):
    agent = CriticAgent(
        client=FakeGroqClient("{}")
    )
    review_context["question"] = "   "

    with pytest.raises(ValueError, match="Question cannot be empty"):
        await agent.review(**review_context)


@pytest.mark.asyncio
async def test_critic_rejects_contradictory_review(review_context):
    response = {
        "passed": True,
        "issues": [],
        "feedback": "Looks good.",
        "needs_revision": True,
    }

    agent = CriticAgent(client=FakeGroqClient(json.dumps(response)))

    with pytest.raises(ValueError, match="contradictory"):
        await agent.review(**review_context)