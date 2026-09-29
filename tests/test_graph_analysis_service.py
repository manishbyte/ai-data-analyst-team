
import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from services.graph_analysis_service import GraphAnalysisService


@pytest.fixture
def fake_database(monkeypatch):
    engine = create_async_engine(
        "postgresql+psycopg://user:password@localhost:5432/testdb"
    )

    class FakeSession:
        def __init__(self, engine):
            self.engine = engine
            self.closed = False

        async def close(self):
            if not self.closed:
                self.closed = True
                await self.engine.dispose()

    fake_session = FakeSession(engine)

    class FakeDatabaseInfo:
        schema = "public"
        table_count = 2
        tables = ["customers", "orders"]

    captured = {}

    async def fake_connect(database_url, *, schema):
        captured["database_url"] = database_url
        captured["schema"] = schema
        return fake_session, FakeDatabaseInfo()

    monkeypatch.setattr(
        "services.graph_analysis_service.DatabaseSession.connect",
        fake_connect,
    )

    return fake_session, captured


class FakeGraph:
    def __init__(self, captured, result=None):
        self.captured = captured
        self.result = result or {
            "status": "completed",
            "final_answer": "Analysis completed.",
            "sql": "SELECT COUNT(*) FROM customers",
            "query_result": {
                "columns": ["count"],
                "rows": [{"count": 12}],
            },
            "result_analysis": "12 customers.",
        }

    async def ainvoke(self, state, *, context):
        self.captured["state"] = state
        self.captured["context"] = context
        return self.result


@pytest.mark.asyncio
async def test_graph_analysis_discovers_tables_automatically(
    monkeypatch,
    fake_database,
):
    fake_session, captured = fake_database

    monkeypatch.setattr(
        "services.graph_analysis_service.build_analyst_graph",
        lambda: FakeGraph(captured),
    )

    service = GraphAnalysisService()

    result = await service.analyze(
        database_url="postgresql://user:password@localhost:5432/testdb",
        question="Count customers",
    )

    assert result["status"] == "completed"
    assert result["query_result"]["rows"] == [{"count": 12}]
    assert captured["context"].engine is fake_session.engine
    assert captured["state"]["allowed_tables"] == {
        "customers",
        "orders",
    }
    assert result["database"]["table_count"] == 2
    assert fake_session.closed is True


@pytest.mark.asyncio
async def test_graph_analysis_preserves_explicit_allowlist(
    monkeypatch,
    fake_database,
):
    fake_session, captured = fake_database

    monkeypatch.setattr(
        "services.graph_analysis_service.build_analyst_graph",
        lambda: FakeGraph(captured),
    )

    service = GraphAnalysisService()

    await service.analyze(
        database_url="postgresql://user:password@localhost:5432/testdb",
        question="Count customers",
        allowed_tables={"customers"},
    )

    assert captured["state"]["allowed_tables"] == {"customers"}
    assert fake_session.closed is True


@pytest.mark.asyncio
async def test_graph_analysis_rejects_unknown_allowlist_tables(
    monkeypatch,
    fake_database,
):
    fake_session, captured = fake_database

    service = GraphAnalysisService()

    with pytest.raises(
        ValueError,
        match="Unknown or unavailable tables",
    ):
        await service.analyze(
            database_url="postgresql://user:password@localhost:5432/testdb",
            question="Count customers",
            allowed_tables={"customers", "secret_table"},
        )

    assert fake_session.closed is True
    assert "state" not in captured


@pytest.mark.asyncio
async def test_graph_analysis_rejects_empty_explicit_allowlist(
    fake_database,
):
    fake_session, _ = fake_database

    service = GraphAnalysisService()

    with pytest.raises(
        ValueError,
        match="allowlist cannot be empty",
    ):
        await service.analyze(
            database_url="postgresql://user:password@localhost:5432/testdb",
            question="Count customers",
            allowed_tables=set(),
        )

    assert fake_session.closed is True


@pytest.mark.asyncio
async def test_graph_analysis_closes_session_on_graph_failure(
    monkeypatch,
    fake_database,
):
    fake_session, captured = fake_database

    failed_graph = FakeGraph(
        captured,
        result={
            "status": "failed",
            "error": "Analysis failed.",
        },
    )

    monkeypatch.setattr(
        "services.graph_analysis_service.build_analyst_graph",
        lambda: failed_graph,
    )

    service = GraphAnalysisService()

    with pytest.raises(RuntimeError, match="Analysis failed"):
        await service.analyze(
            database_url="postgresql://user:password@localhost:5432/testdb",
            question="Count customers",
        )

    assert fake_session.closed is True


@pytest.mark.asyncio
async def test_graph_analysis_rejects_empty_question(fake_database):
    fake_session, captured = fake_database

    service = GraphAnalysisService()

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        await service.analyze(
            database_url="postgresql://user:password@localhost:5432/testdb",
            question="   ",
        )

    assert fake_session.closed is False
    assert "state" not in captured

    await fake_session.close()