
import pytest

from services.database_session import DatabaseSession


@pytest.mark.asyncio
async def test_connect_returns_session_and_schema_info(monkeypatch):
    class FakeSettings:
        db_connect_timeout = 5

    class FakeConnectionResult:
        success = True
        server_version = "PostgreSQL test server"

    class FakeEngine:
        def __init__(self):
            self.disposed = False

        async def dispose(self):
            self.disposed = True

    async def fake_check_connection(*args, **kwargs):
        return FakeConnectionResult()

    async def fake_inspect_schema(engine, schema):
        assert schema == "public"
        return {
            "schema": schema,
            "table_count": 2,
            "tables": [
                {"table_name": "customers"},
                {"table_name": "orders"},
            ],
        }

    engine = FakeEngine()

    monkeypatch.setattr(
        "services.database_session.get_settings",
        lambda: FakeSettings(),
    )
    monkeypatch.setattr(
        "services.database_session.check_database_connection",
        fake_check_connection,
    )
    monkeypatch.setattr(
        "services.database_session.create_database_engine",
        lambda *args, **kwargs: engine,
    )
    monkeypatch.setattr(
        "services.database_session.inspect_schema",
        fake_inspect_schema,
    )

    session, info = await DatabaseSession.connect(
        "postgresql://user:password@localhost:5432/testdb"
    )

    assert session.engine is engine
    assert info.server_version == "PostgreSQL test server"
    assert info.schema == "public"
    assert info.table_count == 2
    assert info.tables == ["customers", "orders"]

    await session.close()
    assert engine.disposed is True


@pytest.mark.asyncio
async def test_inspection_failure_disposes_engine(monkeypatch):
    class FakeSettings:
        db_connect_timeout = 5

    class FakeConnectionResult:
        success = True
        server_version = "PostgreSQL test server"

    class FakeEngine:
        def __init__(self):
            self.disposed = False

        async def dispose(self):
            self.disposed = True

    async def fake_check_connection(*args, **kwargs):
        return FakeConnectionResult()

    async def fake_inspect_schema(*args, **kwargs):
        raise ValueError("Schema unavailable")

    engine = FakeEngine()

    monkeypatch.setattr(
        "services.database_session.get_settings",
        lambda: FakeSettings(),
    )
    monkeypatch.setattr(
        "services.database_session.check_database_connection",
        fake_check_connection,
    )
    monkeypatch.setattr(
        "services.database_session.create_database_engine",
        lambda *args, **kwargs: engine,
    )
    monkeypatch.setattr(
        "services.database_session.inspect_schema",
        fake_inspect_schema,
    )

    with pytest.raises(ValueError, match="Schema unavailable"):
        await DatabaseSession.connect(
            "postgresql://user:password@localhost:5432/testdb"
        )

    assert engine.disposed is True