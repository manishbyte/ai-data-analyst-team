
from typing import Any

from sqlalchemy import Engine, inspect
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine


def _inspect_schema_sync(
    sync_engine_or_connection: Engine,
    schema: str,
) -> dict[str, Any]:
    """Internal synchronous inspection function for run_sync()."""
    if not schema or not schema.strip():
        raise ValueError("Schema name cannot be empty.")

    inspector = inspect(sync_engine_or_connection)

    if schema not in inspector.get_schema_names():
        raise ValueError("The requested schema is not accessible.")

    tables = []

    for table_name in sorted(
        inspector.get_table_names(schema=schema)
    ):
        columns = inspector.get_columns(
            table_name,
            schema=schema,
        )
        primary_key = inspector.get_pk_constraint(
            table_name,
            schema=schema,
        )
        foreign_keys = inspector.get_foreign_keys(
            table_name,
            schema=schema,
        )

        tables.append(
            {
                "table_name": table_name,
                "columns": [
                    {
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column.get("nullable", True),
                        "default": (
                            str(column["default"])
                            if column.get("default") is not None
                            else None
                        ),
                    }
                    for column in columns
                ],
                "primary_key": (
                    primary_key.get("constrained_columns") or []
                ),
                "foreign_keys": [
                    {
                        "columns": fk.get("constrained_columns") or [],
                        "referred_schema": fk.get("referred_schema"),
                        "referred_table": fk.get("referred_table"),
                        "referred_columns": (
                            fk.get("referred_columns") or []
                        ),
                    }
                    for fk in foreign_keys
                ],
            }
        )

    return {
        "schema": schema,
        "table_count": len(tables),
        "tables": tables,
    }


async def inspect_schema(
    engine: AsyncEngine,
    schema: str = "public",
) -> dict[str, Any]:
    """Asynchronously inspect tables and columns visible to this user."""
    if not schema or not schema.strip():
        raise ValueError("Schema name cannot be empty.")

    async with engine.connect() as connection:
        return await connection.run_sync(
            lambda sync_connection: _inspect_schema_sync(
                sync_connection,
                schema,
            )
        )


async def inspect_schema_with_connection(
    connection: AsyncConnection,
    schema: str = "public",
) -> dict[str, Any]:
    """Inspect schema using an existing async connection."""
    if not schema or not schema.strip():
        raise ValueError("Schema name cannot be empty.")

    return await connection.run_sync(
        lambda sync_connection: _inspect_schema_sync(
            sync_connection,
            schema,
        )
    )