
from dataclasses import dataclass

import sqlglot
from sqlglot import exp 

import re
class SQLValidationError(ValueError):
    """Raised when a SQL query violates the execution policy."""



@dataclass(frozen=True)
class ValidatedQuery:
    sql: str
    referenced_tables: tuple[str, ...]
    max_rows: int


_FORBIDDEN_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
)

_FORBIDDEN_FUNCTIONS = {
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "lo_import",
    "lo_export",
    "dblink",
    "dblink_exec",
    "pg_terminate_backend",
    "pg_cancel_backend",
}


def validate_readonly_sql(
    sql: str,
    *,
    allowed_tables: set[str] | None = None,
    max_rows: int = 500,
) -> ValidatedQuery:
    """Validate one SELECT and impose a predictable row cap."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLValidationError("SQL query cannot be empty.")

    if not 1 <= max_rows <= 10_000:
        raise SQLValidationError(
            "max_rows must be between 1 and 10000."
        )

    try:
        statements = sqlglot.parse(sql, read="postgres")
    except sqlglot.errors.ParseError as exc:
        raise SQLValidationError("SQL syntax is invalid.") from exc

    statements = [
        statement for statement in statements
        if statement is not None
    ]

    if len(statements) != 1:
        raise SQLValidationError(
            "Exactly one SQL statement is allowed."
        )

    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union)):
     raise SQLValidationError(
        "Only SELECT queries and UNION operations are allowed."
    )

    normalized_allowlist = (
        {name.lower() for name in allowed_tables}
        if allowed_tables is not None
        else None
    )

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_EXPRESSIONS):
            raise SQLValidationError(
                "The query contains a forbidden operation."
            )

        if isinstance(node, exp.Into):
            raise SQLValidationError("SELECT INTO is not allowed.")

        if isinstance(node, exp.Lock):
            raise SQLValidationError(
                "Locking clauses are not allowed."
            )

        if isinstance(node, exp.Anonymous):
            if node.name.lower() in _FORBIDDEN_FUNCTIONS:
                raise SQLValidationError(
                    "The query contains a forbidden function."
                )

    referenced_tables: set[str] = set()

    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SQLValidationError("Unsupported table source.")

        table_name = table.name.lower()
        schema_name = table.db.lower() if table.db else "public"

        if table.catalog:
            raise SQLValidationError(
                "Cross-database references are not allowed."
            )

        if schema_name != "public":
            raise SQLValidationError(
                "Only tables in the public schema are allowed."
            )

        qualified_name = f"{schema_name}.{table_name}"

        if normalized_allowlist is not None:
            if (
                table_name not in normalized_allowlist
                and qualified_name not in normalized_allowlist
            ):
                raise SQLValidationError(
                    f"Table '{qualified_name}' is not allowed."
                )

        referenced_tables.add(qualified_name)

    fetch_limit = max_rows + 1
    existing_limit = tree.args.get("limit")

    if existing_limit is None:
        tree.set(
            "limit",
            exp.Limit(
                expression=exp.Literal.number(fetch_limit)
            ),
        )
    else:
        limit_expression = existing_limit.expression

        if not isinstance(limit_expression, exp.Literal):
            raise SQLValidationError(
                "LIMIT must be a literal integer."
            )

        if limit_expression.is_string:
            raise SQLValidationError("LIMIT must be an integer.")

        try:
            requested_limit = int(limit_expression.this)
        except (TypeError, ValueError) as exc:
            raise SQLValidationError(
                "LIMIT must be an integer."
            ) from exc

        if requested_limit < 0:
            raise SQLValidationError("LIMIT cannot be negative.")

        if requested_limit > fetch_limit:
            existing_limit.set(
                "expression",
                exp.Literal.number(fetch_limit),
            )


    validated_sql = tree.sql(
    dialect="postgres",
    comments=True,
    )

    validated_sql = tree.sql(dialect="postgres")
    # Preserve SQLAlchemy-style named bind parameters before SQLGlot parses SQL.
    bind_parameters = re.findall(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)", sql)

    # SQLGlot emits Psycopg placeholders (%(name)s).
    # Convert known placeholders back to SQLAlchemy's :name syntax.
    for name in bind_parameters:
        validated_sql = validated_sql.replace(
            f"%({name})s",
            f":{name}",
        )

    return ValidatedQuery(
        sql=validated_sql,
        referenced_tables=tuple(sorted(referenced_tables)),
        max_rows=max_rows,
    )