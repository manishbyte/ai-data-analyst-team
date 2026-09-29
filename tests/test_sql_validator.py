
import pytest

from sql.validator import (
    SQLValidationError,
    validate_readonly_sql,
)


def test_accepts_select():
    result = validate_readonly_sql(
        "SELECT id, name FROM public.customers",
        allowed_tables={"customers"},
    )
    assert result.referenced_tables == ("public.customers",)


def test_adds_limit_when_missing():
    result = validate_readonly_sql(
        "SELECT * FROM customers",
        max_rows=100,
    )
    assert "LIMIT 101" in result.sql.upper()


def test_caps_existing_limit():
    result = validate_readonly_sql(
        "SELECT * FROM customers LIMIT 10000",
        max_rows=100,
    )
    assert "LIMIT 101" in result.sql.upper()


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM customers",
        "UPDATE customers SET name = 'x'",
        "INSERT INTO customers (name) VALUES ('x')",
        "DROP TABLE customers",
        "CREATE TABLE example (id INT)",
        "SELECT * FROM customers; SELECT * FROM orders",
        "SELECT * INTO another_table FROM customers",
    ],
)
def test_rejects_unsafe_statements(query):
    with pytest.raises(SQLValidationError):
        validate_readonly_sql(query)


def test_rejects_disallowed_table():
    with pytest.raises(SQLValidationError):
        validate_readonly_sql(
            "SELECT * FROM payroll",
            allowed_tables={"customers", "orders"},
        )


def test_rejects_non_public_schema():
    with pytest.raises(SQLValidationError):
        validate_readonly_sql(
            "SELECT * FROM private.payroll"
        )


def test_rejects_forbidden_function():
    with pytest.raises(SQLValidationError):
        validate_readonly_sql(
            "SELECT pg_read_file('/etc/passwd')"
        )


def test_rejects_invalid_max_rows():
    with pytest.raises(SQLValidationError):
        validate_readonly_sql("SELECT 1", max_rows=0)


def test_rejects_parameterized_limit():
    with pytest.raises(SQLValidationError):
        validate_readonly_sql(
            "SELECT * FROM customers LIMIT :limit"
        )


def test_accepts_cte():
    result = validate_readonly_sql(
        "WITH totals AS (SELECT 1 AS amount) "
        "SELECT amount FROM totals"
    )
    assert result.sql