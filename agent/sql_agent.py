"""NL→SQL agent using DuckDB and Claude."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import duckdb
import pandas as pd

from agent.analyst import call_claude, load_prompt

logger = logging.getLogger(__name__)

BLOCKED_KEYWORDS = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)


class SQLAgentError(Exception):
    """Raised when SQL generation or extraction fails."""


class SQLExecutionError(Exception):
    """Raised when the generated SQL cannot be executed."""


# ── DuckDB helpers ────────────────────────────────────────────────────────────


def _duckdb_read_expr(file_path: str) -> str:
    """Return the DuckDB read function call for the given file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        return f"read_csv_auto('{file_path}')"
    if ext in {".xlsx", ".xls"}:
        # DuckDB does not natively read Excel — load via pandas first
        return None  # handled separately
    if ext == ".parquet":
        return f"read_parquet('{file_path}')"
    if ext == ".json":
        return f"read_json_auto('{file_path}')"
    raise SQLAgentError(f"Unsupported file format for DuckDB: '{ext}'")


def load_file_to_duckdb(
    file_path: str,
    table_name: str = "data",
    max_rows: int | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> duckdb.DuckDBPyConnection:
    """Load a data file into an in-memory DuckDB table.

    Supports CSV, Parquet, JSON natively via DuckDB. Excel files are loaded
    via pandas and then registered as a DuckDB relation.

    Args:
        file_path: Path to the data file.
        table_name: Name for the DuckDB table (default: ``"data"``).
        max_rows: If set, apply a LIMIT clause (CSV/Parquet/JSON only).
        conn: Existing DuckDB connection to reuse (for multi-file sessions).

    Returns:
        Open DuckDB connection with the table loaded.

    Raises:
        SQLAgentError: If the file cannot be loaded.
    """
    if conn is None:
        conn = duckdb.connect(database=":memory:")

    ext = Path(file_path).suffix.lower()
    limit_clause = f" LIMIT {max_rows}" if max_rows else ""

    try:
        if ext in {".xlsx", ".xls"}:
            # Excel: load with pandas, register as DuckDB view
            import pandas as _pd
            df = _pd.read_excel(file_path, engine="openpyxl")
            if max_rows:
                df = df.head(max_rows)
            conn.register(table_name, df)
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}")
        else:
            read_expr = _duckdb_read_expr(file_path)
            conn.execute(
                f"CREATE TABLE {table_name} AS "
                f"SELECT * FROM {read_expr}{limit_clause}"
            )
        return conn
    except SQLAgentError:
        raise
    except Exception as exc:
        raise SQLAgentError(f"Failed to load '{file_path}' into DuckDB: {exc}") from exc


# Backwards-compatible alias
def load_csv_to_duckdb(
    csv_path: str,
    max_rows: int | None = None,
) -> duckdb.DuckDBPyConnection:
    """Alias for :func:`load_file_to_duckdb` for CSV files."""
    return load_file_to_duckdb(csv_path, max_rows=max_rows)


def get_duckdb_schema(conn: duckdb.DuckDBPyConnection) -> str:
    """Return a human-readable schema string for the ``data`` table.

    Args:
        conn: Open DuckDB connection.

    Returns:
        Multi-line string, one ``column_name (data_type)`` entry per line.
    """
    rows = conn.execute("DESCRIBE data").fetchall()
    return "\n".join(f"{row[0]} ({row[1]})" for row in rows)


def get_sample_rows(conn: duckdb.DuckDBPyConnection, n: int = 3) -> str:
    """Fetch the first ``n`` rows of the ``data`` table as a string.

    Args:
        conn: Open DuckDB connection.
        n: Number of sample rows.

    Returns:
        String representation of the sample DataFrame.
    """
    df = conn.execute(f"SELECT * FROM data LIMIT {n}").df()
    return df.to_string(index=False)


# ── Prompt building ───────────────────────────────────────────────────────────


def build_sql_prompt(schema: str, sample_rows: str, question: str) -> str:
    """Build the SQL generation prompt.

    Args:
        schema: Schema string from :func:`get_duckdb_schema`.
        sample_rows: Sample rows string from :func:`get_sample_rows`.
        question: User's natural language question.

    Returns:
        Fully substituted prompt string.
    """
    template = load_prompt("sql_prompt")
    return (
        template
        .replace("{schema}", schema)
        .replace("{sample_rows}", sample_rows)
        .replace("{question}", question)
    )


# ── SQL extraction & validation ───────────────────────────────────────────────


def extract_sql_query(raw_response: str) -> str:
    """Strip markdown fences and validate the SQL query.

    Only ``SELECT`` statements are permitted. INSERT, UPDATE, DELETE, DROP,
    CREATE, ALTER, TRUNCATE, REPLACE, and MERGE are blocked.

    Args:
        raw_response: Raw text returned by Claude.

    Returns:
        Cleaned SQL query string.

    Raises:
        SQLAgentError: If no valid SELECT statement is found.
    """
    # Strip markdown fences (```sql ... ``` or ``` ... ```)
    cleaned = re.sub(r"```(?:sql)?\s*", "", raw_response, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    if not cleaned:
        raise SQLAgentError("Claude returned an empty response.")

    if BLOCKED_KEYWORDS.match(cleaned):
        raise SQLAgentError(
            f"Blocked non-SELECT SQL statement: {cleaned[:80]!r}"
        )

    if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
        raise SQLAgentError(
            f"Response does not contain a valid SELECT statement: {cleaned[:80]!r}"
        )

    return cleaned


# ── Query execution ───────────────────────────────────────────────────────────


def execute_sql_query(
    conn: duckdb.DuckDBPyConnection,
    query: str,
) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame.

    Args:
        conn: Open DuckDB connection.
        query: Validated SQL query string.

    Returns:
        Query results as a pandas DataFrame.

    Raises:
        SQLExecutionError: If DuckDB raises an error during execution.
    """
    try:
        return conn.execute(query).df()
    except Exception as exc:
        raise SQLExecutionError(f"SQL execution failed: {exc}") from exc


# ── Main pipeline ─────────────────────────────────────────────────────────────


def answer_question(
    file_path: str,
    question: str,
    max_rows: int | None = None,
) -> dict:
    """Full NL→SQL pipeline: load file → generate SQL → execute → return results.

    Supports CSV, Excel, JSON, and Parquet. A fresh DuckDB connection is
    created per call (stateless by default).

    Args:
        file_path: Path to the data file (CSV, XLSX, JSON, or Parquet).
        question: Natural language question from the user.
        max_rows: Optional row limit.

    Returns:
        Dictionary with keys:
        - ``question`` (str)
        - ``sql_query`` (str)
        - ``results`` (pd.DataFrame)
        - ``row_count`` (int)
        - ``error`` (str | None)
    """
    result: dict = {
        "question": question,
        "sql_query": "",
        "results": pd.DataFrame(),
        "row_count": 0,
        "error": None,
    }

    try:
        if max_rows is None:
            max_rows = int(os.getenv("MAX_CSV_ROWS", 0)) or None

        conn = load_file_to_duckdb(file_path, max_rows=max_rows)
        schema = get_duckdb_schema(conn)
        sample_rows = get_sample_rows(conn)
        prompt = build_sql_prompt(schema, sample_rows, question)

        logger.info("Requesting SQL from Claude for question: %r", question)
        raw = call_claude(prompt, max_tokens=512)

        sql = extract_sql_query(raw)
        result["sql_query"] = sql

        df = execute_sql_query(conn, sql)
        result["results"] = df
        result["row_count"] = len(df)

    except (SQLAgentError, SQLExecutionError) as exc:
        logger.error("SQL agent error: %s", exc)
        result["error"] = str(exc)
    except Exception as exc:
        logger.error("Unexpected SQL agent error: %s", exc)
        result["error"] = str(exc)

    return result


class SQLAgent:
    """Convenience class wrapping the NL→SQL pipeline."""

    def __init__(self, file_path: str, max_rows: int | None = None) -> None:
        self.file_path = file_path
        # Legacy alias
        self.csv_path = file_path
        self.max_rows = max_rows

    def ask(self, question: str) -> dict:
        """Answer a natural language question about the loaded file.

        Args:
            question: The user's question.

        Returns:
            Result dict from :func:`answer_question`.
        """
        return answer_question(self.file_path, question, self.max_rows)
