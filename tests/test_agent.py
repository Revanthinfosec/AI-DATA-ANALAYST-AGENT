"""Test suite for the AI Data Analyst Agent.

All tests that touch the Anthropic API use mocker.patch to avoid real
network calls. Tests run offline and do not consume API credits.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Ensure project root is on sys.path when running `pytest` from any directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_CSV = ROOT / "sample_data" / "example.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture
def sample_profile(sample_df) -> dict:
    from agent.profiler import profile_dataframe
    return profile_dataframe(sample_df)


@pytest.fixture
def mock_insights() -> list[dict]:
    return [
        {
            "title": "High Revenue for Widget A",
            "insight": "Widget A generates the highest revenue.",
            "quality_flag": None,
            "followup": "Which region drives Widget A sales?",
        }
    ]


@pytest.fixture
def _mock_claude(mocker):
    """Patch anthropic.Anthropic so no real API calls are made."""
    valid_response = json.dumps([
        {
            "title": "Top Revenue Product",
            "insight": "Widget A has the highest revenue across all regions.",
            "quality_flag": None,
            "followup": "Which region buys the most Widget A?",
        }
    ])
    mock_content = MagicMock()
    mock_content.text = valid_response

    mock_message = MagicMock()
    mock_message.content = [mock_content]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    mocker.patch("agent.analyst.anthropic.Anthropic", return_value=mock_client)
    mocker.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-fake"})
    return mock_client


# ─────────────────────────────────────────────────────────────────────────────
# profiler tests
# ─────────────────────────────────────────────────────────────────────────────


class TestProfiler:
    def test_profile_shape(self, sample_df, sample_profile):
        assert sample_profile["shape"]["rows"] == len(sample_df)
        assert sample_profile["shape"]["cols"] == len(sample_df.columns)

    def test_profile_null_detection(self):
        from agent.profiler import profile_dataframe
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
        profile = profile_dataframe(df)
        assert profile["columns"]["a"]["null_count"] == 1
        assert round(profile["columns"]["a"]["null_pct"], 4) == round(1 / 3, 4)
        assert profile["columns"]["b"]["null_count"] == 1

    def test_profile_empty_csv(self, tmp_path):
        from agent.profiler import load_csv, profile_dataframe, DataProfilerError
        empty = tmp_path / "empty.csv"
        empty.write_text("col1,col2\n")
        df = load_csv(str(empty))
        assert df.empty
        profile = profile_dataframe(df)
        assert profile["shape"]["rows"] == 0

    def test_profile_single_column(self, tmp_path):
        from agent.profiler import load_csv, profile_dataframe
        f = tmp_path / "single.csv"
        f.write_text("score\n10\n20\n30\n")
        df = load_csv(str(f))
        profile = profile_dataframe(df)
        assert profile["shape"]["cols"] == 1
        assert "score" in profile["columns"]
        assert profile["columns"]["score"]["col_type"] == "numeric"

    def test_profile_all_null_column(self):
        from agent.profiler import profile_dataframe
        df = pd.DataFrame({"x": [None, None, None], "y": [1, 2, 3]})
        profile = profile_dataframe(df)
        assert profile["columns"]["x"]["null_pct"] == 1.0
        assert "x" in profile["high_null_columns"]

    def test_profile_wide_csv(self):
        from agent.profiler import profile_dataframe
        import numpy as np
        df = pd.DataFrame(
            np.random.rand(20, 100),
            columns=[f"col_{i}" for i in range(100)],
        )
        profile = profile_dataframe(df)
        assert profile["shape"]["cols"] == 100
        assert len(profile["numeric_columns"]) == 100

    def test_classify_column_numeric(self):
        from agent.profiler import classify_column
        s = pd.Series([1.0, 2.5, 3.7, 4.0])
        assert classify_column(s) == "numeric"

    def test_classify_column_categorical(self):
        from agent.profiler import classify_column
        s = pd.Series(["apple", "banana", "apple", "cherry", "banana"] * 10)
        assert classify_column(s) == "categorical"

    def test_classify_column_datetime(self):
        from agent.profiler import classify_column
        s = pd.Series(["2024-01-01", "2024-02-01", "2024-03-01"])
        assert classify_column(s) == "datetime"

    def test_classify_column_text(self):
        from agent.profiler import classify_column
        # Long unique strings → "text"
        s = pd.Series([f"unique_description_{i}" for i in range(100)])
        assert classify_column(s) == "text"

    def test_mixed_dtype_column(self):
        from agent.profiler import classify_column
        s = pd.Series(["123", "abc", "456", "def", "789"])
        # Mixed → cannot parse as datetime or numeric; few uniques → categorical
        result = classify_column(s)
        assert result in ("categorical", "text")

    def test_numpy_types_are_serialisable(self, sample_profile):
        """Profile dict must be JSON-serialisable (no numpy types)."""
        json.dumps(sample_profile)  # raises TypeError if numpy types present


# ─────────────────────────────────────────────────────────────────────────────
# analyst tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyst:
    def test_generate_insights_calls_claude(self, sample_profile, _mock_claude):
        from agent.analyst import generate_insights
        insights = generate_insights(sample_profile)
        assert isinstance(insights, list)
        assert len(insights) >= 1
        assert "title" in insights[0]
        assert "insight" in insights[0]
        _mock_claude.messages.create.assert_called_once()

    def test_parse_insight_response_valid(self):
        from agent.analyst import parse_insight_response
        raw = json.dumps([
            {"title": "T", "insight": "I", "quality_flag": None, "followup": "F"}
        ])
        result = parse_insight_response(raw)
        assert result[0]["title"] == "T"

    def test_parse_insight_response_with_json_fence(self):
        from agent.analyst import parse_insight_response
        raw = '```json\n[{"title":"T","insight":"I","quality_flag":null,"followup":"F"}]\n```'
        result = parse_insight_response(raw)
        assert result[0]["title"] == "T"

    def test_parse_insight_response_with_plain_fence(self):
        from agent.analyst import parse_insight_response
        raw = '```\n[{"title":"T","insight":"I","quality_flag":null,"followup":"F"}]\n```'
        result = parse_insight_response(raw)
        assert result[0]["title"] == "T"

    def test_parse_insight_response_invalid_returns_fallback(self):
        from agent.analyst import parse_insight_response
        result = parse_insight_response("This is not JSON at all.")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "title" in result[0]

    def test_generate_narrative_returns_string(self, sample_profile, mock_insights, _mock_claude):
        from agent.analyst import generate_narrative
        # Override mock to return a plain string for narrative
        _mock_claude.messages.create.return_value.content[0].text = "A great dataset."
        narrative = generate_narrative(sample_profile, mock_insights)
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    def test_missing_api_key_raises(self, sample_profile, mocker):
        from agent.analyst import generate_insights
        mocker.patch.dict(os.environ, {}, clear=True)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            generate_insights(sample_profile)


# ─────────────────────────────────────────────────────────────────────────────
# visualizer tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVisualizer:
    def test_generate_charts_creates_files(self, sample_df, sample_profile, tmp_path):
        from agent.visualizer import generate_all_charts
        paths = generate_all_charts(sample_df, sample_profile, str(tmp_path))
        assert len(paths) > 0
        for p in paths:
            assert Path(p).exists()
            assert p.endswith(".png")

    def test_plot_histogram_saves_file(self, tmp_path):
        from agent.visualizer import plot_histogram
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 3.5, 2.5])
        path = plot_histogram(s, "test_col", str(tmp_path))
        assert Path(path).exists()

    def test_plot_boxplot_saves_file(self, tmp_path):
        from agent.visualizer import plot_boxplot
        s = pd.Series([10, 20, 30, 15, 25, 100])
        path = plot_boxplot(s, "test_col", str(tmp_path))
        assert Path(path).exists()

    def test_plot_bar_chart_saves_file(self, tmp_path):
        from agent.visualizer import plot_bar_chart
        s = pd.Series(["a", "b", "a", "c", "b", "a"])
        path = plot_bar_chart(s, "category", str(tmp_path))
        assert Path(path).exists()

    def test_plot_pie_chart_saves_file(self, tmp_path):
        from agent.visualizer import plot_pie_chart
        s = pd.Series(["yes", "no", "yes", "yes", "no"])
        path = plot_pie_chart(s, "flag", str(tmp_path))
        assert Path(path).exists()

    def test_chart_cap_at_20(self, tmp_path):
        from agent.visualizer import generate_all_charts
        import numpy as np
        # 15 numeric columns → would produce 30 charts without cap
        df = pd.DataFrame(
            np.random.rand(50, 15),
            columns=[f"n{i}" for i in range(15)],
        )
        profile = {
            "numeric_columns": [f"n{i}" for i in range(15)],
            "categorical_columns": [],
            "datetime_columns": [],
            "columns": {f"n{i}": {"unique_count": 50} for i in range(15)},
        }
        paths = generate_all_charts(df, profile, str(tmp_path))
        assert len(paths) <= 20


# ─────────────────────────────────────────────────────────────────────────────
# sql_agent tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSQLAgent:
    def test_load_csv_to_duckdb(self):
        from agent.sql_agent import load_csv_to_duckdb
        conn = load_csv_to_duckdb(str(SAMPLE_CSV))
        count = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
        assert count == 10  # sample CSV has 10 data rows

    def test_get_duckdb_schema(self):
        from agent.sql_agent import load_csv_to_duckdb, get_duckdb_schema
        conn = load_csv_to_duckdb(str(SAMPLE_CSV))
        schema = get_duckdb_schema(conn)
        assert "date" in schema.lower() or "product" in schema.lower()

    def test_extract_sql_query_valid(self):
        from agent.sql_agent import extract_sql_query
        result = extract_sql_query("SELECT * FROM data WHERE revenue > 10000")
        assert result.strip().upper().startswith("SELECT")

    def test_extract_sql_query_with_sql_fence(self):
        from agent.sql_agent import extract_sql_query
        raw = "```sql\nSELECT product, SUM(revenue) FROM data GROUP BY product\n```"
        result = extract_sql_query(raw)
        assert "SELECT" in result.upper()

    def test_extract_sql_query_blocks_drop(self):
        from agent.sql_agent import extract_sql_query, SQLAgentError
        with pytest.raises(SQLAgentError):
            extract_sql_query("DROP TABLE data")

    def test_extract_sql_query_blocks_insert(self):
        from agent.sql_agent import extract_sql_query, SQLAgentError
        with pytest.raises(SQLAgentError):
            extract_sql_query("INSERT INTO data VALUES (1, 2)")

    def test_extract_sql_query_empty_raises(self):
        from agent.sql_agent import extract_sql_query, SQLAgentError
        with pytest.raises(SQLAgentError):
            extract_sql_query("```sql\n\n```")

    def test_execute_sql_query_returns_dataframe(self):
        from agent.sql_agent import load_csv_to_duckdb, execute_sql_query
        conn = load_csv_to_duckdb(str(SAMPLE_CSV))
        df = execute_sql_query(conn, "SELECT * FROM data LIMIT 3")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_answer_question_mock(self, mocker, tmp_path):
        from agent.sql_agent import answer_question
        mock_sql = "SELECT product, SUM(revenue) AS total FROM data GROUP BY product"
        mock_content = MagicMock()
        mock_content.text = mock_sql
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        mocker.patch("agent.analyst.anthropic.Anthropic", return_value=mock_client)
        mocker.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-fake"})

        result = answer_question(str(SAMPLE_CSV), "What is total revenue per product?")
        assert result["error"] is None
        assert result["sql_query"] == mock_sql
        assert isinstance(result["results"], pd.DataFrame)
        assert result["row_count"] == len(result["results"])

    def test_nl_sql_ambiguous_question(self, mocker):
        """Ambiguous question should not crash — SQL executes or returns error dict."""
        from agent.sql_agent import answer_question
        mock_content = MagicMock()
        mock_content.text = "SELECT * FROM data LIMIT 5"
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        mocker.patch("agent.analyst.anthropic.Anthropic", return_value=mock_client)
        mocker.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-fake"})

        result = answer_question(str(SAMPLE_CSV), "tell me something interesting")
        # Should not raise — either succeeds or returns error key
        assert "error" in result
        assert "results" in result


# ─────────────────────────────────────────────────────────────────────────────
# sandbox tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSandbox:
    def test_execute_code_success(self):
        from agent.sandbox import execute_code
        result = execute_code("print('hello sandbox')", str(SAMPLE_CSV))
        assert result["success"] is True
        assert "hello sandbox" in result["stdout"]

    def test_execute_code_timeout(self):
        from agent.sandbox import execute_code
        infinite_loop = "while True: pass"
        result = execute_code(infinite_loop, str(SAMPLE_CSV), timeout=2)
        assert result["success"] is False
        assert "timed out" in result["stderr"].lower()

    def test_execute_code_logs_before_run(self, mocker):
        from agent import sandbox
        log_spy = mocker.spy(sandbox, "log_execution")
        code = "print(df.shape)"
        sandbox.execute_code(code, str(SAMPLE_CSV))
        # log_execution is called twice: once before (no result) and once after
        assert log_spy.call_count >= 1
        # First call should have no result argument
        first_call_args = log_spy.call_args_list[0]
        assert first_call_args.args[0] == code

    def test_execute_code_syntax_error(self):
        from agent.sandbox import execute_code
        result = execute_code("def broken(: pass", str(SAMPLE_CSV))
        assert result["success"] is False
        assert result["stderr"]  # stderr should contain the syntax error

    def test_execute_code_captures_dataframe_output(self):
        from agent.sandbox import execute_code
        result = execute_code("print(len(df))", str(SAMPLE_CSV))
        assert result["success"] is True
        assert "10" in result["stdout"]  # sample CSV has 10 rows
