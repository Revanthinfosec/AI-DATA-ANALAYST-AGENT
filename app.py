"""Main entry point for the AI Data Analyst Agent.

Supports two modes:
- CLI:       python app.py --file path/to/file.csv [--sql] [--output-dir /tmp/charts]
             python app.py --file a.csv b.xlsx c.parquet   (multiple files)
- Streamlit: python app.py --mode streamlit
             (or: streamlit run app.py)

Supported file formats: CSV, Excel (.xlsx/.xls), JSON, Parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load .env before any env var reads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".json", ".parquet"]
STREAMLIT_TYPES = ["csv", "xlsx", "xls", "json", "parquet"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _chart_dir(override: str | None = None) -> str:
    return override or os.getenv("CHART_OUTPUT_DIR", "/tmp/charts/")


def _print_section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _save_upload(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temp file and return its path."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


# ── Single-file analysis pipeline ─────────────────────────────────────────────


def analyse_file(
    file_path: str,
    enable_sql: bool = False,
    output_dir: str | None = None,
) -> None:
    """Run the full analysis pipeline for a single file in CLI mode.

    Args:
        file_path: Path to the data file (any supported format).
        enable_sql: If True, enter an interactive NL→SQL loop after analysis.
        output_dir: Override for the chart output directory.
    """
    from agent.profiler import DataProfiler
    from agent.analyst import DataAnalyst
    from agent.visualizer import DataVisualizer
    from agent.sql_agent import SQLAgent

    chart_output = _chart_dir(output_dir)
    file_name = Path(file_path).name

    _print_section(f"File: {file_name}")

    # 1. Profile
    _print_section("Step 1/4 — Profiling data")
    profiler = DataProfiler(file_path)
    profile = profiler.run()

    shape = profile["shape"]
    print(f"  Rows: {shape['rows']:,}  |  Cols: {shape['cols']}")
    print(f"  Duplicate rows: {profile['duplicate_rows']}")
    print(f"  Memory: {profile['memory_usage_mb']:.2f} MB")
    print(f"  Numeric columns : {profile['numeric_columns']}")
    print(f"  Categorical cols: {profile['categorical_columns']}")
    print(f"  Datetime columns: {profile['datetime_columns']}")
    if profile["high_null_columns"]:
        print(f"  ⚠  High-null columns (>50%): {profile['high_null_columns']}")

    # 2. Insights
    _print_section("Step 2/4 — Generating insights")
    analyst = DataAnalyst()
    analysis = analyst.run(profile)

    for i, item in enumerate(analysis["insights"], 1):
        print(f"\n  [{i}] {item['title']}")
        print(f"      {item['insight']}")
        if item.get("quality_flag"):
            print(f"      ⚠  Quality: {item['quality_flag']}")
        print(f"      → Follow-up: {item['followup']}")

    # 3. Visualisations
    _print_section("Step 3/4 — Generating charts")
    visualizer = DataVisualizer(chart_output)
    chart_paths = visualizer.run(profiler.df, profile)

    if chart_paths:
        print(f"  {len(chart_paths)} chart(s) saved:")
        for p in chart_paths:
            print(f"    • {p}")
    else:
        print("  No charts generated.")

    # 4. Narrative
    _print_section("Step 4/4 — Summary narrative")
    print(f"\n  {analysis['narrative']}\n")

    # Optional NL→SQL
    if enable_sql:
        _print_section("NL→SQL Interactive Mode  (type 'exit' to quit)")
        sql_agent = SQLAgent(file_path)
        while True:
            try:
                question = input("\n  Ask a question about your data: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Exiting SQL mode.")
                break
            if question.lower() in {"exit", "quit", "q"}:
                break
            if not question:
                continue
            result = sql_agent.ask(question)
            if result["error"]:
                print(f"  ✗ Error: {result['error']}")
            else:
                print(f"\n  SQL: {result['sql_query']}")
                print(f"\n  Results ({result['row_count']} rows):")
                print(result["results"].to_string(index=False))


def run_cli(
    file_paths: list[str],
    enable_sql: bool = False,
    output_dir: str | None = None,
) -> None:
    """Run CLI analysis for one or more files.

    Args:
        file_paths: List of paths to analyse.
        enable_sql: Enable NL→SQL loop (applied to each file in sequence).
        output_dir: Chart output directory override.
    """
    for path in file_paths:
        analyse_file(path, enable_sql=enable_sql, output_dir=output_dir)
        if len(file_paths) > 1:
            print("\n" + "═" * 60 + "\n")


# ── Streamlit mode ─────────────────────────────────────────────────────────────


def _render_single_file_analysis(
    st,
    file_path: str,
    file_label: str,
    chart_dir: str,
) -> None:
    """Render the full analysis UI for one file inside a Streamlit tab/expander."""
    import pandas as pd
    from agent.profiler import DataProfiler
    from agent.analyst import DataAnalyst, chat_with_data
    from agent.visualizer import DataVisualizer
    from agent.sql_agent import SQLAgent

    with st.spinner(f"Profiling {file_label}…"):
        profiler = DataProfiler(file_path)
        profile = profiler.run()

    # Profile summary
    shape = profile["shape"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{shape['rows']:,}")
    c2.metric("Columns", shape["cols"])
    c3.metric("Duplicate Rows", profile["duplicate_rows"])

    with st.expander("Column Details", expanded=False):
        rows = [
            {
                "Column": col,
                "Type": info["col_type"],
                "Null %": f"{info['null_pct'] * 100:.1f}%",
                "Unique": info["unique_count"],
            }
            for col, info in profile["columns"].items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if profile["high_null_columns"]:
        st.warning(f"High-null columns (>50%): {profile['high_null_columns']}")

    # Insights
    st.subheader("AI Insights")
    with st.spinner("Generating insights with Claude…"):
        analyst = DataAnalyst()
        analysis = analyst.run(profile)

    for item in analysis["insights"]:
        with st.expander(f"💡 {item['title']}"):
            st.write(item["insight"])
            if item.get("quality_flag"):
                st.warning(f"Data quality: {item['quality_flag']}")
            st.info(f"Follow-up: {item['followup']}")

    # Narrative
    st.subheader("Summary")
    st.markdown(analysis["narrative"])

    # Charts
    st.subheader("Visualisations")
    with st.spinner("Generating charts…"):
        visualizer = DataVisualizer(chart_dir)
        chart_paths = visualizer.run(profiler.df, profile)

    if chart_paths:
        cols = st.columns(2)
        for i, path in enumerate(chart_paths):
            with cols[i % 2]:
                st.image(path)
    else:
        st.info("No charts were generated for this dataset.")

    # ── Business Advisor Chat ─────────────────────────────────────────────
    st.subheader("💬 Ask Your Data Analyst")
    st.caption(
        "Ask any business question — growth strategies, trends, anomalies, "
        "forecasts, or 'what should I focus on?' — grounded in your data."
    )

    from agent.analyst import chat_with_data

    # Session state key per file so each tab has its own history
    history_key = f"chat_history_{file_label}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    history: list[dict] = st.session_state[history_key]

    # Render existing conversation
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Ask anything about your data… e.g. 'How can I grow revenue?' or 'Which product should I focus on?'",
        key=f"chat_input_{file_label}",
    )

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = chat_with_data(
                        question=user_input,
                        profile=profile,
                        insights=analysis["insights"],
                        history=history[:-1],  # exclude the just-added user msg
                    )
                except Exception as exc:
                    answer = f"Error: {exc}"
            st.markdown(answer)

        history.append({"role": "assistant", "content": answer})
        st.session_state[history_key] = history

    # Clear chat button
    if history and st.button("Clear chat", key=f"clear_{file_label}"):
        st.session_state[history_key] = []
        st.rerun()

    # ── NL→SQL (collapsed) ────────────────────────────────────────────────
    with st.expander("🔍 Run a SQL Query directly"):
        q_key = f"question_{file_label}"
        btn_key = f"run_{file_label}"
        question = st.text_input(
            "SQL question (e.g. 'Total revenue per region?')",
            key=q_key,
        )
        if st.button("Run Query", key=btn_key) and question:
            with st.spinner("Generating SQL and running query…"):
                result = SQLAgent(file_path).ask(question)
            if result["error"]:
                st.error(f"Error: {result['error']}")
            else:
                st.code(result["sql_query"], language="sql")
                st.write(f"Results ({result['row_count']} rows):")
                st.dataframe(result["results"], use_container_width=True)


def run_streamlit() -> None:
    """Run the Streamlit web UI with multi-file support."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Run: pip install streamlit")
        sys.exit(1)

    st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
    st.title("📊 AI Data Analyst Agent")
    st.caption(
        "Upload one or more files (CSV, Excel, JSON, Parquet) to get automated "
        "insights, charts, and a data-aware business advisor chat."
    )

    # ── Password gate ─────────────────────────────────────────────────────
    app_password = os.getenv("APP_PASSWORD")
    if app_password:
        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False
        if not st.session_state["authenticated"]:
            pwd = st.text_input("Enter password to access the app", type="password")
            if st.button("Login"):
                if pwd == app_password:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            st.stop()

    uploaded_files = st.file_uploader(
        "Upload your data files",
        type=STREAMLIT_TYPES,
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more data files to begin.")
        return

    chart_dir = _chart_dir()

    # Save all uploads to temp files upfront
    tmp_paths: list[tuple[str, str]] = []  # (label, tmp_path)
    for uf in uploaded_files:
        tmp_path = _save_upload(uf)
        tmp_paths.append((uf.name, tmp_path))

    if len(tmp_paths) == 1:
        # Single file — render directly (no tabs needed)
        label, path = tmp_paths[0]
        st.subheader(f"Analysis: {label}")
        _render_single_file_analysis(st, path, label, chart_dir)
    else:
        # Multiple files — one tab per file
        tab_labels = [label for label, _ in tmp_paths]
        tabs = st.tabs(tab_labels)
        for tab, (label, path) in zip(tabs, tmp_paths):
            with tab:
                _render_single_file_analysis(st, path, label, chart_dir)


# ── CLI argument parsing ───────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Data Analyst Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python app.py --file data.csv\n"
            "  python app.py --file sales.xlsx customers.csv\n"
            "  python app.py --file data.parquet --sql\n"
            "  python app.py --mode streamlit\n"
            "  streamlit run app.py\n"
        ),
    )
    parser.add_argument(
        "--file",
        nargs="+",
        metavar="PATH",
        help="Path(s) to data file(s) — CSV, Excel, JSON, or Parquet",
    )
    # Legacy alias so existing --csv usage still works
    parser.add_argument("--csv", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        choices=["cli", "streamlit"],
        default="cli",
        help="Run mode (default: cli)",
    )
    parser.add_argument(
        "--sql",
        action="store_true",
        help="Enable interactive NL→SQL loop in CLI mode",
    )
    parser.add_argument(
        "--output-dir",
        help="Override CHART_OUTPUT_DIR for chart output",
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # When launched via `streamlit run app.py`, Streamlit injects its own
    # argv — detect this case and go straight to the Streamlit UI.
    if "streamlit" in sys.modules or any("streamlit" in a for a in sys.argv):
        run_streamlit()
    else:
        args = parse_args()

        if args.mode == "streamlit":
            import subprocess
            subprocess.run(["streamlit", "run", __file__], check=True)
        else:
            # Collect file paths — support both --file and legacy --csv
            files: list[str] = []
            if args.file:
                files.extend(args.file)
            if args.csv:
                files.append(args.csv)

            if not files:
                print(
                    "Error: provide at least one file with --file path [path ...]",
                    file=sys.stderr,
                )
                sys.exit(1)

            missing = [f for f in files if not Path(f).exists()]
            if missing:
                for m in missing:
                    print(f"Error: file not found: {m}", file=sys.stderr)
                sys.exit(1)

            run_cli(files, enable_sql=args.sql, output_dir=args.output_dir)
