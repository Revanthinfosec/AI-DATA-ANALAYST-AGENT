"""Chart generation module — produces PNG visualisations from a profiled DataFrame."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import time
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot — headless / no display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────


def setup_output_dir(chart_dir: str) -> Path:
    """Create the chart output directory if it does not exist.

    Args:
        chart_dir: Path string to the desired directory.

    Returns:
        Path object pointing to the (now-existing) directory.
    """
    path = Path(chart_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_col_name(col_name: str) -> str:
    """Strip characters that are unsafe in file names."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in col_name)


def save_figure(fig: plt.Figure, name: str, chart_dir: str) -> str:
    """Save a matplotlib figure as PNG and close it.

    Args:
        fig: The figure to save.
        name: Base file name (without extension).
        chart_dir: Directory to save to.

    Returns:
        Absolute path to the saved PNG file.
    """
    out_dir = setup_output_dir(chart_dir)
    ts = int(time() * 1000)
    file_path = out_dir / f"{_safe_col_name(name)}_{ts}.png"
    fig.savefig(str(file_path), bbox_inches="tight", dpi=100)
    plt.close(fig)
    return str(file_path)


# ── Individual chart functions ────────────────────────────────────────────────


def plot_histogram(series: pd.Series, col_name: str, chart_dir: str) -> str:
    """Histogram with KDE overlay for a numeric series.

    Args:
        series: Numeric column data.
        col_name: Column name (used in title and file name).
        chart_dir: Directory to save the chart.

    Returns:
        Absolute path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(series.dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {col_name}")
    ax.set_xlabel(col_name)
    ax.set_ylabel("Count")
    return save_figure(fig, f"{col_name}_histogram", chart_dir)


def plot_boxplot(series: pd.Series, col_name: str, chart_dir: str) -> str:
    """Boxplot for outlier visualisation.

    Args:
        series: Numeric column data.
        col_name: Column name.
        chart_dir: Directory to save the chart.

    Returns:
        Absolute path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(series.dropna(), vert=True, patch_artist=True,
               boxprops={"facecolor": "lightblue"})
    ax.set_title(f"Box Plot — {col_name}")
    ax.set_ylabel(col_name)
    ax.set_xticks([])
    return save_figure(fig, f"{col_name}_boxplot", chart_dir)


def plot_bar_chart(series: pd.Series, col_name: str, chart_dir: str) -> str:
    """Horizontal bar chart of the top value counts.

    Args:
        series: Categorical column data.
        col_name: Column name.
        chart_dir: Directory to save the chart.

    Returns:
        Absolute path to saved PNG.
    """
    counts = series.value_counts(dropna=True).head(15)
    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.5)))
    counts.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_title(f"Value Counts — {col_name}")
    ax.set_xlabel("Count")
    return save_figure(fig, f"{col_name}_bar", chart_dir)


def plot_pie_chart(series: pd.Series, col_name: str, chart_dir: str) -> str:
    """Pie chart for categorical columns with ≤ 8 unique values.

    Args:
        series: Categorical column data.
        col_name: Column name.
        chart_dir: Directory to save the chart.

    Returns:
        Absolute path to saved PNG.
    """
    counts = series.value_counts(dropna=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts.values, labels=counts.index.tolist(),
           autopct="%1.1f%%", startangle=140)
    ax.set_title(f"Composition — {col_name}")
    return save_figure(fig, f"{col_name}_pie", chart_dir)


def plot_line_chart(
    df: pd.DataFrame,
    date_col: str,
    numeric_col: str,
    chart_dir: str,
) -> str:
    """Time-series line chart sorted by a datetime column.

    Args:
        df: Source DataFrame.
        date_col: Name of the datetime column.
        numeric_col: Name of the numeric column to plot.
        chart_dir: Directory to save the chart.

    Returns:
        Absolute path to saved PNG.
    """
    tmp = df[[date_col, numeric_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna().sort_values(date_col)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tmp[date_col], tmp[numeric_col], marker="o", color="steelblue")
    ax.set_title(f"{numeric_col} over {date_col}")
    ax.set_xlabel(date_col)
    ax.set_ylabel(numeric_col)
    fig.autofmt_xdate()
    return save_figure(fig, f"{date_col}_{numeric_col}_line", chart_dir)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    chart_dir: str,
) -> str:
    """Scatter plot for two numeric columns.

    Args:
        df: Source DataFrame.
        x_col: Name of the X-axis column.
        y_col: Name of the Y-axis column.
        chart_dir: Directory to save the chart.

    Returns:
        Absolute path to saved PNG.
    """
    tmp = df[[x_col, y_col]].dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=tmp, x=x_col, y=y_col, ax=ax, alpha=0.7, color="steelblue")
    ax.set_title(f"{y_col} vs {x_col}")
    return save_figure(fig, f"{x_col}_{y_col}_scatter", chart_dir)


def plot_correlation_heatmap(df: pd.DataFrame, chart_dir: str) -> str:
    """Correlation heatmap for numeric columns (top 10 by variance).

    Args:
        df: Source DataFrame containing numeric columns.
        chart_dir: Directory to save the chart.

    Returns:
        Absolute path to saved PNG.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Pick top 10 by variance to keep the chart readable
    top_cols = (
        df[numeric_cols].var().nlargest(10).index.tolist()
        if len(numeric_cols) > 10
        else numeric_cols
    )
    corr = df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(max(6, len(top_cols)), max(5, len(top_cols) - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Correlation Matrix")
    return save_figure(fig, "correlation_heatmap", chart_dir)


# ── Master dispatcher ──────────────────────────────────────────────────────────


def generate_all_charts(
    df: pd.DataFrame,
    profile: dict,
    chart_dir: str,
) -> list[str]:
    """Generate all relevant charts for a profiled DataFrame.

    Dispatches to individual plot functions based on detected column types.
    Caps total charts at 20 to avoid runaway generation. Each chart function
    is wrapped in a try/except so one failure does not abort the full pass.

    Args:
        df: Source DataFrame.
        profile: Profile dict produced by :func:`agent.profiler.profile_dataframe`.
        chart_dir: Directory to save charts.

    Returns:
        List of absolute file paths for every chart that was successfully saved.
    """
    paths: list[str] = []
    MAX_CHARTS = 20

    numeric_cols: list[str] = profile.get("numeric_columns", [])
    categorical_cols: list[str] = profile.get("categorical_columns", [])
    datetime_cols: list[str] = profile.get("datetime_columns", [])

    def _try(fn, *args, **kwargs) -> str | None:
        """Call fn; log and skip on any exception."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.warning("Chart generation failed (%s): %s", fn.__name__, exc)
            return None

    # Numeric: histogram + boxplot per column
    for col in numeric_cols:
        if len(paths) >= MAX_CHARTS:
            break
        p = _try(plot_histogram, df[col], col, chart_dir)
        if p:
            paths.append(p)
        if len(paths) < MAX_CHARTS:
            p = _try(plot_boxplot, df[col], col, chart_dir)
            if p:
                paths.append(p)

    # Categorical: bar chart; pie if ≤ 8 unique
    for col in categorical_cols:
        if len(paths) >= MAX_CHARTS:
            break
        p = _try(plot_bar_chart, df[col], col, chart_dir)
        if p:
            paths.append(p)
        unique_count = profile["columns"][col].get("unique_count", 99)
        if unique_count <= 8 and len(paths) < MAX_CHARTS:
            p = _try(plot_pie_chart, df[col], col, chart_dir)
            if p:
                paths.append(p)

    # Datetime × first numeric column: line chart
    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        for num_col in numeric_cols[:3]:  # up to 3 line charts
            if len(paths) >= MAX_CHARTS:
                break
            p = _try(plot_line_chart, df, date_col, num_col, chart_dir)
            if p:
                paths.append(p)

    # Two numeric columns: scatter + heatmap
    if len(numeric_cols) >= 2 and len(paths) < MAX_CHARTS:
        p = _try(plot_scatter, df, numeric_cols[0], numeric_cols[1], chart_dir)
        if p:
            paths.append(p)
    if len(numeric_cols) >= 2 and len(paths) < MAX_CHARTS:
        p = _try(plot_correlation_heatmap, df, chart_dir)
        if p:
            paths.append(p)

    return paths


class DataVisualizer:
    """Convenience class wrapping chart generation."""

    def __init__(self, chart_dir: str | None = None) -> None:
        self.chart_dir = chart_dir or os.getenv("CHART_OUTPUT_DIR", "/tmp/charts/")

    def run(self, df: pd.DataFrame, profile: dict) -> list[str]:
        """Generate all charts and return their file paths."""
        return generate_all_charts(df, profile, self.chart_dir)
