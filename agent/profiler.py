"""Data profiling module — loads CSV/Excel/JSON/Parquet and computes a structured profile dict."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


class DataProfilerError(Exception):
    """Raised when profiling fails."""


def load_file(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load a data file into a DataFrame. Supports CSV, Excel, JSON, and Parquet.

    The format is determined by the file extension. ``max_rows`` is applied
    where possible (Parquet reads the full file then truncates).

    Args:
        path: Path to the data file.
        max_rows: Maximum number of rows to load. Defaults to MAX_CSV_ROWS env var or None.

    Returns:
        Loaded DataFrame.

    Raises:
        DataProfilerError: If the file cannot be read or the format is unsupported.
    """
    if max_rows is None:
        max_rows = int(os.getenv("MAX_CSV_ROWS", 0)) or None

    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise DataProfilerError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    try:
        if ext == ".csv":
            df = pd.read_csv(path, nrows=max_rows)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(path, nrows=max_rows, engine="openpyxl")
        elif ext == ".json":
            df = pd.read_json(path)
            if max_rows:
                df = df.head(max_rows)
        elif ext == ".parquet":
            df = pd.read_parquet(path, engine="pyarrow")
            if max_rows:
                df = df.head(max_rows)
        else:
            raise DataProfilerError(f"Unhandled extension: {ext}")
    except DataProfilerError:
        raise
    except Exception as exc:
        raise DataProfilerError(f"Failed to load '{path}': {exc}") from exc

    return df


# Backwards-compatible alias
def load_csv(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Alias for :func:`load_file` — accepts CSV files (and any other supported format)."""
    return load_file(path, max_rows)


def classify_column(series: pd.Series) -> str:
    """Classify a pandas Series into one of four column types.

    Classification order (first match wins):
    1. datetime  — already a datetime dtype, OR >80% of values parse as dates
    2. numeric   — integer or float dtype
    3. categorical — nunique/count < 5% OR nunique <= 20
    4. text      — everything else

    Args:
        series: The column to classify.

    Returns:
        One of: ``"datetime"``, ``"numeric"``, ``"categorical"``, ``"text"``.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Try to coerce to datetime (skip if already numeric)
    if not pd.api.types.is_numeric_dtype(series):
        try:
            parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
            non_null = series.dropna()
            if len(non_null) > 0 and parsed.notna().sum() / len(non_null) > 0.8:
                return "datetime"
        except Exception:
            pass

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    non_null_count = series.dropna().shape[0]
    if non_null_count == 0:
        return "text"

    n_unique = series.nunique(dropna=True)
    if n_unique / non_null_count < 0.05 or n_unique <= 20:
        return "categorical"

    return "text"


def _to_python(val: Any) -> Any:
    """Convert numpy scalars to native Python types for JSON serialisation."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if pd.isna(val) if not isinstance(val, (list, dict, str)) else False:
        return None
    return val


def compute_column_profile(series: pd.Series, col_type: str) -> dict:
    """Compute per-column statistics.

    Args:
        series: The column data.
        col_type: One of the four types returned by :func:`classify_column`.

    Returns:
        Dictionary with null counts, unique counts, sample values, and
        type-specific descriptive statistics.
    """
    total = len(series)
    null_count = int(series.isna().sum())
    null_pct = round(null_count / total, 4) if total > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))
    sample_values = [_to_python(v) for v in series.dropna().head(5).tolist()]

    profile: dict[str, Any] = {
        "dtype": str(series.dtype),
        "col_type": col_type,
        "null_count": null_count,
        "null_pct": null_pct,
        "unique_count": unique_count,
        "sample_values": sample_values,
        "stats": {},
    }

    if col_type == "numeric":
        desc = series.describe()
        profile["stats"] = {
            "min": _to_python(desc.get("min")),
            "max": _to_python(desc.get("max")),
            "mean": _to_python(desc.get("mean")),
            "median": _to_python(series.median()),
            "std": _to_python(desc.get("std")),
            "q25": _to_python(desc.get("25%")),
            "q75": _to_python(desc.get("75%")),
            "skewness": _to_python(series.skew()),
            "kurtosis": _to_python(series.kurtosis()),
        }

    elif col_type == "categorical":
        top5 = series.value_counts(dropna=True).head(5)
        profile["stats"] = {
            "top_5_values": {str(k): int(v) for k, v in top5.items()},
            "mode": str(series.mode().iloc[0]) if not series.mode().empty else None,
        }

    elif col_type == "datetime":
        dt_series = pd.to_datetime(series, errors="coerce")
        valid = dt_series.dropna()
        profile["stats"] = {
            "min_date": str(valid.min()) if not valid.empty else None,
            "max_date": str(valid.max()) if not valid.empty else None,
            "date_range_days": int((valid.max() - valid.min()).days) if len(valid) > 1 else 0,
        }

    elif col_type == "text":
        lengths = series.dropna().astype(str).str.len()
        profile["stats"] = {
            "avg_length": _to_python(lengths.mean()) if not lengths.empty else 0,
            "max_length": _to_python(lengths.max()) if not lengths.empty else 0,
        }

    return profile


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Build a complete profile of the DataFrame.

    Args:
        df: The loaded DataFrame to profile.

    Returns:
        Structured dict with shape, duplicates, memory usage, per-column
        profiles, column type lists, high-null columns, and correlation matrix.

    Raises:
        DataProfilerError: If profiling fails unexpectedly.
    """
    try:
        rows, cols = df.shape
        duplicate_rows = int(df.duplicated().sum())
        memory_mb = round(float(df.memory_usage(deep=True).sum()) / 1024 ** 2, 4)

        columns: dict[str, dict] = {}
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []
        datetime_columns: list[str] = []
        text_columns: list[str] = []

        for col in df.columns:
            col_type = classify_column(df[col])
            columns[col] = compute_column_profile(df[col], col_type)

            if col_type == "numeric":
                numeric_columns.append(col)
            elif col_type == "categorical":
                categorical_columns.append(col)
            elif col_type == "datetime":
                datetime_columns.append(col)
            else:
                text_columns.append(col)

        high_null_columns = [
            col for col, prof in columns.items() if prof["null_pct"] > 0.5
        ]

        # Correlation matrix — numeric cols only, cap at 10
        correlation_matrix: dict = {}
        top_numeric = numeric_columns[:10]
        if len(top_numeric) >= 2:
            corr = df[top_numeric].corr()
            # Convert to plain Python floats
            correlation_matrix = {
                col: {
                    other: round(v, 4) if (v := _to_python(val)) is not None else None
                    for other, val in row.items()
                }
                for col, row in corr.to_dict().items()
            }

        return {
            "shape": {"rows": rows, "cols": cols},
            "duplicate_rows": duplicate_rows,
            "memory_usage_mb": memory_mb,
            "columns": columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "text_columns": text_columns,
            "high_null_columns": high_null_columns,
            "correlation_matrix": correlation_matrix,
        }

    except Exception as exc:
        raise DataProfilerError(f"Profiling failed: {exc}") from exc


class DataProfiler:
    """Convenience class wrapping the profiler functions."""

    def __init__(self, file_path: str, max_rows: int | None = None) -> None:
        self.file_path = file_path
        # Keep legacy attribute name so existing callers don't break
        self.csv_path = file_path
        self.max_rows = max_rows
        self.df: pd.DataFrame | None = None
        self.profile: dict | None = None

    def run(self) -> dict:
        """Load the file and return the profile dict."""
        self.df = load_file(self.file_path, self.max_rows)
        self.profile = profile_dataframe(self.df)
        return self.profile
