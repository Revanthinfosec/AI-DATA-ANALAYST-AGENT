"""Subprocess-based sandbox for safely executing LLM-generated Python code.

All LLM-generated code is run in a child process — never via exec() or eval()
in the main process. Writes are restricted to /tmp/sandbox/ and execution is
killed after SANDBOX_TIMEOUT seconds.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SANDBOX_DIR = "/tmp/sandbox/"
DEFAULT_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "30"))


class SandboxError(Exception):
    """Raised when the sandbox cannot be prepared or the script cannot be written."""


def prepare_sandbox_dir() -> str:
    """Create the sandbox directory if it does not exist.

    Returns:
        Path string to the sandbox directory.
    """
    path = Path(SANDBOX_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def build_execution_script(code: str, csv_path: str, sandbox_dir: str) -> str:
    """Wrap LLM-generated code in a safety harness.

    The harness:
    - Changes the working directory to ``sandbox_dir`` before running user code
      so any relative-path writes land inside the sandbox.
    - Pre-loads pandas and reads the CSV into ``df`` so LLM code can assume it
      is available.
    - Flushes stdout at the end.

    Args:
        code: Raw Python code string from the LLM.
        csv_path: Absolute path to the CSV file the code should operate on.
        sandbox_dir: Directory the process should chdir into.

    Returns:
        Full Python script as a string.
    """
    header = f"""\
import os
import sys

# Restrict working directory to sandbox
os.chdir({sandbox_dir!r})

import pandas as pd
import numpy as np

# Pre-load the DataFrame so LLM code can use `df` directly
df = pd.read_csv({csv_path!r})

"""
    footer = "\nsys.stdout.flush()\n"
    return header + code + footer


def log_execution(code: str, result: dict | None = None) -> None:
    """Log the code (and optionally its result) to the logger.

    Per CLAUDE.md, all generated code MUST be logged before execution.

    Args:
        code: The Python code that will be / was executed.
        result: Optional execution result dict to log after the run.
    """
    separator = "=" * 60
    logger.info("%s\nSANDBOX CODE TO EXECUTE:\n%s\n%s", separator, code, separator)
    if result is not None:
        status = "SUCCESS" if result.get("success") else "FAILURE"
        logger.info(
            "SANDBOX RESULT [%s] — %.2fs\nstdout: %s\nstderr: %s",
            status,
            result.get("execution_time_seconds", 0),
            result.get("stdout", ""),
            result.get("stderr", ""),
        )


def _find_new_pngs(sandbox_dir: str, before_mtime: float) -> list[str]:
    """Return PNG files in sandbox_dir created/modified after before_mtime."""
    results = []
    try:
        for f in Path(sandbox_dir).iterdir():
            if f.suffix.lower() == ".png" and f.stat().st_mtime >= before_mtime:
                results.append(str(f))
    except Exception:
        pass
    return results


def execute_code(
    code: str,
    csv_path: str,
    timeout: int | None = None,
) -> dict:
    """Execute LLM-generated Python code in an isolated subprocess.

    The code is written to a temporary ``.py`` file inside the sandbox
    directory and run via ``subprocess.run``. The main process never calls
    ``exec()`` or ``eval()``.

    Args:
        code: Python source code string to execute.
        csv_path: Path to the CSV file (injected into the script header).
        timeout: Seconds before the subprocess is killed. Defaults to
            ``SANDBOX_TIMEOUT`` env var (default 30 s).

    Returns:
        Dictionary with keys:
        - ``success`` (bool)
        - ``stdout`` (str)
        - ``stderr`` (str)
        - ``generated_files`` (list[str]) — PNGs created in the sandbox
        - ``execution_time_seconds`` (float)
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    sandbox_dir = prepare_sandbox_dir()

    # Log the code BEFORE execution (required by CLAUDE.md)
    log_execution(code)

    script = build_execution_script(code, csv_path, sandbox_dir)

    # Record sandbox state before running
    before_mtime = time.time()

    # Write to a temp file inside the sandbox
    try:
        with tempfile.NamedTemporaryFile(
            dir=sandbox_dir,
            suffix=".py",
            mode="w",
            delete=False,
        ) as tmp:
            tmp.write(script)
            script_path = tmp.name
    except Exception as exc:
        raise SandboxError(f"Failed to write sandbox script: {exc}") from exc

    start = time.time()
    try:
        proc = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        success = proc.returncode == 0
        result = {
            "success": success,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "generated_files": _find_new_pngs(sandbox_dir, before_mtime),
            "execution_time_seconds": round(elapsed, 3),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        result = {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "generated_files": [],
            "execution_time_seconds": round(elapsed, 3),
        }
    finally:
        # Clean up the temp script
        try:
            os.unlink(script_path)
        except Exception:
            pass

    log_execution(code, result)
    return result
