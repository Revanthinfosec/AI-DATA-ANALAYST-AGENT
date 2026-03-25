"""LLM-powered insight generation using the Claude API."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
MAX_PROMPT_CHARS = 15_000


def _get_model() -> str:
    """Return the model name, preferring MODEL_NAME env var if set."""
    return os.getenv("MODEL_NAME", DEFAULT_MODEL)


# Keep module-level MODEL for backwards compatibility
MODEL = DEFAULT_MODEL


class AnalystError(Exception):
    """Raised when the analyst cannot generate insights."""


# ── Prompt loading ────────────────────────────────────────────────────────────


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        prompt_name: File stem (e.g. ``"insight_prompt"``). The ``.txt``
            extension is added automatically.

    Returns:
        Raw prompt template string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = PROMPTS_DIR / f"{prompt_name}.txt"
    return path.read_text(encoding="utf-8")


# ── Profile truncation ────────────────────────────────────────────────────────


def _truncate_profile(profile: dict) -> dict:
    """Progressively strip the profile until json.dumps fits MAX_PROMPT_CHARS.

    Truncation order:
    1. Drop ``sample_values`` from every column
    2. Drop ``correlation_matrix``
    3. Cap ``columns`` to first 50 entries

    Args:
        profile: Full profile dict.

    Returns:
        Truncated copy that serialises within the character limit.
    """
    import copy

    p = copy.deepcopy(profile)

    def _size(d: dict) -> int:
        return len(json.dumps(d, default=str))

    if _size(p) <= MAX_PROMPT_CHARS:
        return p

    # Step 1 — drop sample_values
    for col_data in p.get("columns", {}).values():
        col_data.pop("sample_values", None)

    if _size(p) <= MAX_PROMPT_CHARS:
        return p

    # Step 2 — drop correlation_matrix
    p.pop("correlation_matrix", None)

    if _size(p) <= MAX_PROMPT_CHARS:
        return p

    # Step 3 — cap columns at 50
    cols = p.get("columns", {})
    if len(cols) > 50:
        p["columns"] = dict(list(cols.items())[:50])

    return p


def build_insight_prompt(profile: dict) -> str:
    """Build the insight prompt by injecting the profile JSON.

    Args:
        profile: Profile dict from :func:`agent.profiler.profile_dataframe`.

    Returns:
        Fully substituted prompt string ready to send to Claude.
    """
    template = load_prompt("insight_prompt")
    truncated = _truncate_profile(profile)
    profile_json = json.dumps(truncated, indent=2, default=str)
    return template.replace("{profile_json}", profile_json)


# ── Claude API call ───────────────────────────────────────────────────────────


def call_claude(
    prompt: str,
    model: str = MODEL,
    max_tokens: int = 2048,
    system: str | None = None,
) -> str:
    """Send a prompt to the Claude API and return the text response.

    Args:
        prompt: The user-turn message.
        model: Claude model identifier.
        max_tokens: Maximum tokens in the response.
        system: Optional system prompt.

    Returns:
        Stripped response text.

    Raises:
        ValueError: If ``ANTHROPIC_API_KEY`` is not set.
        AnalystError: On API call failure.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )

    # Blackboxai uses OpenAI-compatible format with its own base URL
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.blackbox.ai")

    # Use MODEL_NAME env var if set
    if model == DEFAULT_MODEL:
        model = _get_model()

    logger.info("API base_url: %s", base_url)
    logger.info("Model: %s", model)

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise AnalystError(
            f"Claude API call failed: {exc}\n"
            f"  base_url={base_url}\n"
            f"  model={model}"
        ) from exc


# ── Response parsing ──────────────────────────────────────────────────────────


def parse_insight_response(raw_response: str) -> list[dict]:
    """Extract and parse the JSON array from Claude's response.

    Handles three common output patterns:
    - Raw JSON array
    - JSON inside ``\`\`\`json ... \`\`\``` fences
    - JSON inside plain ``\`\`\` ... \`\`\``` fences

    Falls back to a single-item list with the raw text if parsing fails.

    Args:
        raw_response: The text returned by Claude.

    Returns:
        List of insight dicts. Each dict should contain at minimum
        ``title``, ``insight``, and ``followup`` keys.
    """
    # Try to find a JSON array anywhere in the response
    match = re.search(r"\[.*\]", raw_response, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                # Normalise each item to ensure required keys
                result = []
                for item in parsed:
                    if isinstance(item, dict):
                        result.append({
                            "title": item.get("title", "Untitled"),
                            "insight": item.get("insight", ""),
                            "quality_flag": item.get("quality_flag"),
                            "followup": item.get("followup", ""),
                        })
                if result:
                    return result
        except json.JSONDecodeError:
            pass

    # Fallback — return raw text as a single insight
    logger.warning("Could not parse Claude response as JSON; returning raw text.")
    return [
        {
            "title": "Analysis Result",
            "insight": raw_response,
            "quality_flag": None,
            "followup": "Review the raw response above for details.",
        }
    ]


# ── High-level functions ───────────────────────────────────────────────────────


def generate_insights(profile: dict) -> list[dict]:
    """Generate structured insights for a data profile.

    Orchestrates: build prompt → call Claude → parse response.

    Args:
        profile: Profile dict from :func:`agent.profiler.profile_dataframe`.

    Returns:
        List of insight dicts with keys: ``title``, ``insight``,
        ``quality_flag``, ``followup``.
    """
    prompt = build_insight_prompt(profile)
    logger.info("Requesting insights from Claude (%s)…", MODEL)
    raw = call_claude(prompt)
    return parse_insight_response(raw)


def generate_narrative(profile: dict, insights: list[dict]) -> str:
    """Generate a plain-English summary paragraph.

    Makes a second Claude call combining the high-level profile stats and
    the structured insights list into a coherent narrative.

    Args:
        profile: Profile dict.
        insights: List of insight dicts from :func:`generate_insights`.

    Returns:
        Narrative string.
    """
    shape = profile.get("shape", {})
    rows = shape.get("rows", "?")
    cols = shape.get("cols", "?")
    high_null = profile.get("high_null_columns", [])
    duplicates = profile.get("duplicate_rows", 0)

    insight_bullets = "\n".join(
        f"- {i['title']}: {i['insight']}" for i in insights
    )

    prompt = (
        f"You are a data analyst writing a short executive summary.\n\n"
        f"Dataset: {rows} rows × {cols} columns. "
        f"Duplicate rows: {duplicates}. "
        f"High-null columns: {high_null or 'none'}.\n\n"
        f"Key insights discovered:\n{insight_bullets}\n\n"
        f"Write a 3–5 sentence plain-English summary of the most important "
        f"findings and any data quality concerns. Be concise and avoid bullet "
        f"points — use flowing prose."
    )

    logger.info("Requesting narrative from Claude (%s)…", MODEL)
    return call_claude(prompt, max_tokens=512)


class DataAnalyst:
    """Convenience class wrapping insight and narrative generation."""

    def __init__(self) -> None:
        self.insights: list[dict] = []
        self.narrative: str = ""

    def run(self, profile: dict) -> dict:
        """Run insight generation and narrative for the given profile.

        Args:
            profile: Profile dict.

        Returns:
            Dict with ``insights`` (list) and ``narrative`` (str).
        """
        self.insights = generate_insights(profile)
        self.narrative = generate_narrative(profile, self.insights)
        return {"insights": self.insights, "narrative": self.narrative}
