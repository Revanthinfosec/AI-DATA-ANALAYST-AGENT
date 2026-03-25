# CLAUDE.md — Auto Data Analyst Agent

## Project Overview

An agentic data analysis tool where users upload a CSV file and receive automated
insights, visualizations, and (optionally) natural language SQL queries — powered
by Claude + a Python execution sandbox.

---

## Agent Behavior & Goals

You are an expert data analyst agent. When a user uploads a CSV:

1. **Profile the data** — shape, dtypes, nulls, duplicates, basic stats
2. **Generate insights** — trends, correlations, outliers, distributions
3. **Produce charts** — relevant visualizations using matplotlib/seaborn
4. **Summarize findings** — plain-English narrative of what matters most
5. **Suggest next steps** — follow-up analyses or questions worth exploring
6. *(Stretch)* **Answer NL→SQL queries** — translate user questions into SQL

Always be concise, accurate, and avoid hallucinating statistics. If the data is
ambiguous, state your assumptions clearly before proceeding.

---

## Project Structure

```
auto-data-analyst/
├── CLAUDE.md               # This file
├── README.md
├── app.py                  # Main entry point (Streamlit or CLI)
├── agent/
│   ├── __init__.py
│   ├── profiler.py         # Data profiling logic
│   ├── analyst.py          # LLM-powered insight generation
│   ├── visualizer.py       # Chart generation
│   ├── sql_agent.py        # NL → SQL (stretch goal)
│   └── sandbox.py          # Safe Python execution environment
├── prompts/
│   ├── profile_prompt.txt
│   ├── insight_prompt.txt
│   └── sql_prompt.txt
├── tests/
│   └── test_agent.py
├── sample_data/
│   └── example.csv
└── requirements.txt
```

---

## Tech Stack

| Layer         | Tool/Library                        |
|---------------|-------------------------------------|
| LLM           | Claude (`claude-sonnet-4-20250514`) |
| Data          | pandas, numpy                       |
| Visualization | matplotlib, seaborn, plotly         |
| Execution     | subprocess sandbox / RestrictedPython |
| UI (optional) | Streamlit                           |
| SQL (stretch) | DuckDB + Claude NL→SQL              |

---

## Core Modules

### `agent/profiler.py`
- Load CSV with pandas, infer dtypes
- Compute: shape, null %, unique counts, descriptive stats
- Detect column types: numeric, categorical, datetime, text
- Output: structured dict / JSON summary

### `agent/analyst.py`
- Take profile JSON → send to Claude with `insight_prompt.txt`
- Extract: top trends, correlations, anomalies, data quality issues
- Return structured insight list (title + explanation per insight)

### `agent/visualizer.py`
- Generate charts based on column types:
  - Numeric → histogram, boxplot
  - Categorical → bar chart, pie (if ≤ 8 categories)
  - Datetime + Numeric → line chart
  - Two numerics → scatter + correlation heatmap
- Save charts as PNG/SVG; return file paths

### `agent/sandbox.py`
- Accept LLM-generated Python code strings
- Execute in a restricted subprocess with timeout (default: 30s)
- Capture stdout, stderr, and any generated files
- Never execute code with file system writes outside `/tmp/sandbox/`

### `agent/sql_agent.py` *(stretch)*
- Load CSV into DuckDB in-memory table
- Accept natural language question from user
- Prompt Claude to generate a DuckDB-compatible SQL query
- Execute query, return results as DataFrame

---

## Prompting Guidelines

### Insight Prompt (`prompts/insight_prompt.txt`)
```
You are a senior data analyst. Given the following data profile:

{profile_json}

Generate 5–8 concise, specific insights. For each insight:
- Give it a short title
- Explain what the data shows (1–2 sentences)
- Flag any data quality issues
- Suggest a follow-up question

Respond in JSON: [{"title": "...", "insight": "...", "followup": "..."}]
```

### SQL Prompt (`prompts/sql_prompt.txt`)
```
You have a DuckDB table named `data` with the following schema:

{schema}

User question: "{question}"

Write a single valid DuckDB SQL query to answer this question.
Return ONLY the SQL query, no explanation.
```

---

## Agentic Loop

```
User uploads CSV
       │
       ▼
   profiler.py        ← pandas profiling
       │
       ▼
   analyst.py         ← Claude generates insights
       │
       ▼
  visualizer.py       ← charts generated
       │
       ▼
  summarize output    ← Claude writes narrative
       │
       ▼
  [optional] NL→SQL   ← user asks follow-up question
```

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-...        # Required
SANDBOX_TIMEOUT=30              # Seconds before code execution is killed
MAX_CSV_ROWS=100000             # Row limit to avoid context overflow
CHART_OUTPUT_DIR=/tmp/charts/   # Where charts are saved
```

---

## Code Style & Conventions

- Python 3.11+
- Type hints on all function signatures
- Docstrings on all public methods
- Keep LLM prompts in `/prompts/` — never hardcode in Python files
- All sandbox-executed code must be logged before execution
- Never pass raw user input directly to `exec()` or `eval()`

---

## Safety & Sandboxing Rules

- All LLM-generated code runs in a subprocess, never `exec()` in main process
- Sandbox has no network access
- Sandbox write access limited to `/tmp/sandbox/` only
- Kill switch: terminate subprocess if runtime > `SANDBOX_TIMEOUT`
- Log all generated code before execution for auditability

---

## Testing

```bash
# Run all tests
pytest tests/

# Test with sample CSV
python app.py --csv sample_data/example.csv --mode cli
```

Test cases to cover:
- Empty CSV
- Single-column CSV
- All-null column
- Very wide CSV (100+ columns)
- Mixed dtypes in a single column
- NL→SQL with ambiguous question

---

## Stretch Goals (Prioritized)

1. **NL→SQL with DuckDB** — most impactful, relatively easy
2. **Multi-turn conversation** — user can ask follow-up questions about the data
3. **Automated report export** — PDF/HTML with charts + insights
4. **Schema inference for joins** — handle multiple uploaded CSVs
5. **Anomaly detection** — flag statistical outliers automatically

---

## Known Limitations

- Max CSV size: `MAX_CSV_ROWS` rows (configurable)
- Charts are static PNG unless Plotly is used
- NL→SQL may fail on ambiguous column names — prompt user to clarify
- No persistent memory between sessions (stateless by default)