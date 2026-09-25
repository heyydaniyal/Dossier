# Dossier

Multi-Agent AI for Anti-Money-Laundering Alert Investigation — NOVA IMS capstone
(Daniyal Ahmad, Joana Martins, Maiara Almada, Mariana Martins).

Rule-based alerts arrive in a queue; a multi-agent system investigates each alert and hands a cited
case file and a recommendation to a human analyst, who decides. Decision support only: the system
never files SARs, freezes accounts or sets detection thresholds.

**Source of truth for decisions, freezes and results: `PROJECT_STATE.md` (in the Claude project).**

## Setup
```bash
# requires uv (https://docs.astral.sh/uv/) - installs Python 3.11 deps from uv.lock
uv sync
uv run pre-commit install
cp .env.example .env        # fill in keys; .env is gitignored
uv run pytest               # must pass before any merge
uv run streamlit run src/app/streamlit_app.py
uv run python scripts/cost_model.py
```

## Layout
| Path | Purpose |
|---|---|
| `src/contracts` | Shared pydantic interface contracts — **FROZEN v1** |
| `src/data` `src/features` `src/models` | data loading, point-in-time features, detection model + calibration |
| `src/tools` `src/agents` `src/rag` | agent runtime (never imports `src/eval`; enforced by test) |
| `src/eval` | evaluation harness; the only code allowed to read ground truth |
| `src/app` | Streamlit app |
| `prompts/` `configs/` | versioned prompts; YAML configs (ablations = config changes) |
| `tests/` | pytest; runs on every commit via pre-commit |
| `notebooks/` | exploration only, never the source of truth |
| `docs/` | professor brief, related work, ethics statement, team seams, deploy guide, contracts changelog |

## Non-negotiables (see project instructions for the full list)
Point-in-time correctness (`event_ts < as_of`, ties excluded) · runtime never sees labels/patterns ·
temporal splits with embargo · LLMs never do arithmetic · every call logged · hard budgets ·
secrets never committed.
