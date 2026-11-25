# Drone Security Analyst Agent

An AI-powered prototype that processes simulated drone patrol footage, extracts structured detections, fires deterministic security alerts, indexes every frame for search, and answers natural-language questions via a LangChain agent.

---

## Features

- **Frame analysis**: rule-based (offline, no API key) and LLM-backed (Groq/Qwen) analyzers
- **Deterministic alert rules**: loitering (night), perimeter breach (critical), after-hours vehicle, repeated entry, altitude anomaly, unknown vehicle
- **Persistent storage**: PostgreSQL — data survives page refreshes, restarts, and redeployments. Only cleared by the Reset button.
- **Dual-store index**: PostgreSQL for exact queries + ChromaDB for semantic/vector search
- **Nomic embeddings**: `nomic-embed-text-v1.5` via API — separate task types for indexing vs querying
- **Hybrid search**: SQL pre-filter + vector re-rank
- **Entity tracker**: normalises detections across frames ("Blue Ford F150 entered twice today")
- **Hierarchical summaries**: bounded agent prompt size regardless of session length
- **LangChain Q&A agent**: answers questions with frame citations and security recommendations
- **One-sentence video summary** (bonus)
- **Follow-up Q&A with conversation memory** (bonus)
- **Streamlit UI**: five tabs — Pipeline, Alerts, Search, Entities, Agent Chat

---

## Architecture

```
Telemetry Simulator ──┐
                      ├──→ Ingest / Sync ──→ Analyzer ──→ PostgreSQL + ChromaDB index
Frame Source ─────────┘                              ├──→ Context Manager (entity tracker, summaries)
                                                     └──→ Alert Rules Engine ──→ Alert Sink

                                          LangChain Query Agent (reads index + context)
```

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and design decisions.
Architecture diagram: [`docs/architecture.png`](docs/architecture.png)

---

## Setup

**Requirements**: Python 3.10+

```bash
# 1. Clone the repository
git clone <repo-url>
cd drone-security-agent

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env — fill in all three keys (see below)
```

### Required environment variables

| Variable | What it is | Where to get it |
|---|---|---|
| `API_KEY` | Groq API key for LLM analyzer and agent | [console.groq.com](https://console.groq.com) — free |
| `DATABASE_URL` | PostgreSQL connection string | [neon.tech](https://neon.tech) or [supabase.com](https://supabase.com) — free tier |
| `NOMIC_API_KEY` | Nomic API key for embeddings | [atlas.nomic.ai](https://atlas.nomic.ai) → Settings → API Keys |

**Nomic one-time login** (if not using `NOMIC_API_KEY` in `.env`):
```bash
nomic login <your-token>
```

### `.env` example
```
API_KEY=gsk_...
DATABASE_URL=postgresql://user:password@host:5432/dbname
NOMIC_API_KEY=nk-...
CHROMA_PATH=chroma_db
DEFAULT_ANALYZER=rule
```

> Without `DATABASE_URL`, the app falls back to a local SQLite file — data is lost on restart.
> Without `NOMIC_API_KEY` / `nomic login`, semantic search is disabled but all other features work.

---

## Running

### Streamlit UI
```bash
streamlit run app.py
```

### Tests
```bash
pytest tests/ -v
# 41 passed, 3 skipped — no API key or DATABASE_URL needed
```

---

## Scenarios

| File | Description |
|---|---|
| `scenario_day1.json` | Residential — F150 enters twice, midnight loitering, back fence breach, delivery van, cat false-positive |
| `scenario_day2.json` | Industrial facility — Mahindra Bolero tracking, restricted zone, stray dog false-positive |
| `scenario_day3.json` | Residential night — coordinated two-person breach, vehicle loiters and returns, fox/cat false-positives |
| `scenario_day4.json` | Commercial campus — altitude anomaly, rooftop climb, tailgating, pigeon false-positive |

---

## Project Structure

```
drone-security-agent/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py                      # Streamlit UI (5 tabs)
├── docs/
│   ├── feature_spec.md
│   ├── architecture.md
│   ├── architecture.png
│   ├── testing.md
│   ├── ai_usage_log.md
│   └── report.md
├── data/
│   ├── scenario_day1.json
│   ├── scenario_day2.json
│   ├── scenario_day3.json
│   └── scenario_day4.json
├── drone_agent/
│   ├── config.py               # thresholds, paths, DATABASE_URL, API keys
│   ├── models.py               # Pydantic: Telemetry, Frame, Detection, Alert
│   ├── simulator.py            # scenario loader + stress generator
│   ├── analyzer.py             # RuleBasedAnalyzer + LLMAnalyzer (Groq/Qwen)
│   ├── pipeline.py             # ContextManager, AlertRulesEngine, run_pipeline
│   ├── storage.py              # PostgresStore, SQLiteStore (tests), VectorStore, SearchEngine
│   └── agent.py                # LangChain tool-calling agent (6 tools)
└── tests/
    ├── test_core.py            # Analyzer + alert rules + context manager
    ├── test_e2e.py             # Full pipeline on scenario_day1.json
    └── test_storage.py         # SQLite insert/query tests (no DB needed)
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **PostgreSQL over SQLite** | Data persists across sessions, restarts, and Streamlit Cloud redeployments. `clear()` is the only deletion path — called only by the Reset button. |
| **SQLite kept for tests** | Tests run in CI with no database connection needed. |
| **Rules engine in plain Python** | Alerts are deterministic and auditable. The LLM explains; it does not decide. |
| **Nomic `nomic-embed-text-v1.5`** | API-based (no 500 MB local download). Separate `task_type` for indexing vs querying improves recall. |
| **ChromaDB** | Persistent by default; falls back to in-memory on read-only filesystems (Streamlit Cloud). |
| **Entity key normalisation** | Cross-frame vehicle re-ID without an ML tracking model. |
| **Hierarchical summaries** | Keeps agent prompt size O(log n) as session grows. |
| **LangGraph agent** | LangChain 1.x API. Conversation memory via full message list pass-through on each call. |

---

## Testing

41 tests, no API key or database connection required.

```bash
pytest tests/ -v
# 41 passed, 3 skipped
```

See [`docs/testing.md`](docs/testing.md) for the full test plan.

---

## Deploying to Streamlit Cloud

1. Push repo to GitHub (private)
2. Connect repo in [share.streamlit.io](https://share.streamlit.io)
3. Add secrets in **App settings → Secrets**:
```toml
API_KEY = "gsk_..."
DATABASE_URL = "postgresql://..."
NOMIC_API_KEY = "nk-..."
DEFAULT_ANALYZER = "rule"
```
4. The app auto-detects the read-only filesystem and falls back to in-memory ChromaDB. PostgreSQL data persists across redeployments.

---

## AI Tools Used

Built with **Kiro (Claude)**. See [`docs/ai_usage_log.md`](docs/ai_usage_log.md) for the full log.

---

## Links

- **Demo videos**: *(add Google Drive links)*
- **Report**: [`docs/report.md`](docs/report.md)
- **Repo**: private — `assignments@flytbase.com` added as contributor
