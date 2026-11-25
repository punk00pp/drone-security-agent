# Drone Security Analyst Agent

An AI-powered prototype that processes simulated drone patrol footage, extracts structured detections, fires deterministic security alerts, indexes every frame for search, and answers natural-language questions via a LangChain agent.

---

## Features

- **Frame analysis**: rule-based (offline, no API key) and LLM-backed (structured JSON output) analyzers
- **Deterministic alert rules**: loitering (night), perimeter breach (critical), after-hours vehicle, repeated entry, altitude anomaly, unknown vehicle
- **Dual-store index**: SQLite for exact queries + ChromaDB for semantic/vector search
- **Hybrid search**: SQL pre-filter + vector re-rank
- **Entity tracker**: normalises detections across frames ("Blue Ford F150 entered twice today")
- **Hierarchical summaries**: bounded agent prompt size regardless of session length
- **LangChain Q&A agent**: answers questions with frame citations and security recommendations
- **One-sentence video summary** (bonus)
- **Follow-up Q&A with conversation memory** (bonus)
- Streamlit web UI with five tabs: Pipeline, Alerts, Search, Entities, Agent Chat

---

## Architecture

```
Telemetry Simulator ──┐
                      ├──→ Ingest / Sync ──→ Analyzer ──→ SQLite + ChromaDB index
Frame Source ─────────┘                              ├──→ Context Manager (entity tracker, summaries)
                                                     └──→ Alert Rules Engine ──→ Alert Sink

                                          LangChain Query Agent (reads index + context)
```

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and design decisions.
Architecture diagram: [`docs/architecture.png`](docs/architecture.png)

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Text-simulated frames | Deterministic, free, testable without a camera or VLM |
| Rules engine in plain Python | Alerts must be auditable; LLM explains, doesn't decide |
| SQLite | Zero setup, file-portable, standard library |
| ChromaDB | Persistent, metadata filtering, good LangChain integration |
| Nomic `nomic-embed-text-v1.5` embeddings | API-based, no large local model download, Matryoshka model with separate task types for indexing vs querying |
| Entity key normalisation | Cross-frame re-ID without an ML tracking model |
| Hierarchical summaries | Bounded agent prompt size as session grows |
| Groq (Qwen via OpenAI-compatible API) | Fast inference, free tier, single swap point to change providers |

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
# Edit .env — set your API key
```

**`.env` example:**
```
API_KEY=gsk_...          # Groq API key (default provider)
LLM_PROVIDER=openai      # uses OpenAI-compatible endpoint
DEFAULT_ANALYZER=rule
```

**Nomic authentication** (required for semantic search — one-time setup):
```bash
nomic login <your-token>
# Get your token at https://atlas.nomic.ai → Settings → API Keys
```

---

## Running

### Streamlit UI (recommended for demos)
```bash
streamlit run app.py
```

### CLI — pipeline run
```bash
# Rule-based analyzer (no API key needed)
python -m drone_agent run --scenario data/scenario_day1.json --analyzer rule

# LLM analyzer
python -m drone_agent run --scenario data/scenario_day1.json --analyzer llm
```

### CLI — ask a question
```bash
python -m drone_agent ask "show all truck events"
python -m drone_agent ask "what happened at the main gate last night?"
python -m drone_agent ask "were there any critical alerts?"
```

### CLI — video summary
```bash
python -m drone_agent summary
```

### CLI — interactive chat
```bash
python -m drone_agent chat
```

### Run tests
```bash
pytest tests/ -v
```

---

## Scenarios

| File | Description |
|---|---|
| `scenario_day1.json` | 24-hour residential patrol — loitering, F150 entering twice, perimeter breach, after-hours van, cat false-positive |
| `scenario_day2.json` | Industrial facility — Mahindra Bolero tracking, restricted zone, coordinated breach, stray dog false-positive |
| `scenario_day3.json` | Residential compound night — two-person simultaneous breach, vehicle loitering + return, fox and cat false-positives |
| `scenario_day4.json` | Commercial campus day shift — altitude anomaly, rooftop climbing, tailgating, daytime crowd, pigeon false-positive |

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
│   ├── feature_spec.md         # KR-1/KR-2/KR-3 requirements
│   ├── architecture.md         # Mermaid diagram + design decisions
│   ├── architecture.png        # Architecture diagram (PNG)
│   ├── testing.md              # Test plan, dynamic/emergency scenarios
│   ├── ai_usage_log.md         # Prompts, what AI generated, what changed
│   └── report.md               # Full report
├── data/
│   ├── scenario_day1.json      # Residential patrol (20 frames)
│   ├── scenario_day2.json      # Industrial facility (20 frames)
│   ├── scenario_day3.json      # Residential night incident (20 frames)
│   └── scenario_day4.json      # Commercial campus day shift (20 frames)
├── drone_agent/
│   ├── config.py               # all thresholds + paths
│   ├── models.py               # Pydantic: Telemetry, Frame, Detection, Alert
│   ├── simulator.py            # scenario loader + stress generator
│   ├── analyzer.py             # RuleBasedAnalyzer + LLMAnalyzer
│   ├── pipeline.py             # ContextManager, AlertRulesEngine, run_pipeline
│   ├── storage.py              # SQLiteStore, VectorStore (Nomic), SearchEngine
│   ├── agent.py                # LangChain tool-calling agent
│   └── cli.py                  # CLI: run / ask / summary / chat
└── tests/
    ├── test_core.py            # Analyzer + alert rules + context manager
    ├── test_e2e.py             # Full pipeline on scenario_day1.json
    └── test_storage.py         # SQLite insert/query tests
```

---

## Testing

41 tests, all run without an API key (uses `RuleBasedAnalyzer`).

```bash
pytest tests/ -v
# 41 passed, 3 skipped
```

Key test cases:
- `test_truck_logged` — correct detections for "Blue Ford F150 parked at garage"
- `test_loitering_midnight` — 3 night frames fires warning; same frames at noon fires nothing
- `test_perimeter_breach_critical` — climbing fires critical; cat climbing does not
- `test_entity_entered_twice` — F150 entry_count == 2 after two entering frames
- `test_all_frames_indexed` — all 20 frames in SQLite after pipeline run

See [`docs/testing.md`](docs/testing.md) for the full test plan.

---

## AI Tools Used

Built with **Kiro (Claude)** as the primary AI coding assistant. See [`docs/ai_usage_log.md`](docs/ai_usage_log.md) for the complete log.

Key AI contributions:
- Architecture design (pipeline shape, dual-store index, hierarchical summaries)
- Alert rules engine (all rules; night-window wraparound bug caught and fixed manually)
- LLM analyzer JSON prompt and fallback logic
- LangChain agent tool definitions
- Full pytest test suite (edge cases added manually)

---

## Limitations and Future Work

1. **Real-video mode** — OpenCV frame sampling + BLIP/CLIP captioning; the `FrameSource` interface is designed for this
2. **Analyzer comparison experiment** — benchmark rule-based vs LLM accuracy on ambiguous frames
3. **Daily report generation** — structured end-of-day security brief from hierarchical summaries
4. **Natural-language alert rules** — LLM compiles plain-English rules into config entries
5. **Streamlit dashboard improvements** — live map view, alert timeline chart

---

## Links

- **Demo videos**: *(add Google Drive links)*
- **Report**: [`docs/report.md`](docs/report.md)
- **Repo**: private — `assignments@flytbase.com` added as contributor
