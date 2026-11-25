# Report — Drone Security Analyst Agent
**FlytBase AI Engineer Assignment**

---

## Approach

Built a prototype that processes simulated drone telemetry (time, location, altitude) paired with text frame descriptions, analyzes each frame for objects and activities, fires deterministic Python alert rules, indexes every frame in SQLite and ChromaDB, and exposes a LangChain conversational agent. Both bonus features are implemented: one-sentence video summary and multi-turn follow-up Q&A.

The UI is a Streamlit app with five tabs: Pipeline, Alerts, Search, Entities, Agent Chat.

---

## Assumptions

- Frames are simulated as text — deterministic, free, no camera or VLM required
- Telemetry contains time, location, and altitude only (battery excluded — not in the assignment)
- Night hours are 22:00–05:00, configurable in `config.py`
- Known vehicles are hard-coded in an allow-list; unknowns trigger an info alert
- LLM provider is Groq (free tier, OpenAI-compatible); swap by changing `.env` only

---

## Tool and Config Justification

| Choice | Why |
|---|---|
| **SQLite** | Zero setup, no server, standard library. Covers all required queries: time, object, location. |
| **ChromaDB + Nomic `nomic-embed-text-v1.5`** | Enables natural-language queries ("vehicle at night") that SQL cannot express. Nomic uses separate `task_type` for indexing vs querying, improving recall. Falls back to in-memory on read-only filesystems (Streamlit Cloud). |
| **Rule-based analyzer** | Deterministic, offline, no API key. Used in all 41 tests. LLMAnalyzer falls back to it on any error. |
| **LangChain + LangGraph** | Assignment names LangChain. LangGraph compiled graph (LangChain 1.x) replaced `AgentExecutor`. Conversation memory via full message list pass-through. |
| **Groq / Qwen** | Fast inference, free tier. Single swap point to change providers. |
| **Pydantic v2** | Validates LLM JSON responses and scenario files at every boundary. |

---

## Results

**Sample log (scenario_day1.json, rule-based analyzer):**
```
[00:01] Main Gate — Person standing at Main Gate, 00:01.
  !! ALERT [WARNING]   Person loitering at Main Gate, 00:04.
[12:00] Garage    — Blue ford f150 truck parked at Garage, 12:00.
  !! ALERT [INFO]      Blue Ford F150 Truck entered 2 times today at Main Gate, 17:45.
  !! ALERT [CRITICAL]  Person breaching perimeter at Back Fence, 21:00.
  !! ALERT [WARNING]   Delivery van detected after hours at Garage, 23:30.
```

**Agent Q&A:**
- *"Show all truck events"* → returns all 7 truck frames with frame IDs and timestamps
- *"Did the F150 enter more than once?"* → "Yes. entry_count=2. First at 08:30, again at 17:45."
- *"Were there false positives?"* → "Frame 20 — cat at night. No alert fired; labeled animal, not person."

**Video summary (bonus):**
> A Blue Ford F150 entered twice today; a person loitered at the main gate near midnight; an intruder breached the back fence at 21:00; a delivery van appeared after hours at 23:30.

**Tests:** 41 passed, 3 skipped, 0 failures. Run with `pytest tests/ -v`.

---

## What Could Be Better

- **Analyzer comparison** — benchmark rule-based vs LLM accuracy on ambiguous frames with a precision/recall table
- **Real-video mode** — OpenCV frame sampling + BLIP captioning behind the existing `FrameSource` interface
- **Scalability numbers** — 1k/10k/50k frame timing table showing per-frame latency and bounded context prompt size
- **Daily report generation** — structured end-of-day brief from hierarchical summaries
- **Natural-language alert rules** — LLM compiles plain-English rules ("alert if person at gate after 10pm for 2 min") into config

---

## How AI Tools Assisted

Built with **Kiro (Claude)** throughout. Full log in `docs/ai_usage_log.md`.

| What AI generated | What I changed |
|---|---|
| Pipeline architecture, dual-store index, hierarchical summaries | Dropped Kafka; added entity-key normalisation (AI did not suggest this) |
| All 6 alert rule classes | Fixed night-window wraparound bug; removed out-of-scope low battery rule |
| LLM analyzer JSON prompt | Added RuleBasedAnalyzer fallback on any parse/API error |
| LangChain agent tools and system prompt | Removed `return_direct=True` — it bypassed multi-turn reasoning |
| ~40 pytest test functions | Added animal-at-fence edge case; removed battery tests after removing the feature |

**Bugs AI introduced:** night-window logic was `22 <= hour <= 5` (always false) — caught by a failing test. ChromaDB v0.4 `Client()` API — caught at runtime. Duplicate alerts on re-run from UUID-keyed inserts — caught during demo, fixed with `clear()` before each run.
