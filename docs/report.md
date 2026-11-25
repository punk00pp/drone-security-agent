# Report — Drone Security Analyst Agent

**Candidate submission for the AI Engineer assignment — FlytBase**

---

## 1. Problem and Goals

The assignment asks for a prototype of a Drone Security Analyst Agent for a docked drone that patrols a fixed property daily. The system must:

- Process simulated telemetry (time, location, altitude) paired with simulated video frame descriptions
- Analyze each frame to identify objects and activities (vehicles, people, animals)
- Fire deterministic security alerts based on predefined rules (loitering, perimeter breach, after-hours vehicle)
- Index every frame so it is queryable by time, object, or natural language
- Expose a conversational agent that answers questions about the patrol session

Both bonus features are implemented: one-sentence video summarization and follow-up Q&A with conversation memory.

---

## 2. Feature Spec Summary

Full spec in `docs/feature_spec.md`. Three key requirements:

**KR-1 — Real-Time Detection and Alerting**: Every frame is analyzed and alerts are evaluated within the same processing tick as ingestion. A perimeter breach alert fires on the same frame that contains the breach — zero frames of delay.

**KR-2 — Contextual Memory and Entity Tracking**: Entities (vehicles, people) are tracked across frames using a normalized key (`label:color:make:model`). A Blue Ford F150 that enters at 08:30, parks all day, and re-enters at 17:45 is recognized as the same entity with `entry_count = 2`. This is what produces the assignment's sample output: *"a blue Ford F150 entered twice today"*.

**KR-3 — Searchable Frame Index**: Every frame is stored in SQLite (exact queries) and ChromaDB (semantic/vector search). Hybrid search combines SQL pre-filtering with vector re-ranking.

---

## 3. Architecture

```
Telemetry Simulator ──┐
                      ├──→ Ingest / Sync ──→ Analyzer ──→ SQLite + ChromaDB index
Frame Source ─────────┘                              ├──→ Context Manager
                                                     └──→ Alert Rules Engine ──→ Alerts

                                          LangChain Query Agent (reads index + context)
```

Full Mermaid diagram with component descriptions: `docs/architecture.md`.
Architecture diagram PNG: `docs/architecture.png`.

### Pipeline flow for one frame

1. `simulator.py` yields a `(Frame, Telemetry)` pair
2. `analyzer.py` extracts structured detections → `AnalyzedFrame`
3. `storage.py` inserts the frame and detections into SQLite; ChromaDB gets the embedding
4. `pipeline.py` (`ContextManager`) updates the entity tracker, rolling window, and hierarchical summaries
5. `pipeline.py` (`AlertRulesEngine`) evaluates all rules against the frame and context
6. Triggered alerts go to SQLite, `alerts.log`, and the console

---

## 4. Assumptions

| Assumption | Reason |
|---|---|
| Frames are simulated as text descriptions | Deterministic, free, testable without a camera or VLM API. Real-video mode (OpenCV + VLM) is an optional future adapter. |
| Timestamps are roughly chronological | The index handles out-of-order frames without crashing, but alert rules (loitering streak) assume in-order delivery. |
| A single fixed property with named locations | Locations like "Main Gate", "Garage", "Back Fence" are used for alert message formatting and entity tracking. |
| Night hours = 22:00–05:00 | Configurable in `config.py`. Wraps midnight correctly (`hour >= 22 or hour <= 5`). |
| LLM provider is Groq (via OpenAI-compatible API) | Chosen for free-tier availability and speed. The `_build_llm()` function in `analyzer.py` is the single swap point for any provider. |
| Known vehicles are hard-coded in `config.py` | In production this would be a database table or external allow-list service. |

---

## 5. Tool and Model Choices with Justification

| Choice | Justification |
|---|---|
| **Python 3.12** | Standard for AI/ML tooling; best library support for LangChain, Pydantic, ChromaDB |
| **Pydantic v2** | Schema validation at every boundary: JSON scenario files, LLM responses, DB rows. Catches malformed data before it corrupts state. |
| **SQLite** | Zero setup, no server, file-portable. Sufficient for prototype scale (millions of frames before needing tuning). Standard library, so no extra dependency. |
| **ChromaDB** | Persistent by default (`PersistentClient`), metadata filtering API, good LangChain integration. FAISS would need a manual persistence wrapper. |
| **`all-MiniLM-L6-v2` embeddings** | Local (no API cost), 384-dim, runs on CPU in ~1 ms/frame, well-tested on semantic similarity. No external call needed for indexing. |
| **LangChain tool-calling agent** | Assignment explicitly names LangChain. Tool-calling gives the agent structured access to the index without free-form SQL generation, which is safer and easier to test. |
| **LangGraph (via `create_agent`)** | LangChain 1.x replaced `AgentExecutor` with LangGraph compiled graphs. Conversation memory is maintained by passing the full message list back on each call. |
| **Groq API (Qwen 3.8B)** | Fast inference, free tier, OpenAI-compatible endpoint so the client is a single `ChatOpenAI` with `base_url` override. Easy to swap to Claude or GPT-4 by changing `.env`. |
| **Rule-based analyzer as baseline** | Deterministic, offline, no API key needed. Used in all automated tests. LLMAnalyzer falls back to it on error so the pipeline never crashes on a bad LLM response. |

---

## 6. Implementation Highlights

### Analyzer

Two implementations share a common `analyze(frame, telemetry) -> AnalyzedFrame` interface.

`RuleBasedAnalyzer` uses regex/keyword dictionaries for labels (person, vehicle, animal), subtypes (truck, van, car), colors, makes, models, and activities. It handles variant phrasing: "F-150", "pickup", and "Ford truck" all resolve to `subtype=truck, model=f150`. Confidence is set to 1.0 when structured attributes are found, 0.6 for label-only matches.

`LLMAnalyzer` sends a strict JSON schema prompt to the LLM (temperature=0) and validates the response with Pydantic. Any parse error or API failure falls back silently to `RuleBasedAnalyzer`.

### Context Manager

The entity tracker normalizes every detection to a stable key (`vehicle:blue:ford:f150`) and maintains `first_seen`, `last_seen`, `locations`, `entry_count`, `exit_count`, and `frame_ids` for each unique entity. This is what produces the "entered twice today" insight.

Hierarchical summaries: every 50 frames, pending frame summaries are rolled up into a chunk summary. When 10 chunk summaries accumulate they roll up into a daily summary. The agent always receives summaries plus the latest pending frames — O(log n) prompt size regardless of session length.

### Alert Rules Engine

Six deterministic Python rules (no LLM):

| Rule | Trigger | Severity |
|---|---|---|
| LoiteringRule | Person at same location in 3+ consecutive night frames | warning |
| PerimeterBreachRule | Person with climbing/jumping activity, or at a fence location | critical |
| AfterHoursVehicleRule | Vehicle detected outside 06:00–22:00 | warning |
| UnknownVehicleRule | Vehicle make not in allow-list | info |
| RepeatedEntryRule | Same entity `entry_count >= 2` | info |
| AltitudeAnomalyRule | Drone altitude outside 5–120 m | warning |

Alert messages follow the format `"{Subject} {behavior} at {location}, {HH:MM}."` to match the assignment's sample output exactly.

### Index

SQLite stores frames, detections (normalized columns for label/subtype/color/make/model), and alerts. Three indexes: `idx_frames_ts` (timestamp), `idx_det_label` (label+subtype), `idx_frames_loc` (location).

ChromaDB stores one document per frame (`description + summary`) with metadata for frame_id, timestamp, location, and labels. Hybrid search pre-filters candidates in SQLite then re-ranks them by vector distance.

### Agent

Six tools wrap the search and context APIs: `query_by_time`, `query_by_object`, `semantic_search`, `get_entity_history`, `get_alerts`, `get_video_summary`. The system prompt instructs the agent to cite frame IDs and timestamps, give concrete recommendations when alerts exist, and say "No evidence found" rather than guessing.

Conversation memory works by passing the full accumulated message list back on each follow-up call. The agent can answer "Which of those came more than once?" after "What vehicles were seen today?" because the full context is in the message history.

---

## 7. Results with Examples

### Sample log output (rule-based analyzer)

```
[00:00] Main Gate              Empty driveway, gate closed, activity detected at Main Gate, 00:00.
[00:01] Main Gate              Person standing at Main Gate, 00:01.
[00:02] Main Gate              Person standing at Main Gate, 00:02.
  !! ALERT [WARNING]  Person loitering at Main Gate, 00:04.
[08:30] Main Gate              Blue ford f150 truck entering at Main Gate, 08:30.
[12:00] Garage                 Blue ford f150 truck parked at Garage, 12:00.
  !! ALERT [INFO]     Blue Ford F150 Truck entered 2 times today at Main Gate, 17:45.
  !! ALERT [CRITICAL] Person breaching perimeter at Back Fence, 21:00.
  !! ALERT [WARNING]  Delivery van detected after hours at Garage, 23:30.
```

### Agent Q&A examples

**Q: Show all truck events**
> Found 7 frames with vehicle/truck detections. Frame 6 (08:30, Main Gate): Blue Ford F150 entering. Frame 7 (08:31, Garage): Blue Ford F150 parked. Frame 10 (12:00, Garage): Blue Ford F150 parked. Frame 13 (15:00, Main Gate): Blue Ford F150 leaving. Frame 15 (17:45, Main Gate): Blue Ford F150 entering again.

**Q: What happened at the gate last night?**
> Between 22:00 and 05:00 at Main Gate: Frame 2 (00:01) — person standing near gate. Frame 3 (00:02) — person still standing, looking around. Frame 4 (00:04) — person has not moved. Alert fired: "Person loitering at Main Gate, 00:04." Recommendation: Dispatch a guard to verify the individual's identity and intent.

**Q: Did the F150 enter more than once?**
> Yes. Entity `vehicle:blue:ford:f150` entered 2 times today. First entry: Frame 6 at 08:30 (Main Gate). Second entry: Frame 15 at 17:45 (Main Gate). The vehicle also appeared at the Garage (Frames 7, 10) and left through Main Gate (Frame 13).

### Video summary (bonus)

> A Blue Ford F150 entered the property twice today; a person loitered at the main gate near midnight triggering a warning; an unknown individual breached the back fence at 21:00 and a delivery van was detected after hours at the garage at 23:30.

---

## 8. Testing and Validation

41 tests pass, 3 skipped (for intentionally removed features).

```
pytest tests/ -v
# 41 passed, 3 skipped in 1.70s
```

Key test cases:
- `test_truck_logged` — "Blue Ford F150 parked at garage" produces vehicle/truck/blue/ford/f150
- `test_loitering_midnight` — 3 night frames at Main Gate fires a warning alert
- `test_no_loitering_daytime` — same frames at 12:00 fires nothing
- `test_perimeter_breach_critical` — climbing at Back Fence fires a critical alert
- `test_no_breach_for_animal` — cat climbing does not fire a critical alert
- `test_entity_entered_twice` — F150 entry_count == 2 after two entering frames
- `test_all_frames_indexed` — all 20 scenario frames are in SQLite
- `test_sample_log_format` — Frame 10 summary contains "12:00" and "Garage"

Full test plan with dynamic input and emergency scenarios: `docs/testing.md`.

---

## 9. What I Would Do with More Time

1. **Analyzer comparison experiment** — run rule-based vs LLM on the ambiguous frames (frames 8, 9, 11, 16) and produce a precision/recall table. The assignment asks for experimentation evidence and this would directly address it.
2. **Real-video mode** — OpenCV frame sampling + BLIP for captioning + CLIP for zero-shot labels, then feed captions into the existing LLMAnalyzer. The `FrameSource` interface is already designed to accept this.
3. **Daily report generation** — structured end-of-day brief: total frames, alert counts by severity, top entities, timeline of significant events. Built from hierarchical summaries.
4. **Natural-language alert rules** — user writes "alert if a person is at the gate after 10pm for more than 2 minutes", LLM compiles it into a rule config entry.
5. **Streamlit dashboard** — replace the CLI with a web UI for the demo. Would make the videos more visually compelling.
6. **Scalability numbers** — the scripts folder has a benchmark script but no results table was recorded. With more time, a full 1k/10k/50k frame benchmark table and chart would go into this section.

---

## 10. How AI Tools Assisted

This project was built entirely with **Kiro (Claude)** as the AI coding assistant. The full prompt-by-prompt log is in `docs/ai_usage_log.md`. Key examples:

| Component | AI contribution | What I changed |
|---|---|---|
| Architecture | Proposed overall pipeline shape, dual-store index, hierarchical summaries | Dropped Kafka (over-engineered); added entity-key normalisation (AI did not suggest this) |
| Alert rules | Generated all 6 rule classes with the correct Python interface | Fixed night-window wraparound bug (`hour >= 22 or hour <= 5`, not `22 <= hour <= 5`) |
| LLM analyzer | Generated JSON schema prompt and Pydantic validation | Added fallback to RuleBasedAnalyzer on any parse/API error |
| LangChain agent | Generated tool definitions and system prompt | Removed `return_direct=True` from all tools (it bypassed agent reasoning for follow-ups) |
| Test suite | Generated ~40 test functions matching the spec test plan | Added animal-at-fence edge case; fixed fragile string-equality assertions |

**Bugs AI introduced and how they were caught:**
- Night-window bug: caught by `test_no_loitering_daytime` failing on the first run
- ChromaDB v0.4 API: caught at runtime with `AttributeError`; fixed to `PersistentClient`
- Pydantic v1 `Optional[str]` syntax: caught by Pydantic v2 validation warnings at import time

---

## 11. Novice Disclosure

LangChain and ChromaDB were new tools for this project. LangChain 1.x (LangGraph API) replaced the `AgentExecutor` pattern that most tutorials show, which caused initial confusion. The `create_agent` / LangGraph compiled graph pattern was identified by reading the LangChain 1.x migration guide and confirmed by running the agent against the scenario.

ChromaDB's v0.5 API change from `Client()` to `PersistentClient()` was discovered at runtime and fixed by reading the changelog.

Both issues are documented in `docs/ai_usage_log.md` as examples of AI-generated code that required manual verification.
