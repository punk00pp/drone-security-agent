# AI Usage Log

This file records every significant AI-assisted generation step: what was prompted, what was produced, and what the developer changed or verified. Required for the report (Section 10) and for academic honesty.

---

## Log Table

| # | Stage | Tool | Prompt (abbreviated) | What AI produced | What I changed / verified |
|---|---|---|---|---|---|
| 1 | Architecture | Kiro (Claude) | "Propose a pipeline for a drone security analyst agent — telemetry + text frame descriptions, deterministic alerts, LangChain agent. Justify each component." | Overall pipeline shape, dual-store index idea, hierarchical summary pattern | Dropped Kafka (over-engineered for prototype); added entity-key normalisation approach; simplified to synchronous loop |
| 2 | Feature spec | Kiro (Claude) | "Write a feature spec for the drone security agent with 3 key requirements matching the assignment criteria" | KR-1/KR-2/KR-3 structure, acceptance criteria wording | Added KR-4 explainability as optional; tightened acceptance criteria to match exact expected sample outputs |
| 3 | Data models | Kiro (Claude) | "Generate Pydantic v2 models for Telemetry, Frame, Detection, AnalyzedFrame, Alert as per spec §9" | Full models.py with validators | Verified union types use `\|` syntax (Pydantic v2); added `entity_key` computed property to Detection |
| 4 | Scenario data | Kiro (Claude) | "Generate scenario_day1.json covering the full timeline in §10, including noisy/ambiguous descriptions" | 20-frame scenario JSON with all required events | Verified every expected alert frame is present; added battery drain curve and altitude variation |
| 5 | RuleBasedAnalyzer | Kiro (Claude) | "Generate a keyword+regex rule-based analyzer matching the BaseAnalyzer interface, covering vehicles, persons, animals, colors, makes, activities" | Full rule_based.py with dictionaries | Fixed F-150 hyphen variant regex; added "pickup" → truck subtype mapping; tested against all scenario frames |
| 6 | Alert rules | Kiro (Claude) | "Generate Python alert rule classes for loitering (night window), perimeter breach (critical), after-hours vehicle, repeated entry, low battery, altitude anomaly, with cooldown deduplication" | alerts.py with rule interface and all 7 rules | Fixed night-window wraparound bug (22:00–05:00 crossing midnight); verified cooldown logic resets per rule+entity key |
| 7 | SQLite store | Kiro (Claude) | "Generate sqlite_store.py with the schema from §11.5, insert methods, and query methods for time/object/location" | Full sqlite_store.py | Added missing `evidence_frame_ids` JSON serialisation; added `get_alerts` with optional severity filter |
| 8 | Vector store | Kiro (Claude) | "Generate vector_store.py wrapping ChromaDB with sentence-transformers embeddings, upsert and semantic search" | vector_store.py | Verified ChromaDB v0.5 API (`.get_or_create_collection`); added metadata schema documentation |
| 9 | Context manager | Kiro (Claude) | "Generate context.py with entity tracker (normalised keys), rolling window, and hierarchical chunk+daily summaries" | context.py with EntityInfo, ContextManager | Verified entry/exit count logic; added `get_entity_history` returning structured dict for agent tools |
| 10 | LangChain agent | Kiro (Claude) | "Generate a LangChain tool-calling agent with 6 tools wrapping search.py and context.py, system prompt as security analyst, conversation memory" | agent.py with all tools | Verified tool schemas use `@tool` decorator correctly; tested that system prompt instructs agent not to hallucinate |
| 11 | Test suite | Kiro (Claude) | "Generate pytest test cases for all items in §12 test plan, using RuleBasedAnalyzer (no API key), mocking LLM calls" | tests/ with 6 files, ~40 test functions | Added edge cases: empty description, out-of-order timestamps, midnight wraparound; verified all pass with `pytest` |
| 12 | Stress data generator | Kiro (Claude) | "Generate generate_stress_data.py that produces N random but plausible frames with injected known events" | scripts/generate_stress_data.py | Verified injected events appear at predictable positions so benchmark.py can assert correctness |

---

## Errors AI Made and How They Were Caught

1. **Kafka suggestion (Architecture)**: Initial architecture proposal included an event bus. Rejected — adds operational complexity, no benefit at prototype scale.

2. **Night-window wraparound (Alert rules)**: The initial loitering rule used `22 <= hour <= 5`, which is never true. Fixed to `hour >= 22 or hour <= 5`.

3. **ChromaDB v0.4 API (Vector store)**: AI generated code using the deprecated `chromadb.Client()` constructor. Updated to `chromadb.PersistentClient(path=...)` per v0.5 API.

4. **Pydantic v1 syntax (Models)**: AI initially generated `Optional[str]` and `Union[str, None]` patterns. Updated to `str | None` syntax required by Pydantic v2.

5. **LangChain tool decorator (Agent)**: AI used `@tool` with `return_direct=True` on all tools, which bypasses the agent's reasoning for follow-up questions. Removed `return_direct` from tools that should be chained.

---

*Update this log each time AI assistance produces a significant code artefact.*
