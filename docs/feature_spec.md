# Feature Specification — Drone Security Analyst Agent

**Version**: 1.0  
**Date**: 2026-11-24  
**Status**: Approved for prototype

---

## 1. Problem Statement

Property owners who rely on a docked patrol drone accumulate hours of video footage every day. Manual review is impractical: a single 24-hour patrol produces thousands of frames. Important events (a vehicle entering twice, a person loitering at night, an intruder climbing a fence) are buried in routine footage and are only discovered after the fact.

The Drone Security Analyst Agent replaces manual review with automated, real-time analysis. It processes every frame as it arrives, extracts structured detections, fires deterministic alerts when rules are violated, and maintains a searchable history that security personnel can query in natural language.

---

## 2. Value to Property Owners

| Pain today | Value delivered |
|---|---|
| Reviewing hours of video manually | Automated frame-by-frame analysis; only exceptions surfaced |
| Missing suspicious patterns (loitering, repeat entries) | Stateful entity tracker catches cross-frame patterns |
| No searchable record ("when did that truck come in?") | Full-text + semantic search over every indexed frame |
| Alert fatigue from noisy systems | Deterministic rule engine with cooldown deduplication |
| No daily summary | One-sentence video summary + hierarchical context summaries |

---

## 3. Key Requirements

### KR-1 — Real-Time Detection and Alerting

The system **shall** process each incoming frame and its paired telemetry, extract structured detections (object label, subtype, color, activity), and emit any triggered alerts **within the same processing tick** as the frame is ingested.

**Acceptance criteria**:
- A "Person loitering at Main Gate, 00:01." alert fires no later than the third consecutive person-at-gate frame at night.
- A "critical" perimeter-breach alert fires on the same frame that contains a climbing/jumping activity at a fence location.
- Alerts are written to the console (rich-formatted), to `alerts.log`, and to the SQLite `alerts` table.

### KR-2 — Contextual Memory and Entity Tracking

The system **shall** track unique entities (vehicles, persons) across frames and maintain a running history: first seen, last seen, locations visited, entry/exit counts, and all associated frame IDs.

**Acceptance criteria**:
- A Blue Ford F150 that enters the gate at 08:30, parks all day, and re-enters at 17:45 is recognised as the same entity, with `entry_count == 2`.
- The agent correctly answers "A blue Ford F150 entered twice today" when asked.
- Context summaries grow in O(1) prompt size as frame count grows (hierarchical roll-up), not O(n).

### KR-3 — Searchable Frame Index

The system **shall** store every frame with its timestamp, location, detected objects, and a semantic embedding in a dual-store index (SQLite for exact queries, ChromaDB for semantic search).

**Acceptance criteria**:
- `query_by_time(start="08:00", end="09:00")` returns all frames in that window with correct frame IDs.
- `query_by_object(label="vehicle", subtype="truck")` returns every truck frame and no non-truck frames.
- `semantic_search("vehicle at night")` returns the after-hours delivery-van frame in the top-3 results.
- Hybrid search (SQL filter + vector ranking) outperforms pure-semantic on structured queries.

### KR-4 (Optional) — Explainable Alerts

Every alert **should** carry the IDs of the evidence frames that triggered it and the name of the rule that fired, so a security officer can inspect the raw footage and understand why the alert was raised.

---

## 4. Out of Scope (prototype)

- Real video ingestion (OpenCV + VLM) — the prototype uses text-simulated frames.
- Multi-property or multi-drone coordination.
- Cloud deployment or webhook integrations.
- Mobile app or web dashboard (CLI + rich console is the UI).

Future-work items are tracked in `.kiro/steering/future.md`.

---

## 5. Assumptions

1. Frames arrive in roughly chronological order; minor out-of-order timestamps are tolerated.
2. Telemetry and frames share the same timestamp and are paired by the ingestion layer.
3. The property has a fixed set of named locations (Main Gate, Garage, Perimeter North, Back Fence, etc.).
4. "Night hours" are configurable; the prototype defaults to 22:00–05:00.
5. A vehicle is "known" if its make/model is in the allow-list in `config.py`; unknown vehicles generate an info alert.
