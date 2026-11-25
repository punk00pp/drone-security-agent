# Testing Documentation

## Overview

The test suite uses `pytest` and the `RuleBasedAnalyzer` for all deterministic tests (no API key required). LLM-dependent paths are tested with mocked responses via `pytest-mock`. Every test maps directly to a row in the test plan from the assignment spec (§12).

---

## Running the Tests

```bash
# From the drone-security-agent/ directory
pip install -r requirements.txt
pytest tests/ -v
```

Expected output: all tests pass in < 10 seconds on a laptop (no network calls).

---

## Test Files

| File | What it covers |
|---|---|
| `tests/test_analyzer.py` | RuleBasedAnalyzer: detection extraction, variant phrasing, false-positive control |
| `tests/test_alerts.py` | Each alert rule individually: loitering, breach, after-hours, battery, altitude |
| `tests/test_index.py` | SQLite insert/query: time range, object label, location, alert round-trip |
| `tests/test_context.py` | Entity tracker: entry count, locations, rolling window, summaries |
| `tests/test_agent_tools.py` | Agent tools called against a real DB (no LLM) |
| `tests/test_pipeline_e2e.py` | Full pipeline on `scenario_day1.json` — asserts all expected events |

---

## Test Cases (mapping to spec §12)

| Test | File | Method | Verifies |
|---|---|---|---|
| `test_analyzer_truck_logged` | test_analyzer.py | `test_analyzer_truck_logged` | "Blue Ford F150 parked at garage" → vehicle/truck/blue/ford/f150 |
| `test_analyzer_variants` | test_analyzer.py | `test_analyzer_variants_pickup`, `test_analyzer_variants_f150_hyphen` | "pickup", "F-150" map to same entity |
| `test_analyzer_no_false_person` | test_analyzer.py | `test_analyzer_no_false_person` | "cat walking" → animal, NOT person |
| `test_loitering_alert_midnight` | test_alerts.py | `test_loitering_alert_midnight` | 3+ person frames at gate 00:01–00:04 → warning |
| `test_no_loitering_daytime` | test_alerts.py | `test_no_loitering_daytime` | Same pattern at 12:00 → no alert |
| `test_loitering_dedup` | test_alerts.py | `test_loitering_dedup` | Only one alert within cooldown window |
| `test_perimeter_breach_critical` | test_alerts.py | `test_perimeter_breach_critical` | Climbing fence → critical |
| `test_after_hours_vehicle` | test_alerts.py | `test_after_hours_vehicle` | Van at 23:30 → warning |
| `test_entity_entered_twice` | test_alerts.py | `test_entity_entered_twice` | F150 entry_count == 2 |
| `test_index_query_by_time` | test_index.py | `test_index_query_by_time` | Time range returns correct frame IDs |
| `test_index_query_by_object` | test_index.py | `test_index_query_by_object_truck` | "truck" returns all truck frames only |
| `test_agent_tools_return_correct_data` | test_agent_tools.py | multiple | Tool outputs match DB |
| `test_pipeline_e2e` | test_pipeline_e2e.py | multiple | Full scenario → expected logs and alerts |
| `test_malformed_input` | test_analyzer.py | `test_analyzer_malformed_empty_description`, `test_analyzer_garbled_description` | No crash on empty/garbled input |
| `test_out_of_order_frames` | test_index.py | `test_out_of_order_frames` | Timestamps handled correctly |
| `test_telemetry_anomaly` | test_alerts.py | `test_low_battery_alert`, `test_altitude_anomaly_too_high` | Battery/altitude alerts fire |

---

## Dynamic Input Scenarios

These are tested in `test_pipeline_e2e.py` against `scenario_day1.json`:

### Scenario A — Intruder Mid-Stream
Frame 17 injects an "unknown person climbing over the back fence" at 21:00.
Expected: a critical perimeter breach alert fires on the same frame processing tick (verified by `test_e2e_perimeter_breach_critical`).

### Scenario B — False Positive Control
Frame 20 is a "cat walking across driveway" at 23:50 (night time).
Expected: no person-related alert fires (verified by `test_e2e_cat_no_person_alert`).

### Scenario C — Vehicle Re-Entry
Frame 6 (08:30) and Frame 15 (17:45) both contain "Blue Ford F150 entering".
Expected: entity tracker records entry_count == 2; RepeatedEntryRule fires an info alert (verified by `test_e2e_entity_entered_twice`).

---

## Emergency Scenario (Latency Requirement)

The perimeter breach rule is stateless — it evaluates on a single frame with no rolling window dependency. This means the critical alert fires within the same processing tick as Frame 17, with zero frames of delay.

Verified in `test_perimeter_breach_critical`: the alert is returned immediately by `PerimeterBreachRule.evaluate()` on the first relevant frame.

---

## Scalability Test

See `scripts/benchmark.py` for the scalability test.

```bash
python scripts/benchmark.py --sizes 100 1000 10000
```

This measures:
- Per-frame ingestion latency (target: < 5 ms/frame for rule-based analyzer)
- Total pipeline time
- SQLite DB size
- Query latency (time, object, location)
- Context manager prompt size (should stay bounded, not grow linearly)

For the full 50k-frame test used in the demo video:
```bash
python scripts/generate_stress_data.py --n 50000 --out data/scenario_stress_50k.json
python -m drone_agent run --scenario data/scenario_stress_50k.json --analyzer rule
```

---

## Notes on LLM Tests

LLMAnalyzer tests are not included in the CI suite because they require an API key. To test LLMAnalyzer manually:

```bash
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY (or other provider key)
python -c "
from drone_agent.analyzer.llm_analyzer import LLMAnalyzer
from drone_agent.models import Frame, Telemetry
from datetime import datetime

ana = LLMAnalyzer()
frame = Frame(frame_id=1, timestamp=datetime(2024,1,15,12,0), description='Blue Ford F150 parked at garage')
tel = Telemetry(timestamp=datetime(2024,1,15,12,0), location='Garage', altitude_m=25.0)
af = ana.analyze(frame, tel)
print(af.detections)
print(af.summary)
"
```

---

## AI Assistance in Testing

The test suite structure and initial test cases were generated with Kiro (Claude). Key prompt:
> "Write pytest test cases for all items in the test plan in §12, using RuleBasedAnalyzer (no API key), mocking LLM calls where needed."

Changes made after generation:
- Added the `test_no_breach_for_animal` test (AI did not generate edge cases for animals at fence locations).
- Replaced a fragile string-equality assertion on `entity_key` with a substring check to handle order-of-parts variation.
- Added `test_e2e_sample_log_format` to verify the exact expected output format from the assignment spec.
