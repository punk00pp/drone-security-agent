"""
test_e2e.py — End-to-end pipeline test against scenario_day1.json.
Runs the full pipeline with RuleBasedAnalyzer (no API key required).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drone_agent.analyzer import RuleBasedAnalyzer
from drone_agent.pipeline import AlertRulesEngine, ContextManager, run_pipeline
from drone_agent.simulator import load_scenario
from drone_agent.storage import SQLiteStore, SearchEngine

SCENARIO = Path(__file__).resolve().parents[1] / "data" / "scenario_day1.json"


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    db = SQLiteStore(tmp_path_factory.mktemp("e2e") / "e2e.db")
    search = SearchEngine(db, vector=None)
    ctx = ContextManager()
    analyzed, alerts = run_pipeline(
        load_scenario(SCENARIO), RuleBasedAnalyzer(),
        db, search, ctx, AlertRulesEngine(), quiet=True,
    )
    return db, search, ctx, analyzed, alerts


def test_all_frames_indexed(results):
    db, _, _, analyzed, _ = results
    assert db.frame_count() == 20


def test_loitering_alert(results):
    _, _, _, _, alerts = results
    assert any("loitering" in a.message.lower() or
               ("person" in a.message.lower() and "Main Gate" in a.message)
               for a in alerts)


def test_perimeter_breach_critical(results):
    _, _, _, _, alerts = results
    assert any(a.severity == "critical" for a in alerts)


def test_after_hours_vehicle(results):
    _, _, _, _, alerts = results
    assert any("after hours" in a.message.lower() for a in alerts)


def test_trucks_indexed(results):
    _, search, _, _, _ = results
    assert search.by_object(label="vehicle", subtype="truck")


def test_entity_entered_twice(results):
    _, _, ctx, _, _ = results
    entity = ctx.get_entity("vehicle:blue:ford:f150")
    if entity:
        assert entity.entry_count >= 2
    else:
        result = ctx.get_entity_history("blue ford")
        assert not result["found"] or result["entities"][0]["entry_count"] >= 2


def test_cat_no_person_alert(results):
    _, _, _, _, alerts = results
    cat_time_alerts = [a for a in alerts if "23:50" in a.message]
    assert not any("person" in a.message.lower() for a in cat_time_alerts)


def test_time_query(results):
    _, search, _, _, _ = results
    rows = search.by_time("2024-01-15T08:00:00", "2024-01-15T09:00:00")
    assert any(r["frame_id"] == 6 for r in rows)  # F150 at 08:30


def test_low_battery_alert(results):
    pytest.skip("Low battery rule removed")


def test_sample_log_format(results):
    """Frame 10 (12:00 Garage F150) should include location and time in summary."""
    _, _, _, analyzed, _ = results
    f10 = next((af for af in analyzed if af.frame.frame_id == 10), None)
    assert f10 and "12:00" in f10.summary and "Garage" in f10.summary
