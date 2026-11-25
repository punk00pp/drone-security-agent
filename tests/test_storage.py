"""
test_storage.py — SQLiteStore and SearchEngine tests.
No LLM or vector store required.
"""

from __future__ import annotations

import pytest

from drone_agent.analyzer import RuleBasedAnalyzer
from drone_agent.models import Alert, Frame, Telemetry
from drone_agent.storage import SQLiteStore, SearchEngine
from datetime import datetime


def _rba():
    return RuleBasedAnalyzer()


def _frame(fid: int, desc: str, hour: int) -> Frame:
    return Frame(frame_id=fid, timestamp=datetime(2024, 1, 15, hour, 0), description=desc)


def _tel(loc: str, hour: int) -> Telemetry:
    return Telemetry(timestamp=datetime(2024, 1, 15, hour, 0), location=loc,
                     altitude_m=20.0)


@pytest.fixture
def db(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    rba = _rba()
    for fid, desc, hour, loc in [
        (1, "Blue Ford F150 parked at garage", 12, "Garage"),
        (2, "Person standing near the gate", 0, "Main Gate"),
        (3, "Delivery van parked at garage", 23, "Garage"),
        (4, "Cat walking across the driveway", 23, "Main Gate"),
        (5, "Unknown person climbing over the back fence", 21, "Back Fence"),
    ]:
        store.insert_analyzed_frame(rba.analyze(_frame(fid, desc, hour), _tel(loc, hour)))
    yield store
    store.close()


def test_query_by_time(db):
    results = db.query_by_time("2024-01-15T08:00:00", "2024-01-15T14:00:00")
    ids = {r["frame_id"] for r in results}
    assert 1 in ids   # frame at 12:00
    assert 2 not in ids  # frame at 00:01


def test_query_by_object_truck(db):
    results = db.query_by_object(label="vehicle", subtype="truck")
    ids = {r["frame_id"] for r in results}
    assert 1 in ids
    assert 2 not in ids  # person
    assert 4 not in ids  # cat


def test_query_by_object_all_vehicles(db):
    results = db.query_by_object(label="vehicle")
    ids = {r["frame_id"] for r in results}
    assert 1 in ids and 3 in ids


def test_query_by_location(db):
    results = db.query_by_location("Main Gate")
    ids = {r["frame_id"] for r in results}
    assert 2 in ids and 4 in ids
    assert 1 not in ids  # Garage


def test_frame_count(db):
    assert db.frame_count() == 5


def test_alert_round_trip(db):
    alert = Alert(alert_id="t-001", rule="LoiteringRule", severity="warning",
                  message="Person loitering at Main Gate, 00:01.",
                  timestamp=datetime(2024, 1, 15, 0, 1), evidence_frame_ids=[2])
    db.insert_alert(alert)
    alerts = db.get_all_alerts()
    r = next(a for a in alerts if a["alert_id"] == "t-001")
    assert r["severity"] == "warning" and r["evidence_frame_ids"] == [2]


def test_search_engine_by_object(db):
    engine = SearchEngine(db, vector=None)
    results = engine.by_object(label="vehicle", subtype="truck")
    assert any(r["frame_id"] == 1 for r in results)


def test_out_of_order_frames(tmp_path):
    db = SQLiteStore(tmp_path / "oo.db")
    rba = _rba()
    for fid, hour in [(10, 15), (9, 8)]:  # inserted out of order
        db.insert_analyzed_frame(rba.analyze(_frame(fid, "Empty driveway", hour), _tel("Main Gate", hour)))
    results = db.query_by_time("2024-01-15T00:00:00", "2024-01-15T23:59:59")
    timestamps = [r["timestamp"] for r in results]
    assert timestamps == sorted(timestamps)
    db.close()
