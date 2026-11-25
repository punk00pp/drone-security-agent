"""
test_core.py — Analyzer, alert rules, and context manager tests.
All deterministic — no API key or external services required.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from drone_agent.analyzer import RuleBasedAnalyzer
from drone_agent.models import Frame, Telemetry
from drone_agent.pipeline import AlertRulesEngine, ContextManager
import drone_agent.config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frame(fid: int, desc: str, hour: int = 12, minute: int = 0) -> Frame:
    return Frame(frame_id=fid, timestamp=datetime(2024, 1, 15, hour, minute), description=desc)


def tel(location: str = "Main Gate", hour: int = 12, minute: int = 0,
        altitude: float = 20.0) -> Telemetry:
    return Telemetry(timestamp=datetime(2024, 1, 15, hour, minute),
                     location=location, altitude_m=altitude)


def analyze(desc: str, hour: int = 12, minute: int = 0, location: str = "Main Gate",
            altitude: float = 20.0):
    rba = RuleBasedAnalyzer()
    return rba.analyze(frame(1, desc, hour, minute), tel(location, hour, minute, altitude))


# ============================================================================
# Analyzer
# ============================================================================

def test_truck_logged():
    af = analyze("Blue Ford F150 parked at garage", location="Garage")
    v = [d for d in af.detections if d.label == "vehicle"]
    assert v and v[0].subtype == "truck"
    assert v[0].attributes.get("color") == "blue"
    assert v[0].attributes.get("make") == "ford"
    assert v[0].attributes.get("model") == "f150"
    assert v[0].activity == "parked"
    assert "Garage" in af.summary and "12:00" in af.summary


def test_variant_pickup():
    af = analyze("Red pickup truck at the gate")
    assert any(d.label == "vehicle" and d.subtype == "truck" for d in af.detections)


def test_variant_f150_hyphen():
    af = analyze("F-150 entering through the main gate")
    v = [d for d in af.detections if d.label == "vehicle"]
    assert v and v[0].attributes.get("model") == "f150"


def test_no_false_person_for_cat():
    af = analyze("Cat walking across the driveway near the main gate.")
    assert not any(d.label == "person" for d in af.detections)
    assert any(d.label == "animal" for d in af.detections)


def test_person_standing():
    af = analyze("Person standing near the gate, partially obscured.")
    p = [d for d in af.detections if d.label == "person"]
    assert p and p[0].activity == "standing"


def test_climbing_activity():
    af = analyze("Unknown person climbing over the back fence.", location="Back Fence")
    p = [d for d in af.detections if d.label == "person"]
    assert p and p[0].activity == "climbing"


def test_van_subtype():
    af = analyze("Delivery van parked at garage with rear door open.", location="Garage")
    v = [d for d in af.detections if d.label == "vehicle"]
    assert v and v[0].subtype == "van"


def test_empty_description_no_crash():
    af = analyze("")
    assert af.detections and af.summary


def test_summary_format():
    af = analyze("Blue Ford F150 parked at garage", hour=12, location="Garage")
    assert af.summary.endswith(".")
    assert "Garage" in af.summary and "12:00" in af.summary


def test_entity_key_stable():
    af = analyze("Blue Ford F150 parked at garage", location="Garage")
    v = [d for d in af.detections if d.label == "vehicle"]
    assert v
    key = v[0].entity_key
    assert "vehicle" in key and "blue" in key and "ford" in key


# ============================================================================
# Alert rules
# ============================================================================

def _run_loitering(hour: int):
    rba = RuleBasedAnalyzer()
    ctx = ContextManager()
    engine = AlertRulesEngine()
    desc = "Person standing near the gate."
    frames_data = [
        (frame(i, desc, hour, i), tel("Main Gate", hour, i))
        for i in range(3)
    ]
    analyzed = [rba.analyze(f, t) for f, t in frames_data]
    all_alerts = []
    for af in analyzed:
        ctx.update(af)
        all_alerts.extend(engine.evaluate(af, ctx))
    return all_alerts


def test_loitering_midnight():
    alerts = _run_loitering(hour=0)
    assert any(a.severity == "warning" for a in alerts)


def test_no_loitering_daytime():
    alerts = _run_loitering(hour=12)
    assert not any("loitering" in a.message.lower() for a in alerts)


def test_loitering_dedup():
    pytest.skip("Cooldown/dedup removed")


def test_perimeter_breach_critical():
    af = analyze("Unknown person climbing over the back fence.", 21, 0, "Back Fence")
    ctx = ContextManager()
    alerts = AlertRulesEngine().evaluate(af, ctx)
    assert any(a.severity == "critical" for a in alerts)


def test_no_breach_for_animal():
    af = analyze("Cat climbing over the back fence.", 21, 0, "Back Fence")
    ctx = ContextManager()
    alerts = AlertRulesEngine().evaluate(af, ctx)
    assert not any(a.severity == "critical" for a in alerts)


def test_after_hours_vehicle():
    af = analyze("Delivery van parked at garage.", 23, 30, "Garage")
    ctx = ContextManager()
    alerts = AlertRulesEngine().evaluate(af, ctx)
    assert any(a.severity == "warning" and "after hours" in a.message.lower() for a in alerts)


def test_no_after_hours_daytime():
    af = analyze("Blue Ford F150 entering gate", 9, 0, "Main Gate")
    ctx = ContextManager()
    alerts = AlertRulesEngine().evaluate(af, ctx)
    assert not any("after hours" in a.message.lower() for a in alerts)


def test_entity_entered_twice():
    rba = RuleBasedAnalyzer()
    ctx = ContextManager()
    engine = AlertRulesEngine()
    all_alerts = []
    for i, (hour, desc) in enumerate([(8, "Blue Ford F150 entering the main gate"),
                                       (17, "Blue Ford F150 entering gate again")]):
        af = rba.analyze(frame(i + 1, desc, hour), tel("Main Gate", hour))
        ctx.update(af)
        all_alerts.extend(engine.evaluate(af, ctx))
    entity = ctx.get_entity("vehicle:blue:ford:f150")
    assert entity is not None and entity.entry_count == 2


def test_low_battery_alert():
    pytest.skip("Low battery rule removed")


def test_altitude_too_high():
    af = analyze("Empty driveway.", 12, 0, "Main Gate", altitude=150.0)
    ctx = ContextManager()
    alerts = AlertRulesEngine().evaluate(af, ctx)
    assert any("altitude" in a.message.lower() for a in alerts)


# ============================================================================
# Context manager
# ============================================================================

def test_entity_tracker_first_seen():
    ctx = ContextManager()
    af = analyze("Blue Ford F150 entering the main gate", 8, 0, "Main Gate")
    ctx.update(af)
    entity = ctx.get_entity("vehicle:blue:ford:f150")
    assert entity is not None and entity.first_seen.hour == 8


def test_entity_tracker_locations():
    rba = RuleBasedAnalyzer()
    ctx = ContextManager()
    for i, (loc, desc) in enumerate([("Main Gate", "Blue Ford F150 entering the main gate"),
                                      ("Garage", "Blue Ford F150 parked at garage")]):
        af = rba.analyze(frame(i + 1, desc, 8 + i), tel(loc, 8 + i))
        ctx.update(af)
    entity = ctx.get_entity("vehicle:blue:ford:f150")
    assert "Main Gate" in entity.locations and "Garage" in entity.locations


def test_rolling_window_bounded():
    ctx = ContextManager()
    for i in range(cfg.ROLLING_WINDOW_SIZE + 10):
        af = analyze(f"Empty driveway frame {i}", i % 24)
        ctx.update(af)
    assert len(ctx.get_window()) <= cfg.ROLLING_WINDOW_SIZE


def test_chunk_summary_created():
    ctx = ContextManager()
    for i in range(cfg.SUMMARY_CHUNK_SIZE + 1):
        ctx.update(analyze(f"Frame {i}", 12))
    assert len(ctx.get_summaries()["chunk_summaries"]) >= 1


def test_entity_history_fuzzy_match():
    ctx = ContextManager()
    ctx.update(analyze("Blue Ford F150 entering the main gate", 8))
    result = ctx.get_entity_history("blue ford")
    assert result["found"] is True and result["entities"][0]["entry_count"] == 1


def test_entity_history_not_found():
    ctx = ContextManager()
    assert ctx.get_entity_history("purple unicorn")["found"] is False
