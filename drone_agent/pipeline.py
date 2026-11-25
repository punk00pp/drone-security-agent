"""
pipeline.py — Context manager, alert rules, summarizer, and pipeline loop.

Four things live here because they are tightly coupled and each is small:

  ContextManager   — entity tracker + rolling window + hierarchical summaries
  AlertRulesEngine — 7 deterministic Python rules + cooldown deduplication
  summarize_*      — LLM chunk and video summarizer (bonus feature)
  run_pipeline     — the main loop that wires everything together
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from datetime import datetime
from typing import Iterable

import drone_agent.config as cfg
from drone_agent.models import Alert, AnalyzedFrame, Detection, EntityInfo, Frame, Telemetry

logger = logging.getLogger(__name__)

# ============================================================================
# Context Manager
# ============================================================================

class ContextManager:
    """
    Tracks entities across frames, maintains a rolling window for stateful
    rules, and builds hierarchical summaries so the agent prompt stays bounded.
    """

    def __init__(self, summarizer_fn=None):
        self._entities: dict[str, EntityInfo] = {}
        self._window: deque[AnalyzedFrame] = deque(maxlen=cfg.ROLLING_WINDOW_SIZE)
        self._chunk_summaries: list[str] = []
        self._pending: list[str] = []
        self._daily_summary: str = ""
        self._total: int = 0
        self._summarize = summarizer_fn or (lambda xs: " | ".join(x.rstrip(".") for x in xs if x))

    def update(self, af: AnalyzedFrame) -> None:
        self._total += 1
        self._window.append(af)
        for det in af.detections:
            if det.label != "unknown":
                self._track(det, af)
        self._pending.append(af.summary)
        if len(self._pending) >= cfg.SUMMARY_CHUNK_SIZE:
            self._flush_chunk()

    def _track(self, det: Detection, af: AnalyzedFrame) -> None:
        key = det.entity_key
        ts, loc = af.frame.timestamp, af.telemetry.location
        if key not in self._entities:
            self._entities[key] = EntityInfo(entity_key=key, label=det.label,
                                              first_seen=ts, last_seen=ts,
                                              locations=[loc], attributes=dict(det.attributes))
        else:
            info = self._entities[key]
            info.last_seen = ts
            if loc not in info.locations:
                info.locations.append(loc)
            for k, v in det.attributes.items():
                info.attributes.setdefault(k, v)
        info = self._entities[key]
        info.frame_ids.append(af.frame.frame_id)
        if det.activity == "entering":
            info.entry_count += 1
        elif det.activity == "leaving":
            info.exit_count += 1

    def _flush_chunk(self) -> None:
        self._chunk_summaries.append(self._summarize(list(self._pending)))
        self._pending.clear()
        if len(self._chunk_summaries) >= 10:
            self._daily_summary = self._summarize(self._chunk_summaries)
            self._chunk_summaries = self._chunk_summaries[-2:]

    def flush_final(self) -> None:
        if self._pending:
            self._flush_chunk()

    def get_window(self) -> list[AnalyzedFrame]:
        return list(self._window)

    def get_entity(self, key: str) -> EntityInfo | None:
        return self._entities.get(key)

    def get_entity_history(self, query: str) -> dict:
        """Fuzzy match: all query words must appear in some part of the entity key."""
        words = set(query.lower().replace(":", " ").split())
        matched = [
            info for k, info in self._entities.items()
            if all(any(w in part for part in k.lower().split(":")) for w in words)
            or query.lower() in k.lower()
        ]
        if not matched:
            return {"found": False, "query": query}
        return {"found": True, "query": query, "entities": [
            {"entity_key": i.entity_key, "label": i.label,
             "first_seen": i.first_seen.isoformat(), "last_seen": i.last_seen.isoformat(),
             "locations": i.locations, "entry_count": i.entry_count,
             "exit_count": i.exit_count, "total_frames": len(i.frame_ids),
             "frame_ids": i.frame_ids, "attributes": i.attributes}
            for i in matched
        ]}

    def get_summaries(self) -> dict:
        return {"chunk_summaries": list(self._chunk_summaries),
                "daily_summary": self._daily_summary,
                "total_frames": self._total}

    def get_context_for_agent(self) -> str:
        parts = []
        if self._daily_summary:
            parts.append(f"## Daily Summary\n{self._daily_summary}")
        if self._chunk_summaries:
            parts.append("## Recent Activity\n" + "\n".join(f"- {s}" for s in self._chunk_summaries[-3:]))
        if self._pending:
            parts.append("## Latest Frames\n" + "\n".join(f"- {s}" for s in self._pending[-10:]))
        entity_lines = []
        for key, info in list(self._entities.items())[:20]:
            color = info.attributes.get("color", "")
            make = info.attributes.get("make", "").title()
            model = info.attributes.get("model", "").upper()
            name = " ".join(p for p in [color, make, model] if p) or info.label
            entity_lines.append(f"  - {name} ({key}): entries={info.entry_count} "
                                 f"first={info.first_seen.strftime('%H:%M')} last={info.last_seen.strftime('%H:%M')}")
        if entity_lines:
            parts.append("## Tracked Entities\n" + "\n".join(entity_lines))
        return "\n\n".join(parts) or "No session context yet."


# ============================================================================
# Alert Rules Engine
# ============================================================================

def _is_night(dt: datetime) -> bool:
    """True if hour is in the night window (wraps midnight correctly)."""
    h = dt.hour
    return h >= cfg.NIGHT_START_HOUR or h <= cfg.NIGHT_END_HOUR


def _make_alert(rule: str, severity: str, message: str,
                timestamp: datetime, evidence: list[int]) -> Alert:
    return Alert(alert_id=str(uuid.uuid4()), rule=rule, severity=severity,  # type: ignore[arg-type]
                 message=message, timestamp=timestamp, evidence_frame_ids=evidence)


class AlertRulesEngine:
    """
    Runs deterministic Python rules against each AnalyzedFrame.
    Rules are plain Python — no LLM. Fast, auditable, reproducible.
    """

    def __init__(self):
        # Tracks the last frame_id on which each loitering location fired,
        # so the alert only fires once per continuous streak.
        self._loitering_fired: dict[str, int] = {}

    def evaluate(self, af: AnalyzedFrame, ctx: ContextManager) -> list[Alert]:
        alerts: list[Alert] = []
        try:
            alerts += self._loitering(af, ctx)
            alerts += self._perimeter_breach(af)
            alerts += self._after_hours_vehicle(af)
            alerts += self._unknown_vehicle(af)
            alerts += self._repeated_entry(af, ctx)
            alerts += self._altitude(af)
        except Exception as exc:
            logger.error("Alert rule exception: %s", exc, exc_info=True)
        return alerts

    def _loitering(self, af: AnalyzedFrame, ctx: ContextManager) -> list[Alert]:
        if not _is_night(af.frame.timestamp):
            return []
        alerts = []
        loc = af.telemetry.location
        for det in af.detections:
            if det.label != "person":
                continue
            # Count consecutive person-at-same-location frames in window + current
            streak, evidence = 0, []
            for past in reversed(ctx.get_window() + [af]):
                if any(d.label == "person" for d in past.detections) and past.telemetry.location == loc:
                    streak += 1
                    evidence.insert(0, past.frame.frame_id)
                else:
                    break
            if streak >= cfg.LOITERING_FRAME_THRESHOLD:
                # Only fire once per continuous streak — suppress if the
                # previous alert was on the immediately preceding frame.
                last_fired = self._loitering_fired.get(loc, -1)
                if af.frame.frame_id - last_fired <= 1:
                    continue
                self._loitering_fired[loc] = af.frame.frame_id
                ts = af.frame.timestamp.strftime("%H:%M")
                alerts.append(_make_alert("LoiteringRule", "warning",
                                          f"Person loitering at {loc}, {ts}.",
                                          af.frame.timestamp, evidence))
        return alerts

    def _perimeter_breach(self, af: AnalyzedFrame) -> list[Alert]:
        loc = af.telemetry.location.lower()
        is_breach_loc = any(w in loc for w in ("fence", "perimeter", "wall"))
        alerts = []
        for det in af.detections:
            if det.label != "person":
                continue
            if det.activity in ("climbing", "jumping", "cutting") or is_breach_loc:
                ts = af.frame.timestamp.strftime("%H:%M")
                alerts.append(_make_alert("PerimeterBreachRule", "critical",
                                          f"Person breaching perimeter at {af.telemetry.location}, {ts}.",
                                          af.frame.timestamp, [af.frame.frame_id]))
        return alerts

    def _after_hours_vehicle(self, af: AnalyzedFrame) -> list[Alert]:
        h = af.frame.timestamp.hour
        if cfg.VEHICLE_ALLOWED_START_HOUR <= h <= cfg.VEHICLE_ALLOWED_END_HOUR:
            return []
        alerts = []
        for det in af.detections:
            if det.label != "vehicle":
                continue
            sub = det.subtype or "vehicle"
            color = det.attributes.get("color", "")
            subject = f"{color} {sub}".strip().title()
            ts = af.frame.timestamp.strftime("%H:%M")
            alerts.append(_make_alert("AfterHoursVehicleRule", "warning",
                                      f"{subject} detected after hours at {af.telemetry.location}, {ts}.",
                                      af.frame.timestamp, [af.frame.frame_id]))
        return alerts

    def _unknown_vehicle(self, af: AnalyzedFrame) -> list[Alert]:
        alerts = []
        for det in af.detections:
            if det.label != "vehicle":
                continue
            make = det.attributes.get("make", "").lower()
            model = det.attributes.get("model", "").lower()
            if not make:
                continue
            vid = f"{make}:{model}" if model else make
            if vid not in cfg.KNOWN_VEHICLES:
                ts = af.frame.timestamp.strftime("%H:%M")
                alerts.append(_make_alert("UnknownVehicleRule", "info",
                                          f"Unknown vehicle ({make} {model}) at {af.telemetry.location}, {ts}.",
                                          af.frame.timestamp, [af.frame.frame_id]))
        return alerts

    def _repeated_entry(self, af: AnalyzedFrame, ctx: ContextManager) -> list[Alert]:
        alerts = []
        for det in af.detections:
            if det.activity != "entering":
                continue
            info = ctx.get_entity(det.entity_key)
            if info and info.entry_count >= 2:
                sub = det.subtype or det.label
                color = det.attributes.get("color", "")
                make = det.attributes.get("make", "").title()
                model = det.attributes.get("model", "").upper()
                name = " ".join(p for p in [color, make, model, sub] if p) or "Vehicle"
                ts = af.frame.timestamp.strftime("%H:%M")
                alerts.append(_make_alert("RepeatedEntryRule", "info",
                                          f"{name} entered {info.entry_count} times today at {af.telemetry.location}, {ts}.",
                                          af.frame.timestamp, info.frame_ids[-2:]))
        return alerts

    def _altitude(self, af: AnalyzedFrame) -> list[Alert]:
        alt = af.telemetry.altitude_m
        if cfg.ALTITUDE_MIN_M <= alt <= cfg.ALTITUDE_MAX_M:
            return []
        ts = af.frame.timestamp.strftime("%H:%M")
        return [_make_alert("AltitudeAnomalyRule", "warning",
                            f"Drone altitude anomaly ({alt:.1f} m) at {af.telemetry.location}, {ts}.",
                            af.frame.timestamp, [af.frame.frame_id])]


# ============================================================================
# Summarizer (bonus)
# ============================================================================

def summarize_chunk(summaries: list[str]) -> str:
    """Summarise a batch of frame summaries into one paragraph via LLM."""
    if not summaries:
        return ""
    try:
        from drone_agent.analyzer import _build_llm
        from langchain_core.messages import HumanMessage
        joined = "\n".join(f"- {s}" for s in summaries)
        prompt = (
            "Summarise this portion of a security drone patrol log in 2-3 sentences. "
            "Focus on who/what was detected, where, and notable activities. "
            f"Skip empty observations.\n\nLog:\n{joined}\n\nSummary:"
        )
        return _build_llm().invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as exc:
        logger.warning("Chunk summariser failed: %s", exc)
        return " | ".join(s.rstrip(".") for s in summaries if s)


def summarize_video(chunk_summaries: list[str], daily_summary: str = "") -> str:
    """Produce one sentence covering the whole patrol session."""
    sections = daily_summary or "\n".join(f"- {s}" for s in chunk_summaries if s)
    if not sections:
        return "No significant events recorded."
    try:
        from drone_agent.analyzer import _build_llm
        from langchain_core.messages import HumanMessage
        prompt = (
            "Given these security patrol section summaries, write ONE sentence "
            "describing the most important events of the day.\n\n"
            f"{sections}\n\nOne-sentence summary:"
        )
        return _build_llm().invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as exc:
        logger.warning("Video summariser failed: %s", exc)
        lines = [s.split(".")[0].strip() for s in chunk_summaries[:3] if s]
        return "; ".join(lines) + "." if lines else "Patrol completed."


# ============================================================================
# Pipeline loop
# ============================================================================

def run_pipeline(
    frames_telemetry: Iterable[tuple[Frame, Telemetry]],
    analyzer,           # RuleBasedAnalyzer or LLMAnalyzer
    sqlite,             # SQLiteStore
    search,             # SearchEngine
    context: ContextManager,
    rules: AlertRulesEngine,
    quiet: bool = False,
) -> tuple[list[AnalyzedFrame], list[Alert]]:
    """
    Main pipeline loop: for each (Frame, Telemetry) pair —
      analyze → index (SQLite + vector) → update context → evaluate rules → emit alerts.
    Returns (analyzed_frames, alerts) for callers to inspect.
    """
    all_analyzed: list[AnalyzedFrame] = []
    all_alerts: list[Alert] = []

    with open(cfg.ALERTS_LOG_PATH, "a", encoding="utf-8") as log_file:
        for frame, telemetry in frames_telemetry:
            af = analyzer.analyze(frame, telemetry)
            all_analyzed.append(af)

            sqlite.insert_analyzed_frame(af)

            vec = getattr(search, "_vec", None)
            if vec is not None:
                vec.upsert_frame(
                    frame_id=af.frame.frame_id,
                    description=af.frame.description,
                    summary=af.summary,
                    timestamp=af.frame.timestamp.isoformat(),
                    location=af.telemetry.location,
                    labels=[d.label for d in af.detections],
                )

            context.update(af)
            frame_alerts = rules.evaluate(af, context)

            for alert in frame_alerts:
                sqlite.insert_alert(alert)
                all_alerts.append(alert)
                log_file.write(f"[{alert.severity.upper()}] {alert.message} "
                               f"(rule={alert.rule}, frames={alert.evidence_frame_ids})\n")
                log_file.flush()
                if not quiet:
                    print(f"  !! ALERT [{alert.severity.upper()}]  {alert.message}")

            if not quiet:
                ts = af.frame.timestamp.strftime("%H:%M")
                print(f"[{ts}] {af.telemetry.location:20s}  {af.summary}")

    context.flush_final()

    if not quiet:
        print(f"\n--- Pipeline complete: {len(all_analyzed)} frames, {len(all_alerts)} alerts ---")
        for a in all_alerts:
            print(f"  [{a.severity.upper()}] {a.message}")

    return all_analyzed, all_alerts
