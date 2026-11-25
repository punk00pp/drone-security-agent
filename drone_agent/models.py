"""
models.py — Pydantic data models for the drone security agent.

WHY Pydantic:
  - Validates incoming JSON (from scenario files and LLM responses) at the boundary.
  - Provides typed, IDE-navigable objects throughout the codebase.
  - Makes test assertions readable: `frame.detections[0].label == "vehicle"`.

All datetime fields use timezone-naive UTC datetimes for simplicity.
The scenario loader normalises HH:MM strings to datetime objects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class Telemetry(BaseModel):
    """
    One telemetry sample from the drone at a given moment.
    Timestamps match frames so the ingestion layer can pair them.
    """
    timestamp: datetime
    location: str                           # e.g. "Main Gate", "Garage"
    altitude_m: float = 20.0
    lat: float | None = None
    lon: float | None = None


# ---------------------------------------------------------------------------
# Frame (raw input)
# ---------------------------------------------------------------------------

class Frame(BaseModel):
    """
    A single video frame, represented as a text description in simulation mode.
    In real-video mode, image_path points to a file and description may be
    populated by the VLMAnalyzer caption.
    """
    frame_id: int
    timestamp: datetime
    description: str                        # "Blue Ford F150 parked at garage"
    image_path: str | None = None           # real-video mode only


# ---------------------------------------------------------------------------
# Detection (structured output from the analyzer)
# ---------------------------------------------------------------------------

class Detection(BaseModel):
    """
    One detected entity within a frame.
    The analyzer extracts these from the frame description (or image).

    WHY computed entity_key:
      Normalising to a stable string key (e.g. "vehicle:blue:ford:f150") lets
      the entity tracker in context.py recognise the same object across frames
      even if the descriptions vary in phrasing ("truck", "F-150", "pickup").
    """
    label: str                              # person | vehicle | animal | unknown
    subtype: str | None = None             # truck | car | van | motorcycle | ...
    attributes: dict = Field(default_factory=dict)
    # attributes keys: color, make, model, plate, etc.
    confidence: float = 1.0
    activity: str | None = None            # parked | entering | leaving | standing |
                                           # walking | climbing | running | ...

    @computed_field                        # type: ignore[misc]
    @property
    def entity_key(self) -> str:
        """
        A normalised, stable key used by the entity tracker.
        Format: label:color:make:model  (empty parts are omitted).
        Example: "vehicle:blue:ford:f150"
        """
        parts = [self.label]
        for attr in ("color", "make", "model"):
            val = self.attributes.get(attr, "").strip().lower()
            if val:
                parts.append(val)
        return ":".join(parts)


# ---------------------------------------------------------------------------
# AnalyzedFrame (output of the analyzer pipeline)
# ---------------------------------------------------------------------------

class AnalyzedFrame(BaseModel):
    """
    The result of running an analyzer on a Frame + Telemetry pair.
    This is what all downstream components (index, context, alerts) consume.
    """
    frame: Frame
    telemetry: Telemetry
    detections: list[Detection]
    summary: str                            # human-readable one-line summary


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    """
    A security alert fired by the rules engine.

    WHY evidence_frame_ids:
      Explainability — the officer can pull the exact frames that triggered the
      alert and verify the footage manually. This satisfies KR-4.

    Message format: "{Subject} {behavior} at {location}, {HH:MM}."
    Example: "Person loitering at Main Gate, 00:01."
    """
    alert_id: str                           # uuid4 string
    rule: str                               # rule class name, e.g. "LoiteringRule"
    severity: Literal["info", "warning", "critical"]
    message: str
    timestamp: datetime
    evidence_frame_ids: list[int]


# ---------------------------------------------------------------------------
# EntityInfo (used by the context manager)
# ---------------------------------------------------------------------------

class EntityInfo(BaseModel):
    """
    Tracks a unique entity (vehicle, person) across the entire session.
    The entity_key from Detection is used as the dict key in ContextManager.
    """
    entity_key: str
    label: str
    first_seen: datetime
    last_seen: datetime
    locations: list[str] = Field(default_factory=list)
    entry_count: int = 0
    exit_count: int = 0
    frame_ids: list[int] = Field(default_factory=list)
    attributes: dict = Field(default_factory=dict)
