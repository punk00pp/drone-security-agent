"""
simulator.py — Scenario loader and stress-data generator.

WHY this module exists:
  Separates data sourcing from pipeline logic.  The rest of the pipeline only
  sees (Frame, Telemetry) pairs; it does not care whether they came from a JSON
  scenario file, a stress generator, or (in future) a live drone stream.

Public API:
  load_scenario(path)         -> Iterator[tuple[Frame, Telemetry]]
  generate_stress(n, seed)    -> list[tuple[Frame, Telemetry]]
"""

from __future__ import annotations

import json
import random
import time as time_module
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Iterator

from drone_agent.models import Frame, Telemetry

# Reference date used when scenario only specifies HH:MM times.
_SCENARIO_DATE = date(2024, 1, 15)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_time(time_str: str, ref_date: date = _SCENARIO_DATE) -> datetime:
    """
    Convert "HH:MM" string to a timezone-naive datetime on ref_date.
    WHY: The scenario JSON stores times as "08:30" for readability; Pydantic
    models require datetime objects.
    """
    h, m = map(int, time_str.split(":"))
    return datetime(ref_date.year, ref_date.month, ref_date.day, h, m)


# ---------------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------------

def load_scenario(
    path: str | Path,
    realtime: bool = False,
    speed_factor: float = 1.0,
) -> Iterator[tuple[Frame, Telemetry]]:
    """
    Load a scenario JSON file and yield (Frame, Telemetry) pairs in order.

    Args:
        path:          Path to the scenario JSON (see data/scenario_day1.json).
        realtime:      If True, sleep between frames to simulate a live stream.
                       Useful for demo videos.
        speed_factor:  Multiplier for sleep duration (< 1 = faster, default 1.0).
                       Only used when realtime=True.

    Yields:
        (Frame, Telemetry) pairs, one per scenario entry.

    WHY sorted by timestamp:
      Scenario files are authored chronologically, but we sort defensively so
      out-of-order entries in test scenarios are handled gracefully.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    ref_date_str: str = data.get("date", "2024-01-15")
    ref_date = date.fromisoformat(ref_date_str)

    entries = sorted(data["frames"], key=lambda e: e["time"])
    prev_dt: datetime | None = None

    for entry in entries:
        dt = _parse_time(entry["time"], ref_date)
        frame = Frame(
            frame_id=entry["frame_id"],
            timestamp=dt,
            description=entry["description"],
            image_path=entry.get("image_path"),
        )
        telemetry = Telemetry(
            timestamp=dt,
            location=entry["location"],
            altitude_m=float(entry.get("altitude_m", 20.0)),
        )

        if realtime and prev_dt is not None:
            gap_seconds = (dt - prev_dt).total_seconds()
            # Cap sleep at 5 s even with speed_factor to keep demos snappy.
            sleep_time = min(gap_seconds / max(speed_factor, 1.0), 5.0)
            if sleep_time > 0:
                time_module.sleep(sleep_time)

        prev_dt = dt
        yield frame, telemetry


# ---------------------------------------------------------------------------
# Stress data generator
# ---------------------------------------------------------------------------

# Pool of realistic frame descriptions for random generation.
_DESCRIPTION_POOL = [
    "Empty driveway, no activity.",
    "Person walking near the main gate.",
    "Blue Ford F150 parked at garage.",
    "Unknown vehicle at side entrance.",
    "Cat walking across the driveway.",
    "Delivery van parked outside garage.",
    "Person standing near the perimeter fence.",
    "Silver sedan leaving the parking lot.",
    "Security guard making rounds.",
    "Bicycle leaning against the wall.",
    "Dog running across the lawn.",
    "Garbage truck at the main gate.",
    "Person jogging along the perimeter.",
    "Red pickup truck entering gate.",
    "Drone returning to dock, no activity.",
    "Person climbing over the back fence.",
    "SUV parked at the side entrance.",
    "Two people talking near the garage.",
    "Delivery person at the front door.",
    "Empty parking lot, lights on.",
]

_LOCATIONS = [
    "Main Gate", "Garage", "Back Fence",
    "Perimeter North", "Perimeter South",
    "Driveway", "Parking Lot", "Side Entrance",
]


def generate_stress(
    n_frames: int,
    seed: int = 42,
    inject_known_events: bool = True,
    ref_date: date | None = None,
) -> list[tuple[Frame, Telemetry]]:
    """
    Generate a list of (Frame, Telemetry) pairs for scalability testing.

    WHY inject_known_events:
      Random-only data makes it impossible to assert correctness in the
      benchmark.  Known events (loitering, perimeter breach) are injected at
      predictable positions so benchmark.py can verify they were detected.

    Args:
        n_frames:            Total number of frames to generate.
        seed:                Random seed for reproducibility.
        inject_known_events: If True, inject a loitering sequence and a
                             perimeter breach at fixed positions.
        ref_date:            Base date for timestamps (defaults to today).

    Returns:
        List of (Frame, Telemetry) pairs.
    """
    rng = random.Random(seed)
    if ref_date is None:
        ref_date = date.today()

    pairs: list[tuple[Frame, Telemetry]] = []
    base_dt = datetime(ref_date.year, ref_date.month, ref_date.day, 0, 0)

    for i in range(n_frames):
        dt = base_dt + timedelta(seconds=i * 30)
        altitude = rng.uniform(15.0, 40.0)
        location = rng.choice(_LOCATIONS)
        description = rng.choice(_DESCRIPTION_POOL)

        frame = Frame(
            frame_id=i + 1,
            timestamp=dt,
            description=description,
        )
        telemetry = Telemetry(
            timestamp=dt,
            location=location,
            altitude_m=altitude,
        )
        pairs.append((frame, telemetry))

    if inject_known_events and n_frames >= 10:
        night_dt = datetime(ref_date.year, ref_date.month, ref_date.day, 1, 0)
        for offset, pos in enumerate([5, 6, 7]):
            if pos < n_frames:
                dt = night_dt + timedelta(minutes=offset)
                pairs[pos] = (
                    Frame(
                        frame_id=pairs[pos][0].frame_id,
                        timestamp=dt,
                        description="Person standing near the main gate, not moving.",
                    ),
                    Telemetry(
                        timestamp=dt,
                        location="Main Gate",
                        altitude_m=20.0,
                    ),
                )

        mid = n_frames // 2
        breach_dt = datetime(ref_date.year, ref_date.month, ref_date.day, 21, 0)
        pairs[mid] = (
            Frame(
                frame_id=pairs[mid][0].frame_id,
                timestamp=breach_dt,
                description="Unknown person climbing over the back fence.",
            ),
            Telemetry(
                timestamp=breach_dt,
                location="Back Fence",
                altitude_m=15.0,
            ),
        )

    return pairs
