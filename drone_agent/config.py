"""
config.py — Centralised configuration for the drone security agent.

WHY a single config module:
  All thresholds, paths, and behavioural constants live here so they can be
  tuned without touching rule logic.  Tests can patch these values directly.
  API keys and file paths are read from .env; the rest has sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # drone-security-agent/
DATA_DIR = PROJECT_ROOT / "data"

# On read-only filesystems (e.g. Streamlit Cloud) fall back to /tmp
def _writable_path(default: Path) -> Path:
    try:
        default.parent.mkdir(parents=True, exist_ok=True)
        default.touch(exist_ok=True)
        return default
    except OSError:
        return Path("/tmp") / default.name

DB_PATH = _writable_path(Path(os.getenv("DB_PATH", str(PROJECT_ROOT / "drone_security.db"))))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db")))
ALERTS_LOG_PATH = _writable_path(PROJECT_ROOT / "alerts.log")

# PostgreSQL connection string — if set, PostgresStore is used instead of SQLiteStore.
# Format: postgresql://user:password@host:port/dbname
DATABASE_URL: str | None = os.getenv("DATABASE_URL")
# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

# API key for Groq (used by LLMAnalyzer and the agent via langchain-openai).
# Read directly in analyzer.py via os.getenv('API_KEY').

# ---------------------------------------------------------------------------
# Default analyzer
# ---------------------------------------------------------------------------

DEFAULT_ANALYZER: str = os.getenv("DEFAULT_ANALYZER", "rule")  # rule | llm

# ---------------------------------------------------------------------------
# Alert rule thresholds
# ---------------------------------------------------------------------------

# Night window (inclusive hours, local time).
# Wraparound handled in the rule: hour >= NIGHT_START or hour <= NIGHT_END.
NIGHT_START_HOUR: int = 22
NIGHT_END_HOUR: int = 5

# Loitering: person seen at the same location in this many consecutive frames.
LOITERING_FRAME_THRESHOLD: int = 3

# After-hours vehicle window (vehicles ALLOWED between these hours).
VEHICLE_ALLOWED_START_HOUR: int = 6
VEHICLE_ALLOWED_END_HOUR: int = 22

# Altitude bounds (metres).
ALTITUDE_MIN_M: float = 5.0
ALTITUDE_MAX_M: float = 120.0

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = "nomic-embed-text-v1.5"  # used by VectorStore via Nomic API

# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

# Rolling window size (number of AnalyzedFrames kept in memory for stateful rules).
ROLLING_WINDOW_SIZE: int = 20

# Hierarchical summary: produce a chunk summary every K frames.
SUMMARY_CHUNK_SIZE: int = 50

# ---------------------------------------------------------------------------
# Known vehicles allow-list (make:model pairs, lowercase).
# Vehicles not on this list trigger an "unknown vehicle" info alert.
# ---------------------------------------------------------------------------

KNOWN_VEHICLES: set[str] = {
    "ford:f150",
    "ford:f-150",
    "toyota:camry",
    "honda:civic",
}

# ---------------------------------------------------------------------------
# Property locations (used for display and location matching).
# ---------------------------------------------------------------------------

KNOWN_LOCATIONS: list[str] = [
    "Main Gate",
    "Garage",
    "Back Fence",
    "Perimeter North",
    "Perimeter South",
    "Driveway",
    "Parking Lot",
    "Side Entrance",
]
