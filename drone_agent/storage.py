"""
storage.py — SQLite store, ChromaDB vector store, and unified search API.

Three things live here because they form one cohesive storage layer:

  SQLiteStore   — frames, detections, alerts persisted to a local .db file
  VectorStore   — ChromaDB + sentence-transformers for semantic search
  SearchEngine  — unified API over both stores, plus hybrid search
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from PIL.ImagePalette import load
import nomic
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
nomic.login(os.getenv("NOMIC_API_KEY"))

from drone_agent.models import Alert, AnalyzedFrame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------

class SQLiteStore:
    """SQLite persistence for frames, detections, and alerts."""

    def __init__(self, db_path: str | Path):
        import sqlite3
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id   INTEGER PRIMARY KEY,
                timestamp  TEXT NOT NULL,
                location   TEXT,
                altitude_m REAL,
                description TEXT,
                summary    TEXT,
                image_path TEXT
            );
            CREATE TABLE IF NOT EXISTS detections (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id   INTEGER REFERENCES frames(frame_id),
                label      TEXT, subtype TEXT, color TEXT,
                make       TEXT, model   TEXT, activity TEXT,
                confidence REAL, attributes_json TEXT
            );
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id           TEXT PRIMARY KEY,
                rule               TEXT, severity TEXT,
                message            TEXT, timestamp TEXT,
                evidence_frame_ids TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_frames_ts  ON frames(timestamp);
            CREATE INDEX IF NOT EXISTS idx_det_label  ON detections(label, subtype);
            CREATE INDEX IF NOT EXISTS idx_frames_loc ON frames(location);
        """)
        self._conn.commit()

    def insert_analyzed_frame(self, af: AnalyzedFrame) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO frames VALUES (?,?,?,?,?,?,?)",
            (af.frame.frame_id, af.frame.timestamp.isoformat(), af.telemetry.location,
             af.telemetry.altitude_m, af.frame.description, af.summary, af.frame.image_path),
        )
        for d in af.detections:
            cur.execute(
                "INSERT INTO detections (frame_id,label,subtype,color,make,model,activity,confidence,attributes_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (af.frame.frame_id, d.label, d.subtype, d.attributes.get("color"),
                 d.attributes.get("make"), d.attributes.get("model"), d.activity,
                 d.confidence, json.dumps(d.attributes)),
            )
        self._conn.commit()

    def insert_alert(self, alert: Alert) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?)",
            (alert.alert_id, alert.rule, alert.severity, alert.message,
             alert.timestamp.isoformat(), json.dumps(alert.evidence_frame_ids)),
        )
        self._conn.commit()

    def query_by_time(self, start: str | datetime, end: str | datetime) -> list[dict]:
        s = start.isoformat() if isinstance(start, datetime) else start
        e = end.isoformat() if isinstance(end, datetime) else end
        rows = self._conn.execute(
            "SELECT frame_id,timestamp,location,altitude_m,description,summary "
            "FROM frames WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp", (s, e)
        ).fetchall()
        return [self._row(r) for r in rows]

    def query_by_object(self, label=None, subtype=None, color=None) -> list[dict]:
        conds, params = [], []
        for col, val in [("d.label", label), ("d.subtype", subtype), ("d.color", color)]:
            if val:
                conds.append(f"{col} = ?"); params.append(val.lower())
        where = " AND ".join(conds) if conds else "1=1"
        rows = self._conn.execute(
            f"SELECT DISTINCT f.frame_id,f.timestamp,f.location,f.altitude_m,f.description,f.summary "
            f"FROM frames f JOIN detections d ON d.frame_id=f.frame_id WHERE {where} ORDER BY f.timestamp",
            params,
        ).fetchall()
        return [self._row(r) for r in rows]

    def query_by_location(self, name: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT frame_id,timestamp,location,altitude_m,description,summary "
            "FROM frames WHERE LOWER(location) LIKE ? ORDER BY timestamp",
            (f"%{name.lower()}%",),
        ).fetchall()
        return [self._row(r) for r in rows]

    def get_all_alerts(self, severity: str | None = None, since: str | datetime | None = None) -> list[dict]:
        conds, params = [], []
        if severity:
            conds.append("severity = ?"); params.append(severity.lower())
        if since:
            conds.append("timestamp >= ?")
            params.append(since.isoformat() if isinstance(since, datetime) else since)
        where = " AND ".join(conds) if conds else "1=1"
        rows = self._conn.execute(
            f"SELECT alert_id,rule,severity,message,timestamp,evidence_frame_ids FROM alerts WHERE {where} ORDER BY timestamp",
            params,
        ).fetchall()
        return [{"alert_id": r[0], "rule": r[1], "severity": r[2], "message": r[3],
                 "timestamp": r[4], "evidence_frame_ids": json.loads(r[5])} for r in rows]

    def frame_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def clear(self) -> None:
        """Wipe all data — called at the start of each pipeline run."""
        self._conn.executescript("""
            DELETE FROM alerts;
            DELETE FROM detections;
            DELETE FROM frames;
        """)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row(r: tuple) -> dict:
        return {"frame_id": r[0], "timestamp": r[1], "location": r[2],
                "altitude_m": r[3], "description": r[4], "summary": r[5]}


# ---------------------------------------------------------------------------
# Vector store (ChromaDB + Nomic embed for semantic search)
# ---------------------------------------------------------------------------

class VectorStore:
    """ChromaDB + Nomic embed text for semantic search over frame descriptions.

    Embeddings are generated via the Nomic API (nomic-embed-text-v1.5).
    Requires: pip install nomic chromadb
    Requires: nomic login <token>  (run once in the terminal)

    WHY nomic-embed-text-v1.5:
      - Matryoshka model — dimensionality can be reduced without quality loss
      - task_type parameter distinguishes indexing vs querying, improving recall
      - API-based, no large local model download required
    """

    def __init__(self, persist_path: str | Path):
        try:
            import chromadb
            import nomic.embed  # noqa: F401 — validate import early
        except ImportError:
            raise RuntimeError("Run: pip install chromadb nomic")
        self._persist_path = str(persist_path)
        import chromadb as _chroma
        self._client = _chroma.PersistentClient(path=str(persist_path))
        self._col = self._client.get_or_create_collection(
            "drone_frames", metadata={"hnsw:space": "cosine"})

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for indexing — task_type=search_document."""
        from nomic import embed
        result = embed.text(texts, model="nomic-embed-text-v1.5",
                            task_type="search_document")
        return result["embeddings"]

    def _embed_query(self, query: str) -> list[float]:
        """Embed a single query string — task_type=search_query."""
        from nomic import embed
        result = embed.text([query], model="nomic-embed-text-v1.5",
                            task_type="search_query")
        return result["embeddings"][0]

    def upsert_frame(self, frame_id: int, description: str, summary: str,
                     timestamp: str, location: str, labels: list[str]) -> None:
        doc = f"{description} {summary}".strip()
        emb = self._embed_documents([doc])[0]
        self._col.upsert(ids=[str(frame_id)], embeddings=[emb], documents=[doc],
                         metadatas=[{"frame_id": frame_id, "timestamp": timestamp,
                                     "location": location, "labels": ",".join(labels)}])

    def semantic_search(self, query: str, k: int = 5, where: dict | None = None) -> list[dict]:
        emb = self._embed_query(query)
        kw: dict = {"query_embeddings": [emb],
                    "n_results": min(k, max(1, self._col.count())),
                    "include": ["metadatas", "documents", "distances"]}
        if where:
            kw["where"] = where
        res = self._col.query(**kw)
        return [{"frame_id": m["frame_id"], "timestamp": m["timestamp"],
                 "location": m["location"], "labels": m["labels"],
                 "document": res["documents"][0][i], "distance": res["distances"][0][i]}
                for i, m in enumerate(res["metadatas"][0])]

    def count(self) -> int:
        return self._col.count()


# ---------------------------------------------------------------------------
# Search engine — unified API over SQLite + optional vector store
# ---------------------------------------------------------------------------

class SearchEngine:
    """Single entry point for all queries. Used by agent tools and CLI."""

    def __init__(self, sqlite: SQLiteStore, vector: VectorStore | None = None):
        self._db = sqlite
        self._vec = vector

    def by_time(self, start, end) -> list[dict]:
        return self._db.query_by_time(start, end)

    def by_object(self, label=None, subtype=None, color=None) -> list[dict]:
        return self._db.query_by_object(label=label, subtype=subtype, color=color)

    def by_location(self, name: str) -> list[dict]:
        return self._db.query_by_location(name)

    def get_alerts(self, severity=None, since=None) -> list[dict]:
        return self._db.get_all_alerts(severity=severity, since=since)

    def semantic(self, query: str, k: int = 5) -> list[dict]:
        if self._vec is None:
            return []
        return self._vec.semantic_search(query, k=k)

    def hybrid(self, query: str, filters: dict | None = None, k: int = 5) -> list[dict]:
        """SQL pre-filter then vector re-rank — the innovation angle."""
        if self._vec is None:
            return self._sql_filter(filters or {})[:k]
        candidates = self._sql_filter(filters or {})
        if not candidates:
            return []
        cand_ids = {str(c["frame_id"]) for c in candidates}
        raw = self._vec.semantic_search(query, k=max(k * 5, 50))
        filtered = sorted(
            [r for r in raw if str(r["frame_id"]) in cand_ids],
            key=lambda r: r["distance"],
        )
        id_to_sql = {c["frame_id"]: c for c in candidates}
        return [{**id_to_sql.get(r["frame_id"], {}), "semantic_distance": r["distance"]}
                for r in filtered[:k]]

    def _sql_filter(self, f: dict) -> list[dict]:
        if f.get("label") or f.get("subtype") or f.get("color"):
            return self._db.query_by_object(label=f.get("label"), subtype=f.get("subtype"), color=f.get("color"))
        if f.get("start_time") and f.get("end_time"):
            return self._db.query_by_time(f["start_time"], f["end_time"])
        if f.get("location"):
            return self._db.query_by_location(f["location"])
        return self._db.query_by_time("1970-01-01T00:00:00", "2099-12-31T23:59:59")
