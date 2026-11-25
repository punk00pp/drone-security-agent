"""
storage.py — PostgreSQL store, ChromaDB vector store, and unified search API.

  PostgresStore — frames, detections, alerts in a persistent PostgreSQL database.
                  Data survives restarts, redeployments, and multiple sessions.
                  Only cleared explicitly via clear() — called only by the Reset button.
  SQLiteStore   — kept for tests only (no DATABASE_URL needed in CI).
  VectorStore   — ChromaDB + Nomic API embeddings for semantic search.
  SearchEngine  — unified API over both stores.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from drone_agent.models import Alert, AnalyzedFrame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PostgresStore — production store
# ---------------------------------------------------------------------------

class PostgresStore:
    """
    PostgreSQL persistence for frames, detections, and alerts.

    WHY PostgreSQL over SQLite:
      - Data persists across sessions, restarts, and Streamlit Cloud redeployments.
      - Multiple clients can connect simultaneously.
      - clear() is the ONLY way to delete data — called exclusively from the
        Reset button in app.py. Nothing else touches it.

    Connection string read from DATABASE_URL in .env.
    Format: postgresql://user:password@host:port/dbname
    """

    _DDL = """
        CREATE TABLE IF NOT EXISTS frames (
            frame_id    INTEGER PRIMARY KEY,
            timestamp   TEXT NOT NULL,
            location    TEXT,
            altitude_m  REAL,
            description TEXT,
            summary     TEXT,
            image_path  TEXT
        );
        CREATE TABLE IF NOT EXISTS detections (
            id               SERIAL PRIMARY KEY,
            frame_id         INTEGER REFERENCES frames(frame_id),
            label            TEXT,
            subtype          TEXT,
            color            TEXT,
            make             TEXT,
            model            TEXT,
            activity         TEXT,
            confidence       REAL,
            attributes_json  TEXT
        );
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id           TEXT PRIMARY KEY,
            rule               TEXT,
            severity           TEXT,
            message            TEXT,
            timestamp          TEXT,
            evidence_frame_ids TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_frames_ts  ON frames(timestamp);
        CREATE INDEX IF NOT EXISTS idx_det_label  ON detections(label, subtype);
        CREATE INDEX IF NOT EXISTS idx_frames_loc ON frames(location);
    """

    def __init__(self, database_url: str):
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise RuntimeError("Run: pip install psycopg2-binary")
        self._url = database_url
        self._conn = psycopg2.connect(database_url)
        self._conn.autocommit = False
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self._DDL)
        self._conn.commit()

    def _cur(self):
        import psycopg2
        try:
            return self._conn.cursor()
        except psycopg2.InterfaceError:
            # Reconnect if connection was dropped
            import psycopg2 as pg
            self._conn = pg.connect(self._url)
            self._conn.autocommit = False
            return self._conn.cursor()

    def insert_analyzed_frame(self, af: AnalyzedFrame) -> None:
        with self._cur() as cur:
            cur.execute(
                """INSERT INTO frames (frame_id,timestamp,location,altitude_m,description,summary,image_path)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (frame_id) DO UPDATE SET
                     timestamp=EXCLUDED.timestamp, location=EXCLUDED.location,
                     altitude_m=EXCLUDED.altitude_m, description=EXCLUDED.description,
                     summary=EXCLUDED.summary, image_path=EXCLUDED.image_path""",
                (af.frame.frame_id, af.frame.timestamp.isoformat(), af.telemetry.location,
                 af.telemetry.altitude_m, af.frame.description, af.summary, af.frame.image_path),
            )
            for d in af.detections:
                cur.execute(
                    """INSERT INTO detections
                       (frame_id,label,subtype,color,make,model,activity,confidence,attributes_json)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (af.frame.frame_id, d.label, d.subtype, d.attributes.get("color"),
                     d.attributes.get("make"), d.attributes.get("model"), d.activity,
                     d.confidence, json.dumps(d.attributes)),
                )
        self._conn.commit()

    def insert_alert(self, alert: Alert) -> None:
        with self._cur() as cur:
            cur.execute(
                """INSERT INTO alerts (alert_id,rule,severity,message,timestamp,evidence_frame_ids)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (alert_id) DO NOTHING""",
                (alert.alert_id, alert.rule, alert.severity, alert.message,
                 alert.timestamp.isoformat(), json.dumps(alert.evidence_frame_ids)),
            )
        self._conn.commit()

    def query_by_time(self, start: str | datetime, end: str | datetime) -> list[dict]:
        s = start.isoformat() if isinstance(start, datetime) else start
        e = end.isoformat() if isinstance(end, datetime) else end
        with self._cur() as cur:
            cur.execute(
                "SELECT frame_id,timestamp,location,altitude_m,description,summary "
                "FROM frames WHERE timestamp BETWEEN %s AND %s ORDER BY timestamp", (s, e)
            )
            return [self._row(r) for r in cur.fetchall()]

    def query_by_object(self, label=None, subtype=None, color=None) -> list[dict]:
        conds, params = [], []
        for col, val in [("d.label", label), ("d.subtype", subtype), ("d.color", color)]:
            if val:
                conds.append(f"{col} = %s")
                params.append(val.lower())
        where = " AND ".join(conds) if conds else "1=1"
        with self._cur() as cur:
            cur.execute(
                f"SELECT DISTINCT f.frame_id,f.timestamp,f.location,f.altitude_m,f.description,f.summary "
                f"FROM frames f JOIN detections d ON d.frame_id=f.frame_id WHERE {where} ORDER BY f.timestamp",
                params,
            )
            return [self._row(r) for r in cur.fetchall()]

    def query_by_location(self, name: str) -> list[dict]:
        with self._cur() as cur:
            cur.execute(
                "SELECT frame_id,timestamp,location,altitude_m,description,summary "
                "FROM frames WHERE LOWER(location) LIKE %s ORDER BY timestamp",
                (f"%{name.lower()}%",),
            )
            return [self._row(r) for r in cur.fetchall()]

    def get_all_alerts(self, severity: str | None = None,
                       since: str | datetime | None = None) -> list[dict]:
        conds, params = [], []
        if severity:
            conds.append("severity = %s")
            params.append(severity.lower())
        if since:
            conds.append("timestamp >= %s")
            params.append(since.isoformat() if isinstance(since, datetime) else since)
        where = " AND ".join(conds) if conds else "1=1"
        with self._cur() as cur:
            cur.execute(
                f"SELECT alert_id,rule,severity,message,timestamp,evidence_frame_ids "
                f"FROM alerts WHERE {where} ORDER BY timestamp",
                params,
            )
            return [{"alert_id": r[0], "rule": r[1], "severity": r[2], "message": r[3],
                     "timestamp": r[4], "evidence_frame_ids": json.loads(r[5])}
                    for r in cur.fetchall()]

    def frame_count(self) -> int:
        with self._cur() as cur:
            cur.execute("SELECT COUNT(*) FROM frames")
            return cur.fetchone()[0]

    def clear(self) -> None:
        """
        Delete ALL data from the database.
        ONLY called from the Reset button in app.py.
        Nothing in the pipeline or agent calls this — data is preserved across runs.
        """
        with self._cur() as cur:
            cur.execute("DELETE FROM alerts")
            cur.execute("DELETE FROM detections")
            cur.execute("DELETE FROM frames")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row(r: tuple) -> dict:
        return {"frame_id": r[0], "timestamp": r[1], "location": r[2],
                "altitude_m": r[3], "description": r[4], "summary": r[5]}


# ---------------------------------------------------------------------------
# SQLiteStore — used by tests only (no DATABASE_URL required)
# ---------------------------------------------------------------------------

class SQLiteStore:
    """SQLite store for automated tests. Not used in production."""

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
                make       TEXT, model TEXT, activity TEXT,
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
                conds.append(f"{col} = ?")
                params.append(val.lower())
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

    def get_all_alerts(self, severity: str | None = None,
                       since: str | datetime | None = None) -> list[dict]:
        conds, params = [], []
        if severity:
            conds.append("severity = ?")
            params.append(severity.lower())
        if since:
            conds.append("timestamp >= ?")
            params.append(since.isoformat() if isinstance(since, datetime) else since)
        where = " AND ".join(conds) if conds else "1=1"
        rows = self._conn.execute(
            f"SELECT alert_id,rule,severity,message,timestamp,evidence_frame_ids "
            f"FROM alerts WHERE {where} ORDER BY timestamp",
            params,
        ).fetchall()
        return [{"alert_id": r[0], "rule": r[1], "severity": r[2], "message": r[3],
                 "timestamp": r[4], "evidence_frame_ids": json.loads(r[5])} for r in rows]

    def frame_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def clear(self) -> None:
        self._conn.executescript("DELETE FROM alerts; DELETE FROM detections; DELETE FROM frames;")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row(r: tuple) -> dict:
        return {"frame_id": r[0], "timestamp": r[1], "location": r[2],
                "altitude_m": r[3], "description": r[4], "summary": r[5]}


# ---------------------------------------------------------------------------
# Factory — returns PostgresStore if DATABASE_URL is set, else SQLiteStore
# ---------------------------------------------------------------------------

def make_store(database_url: str | None = None, db_path: Path | None = None):
    """
    Returns PostgresStore when DATABASE_URL is configured, SQLiteStore otherwise.
    This is the single place that decides which backend to use.
    """
    if database_url:
        return PostgresStore(database_url)
    from drone_agent.config import DB_PATH
    return SQLiteStore(db_path or DB_PATH)


# ---------------------------------------------------------------------------
# Vector store (ChromaDB + Nomic embed for semantic search)
# ---------------------------------------------------------------------------

class VectorStore:
    """
    ChromaDB + Nomic embed text for semantic search over frame descriptions.

    WHY nomic-embed-text-v1.5:
      - Matryoshka model — dimensionality can be reduced without quality loss
      - Separate task_type for indexing (search_document) vs querying (search_query)
      - API-based, no large local model download required
    """

    def __init__(self, persist_path: str | Path):
        try:
            import chromadb
            import nomic.embed  # noqa: F401
        except ImportError:
            raise RuntimeError("Run: pip install chromadb nomic")
        import chromadb as _chroma
        try:
            self._client = _chroma.PersistentClient(path=str(persist_path))
        except Exception:
            # Read-only filesystem (Streamlit Cloud) — in-memory fallback.
            self._client = _chroma.EphemeralClient()
        self._col = self._client.get_or_create_collection(
            "drone_frames", metadata={"hnsw:space": "cosine"})

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        from nomic import embed
        return embed.text(texts, model="nomic-embed-text-v1.5",
                          task_type="search_document")["embeddings"]

    def _embed_query(self, query: str) -> list[float]:
        from nomic import embed
        return embed.text([query], model="nomic-embed-text-v1.5",
                          task_type="search_query")["embeddings"][0]

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
# SearchEngine — unified API over store + optional vector store
# ---------------------------------------------------------------------------

class SearchEngine:
    """Single entry point for all queries. Accepts PostgresStore or SQLiteStore."""

    def __init__(self, store: PostgresStore | SQLiteStore, vector: VectorStore | None = None):
        self._db = store
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
        """SQL pre-filter then vector re-rank."""
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
            return self._db.query_by_object(label=f.get("label"),
                                             subtype=f.get("subtype"),
                                             color=f.get("color"))
        if f.get("start_time") and f.get("end_time"):
            return self._db.query_by_time(f["start_time"], f["end_time"])
        if f.get("location"):
            return self._db.query_by_location(f["location"])
        return self._db.query_by_time("1970-01-01T00:00:00", "2099-12-31T23:59:59")
