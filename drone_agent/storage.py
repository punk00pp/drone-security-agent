"""
storage.py — PostgreSQL store, ChromaDB vector store, and unified search API.

  PostgresStore — frames, detections, alerts scoped by run_id (UUID).
                  Each pipeline run gets its own run_id so frame_id 1-20 from
                  different scenarios never collide. The session always sees
                  only its own run. Reset deletes only that run's data.
  SQLiteStore   — kept for tests only (no DATABASE_URL needed in CI).
  VectorStore   — ChromaDB + Nomic API embeddings for semantic search.
  SearchEngine  — unified API over both stores.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from drone_agent.models import Alert, AnalyzedFrame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PostgresStore — production store, scoped by run_id
# ---------------------------------------------------------------------------

class PostgresStore:
    """
    PostgreSQL persistence scoped by run_id.

    WHY run_id:
      All scenarios use frame_id 1-20. Without run_id, running scenario_day2
      after scenario_day1 silently overwrites every row. With run_id, each run
      lives in its own namespace — the DB accumulates all runs and each session
      only sees its own.

    Schema:
      runs(run_id PK, scenario, analyzer, created_at)
      frames(run_id, frame_id, ...)   PK = (run_id, frame_id)
      detections(id SERIAL, run_id, frame_id, ...)
      alerts(alert_id PK, run_id, ...)

    clear() deletes only the rows belonging to this instance's run_id.
    It is called ONLY from the Reset button in app.py.
    """

    _DDL = """
        CREATE TABLE IF NOT EXISTS runs (
            run_id      TEXT PRIMARY KEY,
            scenario    TEXT,
            analyzer    TEXT,
            created_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS frames (
            run_id      TEXT NOT NULL,
            frame_id    INTEGER NOT NULL,
            timestamp   TEXT NOT NULL,
            location    TEXT,
            altitude_m  REAL,
            description TEXT,
            summary     TEXT,
            image_path  TEXT,
            PRIMARY KEY (run_id, frame_id)
        );
        CREATE TABLE IF NOT EXISTS detections (
            id               SERIAL PRIMARY KEY,
            run_id           TEXT NOT NULL,
            frame_id         INTEGER NOT NULL,
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
            run_id             TEXT NOT NULL,
            rule               TEXT,
            severity           TEXT,
            message            TEXT,
            timestamp          TEXT,
            evidence_frame_ids TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_frames_ts   ON frames(run_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_det_label   ON detections(run_id, label, subtype);
        CREATE INDEX IF NOT EXISTS idx_frames_loc  ON frames(run_id, location);
        CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
    """

    def __init__(self, database_url: str, run_id: str | None = None):
        """
        database_url — PostgreSQL connection string.
        run_id       — UUID for this session's run. If None a new one is created.
                       Pass an existing run_id to reconnect to a previous run
                       (e.g. after a page refresh).
        """
        try:
            import psycopg2
        except ImportError:
            raise RuntimeError("Run: pip install psycopg2-binary")
        self._url = database_url
        self._conn = psycopg2.connect(database_url)
        self._conn.autocommit = False
        self._ensure_schema()
        self.run_id: str = run_id or str(uuid.uuid4())

    def _ensure_schema(self) -> None:
        """
        Migrate the schema to the run_id-scoped version.
        Drops old tables if they exist without run_id, then recreates them.
        """
        with self._conn.cursor() as cur:
            # Check if frames table already has run_id column
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name='frames' AND column_name='run_id'
            """)
            has_run_id = cur.fetchone() is not None

            if not has_run_id:
                # Old schema — drop everything and start fresh
                cur.execute("DROP TABLE IF EXISTS alerts CASCADE")
                cur.execute("DROP TABLE IF EXISTS detections CASCADE")
                cur.execute("DROP TABLE IF EXISTS frames CASCADE")
                cur.execute("DROP TABLE IF EXISTS runs CASCADE")

            cur.execute(self._DDL)
        self._conn.commit()

    def _cur(self):
        import psycopg2
        try:
            # Test connection is alive
            self._conn.cursor().execute("SELECT 1")
            return self._conn.cursor()
        except Exception:
            import psycopg2 as pg
            self._conn = pg.connect(self._url)
            self._conn.autocommit = False
            return self._conn.cursor()

    def register_run(self, scenario: str, analyzer: str) -> None:
        """Insert a row into runs for this run_id. Called once at pipeline start."""
        with self._cur() as cur:
            cur.execute(
                """INSERT INTO runs (run_id, scenario, analyzer, created_at)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (run_id) DO NOTHING""",
                (self.run_id, scenario, analyzer, datetime.utcnow().isoformat()),
            )
        self._conn.commit()

    def get_latest_run_id(self) -> str | None:
        """Return the most recently created run_id, or None if no runs exist."""
        with self._cur() as cur:
            cur.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()
            return row[0] if row else None

    def insert_analyzed_frame(self, af: AnalyzedFrame) -> None:
        with self._cur() as cur:
            cur.execute(
                """INSERT INTO frames
                   (run_id,frame_id,timestamp,location,altitude_m,description,summary,image_path)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (run_id, frame_id) DO UPDATE SET
                     timestamp=EXCLUDED.timestamp, location=EXCLUDED.location,
                     altitude_m=EXCLUDED.altitude_m, description=EXCLUDED.description,
                     summary=EXCLUDED.summary, image_path=EXCLUDED.image_path""",
                (self.run_id, af.frame.frame_id, af.frame.timestamp.isoformat(),
                 af.telemetry.location, af.telemetry.altitude_m, af.frame.description,
                 af.summary, af.frame.image_path),
            )
            for d in af.detections:
                cur.execute(
                    """INSERT INTO detections
                       (run_id,frame_id,label,subtype,color,make,model,activity,confidence,attributes_json)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (self.run_id, af.frame.frame_id, d.label, d.subtype,
                     d.attributes.get("color"), d.attributes.get("make"),
                     d.attributes.get("model"), d.activity,
                     d.confidence, json.dumps(d.attributes)),
                )
        self._conn.commit()

    def insert_alert(self, alert: Alert) -> None:
        with self._cur() as cur:
            cur.execute(
                """INSERT INTO alerts
                   (alert_id,run_id,rule,severity,message,timestamp,evidence_frame_ids)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (alert_id) DO NOTHING""",
                (alert.alert_id, self.run_id, alert.rule, alert.severity,
                 alert.message, alert.timestamp.isoformat(),
                 json.dumps(alert.evidence_frame_ids)),
            )
        self._conn.commit()

    def query_by_time(self, start: str | datetime, end: str | datetime) -> list[dict]:
        s = start.isoformat() if isinstance(start, datetime) else start
        e = end.isoformat() if isinstance(end, datetime) else end
        with self._cur() as cur:
            cur.execute(
                "SELECT frame_id,timestamp,location,altitude_m,description,summary "
                "FROM frames WHERE run_id=%s AND timestamp BETWEEN %s AND %s "
                "ORDER BY timestamp",
                (self.run_id, s, e),
            )
            return [self._row(r) for r in cur.fetchall()]

    def query_by_object(self, label=None, subtype=None, color=None) -> list[dict]:
        conds = ["f.run_id = %s"]
        params: list = [self.run_id]
        for col, val in [("d.label", label), ("d.subtype", subtype), ("d.color", color)]:
            if val:
                conds.append(f"{col} = %s")
                params.append(val.lower())
        where = " AND ".join(conds)
        with self._cur() as cur:
            cur.execute(
                f"SELECT DISTINCT f.frame_id,f.timestamp,f.location,f.altitude_m,"
                f"f.description,f.summary "
                f"FROM frames f JOIN detections d "
                f"ON d.run_id=f.run_id AND d.frame_id=f.frame_id "
                f"WHERE {where} ORDER BY f.timestamp",
                params,
            )
            return [self._row(r) for r in cur.fetchall()]

    def query_by_location(self, name: str) -> list[dict]:
        with self._cur() as cur:
            cur.execute(
                "SELECT frame_id,timestamp,location,altitude_m,description,summary "
                "FROM frames WHERE run_id=%s AND LOWER(location) LIKE %s "
                "ORDER BY timestamp",
                (self.run_id, f"%{name.lower()}%"),
            )
            return [self._row(r) for r in cur.fetchall()]

    def get_all_alerts(self, severity: str | None = None,
                       since: str | datetime | None = None) -> list[dict]:
        conds = ["run_id = %s"]
        params: list = [self.run_id]
        if severity:
            conds.append("severity = %s")
            params.append(severity.lower())
        if since:
            conds.append("timestamp >= %s")
            params.append(since.isoformat() if isinstance(since, datetime) else since)
        where = " AND ".join(conds)
        with self._cur() as cur:
            cur.execute(
                f"SELECT alert_id,rule,severity,message,timestamp,evidence_frame_ids "
                f"FROM alerts WHERE {where} ORDER BY timestamp",
                params,
            )
            return [{"alert_id": r[0], "rule": r[1], "severity": r[2],
                     "message": r[3], "timestamp": r[4],
                     "evidence_frame_ids": json.loads(r[5])}
                    for r in cur.fetchall()]

    def get_all_alerts_global(self, severity: str | None = None,
                              since: str | datetime | None = None) -> list[dict]:
        """Return alerts across ALL runs — used by the UI to show the full history."""
        conds, params = [], []
        if severity:
            conds.append("severity = %s")
            params.append(severity.lower())
        if since:
            conds.append("timestamp >= %s")
            params.append(since.isoformat() if isinstance(since, datetime) else since)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        with self._cur() as cur:
            cur.execute(
                f"SELECT alert_id,rule,severity,message,timestamp,evidence_frame_ids "
                f"FROM alerts {where} ORDER BY timestamp",
                params,
            )
            return [{"alert_id": r[0], "rule": r[1], "severity": r[2],
                     "message": r[3], "timestamp": r[4],
                     "evidence_frame_ids": json.loads(r[5])}
                    for r in cur.fetchall()]

    def frame_count_global(self) -> int:
        """Total frames across ALL runs."""
        with self._cur() as cur:
            cur.execute("SELECT COUNT(*) FROM frames")
            return cur.fetchone()[0]

    def frame_count(self) -> int:
        with self._cur() as cur:
            cur.execute("SELECT COUNT(*) FROM frames WHERE run_id=%s", (self.run_id,))
            return cur.fetchone()[0]

    def clear(self) -> None:
        """
        Delete all data for this run_id only.
        ONLY called from the Reset button in app.py.
        Other runs in the DB are untouched.
        """
        with self._cur() as cur:
            cur.execute("DELETE FROM alerts     WHERE run_id=%s", (self.run_id,))
            cur.execute("DELETE FROM detections WHERE run_id=%s", (self.run_id,))
            cur.execute("DELETE FROM frames     WHERE run_id=%s", (self.run_id,))
            cur.execute("DELETE FROM runs       WHERE run_id=%s", (self.run_id,))
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
                "INSERT INTO detections (frame_id,label,subtype,color,make,model,"
                "activity,confidence,attributes_json) VALUES (?,?,?,?,?,?,?,?,?)",
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
            f"SELECT DISTINCT f.frame_id,f.timestamp,f.location,f.altitude_m,"
            f"f.description,f.summary "
            f"FROM frames f JOIN detections d ON d.frame_id=f.frame_id "
            f"WHERE {where} ORDER BY f.timestamp",
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
        self._conn.executescript(
            "DELETE FROM alerts; DELETE FROM detections; DELETE FROM frames;")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row(r: tuple) -> dict:
        return {"frame_id": r[0], "timestamp": r[1], "location": r[2],
                "altitude_m": r[3], "description": r[4], "summary": r[5]}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_store(database_url: str | None = None,
               db_path: Path | None = None,
               run_id: str | None = None):
    """
    Returns PostgresStore(run_id) when DATABASE_URL is set, SQLiteStore otherwise.
    Pass run_id to reconnect to an existing run (e.g. after a page refresh).
    """
    if database_url:
        return PostgresStore(database_url, run_id=run_id)
    from drone_agent.config import DB_PATH
    return SQLiteStore(db_path or DB_PATH)


# ---------------------------------------------------------------------------
# Vector store
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
# SearchEngine
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

    def get_alerts_global(self, severity=None, since=None) -> list[dict]:
        """All alerts across every run — for the Alerts tab and sidebar counts."""
        if hasattr(self._db, 'get_all_alerts_global'):
            return self._db.get_all_alerts_global(severity=severity, since=since)
        return self._db.get_all_alerts(severity=severity, since=since)

    def frame_count_global(self) -> int:
        """Total frames across all runs."""
        if hasattr(self._db, 'frame_count_global'):
            return self._db.frame_count_global()
        return self._db.frame_count()

    def semantic(self, query: str, k: int = 5) -> list[dict]:
        if self._vec is None:
            return []
        return self._vec.semantic_search(query, k=k)

    def hybrid(self, query: str, filters: dict | None = None, k: int = 5) -> list[dict]:
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
