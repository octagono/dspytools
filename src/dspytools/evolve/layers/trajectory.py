"""H3 Trajectory Layer — execution trace storage, replay, and diff.

harness-so pattern: every execution is stored as a trajectory.
  - record(inputs, outputs, metadata) → SQLite storage
  - replay(run_id) → replay entire trajectory
  - diff(run_a, run_b) → compare two trajectories
  - search(query) → find patterns across trajectories
"""

from __future__ import annotations

import json as _json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class TrajectoryLayer:
    """H3: Stores execution traces for analysis, replay, and pattern mining.

    Every action invocation stores: timestamp, inputs, outputs, score, metadata.
    Enables: replay, diff-ing runs, searching execution patterns.
    """

    _conn: sqlite3.Connection | None = None
    _lock = threading.Lock()

    @classmethod
    def _db_path(cls) -> Path:
        from dspytools.config.settings import trajectories_db_path as _path

        return _path()

    @classmethod
    def _get_db(cls) -> sqlite3.Connection:
        if cls._conn is None:
            db_path = cls._db_path()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            cls._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            cls._conn.row_factory = sqlite3.Row
            cls._init_db()
        return cls._conn

    @classmethod
    def _init_db(cls) -> None:
        db = cls._get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                action_name TEXT NOT NULL,
                timestamp REAL NOT NULL,
                inputs TEXT NOT NULL,
                outputs TEXT NOT NULL,
                score REAL DEFAULT 0.0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON trajectories(run_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_action ON trajectories(action_name)")
        db.commit()

    @classmethod
    def record(
        cls,
        run_id: str,
        action_name: str,
        inputs: dict,
        outputs: dict,
        score: float = 0.0,
        metadata: dict | None = None,
    ) -> int:
        """Record an execution trace."""
        with cls._lock:
            db = cls._get_db()
            cursor = db.execute(
                "INSERT INTO trajectories (run_id, action_name, timestamp, inputs, outputs, score, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    action_name,
                    time.time(),
                    _json.dumps(inputs, default=str),
                    _json.dumps(outputs, default=str),
                    score,
                    _json.dumps(metadata or {}, default=str),
                ),
            )
            db.commit()
            return cursor.lastrowid or 0

    @classmethod
    def replay(cls, run_id: str) -> list[dict]:
        """Replay an entire execution trajectory."""
        with cls._lock:
            db = cls._get_db()
            rows = db.execute(
                "SELECT * FROM trajectories WHERE run_id = ? ORDER BY timestamp",
                (run_id,),
            ).fetchall()
        return [cls._row_to_dict(r) for r in rows]

    @classmethod
    def diff(cls, run_a: str, run_b: str) -> dict:
        """Compare two trajectories."""
        traj_a = cls.replay(run_a)
        traj_b = cls.replay(run_b)

        return {
            "run_a": {
                "id": run_a,
                "steps": len(traj_a),
                "avg_score": cls._avg_score(traj_a),
            },
            "run_b": {
                "id": run_b,
                "steps": len(traj_b),
                "avg_score": cls._avg_score(traj_b),
            },
            "winner": "a" if cls._avg_score(traj_a) >= cls._avg_score(traj_b) else "b",
            "score_diff": cls._avg_score(traj_a) - cls._avg_score(traj_b),
        }

    @classmethod
    def search(
        cls, action_name: str | None = None, min_score: float = 0.0, limit: int = 20
    ) -> list[dict]:
        """Search across trajectories."""
        query = "SELECT * FROM trajectories WHERE score >= ?"
        params: list[Any] = [min_score]
        if action_name:
            query += " AND action_name = ?"
            params.append(action_name)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with cls._lock:
            db = cls._get_db()
            rows = db.execute(query, params).fetchall()
        return [cls._row_to_dict(r) for r in rows]

    @classmethod
    def stats(cls, action_name: str | None = None) -> dict:
        """Aggregate statistics for trajectories."""
        where = "WHERE action_name = ?" if action_name else ""
        params = [action_name] if action_name else []

        with cls._lock:
            db = cls._get_db()
            total = db.execute(
                f"SELECT COUNT(*) as c FROM trajectories {where}", params
            ).fetchone()
            avg = db.execute(
                f"SELECT AVG(score) as a FROM trajectories {where}", params
            ).fetchone()
            recent = db.execute(
                f"SELECT * FROM trajectories {where} ORDER BY timestamp DESC LIMIT 5",
                params,
            ).fetchall()

        return {
            "total_traces": total["c"] if total else 0,
            "average_score": round(avg["a"], 3) if avg and avg["a"] else 0.0,
            "recent": [cls._row_to_dict(r) for r in recent],
        }

    @classmethod
    def _row_to_dict(cls, row: sqlite3.Row) -> dict:
        d = dict(row)
        inp = d.get("inputs", "{}")
        if isinstance(inp, str):
            d["inputs"] = _json.loads(inp)
        out = d.get("outputs", "{}")
        if isinstance(out, str):
            d["outputs"] = _json.loads(out)
        return d

    @staticmethod
    def _avg_score(trajs: list[dict]) -> float:
        if not trajs:
            return 0.0
        return sum(t.get("score", 0) for t in trajs) / len(trajs)
