"""SQLite 存储层（标准库 sqlite3，线程安全封装）。"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "detections.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                    TEXT PRIMARY KEY,
    series_id             TEXT,
    created_at            TEXT NOT NULL,
    request_params        TEXT NOT NULL,
    result_json           TEXT NOT NULL,
    point_count           INTEGER NOT NULL,
    anomaly_segment_count INTEGER NOT NULL,
    anomaly_point_count   INTEGER NOT NULL,
    threshold             REAL NOT NULL,
    window_size           INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_series_id ON detections(series_id);
CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections(created_at);
"""


def init_db(db_path: str | None = None) -> None:
    """建库建表并启用 WAL。"""
    db_path = db_path or DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)


@contextmanager
def get_conn(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path or DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=15000")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_detection(
    detection_id: str,
    series_id: str | None,
    request_params: dict[str, Any],
    result: dict[str, Any],
    db_path: str | None = None,
) -> str:
    created_at = datetime.now(timezone.utc).isoformat()
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO detections (
                id, series_id, created_at, request_params, result_json,
                point_count, anomaly_segment_count, anomaly_point_count,
                threshold, window_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                detection_id,
                series_id,
                created_at,
                json.dumps(request_params, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                result["point_count"],
                result["anomaly_segment_count"],
                result["anomaly_point_count"],
                result["threshold"],
                result["window_size"],
            ),
        )
    return created_at


def get_detection(detection_id: str, db_path: str | None = None) -> dict[str, Any] | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM detections WHERE id = ?", (detection_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "series_id": row["series_id"],
        "created_at": row["created_at"],
        "request_params": json.loads(row["request_params"]),
        "result": json.loads(row["result_json"]),
    }


def list_detections(
    series_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db_path: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    with get_conn(db_path) as conn:
        if series_id is not None:
            total = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE series_id = ?", (series_id,)
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT id, series_id, created_at, point_count,
                       anomaly_segment_count, anomaly_point_count,
                       threshold, window_size
                FROM detections WHERE series_id = ?
                ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                (series_id, limit, offset),
            ).fetchall()
        else:
            total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            rows = conn.execute(
                """
                SELECT id, series_id, created_at, point_count,
                       anomaly_segment_count, anomaly_point_count,
                       threshold, window_size
                FROM detections ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
    return [dict(r) for r in rows], total
