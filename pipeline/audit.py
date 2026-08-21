"""SQLite audit log — every decision at every stage."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_DB_PATH = Path(__file__).resolve().parent.parent / "audit" / "audit_log.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT,
  timestamp TEXT,
  record_id TEXT,
  side TEXT,
  matched_to TEXT,
  decision TEXT,
  resolved_by TEXT,
  confidence REAL,
  reason_code TEXT,
  rationale TEXT,
  shap_json TEXT,
  tool_trace_json TEXT
);
"""


class AuditLogger:
    def __init__(self, run_id: str, db_path: Path | None = None):
        self.run_id = run_id
        self.db_path = db_path or AUDIT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # Add tool_trace_json column if running against an existing DB
            try:
                conn.execute("ALTER TABLE audit_log ADD COLUMN tool_trace_json TEXT")
            except Exception:
                pass  # column already exists

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log(
        self,
        *,
        record_id: str,
        side: str,
        decision: str,
        resolved_by: str,
        matched_to: str | None = None,
        confidence: float | None = None,
        reason_code: str | None = None,
        rationale: str | None = None,
        shap_json: dict[str, float] | None = None,
        tool_trace: list[dict] | None = None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        shap_str = json.dumps(shap_json) if shap_json else None
        trace_str = json.dumps(tool_trace) if tool_trace else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_log
                (run_id, timestamp, record_id, side, matched_to, decision,
                 resolved_by, confidence, reason_code, rationale, shap_json, tool_trace_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    ts,
                    record_id,
                    side,
                    matched_to,
                    decision,
                    resolved_by,
                    confidence,
                    reason_code,
                    rationale,
                    shap_str,
                    trace_str,
                ),
            )

    def fetch_run(self, run_id: str | None = None) -> list[dict[str, Any]]:
        rid = run_id or self.run_id
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE run_id = ? ORDER BY id",
                (rid,),
            ).fetchall()
        return [dict(r) for r in rows]
