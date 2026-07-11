"""Append-only, hash-chained SQLite audit log — the single source of truth.

Every observation, decision, guard verdict, order, fill, and breach is persisted
here as a tamper-evident chain. Account state is *derived* by replaying this log,
never trusted from a mutable field.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterator, Optional

from trading_agent.models import AuditEvent, EventKind, utcnow


def _canon(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


class AuditLog:
    GENESIS = "0" * 64

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT NOT NULL,
                cycle_id  TEXT NOT NULL,
                kind      TEXT NOT NULL,
                payload   TEXT NOT NULL,
                llm_raw   TEXT,
                prev_hash TEXT NOT NULL,
                hash      TEXT NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind)")
        self.conn.commit()

    def _last_hash(self) -> str:
        row = self.conn.execute("SELECT hash FROM events ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else self.GENESIS

    def append(
        self,
        kind: EventKind | str,
        payload: dict,
        cycle_id: str = "",
        llm_raw: Optional[str] = None,
    ) -> AuditEvent:
        kind = kind.value if isinstance(kind, EventKind) else str(kind)
        ts = utcnow().isoformat()
        payload_json = _canon(payload)
        prev_hash = self._last_hash()
        h = hashlib.sha256(f"{prev_hash}|{kind}|{cycle_id}|{payload_json}".encode()).hexdigest()

        cur = self.conn.execute(
            "INSERT INTO events (ts, cycle_id, kind, payload, llm_raw, prev_hash, hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts, cycle_id, kind, payload_json, llm_raw, prev_hash, h),
        )
        self.conn.commit()
        return AuditEvent(
            id=cur.lastrowid,
            ts=_parse_ts(ts),
            cycle_id=cycle_id,
            kind=EventKind(kind),
            payload=payload,
            llm_raw=llm_raw,
            prev_hash=prev_hash,
            hash=h,
        )

    def iter_events(self) -> Iterator[AuditEvent]:
        for row in self.conn.execute(
            "SELECT id, ts, cycle_id, kind, payload, llm_raw, prev_hash, hash FROM events ORDER BY id"
        ):
            yield AuditEvent(
                id=row[0],
                ts=_parse_ts(row[1]),
                cycle_id=row[2],
                kind=EventKind(row[3]),
                payload=json.loads(row[4]),
                llm_raw=row[5],
                prev_hash=row[6],
                hash=row[7],
            )

    def verify(self) -> bool:
        """Recompute every hash from genesis; True if the chain is intact."""
        prev = self.GENESIS
        for row in self.conn.execute(
            "SELECT kind, cycle_id, payload, prev_hash, hash FROM events ORDER BY id"
        ):
            kind, cycle_id, payload_json, prev_hash, h = row
            if prev_hash != prev:
                return False
            expected = hashlib.sha256(
                f"{prev}|{kind}|{cycle_id}|{payload_json}".encode()
            ).hexdigest()
            if expected != h:
                return False
            prev = h
        return True

    def close(self) -> None:
        self.conn.close()


def _parse_ts(s: str):
    from datetime import datetime

    return datetime.fromisoformat(s)
