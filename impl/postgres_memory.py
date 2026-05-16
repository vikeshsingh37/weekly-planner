"""
Postgres-backed session manager.

Stores each user's full session state as a JSONB column in `planner_sessions`.
All in-memory operations (task mutations, locking) are inherited from
_BaseSessionManager — only persistence is different here.
"""

import json

import psycopg

from impl.memory import _BaseSessionManager

_DDL = """
CREATE TABLE IF NOT EXISTS planner_sessions (
    user_id    TEXT PRIMARY KEY,
    state      JSONB        NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

# Table is created once per process, not once per request.
_table_ensured: set[str] = set()


def ensure_table(conninfo: str) -> None:
    if conninfo in _table_ensured:
        return
    with psycopg.connect(conninfo) as conn:
        conn.execute(_DDL)
    _table_ensured.add(conninfo)


class PostgresSessionManager(_BaseSessionManager):
    def __init__(self, user_id: str, conninfo: str):
        super().__init__()
        self._user_id = user_id
        self._conninfo = conninfo
        ensure_table(conninfo)
        self._load()

    def save(self) -> None:
        data = self._snapshot()
        with psycopg.connect(self._conninfo) as conn:
            conn.execute(
                """
                INSERT INTO planner_sessions (user_id, state, updated_at)
                VALUES (%s, %s::jsonb, NOW())
                ON CONFLICT (user_id) DO UPDATE
                    SET state = EXCLUDED.state, updated_at = NOW()
                """,
                (self._user_id, json.dumps(data)),
            )

    def reload(self) -> None:
        self._load()

    def _load(self) -> None:
        with psycopg.connect(self._conninfo) as conn:
            row = conn.execute(
                "SELECT state FROM planner_sessions WHERE user_id = %s",
                (self._user_id,),
            ).fetchone()
        if row:
            self._restore(row[0])