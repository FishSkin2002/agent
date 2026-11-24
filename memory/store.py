import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict


DEFAULT_SHORT_TERM_TURNS = 10
LONG_TERM_IMPORTANCE = 0.8


class Memory(TypedDict):
    id: int
    content: str
    importance: float
    kind: str
    written_turn: int


class MemoryStore:
    def __init__(self, path: Path, short_term_turns: int = DEFAULT_SHORT_TERM_TURNS) -> None:
        self.short_term_turns = short_term_turns
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.create_function("casefold", 1, str.casefold, deterministic=True)
        try:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS memory_clock (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    completed_turns INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO memory_clock (id, completed_turns) VALUES (1, 0);

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL CHECK (importance BETWEEN 0 AND 1),
                    kind TEXT NOT NULL CHECK (kind IN ('short_term', 'long_term')),
                    written_turn INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS short_memory_expiration
                    ON memories (written_turn) WHERE kind = 'short_term';
            """)
            with self._db:
                self._expire_short_term()
        except sqlite3.Error:
            self.close()
            raise

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def turn(self) -> Iterator[None]:
        """Commit memory changes and advance the clock together after a successful turn."""
        with self._db:
            yield
            self._db.execute("UPDATE memory_clock SET completed_turns = completed_turns + 1")
            self._expire_short_term()

    def _expire_short_term(self) -> None:
        self._db.execute(
            """
            DELETE FROM memories
            WHERE kind = 'short_term'
                AND written_turn <= (SELECT completed_turns - ? FROM memory_clock)
            """,
            (self.short_term_turns,),
        )

    def save(self, content: str, importance: float, memory_id: int | None = None) -> Memory | None:
        kind = "long_term" if importance >= LONG_TERM_IMPORTANCE else "short_term"
        if memory_id is None:
            row = self._db.execute(
                """
                INSERT INTO memories (content, importance, kind, written_turn)
                VALUES (?, ?, ?, (SELECT completed_turns + 1 FROM memory_clock))
                RETURNING *
                """,
                (content, importance, kind),
            ).fetchone()
        else:
            row = self._db.execute(
                """
                UPDATE memories
                SET content = ?, importance = ?,
                    kind = CASE WHEN kind = 'long_term' THEN kind ELSE ? END,
                    written_turn = (SELECT completed_turns + 1 FROM memory_clock)
                WHERE id = ?
                RETURNING *
                """,
                (content, importance, kind, memory_id),
            ).fetchone()
        return self._record(row) if row is not None else None

    def search(self, query: str, limit: int = 5) -> list[Memory]:
        rows = self._db.execute(
            """
            SELECT id, content, importance, kind, written_turn
            FROM memories
            WHERE instr(casefold(content), ?) > 0
            ORDER BY importance DESC, written_turn DESC, id DESC
            LIMIT ?
            """,
            (query.casefold(), limit),
        )
        return [self._record(row) for row in rows]

    def forget(self, memory_id: int) -> bool:
        result = self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return result.rowcount == 1

    @staticmethod
    def _record(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            importance=row["importance"],
            kind=row["kind"],
            written_turn=row["written_turn"],
        )
