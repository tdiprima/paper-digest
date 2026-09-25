"""SQLite + FTS5 store over PDF page chunks. One chunk = a few hundred words from one page.

FTS5 gives BM25 ranking with no embedding model, which keeps the install small
and works offline. Technical queries (method names, dataset names) rank well
under BM25; the agent is told to rephrase queries to compensate for the lack of
semantic matching.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date

from pathlib import Path

from .config import INDEX_DIR

DB_PATH = INDEX_DIR / "papers.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id  TEXT PRIMARY KEY,
    title     TEXT NOT NULL,
    published TEXT NOT NULL,
    pages     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    arxiv_id  TEXT NOT NULL REFERENCES papers(arxiv_id),
    page      INTEGER NOT NULL,
    text      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_paper_page ON chunks(arxiv_id, page);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


@dataclass
class Chunk:
    arxiv_id: str
    title: str
    published: str  # ISO date
    page: int
    text: str
    score: float | None = None


@dataclass
class Paper:
    arxiv_id: str
    title: str
    published: str
    pages: int


def _fts_query(q: str) -> str:
    """Turn free text into an OR-joined FTS5 query so partial matches still rank."""
    terms = [t for t in re.findall(r"[A-Za-z0-9_-]+", q) if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in terms) or '""'


class PaperStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    # ---- writes -------------------------------------------------------

    def has_paper(self, arxiv_id: str) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return row is not None

    def add_chunks(self, chunks: list[Chunk], pages: int) -> None:
        if not chunks:
            return
        first = chunks[0]
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO papers VALUES (?, ?, ?, ?)",
                (first.arxiv_id, first.title, first.published, pages),
            )
            self._db.executemany(
                "INSERT INTO chunks (arxiv_id, page, text) VALUES (?, ?, ?)",
                [(c.arxiv_id, c.page, c.text) for c in chunks],
            )

    # ---- reads --------------------------------------------------------

    def search(self, query: str, k: int = 8, since: date | None = None) -> list[Chunk]:
        # bm25() must be computed in a query where the FTS table is the only
        # source; joining first can make SQLite return NULL for it.
        sql = """
            SELECT c.arxiv_id, p.title, p.published, c.page, c.text, f.score
            FROM (SELECT rowid, bm25(chunks_fts) AS score
                  FROM chunks_fts WHERE chunks_fts MATCH ?) f
            JOIN chunks c ON c.id = f.rowid
            JOIN papers p ON p.arxiv_id = c.arxiv_id
            WHERE 1 = 1
        """
        args: list = [_fts_query(query)]
        if since:
            sql += " AND p.published >= ?"
            args.append(since.isoformat())
        sql += " ORDER BY f.score LIMIT ?"
        args.append(k)
        rows = self._db.execute(sql, args).fetchall()
        return [
            Chunk(r["arxiv_id"], r["title"], r["published"], r["page"], r["text"], -r["score"])
            for r in rows
        ]

    def list_papers(self, since: date | None = None) -> list[Paper]:
        sql = "SELECT * FROM papers"
        args: list = []
        if since:
            sql += " WHERE published >= ?"
            args.append(since.isoformat())
        sql += " ORDER BY published DESC"
        return [Paper(**dict(r)) for r in self._db.execute(sql, args)]

    def get_pages(self, arxiv_id: str, pages: list[int]) -> list[Chunk]:
        if not pages:
            return []
        marks = ",".join("?" * len(pages))
        rows = self._db.execute(
            f"""
            SELECT c.arxiv_id, p.title, p.published, c.page, c.text
            FROM chunks c JOIN papers p ON p.arxiv_id = c.arxiv_id
            WHERE c.arxiv_id = ? AND c.page IN ({marks})
            ORDER BY c.page, c.id
            """,
            [arxiv_id, *pages],
        ).fetchall()
        return [Chunk(r["arxiv_id"], r["title"], r["published"], r["page"], r["text"]) for r in rows]
