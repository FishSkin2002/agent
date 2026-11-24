import json
import sqlite3
from heapq import nlargest
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Paper(TypedDict):
    paper_id: str
    title: str
    authors: list[str]
    submitted_at: str
    abstract: str
    url: str
    journal_ref: str | None


class EmbeddingError(RuntimeError):
    """The local embedding model could not return usable vectors."""


class PaperLibrary:
    def __init__(self, path: Path) -> None:
        self._model = None
        self._model_cache = path.parent / "embedding-models"
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        try:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding BLOB
                );
                CREATE TABLE IF NOT EXISTS passages (
                    paper_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    page INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (paper_id, position)
                );
            """)
        except sqlite3.Error:
            self.close()
            raise

    def close(self) -> None:
        self._db.close()

    def save_papers(self, papers: list[Paper]) -> None:
        with self._db:
            self._db.executemany(
                """
                INSERT INTO papers (paper_id, title, abstract, metadata) VALUES (?, ?, ?, ?)
                ON CONFLICT (paper_id) DO UPDATE SET
                    title = excluded.title,
                    abstract = excluded.abstract,
                    metadata = excluded.metadata,
                    embedding = CASE
                        WHEN papers.title = excluded.title AND papers.abstract = excluded.abstract
                        THEN papers.embedding
                    END
                """,
                [(p["paper_id"], p["title"], p["abstract"], json.dumps(p, ensure_ascii=False)) for p in papers],
            )

    def search_papers(self, query: str) -> list[dict[str, Any]]:
        # Embed newly discovered abstracts on the first local search.
        pending = self._db.execute(
            "SELECT paper_id, title, abstract FROM papers WHERE embedding IS NULL"
        ).fetchall()
        if pending:
            texts = [f'{row["title"]}\n\n{row["abstract"]}' for row in pending]
            vectors = self._embed(texts)
            with self._db:
                self._db.executemany(
                    """
                    UPDATE papers SET embedding = ?
                    WHERE paper_id = ? AND title = ? AND abstract = ? AND embedding IS NULL
                    """,
                    [
                        (vector.tobytes(), row["paper_id"], row["title"], row["abstract"])
                        for row, vector in zip(pending, vectors, strict=True)
                    ],
                )
        rows = self._db.execute(
            """
            SELECT metadata, embedding, EXISTS (
                SELECT 1 FROM passages WHERE passages.paper_id = papers.paper_id
            ) AS has_full_text
            FROM papers WHERE embedding IS NOT NULL ORDER BY paper_id
            """
        ).fetchall()
        return [
            {**json.loads(row["metadata"]), "has_full_text": bool(row["has_full_text"])}
            for row, _ in self._rank(query, rows, 5)
        ]

    def has_passages(self, paper_id: str) -> bool:
        return self._db.execute(
            "SELECT 1 FROM passages WHERE paper_id = ? LIMIT 1", (paper_id,)
        ).fetchone() is not None

    def save_passages(self, paper_id: str, passages: list[tuple[int, str]]) -> None:
        vectors = self._embed([content for _, content in passages])
        # Inserting all passages in one transaction leaves failed imports retryable.
        with self._db:
            self._db.executemany(
                """
                INSERT INTO passages (paper_id, position, page, content, embedding) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (paper_id, position) DO NOTHING
                """,
                [
                    (paper_id, position, page, content, vector.tobytes())
                    for position, ((page, content), vector) in enumerate(zip(passages, vectors, strict=True))
                ],
            )

    def search_passages(self, query: str, paper_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT paper_id, page, content, embedding FROM passages"
        if paper_id is None:
            rows = self._db.execute(sql + " ORDER BY paper_id, position").fetchall()
        else:
            rows = self._db.execute(sql + " WHERE paper_id = ? ORDER BY position", (paper_id,)).fetchall()
        return [
            {
                "paper_id": row["paper_id"],
                "page": row["page"],
                "text": row["content"],
                "url": f'https://arxiv.org/pdf/{row["paper_id"]}#page={row["page"]}',
                "similarity": score,
            }
            for row, score in self._rank(query, rows, 3)
        ]

    def _rank(
        self, query: str, rows: list[sqlite3.Row], limit: int
    ) -> list[tuple[sqlite3.Row, float]]:
        if not rows:
            return []
        query_vector = self._embed([QUERY_INSTRUCTION + query])[0]
        vectors = np.stack([np.frombuffer(row["embedding"], dtype="<f4") for row in rows])
        scores = (vectors @ query_vector).tolist()
        return nlargest(limit, zip(rows, scores, strict=True), key=lambda result: result[1])

    def _embed(self, texts: list[str]) -> NDArray[np.float32]:
        try:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(
                    model_name=EMBEDDING_MODEL, cache_dir=str(self._model_cache), threads=2
                )
            vectors = np.asarray(list(self._model.embed(texts, batch_size=16)), dtype="<f4")
        except Exception as error:
            raise EmbeddingError(f"本地嵌入模型运行失败（{error}）。首次使用需下载模型。") from error
        if vectors.shape != (len(texts), 384):
            raise EmbeddingError("本地嵌入模型返回了非预期的向量维度。")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms) & (norms > 0)):
            raise EmbeddingError("本地嵌入模型返回了无效向量。")
        return (vectors / norms).astype("<f4", copy=False)
