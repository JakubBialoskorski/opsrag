from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from opsrag.chunking import Chunk
from opsrag.sources import Document, with_anchor


@dataclass
class Hit:
    chunk_id: int
    source: str
    path: str
    url: str
    title: str
    heading_path: str
    content: str
    anchor: str | None
    score: float

    @property
    def link(self) -> str:
        return with_anchor(self.url, self.anchor)


HIT_COLUMNS = "c.id, d.source, d.path, d.url, d.title, c.heading_path, c.content, c.anchor"


def connect(url: str) -> psycopg.Connection:
    conn = psycopg.connect(url)
    register_vector(conn)
    # Without this, filtering by source after an HNSW scan can return fewer rows than LIMIT
    conn.execute("SET hnsw.iterative_scan = relaxed_order")
    return conn


def migrate(conn: psycopg.Connection) -> None:
    """Schema changes for databases created by an older version (db/init.sql only runs once)."""
    conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS anchor text")
    conn.commit()


def get_meta(conn: psycopg.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = %s", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: psycopg.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (key, value),
    )


def delete_meta(conn: psycopg.Connection, key: str) -> None:
    conn.execute("DELETE FROM meta WHERE key = %s", (key,))


def reset(conn: psycopg.Connection, dim: int) -> None:
    """Drop all indexed data and resize the vector column for a new embedding model."""
    conn.execute("TRUNCATE documents, meta CASCADE")
    conn.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({int(dim)})")
    conn.execute(f"ALTER TABLE questions ALTER COLUMN embedding TYPE vector({int(dim)}) USING NULL")


def known_documents(conn: psycopg.Connection, source: str) -> dict[str, tuple[str, str]]:
    """{path: (content_hash, url)} of what is already indexed."""
    rows = conn.execute(
        "SELECT path, content_hash, url FROM documents WHERE source = %s", (source,)
    ).fetchall()
    return {path: (content_hash, url) for path, content_hash, url in rows}


def update_url(conn: psycopg.Connection, source: str, path: str, url: str) -> None:
    conn.execute(
        "UPDATE documents SET url = %s WHERE source = %s AND path = %s", (url, source, path)
    )


def replace_document(
    conn: psycopg.Connection, doc: Document, chunks: list[Chunk], embeddings: list[list[float]]
) -> None:
    conn.execute("DELETE FROM documents WHERE source = %s AND path = %s", (doc.source, doc.path))
    (doc_id,) = conn.execute(
        "INSERT INTO documents (source, path, url, title, content_hash) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (doc.source, doc.path, doc.url, doc.title, doc.content_hash),
    ).fetchone()
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, ord, heading_path, content, anchor, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [
                (doc_id, i, c.heading_path, c.content, c.anchor, np.array(e, dtype=np.float32))
                for i, (c, e) in enumerate(zip(chunks, embeddings, strict=True))
            ],
        )


def chunk_anchors(conn: psycopg.Connection, source: str) -> dict[str, list[str | None]]:
    """{path: [anchor of chunk 0, chunk 1, ...]} of what is already indexed."""
    rows = conn.execute(
        "SELECT d.path, c.anchor FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE d.source = %s ORDER BY d.path, c.ord",
        (source,),
    ).fetchall()
    anchors: dict[str, list[str | None]] = {}
    for path, anchor in rows:
        anchors.setdefault(path, []).append(anchor)
    return anchors


def update_anchors(
    conn: psycopg.Connection, source: str, path: str, anchors: list[str | None]
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks c SET anchor = %s FROM documents d "
            "WHERE d.id = c.document_id AND d.source = %s AND d.path = %s AND c.ord = %s",
            [(anchor, source, path, i) for i, anchor in enumerate(anchors)],
        )


def delete_documents(conn: psycopg.Connection, source: str, paths: list[str]) -> None:
    conn.execute(
        "DELETE FROM documents WHERE source = %s AND path = ANY(%s)", (source, list(paths))
    )


def vector_search(
    conn: psycopg.Connection, embedding: list[float], limit: int, source: str | None = None
) -> list[Hit]:
    # <=> is cosine distance, so similarity = 1 - distance
    rows = conn.execute(
        f"SELECT {HIT_COLUMNS}, 1 - (c.embedding <=> %(e)s) AS score "
        "FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE %(source)s::text IS NULL OR d.source = %(source)s "
        "ORDER BY c.embedding <=> %(e)s LIMIT %(limit)s",
        {"e": np.array(embedding, dtype=np.float32), "source": source, "limit": limit},
    ).fetchall()
    return [Hit(*row) for row in rows]


def fts_search(
    conn: psycopg.Connection, query: str, limit: int, source: str | None = None
) -> list[Hit]:
    # plainto_tsquery joins words with AND, which is too strict for natural language
    # questions, so we turn it into OR and let ts_rank put the best matches first
    rows = conn.execute(
        f"SELECT {HIT_COLUMNS}, ts_rank(c.tsv, q) AS score "
        "FROM chunks c JOIN documents d ON d.id = c.document_id, "
        "  to_tsquery('english', replace(plainto_tsquery('english', %(q)s)::text, '&', '|')) q "
        "WHERE c.tsv @@ q AND (%(source)s::text IS NULL OR d.source = %(source)s) "
        "ORDER BY score DESC LIMIT %(limit)s",
        {"q": query, "source": source, "limit": limit},
    ).fetchall()
    return [Hit(*row) for row in rows]


def stats(conn: psycopg.Connection) -> list[tuple[str, int, int]]:
    return conn.execute(
        "SELECT d.source, count(DISTINCT d.id), count(c.id) "
        "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
        "GROUP BY d.source ORDER BY d.source"
    ).fetchall()
