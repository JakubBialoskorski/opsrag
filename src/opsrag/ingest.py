import contextlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import psycopg

from opsrag import store
from opsrag.chunking import SLUGGERS, chunk_markdown
from opsrag.config import Settings
from opsrag.ollama import Ollama
from opsrag.sources import iter_documents, load_lock, load_sources, sync

PROGRESS_KEY = "ingest_progress"


@dataclass
class IngestResult:
    source: str
    version: str
    updated: int = 0
    unchanged: int = 0
    relinked: int = 0
    deleted: int = 0
    chunks: int = 0


def check_embedding_model(conn: psycopg.Connection, settings: Settings, reindex: bool) -> None:
    """Vectors from different models are not comparable, so a model change means a full rebuild."""
    wanted = f"{settings.embed_model}:{settings.embed_dim}"
    current = store.get_meta(conn, "embedding_model")
    if current not in (None, wanted) and not reindex:
        raise RuntimeError(
            f"index was built with {current}, but settings use {wanted}; run with --reindex"
        )
    if reindex:
        store.reset(conn, settings.embed_dim)
    store.set_meta(conn, "embedding_model", wanted)
    conn.commit()


def minutes_left(started: float, done: int, total: int) -> int:
    elapsed = time.monotonic() - started
    return math.ceil(elapsed / done * (total - done) / 60)


def save_progress(conn: psycopg.Connection, source: str, done: int, total: int, eta: int) -> None:
    """Stored in the database so the web UI can show indexing progress."""
    progress = {"source": source, "done": done, "total": total, "minutes_left": eta}
    store.set_meta(conn, PROGRESS_KEY, json.dumps({**progress, "updated_at": time.time()}))


def ingest(
    conn: psycopg.Connection,
    ollama: Ollama,
    settings: Settings,
    only: list[str] | None = None,
    reindex: bool = False,
    log: Callable[[str], None] = print,
) -> list[IngestResult]:
    store.migrate(conn)
    check_embedding_model(conn, settings, reindex)
    lock = load_lock(settings.sources_file)
    sources = [s for s in load_sources(settings.sources_file) if not only or s.name in only]
    unknown = set(only or []) - {s.name for s in sources}
    if unknown:
        raise ValueError(f"unknown source: {', '.join(sorted(unknown))}")

    results = []
    try:
        for source in sources:
            if source.name not in lock:
                raise RuntimeError(
                    f"{source.name} is not in sources.lock.yaml, run `opsrag update`"
                )
            locked = lock[source.name]
            result = IngestResult(source=source.name, version=locked.version)
            log(f"{source.name}: downloading docs ({locked.version})")
            checkout = sync(source, locked, settings.data_dir)

            slug = SLUGGERS[source.anchors]
            known = store.known_documents(conn, source.name)
            anchors = store.chunk_anchors(conn, source.name)
            docs = list(iter_documents(source, checkout))
            changed = [d for d in docs if known.get(d.path, ("", ""))[0] != d.content_hash]
            result.unchanged = len(docs) - len(changed)

            # Link fixes (page URL or #section ids) don't need new embeddings
            changed_paths = {d.path for d in changed}
            for doc in docs:
                if doc.path not in known or doc.path in changed_paths:
                    continue
                chunks = chunk_markdown(doc.title, doc.body, slug, doc.heading_ids)
                new_anchors = [c.anchor for c in chunks]
                old_anchors = anchors.get(doc.path, [])
                if len(new_anchors) != len(old_anchors):
                    # The chunking itself changed, so this document needs new embeddings
                    changed.append(doc)
                    result.unchanged -= 1
                    continue
                url_changed = known[doc.path][1] != doc.url
                if url_changed:
                    store.update_url(conn, source.name, doc.path, doc.url)
                if new_anchors != old_anchors:
                    store.update_anchors(conn, source.name, doc.path, new_anchors)
                if url_changed or new_anchors != old_anchors:
                    result.relinked += 1
            conn.commit()
            if changed:
                log(f"{source.name}: indexing {len(changed)} new or changed documents")

            started = time.monotonic()
            for doc in changed:
                # Documents without usable text are stored with no chunks, so they count as
                # unchanged next time instead of being re-processed on every run
                chunks = chunk_markdown(doc.title, doc.body, slug, doc.heading_ids)
                texts = [f"{settings.doc_prefix}{c.heading_path}\n\n{c.content}" for c in chunks]
                store.replace_document(conn, doc, chunks, ollama.embed(texts))
                result.updated += 1
                result.chunks += len(chunks)
                if result.updated % 25 == 0 or result.updated == len(changed):
                    eta = minutes_left(started, result.updated, len(changed))
                    save_progress(conn, source.name, result.updated, len(changed), eta)
                    log(
                        f"{source.name}: {result.updated}/{len(changed)} documents, ~{eta} min left"
                    )
                # Commit per document, so an interrupted ingest can simply be restarted
                conn.commit()

            removed = sorted(set(known) - {d.path for d in docs})
            if removed:
                store.delete_documents(conn, source.name, removed)
            result.deleted = len(removed)
            store.set_meta(conn, f"version:{source.name}", locked.version)
            conn.commit()
            results.append(result)
    finally:
        # Best effort: the connection may already be broken
        with contextlib.suppress(psycopg.Error):
            conn.rollback()
            store.delete_meta(conn, PROGRESS_KEY)
            conn.commit()
    return results
