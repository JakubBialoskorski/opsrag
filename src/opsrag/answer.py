import re
import time
from collections.abc import Iterator

import psycopg

from opsrag import retrieval
from opsrag.config import Settings
from opsrag.ollama import Ollama
from opsrag.store import Hit

NOT_FOUND = "I couldn't find this in the documentation."

SYSTEM_PROMPT = f"""You are a DevOps documentation assistant.
Answer the question using ONLY the numbered context passages you are given.
The question can also be just a term or a few keywords, like "daemonset" or "helm hooks".
Then explain what it is and how it is used.

Rules:
- After each statement, cite the passage(s) it comes from, like [1] or [2][3].
- If the passages don't contain the answer, reply exactly: "{NOT_FOUND}"
  Do not use outside knowledge.
- Be concise. Include commands or YAML from the passages when they help.
- Only use config keys, annotations, flags, actions, functions, file paths and commands that
  appear in the passages. Never make up an example with syntax the passages don't show.
  If they describe a feature without its exact syntax, say so and point to the source."""

# 5 chunks x 2000 chars is ~2500 tokens, leaving room for the answer in a 4096 token window
MAX_CHUNK_CHARS = 2000

CITATION_RE = re.compile(r"\[(\d+)\]")


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    passages = "\n\n".join(
        f"[{i}] {hit.source}: {hit.heading_path}\n{hit.content[:MAX_CHUNK_CHARS]}"
        for i, hit in enumerate(hits, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{passages}\n\nQuestion: {question}"},
    ]


def cited(text: str, count: int) -> list[int]:
    """Passage numbers the model actually cited, ignoring made-up ones."""
    numbers = {int(n) for n in CITATION_RE.findall(text)}
    return sorted(n for n in numbers if 1 <= n <= count)


def ask(
    conn: psycopg.Connection,
    ollama: Ollama,
    settings: Settings,
    question: str,
    source: str | None = None,
) -> Iterator[dict]:
    """Run the whole RAG pipeline, yielding events: sources, then tokens, then done."""
    start = time.perf_counter()
    pool = retrieval.search(
        conn, ollama, settings, question, "hybrid", settings.rerank_pool, source
    )
    retrieve_s = time.perf_counter() - start
    start = time.perf_counter()
    hits = retrieval.reorder(settings, question, pool)[: settings.top_k]
    rerank_s = time.perf_counter() - start
    yield {
        "type": "sources",
        "sources": [
            {"n": i, "source": h.source, "title": h.heading_path, "url": h.link}
            for i, h in enumerate(hits, start=1)
        ],
    }

    start = time.perf_counter()
    text = ""
    if hits:
        for token in ollama.chat_stream(build_messages(question, hits)):
            text += token
            yield {"type": "token", "text": token}
    else:
        text = NOT_FOUND
        yield {"type": "token", "text": text}

    yield {
        "type": "done",
        "cited": cited(text, len(hits)),
        "timings": {
            "retrieve": retrieve_s,
            "rerank": rerank_s,
            "generate": time.perf_counter() - start,
        },
    }
