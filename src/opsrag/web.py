import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, SimpleQueue

import psycopg
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from opsrag import answer, rerank, store
from opsrag.config import get_settings
from opsrag.ingest import PROGRESS_KEY
from opsrag.ollama import Ollama, OllamaError
from opsrag.queue import AnswerQueue, QueueFull

STATIC = Path(__file__).parent / "static"

QUESTIONS = Counter("opsrag_questions_total", "Questions asked", ["status"])
STAGE_SECONDS = Histogram(
    "opsrag_stage_seconds",
    "Time spent in each pipeline stage",
    ["stage"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with store.connect(get_settings().database_url) as conn:
        store.migrate(conn)
    yield


app = FastAPI(title="opsrag", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

answers = AnswerQueue()


# Questions are short; a pasted log would only hold up the queue and overflow the models
MAX_QUESTION_CHARS = 1000
OLLAMA_DOWN = "Ollama is not running, so questions can't be answered. Start it with: ollama serve"


class Question(BaseModel):
    question: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz() -> JSONResponse:
    try:
        with store.connect(get_settings().database_url) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/status")
def status() -> dict:
    """What the UI needs to tell the user whether it's ready: indexed sources, ongoing
    indexing, whether Ollama has the models and whether the reranker is downloaded."""
    settings = get_settings()
    with store.connect(settings.database_url) as conn:
        sources = [source for source, _, chunks in store.stats(conn) if chunks]
        progress = json.loads(store.get_meta(conn, PROGRESS_KEY) or "null")
    # Ignore progress left behind by a crashed ingest
    if progress and time.time() - progress["updated_at"] > 600:
        progress = None
    try:
        missing = Ollama.from_settings(settings).missing_models()
        ollama = f"missing models: {', '.join(missing)}" if missing else "ok"
    except OllamaError:
        ollama = "unreachable"
    reranker = "ok" if rerank.is_installed(settings) or not settings.rerank else "missing"
    return {"sources": sources, "indexing": progress, "ollama": ollama, "reranker": reranker}


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


# Cloudflare drops a response that sends nothing for ~100 s. On two CPU cores the model reads a
# long prompt for minutes before its first word, so the browser gets an SSE comment meanwhile.
KEEPALIVE_SECONDS = 15
_END = object()


def with_keepalive(events: Callable[[], Iterator[dict]], every: float) -> Iterator[dict | None]:
    """Run a slow event source in a thread, yield its events, and None whenever it has been
    quiet for `every` seconds. Closing this generator stops the source at its next event."""
    channel: SimpleQueue = SimpleQueue()
    stop = threading.Event()

    def produce() -> None:
        try:
            for event in events():
                channel.put(event)
                if stop.is_set():
                    return  # closes the source too, which ends the request to Ollama
        except Exception as e:  # handed over to the consumer
            channel.put(e)
        finally:
            channel.put(_END)

    threading.Thread(target=produce, daemon=True).start()
    try:
        while True:
            try:
                item = channel.get(timeout=every)
            except Empty:
                yield None
                continue
            if item is _END:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()


def stream_answer(q: Question) -> Iterator[str]:
    # The ticket is taken here, not in ask(): a generator that never starts (the browser left
    # right away) would never run its finally and the ticket would block the queue forever
    try:
        ticket = answers.enter()
    except QueueFull as e:
        yield sse({"type": "error", "message": f"Too busy right now, {e}. Try again later."})
        return
    finished = False
    queued_at = time.monotonic()
    settings = get_settings()
    ollama = Ollama.from_settings(settings)
    try:
        # Waiting requests hear from us every few seconds: position and estimated wait
        ahead = answers.ahead(ticket)
        while ahead:
            yield sse({"type": "queue", "ahead": ahead, "eta": answers.eta(ahead)})
            ahead = answers.wait(ticket, timeout=3)
        answers.start()
        STAGE_SECONDS.labels("queue").observe(time.monotonic() - queued_at)
        yield sse({"type": "started"})

        def pipeline() -> Iterator[dict]:
            with store.connect(settings.database_url) as conn:
                yield from answer.ask(conn, ollama, settings, q.question)

        for event in with_keepalive(pipeline, KEEPALIVE_SECONDS):
            if event is None:
                yield ": still working\n\n"  # an SSE comment, ignored by the browser
                continue
            if event["type"] == "done":
                for stage, seconds in event["timings"].items():
                    STAGE_SECONDS.labels(stage).observe(seconds)
            yield sse(event)
        QUESTIONS.labels("ok").inc()
        finished = True
    except OllamaError as e:
        QUESTIONS.labels("error").inc()
        yield sse({"type": "error", "message": str(e)})
    finally:
        # Also runs when the browser goes away, so a closed tab doesn't hold up the queue
        answers.leave(ticket, finished)


@app.post("/api/ask", response_model=None)
def ask(q: Question) -> StreamingResponse | JSONResponse:
    question = q.question.strip()
    if not question:
        return JSONResponse({"detail": "Type a question first."}, 400)
    if len(question) > MAX_QUESTION_CHARS:
        detail = f"The question is too long, keep it under {MAX_QUESTION_CHARS} characters."
        return JSONResponse({"detail": detail}, 400)
    # Fail at once instead of queueing a question that can only end in an error
    if not Ollama.from_settings(get_settings()).reachable():
        return JSONResponse({"detail": OLLAMA_DOWN}, 503)
    if answers.full():
        return JSONResponse({"detail": "Too busy right now, try again in a few minutes."}, 503)
    # Server-Sent Events: the browser shows tokens as soon as the model produces them
    return StreamingResponse(
        stream_answer(Question(question=question)), media_type="text/event-stream"
    )
