import contextlib
import json
import threading
import time

import pytest

from opsrag import web
from opsrag.queue import AnswerQueue, QueueFull


def events(chunks):
    return [json.loads(chunk.removeprefix("data: ")) for chunk in chunks]


def test_tickets_are_served_in_order():
    queue = AnswerQueue()
    first, second, third = queue.enter(), queue.enter(), queue.enter()
    assert [queue.ahead(t) for t in (first, second, third)] == [0, 1, 2]
    queue.leave(first, finished=True)
    assert queue.ahead(third) == 1


def test_queue_has_a_limit():
    queue = AnswerQueue(max_waiting=1)
    queue.enter()
    queue.enter()
    assert queue.full()
    with pytest.raises(QueueFull):
        queue.enter()


def test_eta_comes_from_recent_answers():
    queue = AnswerQueue()
    assert queue.eta(1) is None
    queue.durations.extend([60.0, 120.0])
    # Nothing is running: the current answer counts as a whole one
    assert queue.eta(3) == 90 * 3


def test_wait_wakes_up_when_the_turn_comes():
    queue = AnswerQueue()
    first, second = queue.enter(), queue.enter()
    threading.Timer(0.05, queue.leave, args=(first,)).start()
    assert queue.wait(second, timeout=5) == 0


@pytest.fixture
def fake_pipeline(monkeypatch):
    monkeypatch.setattr(web, "answers", AnswerQueue())
    monkeypatch.setattr(web.store, "connect", lambda url: contextlib.nullcontext())
    timings = {"retrieve": 0, "rerank": 0, "generate": 0}
    done = {"type": "done", "cited": [], "timings": timings}
    monkeypatch.setattr(web.answer, "ask", lambda *a: iter([dict(done)]))


def test_second_question_waits_and_a_closed_tab_frees_the_queue(fake_pipeline):
    q = web.Question(question="q")
    first = web.stream_answer(q)
    assert events([next(first)]) == [{"type": "started"}]

    second = web.stream_answer(q)
    [waiting] = events([next(second)])
    assert waiting["type"] == "queue" and waiting["ahead"] == 1

    first.close()  # the browser tab was closed while its answer was being written
    assert {"type": "started"} in events(second)
    assert web.answers.tickets == []


def slow(events, delay):
    def source():
        for event in events:
            time.sleep(delay)
            yield event

    return source


def test_keepalive_fills_silence():
    items = list(web.with_keepalive(slow([{"n": 1}, {"n": 2}], delay=0.25), every=0.1))
    assert [i for i in items if i is not None] == [{"n": 1}, {"n": 2}]
    assert items.count(None) >= 2


def test_keepalive_passes_errors_on():
    def failing():
        yield {"n": 1}
        raise web.OllamaError("down")

    with pytest.raises(web.OllamaError, match="down"):
        list(web.with_keepalive(failing, every=1))


def test_closing_keepalive_stops_the_source():
    produced = []

    def endless():
        for n in range(1000):
            produced.append(n)
            yield {"n": n}
            time.sleep(0.01)

    stream = web.with_keepalive(endless, every=1)
    next(stream)
    stream.close()
    time.sleep(0.1)
    assert len(produced) < 20


@pytest.fixture
def ollama_up(monkeypatch):
    monkeypatch.setattr(web.Ollama, "reachable", lambda self: True)


@pytest.mark.parametrize("question", ["", "   ", "a" * (web.MAX_QUESTION_CHARS + 1)])
def test_empty_and_huge_questions_are_rejected(ollama_up, question):
    response = web.ask(web.Question(question=question))
    assert response.status_code == 400
    assert web.answers.tickets == []


def test_question_is_rejected_at_once_when_ollama_is_down(monkeypatch):
    monkeypatch.setattr(web.Ollama, "reachable", lambda self: False)
    response = web.ask(web.Question(question="how do I roll back?"))
    assert response.status_code == 503
    assert b"ollama serve" in response.body
    assert web.answers.tickets == []
