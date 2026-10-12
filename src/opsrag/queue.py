import threading
import time
from collections import deque


class QueueFull(RuntimeError):
    pass


class AnswerQueue:
    """Answers one question at a time, first come first served.

    On a CPU-only server two answers generated at once share the cores and both finish later
    than they would one after another. Waiting requests learn their position and an estimate
    based on how long recent answers took.
    """

    def __init__(self, max_waiting: int = 10, history: int = 10):
        self.max_waiting = max_waiting
        self.lock = threading.Condition()
        self.tickets: list[object] = []  # tickets[0] is being answered
        self.started_at: float | None = None
        self.durations: deque[float] = deque(maxlen=history)

    def full(self) -> bool:
        with self.lock:
            return len(self.tickets) > self.max_waiting

    def enter(self) -> object:
        with self.lock:
            if len(self.tickets) > self.max_waiting:
                raise QueueFull(f"{self.max_waiting} questions are already waiting")
            ticket = object()
            self.tickets.append(ticket)
            return ticket

    def ahead(self, ticket: object) -> int:
        """How many questions will be answered before this one."""
        with self.lock:
            return self.tickets.index(ticket)

    def wait(self, ticket: object, timeout: float) -> int:
        """Wait until it's this ticket's turn or `timeout` passes, return `ahead()`."""
        with self.lock:
            self.lock.wait_for(lambda: self.tickets.index(ticket) == 0, timeout)
            return self.tickets.index(ticket)

    def start(self) -> None:
        """The first ticket is being answered now, time it."""
        with self.lock:
            self.started_at = time.monotonic()

    def leave(self, ticket: object, finished: bool = False) -> None:
        with self.lock:
            if self.tickets and self.tickets[0] is ticket:
                if finished and self.started_at is not None:
                    self.durations.append(time.monotonic() - self.started_at)
                self.started_at = None
            self.tickets.remove(ticket)
            self.lock.notify_all()

    def eta(self, ahead: int) -> float | None:
        """Seconds until a ticket with `ahead` questions before it starts, None before the
        first answer has been timed."""
        with self.lock:
            if not self.durations:
                return None
            average = sum(self.durations) / len(self.durations)
            running = time.monotonic() - self.started_at if self.started_at else 0.0
            # The running answer is probably partly done, but never count it as finished
            return max(average - running, 5.0) + (ahead - 1) * average
