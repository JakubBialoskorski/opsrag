from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
import yaml

from opsrag import answer, rerank, retrieval
from opsrag.config import Settings
from opsrag.ollama import Ollama
from opsrag.retrieval import Mode
from opsrag.store import Hit


@dataclass
class Case:
    question: str
    expected: list[str]  # "source:path" entries, empty for out-of-scope questions
    # The syntax a correct answer has to contain, e.g. "offset" for "value one hour ago"
    must_include: list[str] = field(default_factory=list)
    # A known wrong answer: reported but not failing, until the case starts passing
    known_issue: str | None = None


@dataclass
class RetrievalScore:
    mode: str
    hit_rate: float  # share of questions with a relevant document in the top k
    mrr: float  # mean of 1/rank of the first relevant document (0 when missing)
    misses: list[str] = field(default_factory=list)


def load_cases(path: Path) -> list[Case]:
    return [Case(**item) for item in yaml.safe_load(path.read_text())]


def first_relevant_rank(hits: list[Hit], expected: list[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if f"{hit.source}:{hit.path}" in expected:
            return rank
    return None


def score_retrieval(
    conn: psycopg.Connection,
    ollama: Ollama,
    settings: Settings,
    cases: list[Case],
    mode: Mode,
    k: int,
) -> RetrievalScore:
    in_scope = [c for c in cases if c.expected]
    reciprocal_ranks = []
    misses = []
    for case in in_scope:
        hits = retrieval.search(conn, ollama, settings, case.question, mode, limit=k)
        rank = first_relevant_rank(hits, case.expected)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if rank is None:
            misses.append(case.question)
    hit_rate = sum(rr > 0 for rr in reciprocal_ranks) / len(in_scope)
    mrr = sum(reciprocal_ranks) / len(in_scope)
    return RetrievalScore(mode, hit_rate, mrr, misses)


def judge_answer(case: Case, text: str, cited: list[int]) -> str | None:
    """Return what is wrong with an answer, or None if it's fine."""
    if "<think>" in text or "</think>" in text:
        return "reasoning leaked into the answer (use an instruct model)"
    refused = answer.NOT_FOUND in text
    if not case.expected:
        return None if refused else "answered an out-of-scope question"
    if refused:
        return "refused although the docs contain the answer"
    if not cited:
        return "answer cites no sources"
    missing = [m for m in case.must_include if m not in text]
    if missing:
        return f"answer doesn't mention {', '.join(missing)}"
    return None


def check_answer(
    conn: psycopg.Connection, ollama: Ollama, settings: Settings, case: Case
) -> str | None:
    text, cited = "", []
    for event in answer.ask(conn, ollama, settings, case.question):
        if event["type"] == "token":
            text += event["text"]
        elif event["type"] == "done":
            cited = event["cited"]
    return judge_answer(case, text, cited)


def run_checks(
    conn: psycopg.Connection,
    ollama: Ollama,
    settings: Settings,
    cases: list[Case],
    k: int = 5,
    min_hit_rate: float = 0.0,
    answers: bool = False,
    log: Callable[[str], None] = print,
) -> bool:
    """Print a report and return False if any check failed."""
    if settings.rerank and rerank.load(settings) is None:
        # Otherwise "rerank" would silently measure plain hybrid search
        log("FAIL: the reranker is not downloaded, run `opsrag setup` first")
        return False
    passed = True
    scores = [
        score_retrieval(conn, ollama, settings, cases, mode, k)
        for mode in ("vector", "fts", "hybrid", "rerank")
    ]
    log(f"| mode | hit@{k} | MRR |\n|---|---|---|")
    for s in scores:
        log(f"| {s.mode} | {s.hit_rate:.2f} | {s.mrr:.2f} |")
    final = scores[-1]
    for question in final.misses:
        log(f"{final.mode} missed: {question}")
    if final.hit_rate < min_hit_rate:
        log(f"FAIL: {final.mode} hit@{k} {final.hit_rate:.2f} is below {min_hit_rate}")
        passed = False

    if answers:
        log("")
        for case in cases:
            problem = check_answer(conn, ollama, settings, case)
            if case.known_issue and problem:
                log(f"known {case.question}\n      {problem} ({case.known_issue})")
                continue
            if case.known_issue:
                problem = "passes now, remove its known_issue so it can't break again"
            log(f"{'FAIL' if problem else 'ok  '}  {case.question}")
            if problem:
                log(f"      {problem}")
                passed = False
    return passed
