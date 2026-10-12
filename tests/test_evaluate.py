from pathlib import Path
from types import SimpleNamespace

from opsrag.answer import NOT_FOUND
from opsrag.config import Settings
from opsrag.evaluate import (
    Case,
    RetrievalScore,
    first_relevant_rank,
    judge_answer,
    load_cases,
    run_checks,
)
from opsrag.store import Hit


def make_hit(source: str, path: str) -> Hit:
    return Hit(0, source, path, "url", "title", "heading", "content", None, 0.0)


def test_first_relevant_rank():
    hits = [make_hit("helm", "a.md"), make_hit("kubernetes", "b.md")]
    assert first_relevant_rank(hits, ["kubernetes:b.md"]) == 2
    assert first_relevant_rank(hits, ["kubernetes:c.md"]) is None


def test_question_set_is_valid():
    cases = load_cases(Path(__file__).parent.parent / "eval" / "questions.yaml")
    assert len(cases) >= 30
    for case in cases:
        for expected in case.expected:
            source, _, path = expected.partition(":")
            assert source in {"kubernetes", "prometheus", "helm"} and path, expected


def test_judge_answer():
    in_scope = Case("daemonset", ["kubernetes:concepts/workloads/controllers/daemonset.md"])
    out_of_scope = Case("pancakes", [])
    assert judge_answer(in_scope, "A DaemonSet runs a Pod on every node [1].", [1]) is None
    assert "refused" in judge_answer(in_scope, NOT_FOUND, [])
    assert "no sources" in judge_answer(in_scope, "A DaemonSet runs a Pod on every node.", [])
    assert "leaked" in judge_answer(in_scope, "</think> A DaemonSet [1]", [1])
    assert judge_answer(out_of_scope, NOT_FOUND, []) is None
    assert "out-of-scope" in judge_answer(out_of_scope, "Mix flour and eggs.", [])


def test_checks_fail_without_the_reranker(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    lines = []
    assert run_checks(None, None, settings, [], log=lines.append) is False
    assert "reranker is not downloaded" in lines[0]


def test_judge_answer_checks_required_syntax():
    case = Case("value one hour ago", ["prometheus:querying/basics.md"], must_include=["offset"])
    assert judge_answer(case, "Use `metric offset 1h` [1].", [1]) is None
    assert "offset" in judge_answer(case, "Use `metric[1h]` [1].", [1])


def test_known_issues_are_reported_but_never_fail(tmp_path, monkeypatch):
    monkeypatch.setattr("opsrag.evaluate.rerank.load", lambda settings: object())
    monkeypatch.setattr("opsrag.evaluate.score_retrieval", lambda *a: RetrievalScore("x", 1, 1))
    settings = Settings(_env_file=None, data_dir=tmp_path)
    case = Case("hooks", ["helm:a.md"], known_issue="chunk not found")

    monkeypatch.setattr("opsrag.evaluate.check_answer", lambda *a: "answer cites no sources")
    lines = []
    assert run_checks(None, None, settings, [case], answers=True, log=lines.append)
    assert any(line.startswith("known") for line in lines)

    monkeypatch.setattr("opsrag.evaluate.check_answer", lambda *a: None)
    lines = []
    assert run_checks(None, None, settings, [case], answers=True, log=lines.append)
    assert any("passes now" in line for line in lines)


def test_failed_answers_are_retried_with_other_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr("opsrag.evaluate.rerank.load", lambda settings: object())
    monkeypatch.setattr("opsrag.evaluate.score_retrieval", lambda *a: RetrievalScore("x", 1, 1))
    settings = Settings(_env_file=None, data_dir=tmp_path)
    ollama = SimpleNamespace(seed=42)
    case = Case("rate irate", ["prometheus:a.md"])

    # Fails only with the default seed: noise, the run passes
    seeds = []

    def flaky(conn, ollama, settings, case):
        seeds.append(ollama.seed)
        return "answer cites no sources" if ollama.seed == 42 else None

    monkeypatch.setattr("opsrag.evaluate.check_answer", flaky)
    lines = []
    assert run_checks(None, ollama, settings, [case], answers=True, log=lines.append)
    assert seeds == [42, 43]
    assert any(line.startswith("flaky") for line in lines)
    assert ollama.seed == 42

    # Fails with every seed: a real regression, the run fails
    monkeypatch.setattr("opsrag.evaluate.check_answer", lambda *a: "answer cites no sources")
    lines = []
    assert not run_checks(None, ollama, settings, [case], answers=True, log=lines.append)
    assert any(line.startswith("FAIL") for line in lines)
    assert ollama.seed == 42
