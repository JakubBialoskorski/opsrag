from opsrag.ollama import (
    full_name,
    load_model_lock,
    lock_problems,
    models_to_pull,
    save_model_lock,
)


def test_full_name_adds_latest_tag():
    assert full_name("nomic-embed-text") == "nomic-embed-text:latest"
    assert full_name("qwen3:4b-instruct") == "qwen3:4b-instruct"


def test_lock_problems_detects_republished_model():
    lock = {"qwen3:4b-instruct": "aaa111", "nomic-embed-text:latest": "bbb222"}
    models = list(lock)
    assert lock_problems(lock, dict(lock), models) == []
    problems = lock_problems(lock, {**lock, "qwen3:4b-instruct": "ccc333"}, models)
    assert problems == ["qwen3:4b-instruct is ccc333, models.lock.yaml expects aaa111"]


def test_unpinned_models_are_not_problems():
    assert lock_problems({}, {"other:1b": "x"}, ["other:1b"]) == []


def test_model_lock_roundtrip(tmp_path):
    path = tmp_path / "models.lock.yaml"
    save_model_lock(path, {"qwen3:4b-instruct": "aaa111"})
    assert load_model_lock(path) == {"qwen3:4b-instruct": "aaa111"}
    assert load_model_lock(tmp_path / "missing.yaml") == {}


def test_models_to_pull():
    lock = {"qwen3:4b-instruct": "new", "nomic-embed-text:latest": "emb"}
    models = list(lock)
    registry = {"qwen3:4b-instruct": "new", "nomic-embed-text:latest": "emb"}
    # A merged "Update Ollama models" PR: the old version is installed, the registry has the new one
    assert models_to_pull(models, {**lock, "qwen3:4b-instruct": "old"}, lock, registry.get) == [
        ("qwen3:4b-instruct", "switching to the version in models.lock.yaml")
    ]
    # Fresh machine
    assert models_to_pull(models, {}, lock, registry.get) == [
        ("qwen3:4b-instruct", "not installed"),
        ("nomic-embed-text:latest", "not installed"),
    ]
    # The registry serves something nobody tested yet: don't pull, setup reports it
    untested = {**registry, "qwen3:4b-instruct": "newer"}
    assert models_to_pull(models, {**lock, "qwen3:4b-instruct": "old"}, lock, untested.get) == []
    # Everything matches
    assert models_to_pull(models, dict(lock), lock, registry.get) == []
