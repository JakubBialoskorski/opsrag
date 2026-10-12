from opsrag.config import Settings


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("CHAT_MODEL", "llama3.2:3b")
    settings = Settings(_env_file=None)
    assert settings.top_k == 3
    assert settings.chat_model == "llama3.2:3b"
    assert settings.embed_dim == 768


def test_top_k_keeps_answers_the_small_ci_index_cant_check():
    # CI only indexes Helm and Prometheus. On the full index fewer than 5 chunks lose the offset
    # chunk (4th) and rate() for "rate irate" (5th, behind Kubernetes "Rate limits" chunks)
    assert Settings(_env_file=None).top_k >= 5
