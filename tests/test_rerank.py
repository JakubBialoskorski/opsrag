import hashlib

import httpx
import numpy as np
import pytest

from opsrag import rerank, retrieval
from opsrag.config import Settings
from opsrag.store import Hit


def hit(chunk_id: int, content: str) -> Hit:
    return Hit(chunk_id, "helm", "a.md", "url", "Title", "Title", content, None, 0.0)


class FakeReranker(rerank.Reranker):
    def __init__(self):
        pass

    def scores(self, query, passages):
        # The longer the passage, the better, just to get a known order
        return np.array([float(len(p)) for p in passages])


def test_rerank_orders_by_score_and_keeps_scores():
    hits = [hit(1, "a"), hit(2, "ccc"), hit(3, "bb")]
    ranked = FakeReranker().rerank("q", hits)
    assert [h.chunk_id for h in ranked] == [2, 3, 1]
    assert ranked[0].score > ranked[1].score > ranked[2].score


def test_reorder_keeps_order_without_a_reranker(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    hits = [hit(1, "a"), hit(2, "ccc")]
    assert retrieval.reorder(settings, "q", hits) == hits
    off = Settings(_env_file=None, data_dir=tmp_path, rerank=False)
    assert rerank.load(off) is None


def test_files_match_the_cpu(monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    assert rerank.files()[0][0].endswith("arm64.onnx")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    assert rerank.files()[0][0].endswith("avx2.onnx")


def test_download_refuses_a_wrong_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: httpx.Response(
            200, content=b"not the model", request=httpx.Request("GET", url)
        ),
    )
    settings = Settings(_env_file=None, data_dir=tmp_path)
    with pytest.raises(RuntimeError, match="sha256"):
        rerank.download(settings)
    assert not rerank.is_installed(settings)


def test_download_keeps_files_with_the_pinned_checksum(tmp_path, monkeypatch):
    content = b"pinned"
    pinned = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        rerank, "files", lambda: [("onnx/model.onnx", pinned), ("tokenizer.json", pinned)]
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: httpx.Response(200, content=content, request=httpx.Request("GET", url)),
    )
    settings = Settings(_env_file=None, data_dir=tmp_path)
    rerank.download(settings)
    assert rerank.is_installed(settings)
