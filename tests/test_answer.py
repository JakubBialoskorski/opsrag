from opsrag.answer import MAX_CHUNK_CHARS, build_messages, cited
from opsrag.store import Hit


def make_hit(chunk_id: int, content: str) -> Hit:
    return Hit(chunk_id, "helm", "a.md", "url", "Title", "Title > Section", content, None, 0.0)


def test_passages_are_numbered_in_order():
    messages = build_messages("How?", [make_hit(1, "first"), make_hit(2, "second")])
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "[1] helm: Title > Section\nfirst" in user
    assert "[2] helm: Title > Section\nsecond" in user
    assert user.endswith("Question: How?")


def test_long_chunks_are_truncated():
    messages = build_messages("How?", [make_hit(1, "x" * (MAX_CHUNK_CHARS + 500))])
    assert "x" * MAX_CHUNK_CHARS + "\n" in messages[1]["content"]
    assert "x" * (MAX_CHUNK_CHARS + 1) not in messages[1]["content"]


def test_cited_ignores_out_of_range_numbers():
    assert cited("Use rollout [2]. Also [1][2] and [9].", count=3) == [1, 2]
    assert cited("No citations here.", count=3) == []
