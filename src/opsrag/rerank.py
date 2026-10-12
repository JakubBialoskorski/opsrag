import hashlib
import platform
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from opsrag.config import Settings
from opsrag.store import Hit

# A cross-encoder reads the question and a passage together and scores how well the passage
# answers it. Much more accurate than comparing two separate embeddings, but too slow for the
# whole index, so it only reorders the candidates found by hybrid search.
# Pinned like models.lock.yaml: a fixed revision and the sha256 of every file.
REPO = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
TOKENIZER = ("tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66")
# int8 builds of the same model, ~4x smaller and ~2x faster than float32 on a CPU
MODELS = {
    "arm64": (
        "onnx/model_qint8_arm64.onnx",
        "3573b6b9593cb2f75987a31815d409ca3dd8808629118fd20451bb1a5d90cec7",
    ),
    "x86_64": (
        "onnx/model_quint8_avx2.onnx",
        "c80a8b34256ea453093d612e3ac48d3d965a0c0a48c7906709af8b8e28461bf9",
    ),
}
# Passages are cut to 256 tokens: the start of a chunk (breadcrumb + first paragraph) says what
# it is about, and attention cost grows with the square of the length
MAX_TOKENS = 256
BATCH = 16


def files() -> list[tuple[str, str]]:
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    return [MODELS[arch], TOKENIZER]


def model_dir(settings: Settings) -> Path:
    return settings.data_dir / "models" / REPO.replace("/", "--")


def is_installed(settings: Settings) -> bool:
    return all((model_dir(settings) / Path(name).name).exists() for name, _ in files())


def download(settings: Settings) -> None:
    """Fetch the pinned files once, refusing anything whose checksum doesn't match."""
    target = model_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    for name, sha256 in files():
        path = target / Path(name).name
        if path.exists():
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        response = httpx.get(url, follow_redirects=True, timeout=120)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != sha256:
            raise RuntimeError(f"{url} doesn't match its pinned sha256, not using it")
        path.write_bytes(response.content)


class Reranker:
    def __init__(self, directory: Path):
        model, _ = files()[0]
        self.session = ort.InferenceSession(str(directory / Path(model).name))
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(directory / TOKENIZER[0]))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding()

    def scores(self, query: str, passages: list[str]) -> np.ndarray:
        # Small batches, a single batch of 60 long passages takes gigabytes of memory
        return np.concatenate(
            [self._scores(query, passages[i : i + BATCH]) for i in range(0, len(passages), BATCH)]
        )

    def _scores(self, query: str, passages: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch([(query, p) for p in passages])
        feed = {
            "input_ids": [e.ids for e in encoded],
            "attention_mask": [e.attention_mask for e in encoded],
            "token_type_ids": [e.type_ids for e in encoded],
        }
        feed = {k: np.array(v, dtype=np.int64) for k, v in feed.items() if k in self.inputs}
        return self.session.run(None, feed)[0][:, 0]

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = self.scores(query, [f"{h.heading_path}\n{h.content}" for h in hits])
        order = np.argsort(-scores, kind="stable")
        return [replace(hits[i], score=float(scores[i])) for i in order]


@lru_cache
def _load(directory: Path) -> Reranker:
    return Reranker(directory)


def load(settings: Settings) -> Reranker | None:
    """The reranker, or None when it's switched off or not downloaded yet (`opsrag setup`)."""
    if not settings.rerank or not is_installed(settings):
        return None
    return _load(model_dir(settings))
