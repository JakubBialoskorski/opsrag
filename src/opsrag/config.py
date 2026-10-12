from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    # nomic-embed-text expects these task prefixes, other models may need none
    doc_prefix: str = "search_document: "
    query_prefix: str = "search_query: "
    chat_model: str = "qwen3:4b-instruct"
    num_ctx: int = 4096

    database_url: str = "postgresql://opsrag:opsrag@localhost:5432/opsrag"

    data_dir: Path = Path("./data")
    sources_file: Path = Path("./sources.yaml")
    models_lock_file: Path = Path("./models.lock.yaml")

    # Each retriever returns `candidates` chunks, the best `rerank_pool` of the merged list are
    # reordered by the reranker and the first `top_k` go into the prompt. A wide net matters:
    # the right chunk for "value one hour ago" (offset) is only ~60th by meaning or keywords.
    candidates: int = 120
    rerank_pool: int = 60
    rerank: bool = True
    # Every chunk costs ~35 s of prompt reading on two CPU cores, but fewer lose answers on the
    # full index: offset is 4th (bigger rerankers put it lower), rate() for "rate irate" is 5th.
    top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
