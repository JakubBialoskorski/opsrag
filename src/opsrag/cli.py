from pathlib import Path

import psycopg
import typer

from opsrag import answer, evaluate, linkcheck, rerank, retrieval, store
from opsrag.chunking import SLUGGERS, chunk_markdown
from opsrag.config import get_settings
from opsrag.ingest import ingest as run_ingest
from opsrag.ollama import (
    Ollama,
    OllamaError,
    full_name,
    load_model_lock,
    lock_problems,
    models_to_pull,
    remote_model,
    save_model_lock,
)
from opsrag.sources import (
    iter_documents,
    load_lock,
    load_sources,
    resolve,
    save_lock,
    sync,
    with_anchor,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)


def run() -> None:
    """Entry point: show expected failures as one readable line instead of a traceback."""
    try:
        app()
    except OllamaError as e:
        typer.secho(f"error: {e}. Is Ollama running?", fg="red", err=True)
        raise SystemExit(1) from None
    except psycopg.OperationalError as e:
        reason = str(e).splitlines()[0]
        typer.secho(
            f"error: cannot connect to Postgres ({reason}). Did you run ./start.sh?",
            fg="red",
            err=True,
        )
        raise SystemExit(1) from None
    except (RuntimeError, ValueError) as e:
        typer.secho(f"error: {e}", fg="red", err=True)
        raise SystemExit(1) from None


@app.callback()
def main() -> None:
    """Local RAG assistant for DevOps documentation."""


@app.command()
def update() -> None:
    """Resolve the newest version of every source and write sources.lock.yaml."""
    settings = get_settings()
    lock = load_lock(settings.sources_file)
    for source in load_sources(settings.sources_file):
        new = resolve(source)
        old = lock.get(source.name)
        if old and old.ref == new.ref:
            typer.echo(f"{source.name}: {new.version} (up to date)")
        else:
            typer.echo(f"{source.name}: {old.version if old else '-'} -> {new.version}")
        lock[source.name] = new
    save_lock(settings.sources_file, lock)


@app.command()
def ingest(
    source: list[str] | None = typer.Option(None, help="Only ingest this source (repeatable)"),
    reindex: bool = typer.Option(False, help="Drop everything and rebuild the index"),
) -> None:
    """Download documentation, chunk it, embed it and store it in Postgres."""
    settings = get_settings()
    with store.connect(settings.database_url) as conn:
        results = run_ingest(
            conn, Ollama.from_settings(settings), settings, source, reindex, typer.echo
        )
    for r in results:
        typer.echo(
            f"{r.source} {r.version}: {r.updated} updated ({r.chunks} chunks), "
            f"{r.unchanged} unchanged ({r.relinked} with a new link), {r.deleted} deleted"
        )


@app.command()
def setup(
    source: list[str] | None = typer.Option(None, help="Only index this source (repeatable)"),
) -> None:
    """First run: download missing Ollama models and the reranker, check the models against
    models.lock.yaml, then build the index."""
    settings = get_settings()
    ollama = Ollama.from_settings(settings)
    lock = load_model_lock(settings.models_lock_file)
    pulls = models_to_pull(
        ollama.models, ollama.installed(), lock, lambda m: remote_model(m).digest
    )
    for model, why in pulls:
        typer.echo(f"{model}: {why}, downloading (a few minutes)")
        for status in ollama.pull(model):
            typer.echo(f"  {status}")

    problems = lock_problems(lock, ollama.installed(), ollama.models)
    if problems:
        raise RuntimeError(
            "a model is not the one this release was tested with:\n  "
            + "\n  ".join(problems)
            + "\nThe registry already serves a newer, untested version. "
            "Run `make update-models` to test it and pin it."
        )
    for model in ollama.models:
        if model not in lock:
            typer.secho(f"warning: {model} is not pinned in models.lock.yaml", fg="yellow")
    if not rerank.is_installed(settings):
        typer.echo(f"reranker: downloading {rerank.REPO} (23 MB)")
        rerank.download(settings)

    embed_digest = ollama.installed()[full_name(settings.embed_model)]
    with store.connect(settings.database_url) as conn:
        built_with = store.get_meta(conn, "embedding_digest")
    rebuild = built_with is not None and built_with != embed_digest
    if rebuild:
        typer.echo("the embedding model changed, rebuilding the whole index")
    ingest(source=source, reindex=rebuild)
    with store.connect(settings.database_url) as conn:
        store.set_meta(conn, "embedding_digest", embed_digest)


@app.command(name="update-models")
def update_models(
    yes: bool = typer.Option(False, "--yes", help="Don't ask, for scripts"),
    source: list[str] | None = typer.Option(
        None, help="Only (re)index this source for the checks (repeatable), used in CI"
    ),
) -> None:
    """Show which models have a newer version, and on confirmation download them, check
    that answers are still good and pin them in models.lock.yaml."""
    settings = get_settings()
    ollama = Ollama.from_settings(settings)
    lock = load_model_lock(settings.models_lock_file)
    local = ollama.local_models()

    updates = []
    download = False
    for model in ollama.models:
        remote = remote_model(model)
        current = local.get(model)
        if current and current.digest == remote.digest == lock.get(model):
            typer.echo(f"{model}: up to date ({remote.digest[:12]})")
            continue
        typer.secho(f"\n{model}", bold=True)
        tested = lock.get(model, "")[:12] or "not pinned yet"
        typer.echo(f"  tested (models.lock.yaml):  {tested}")
        typer.echo(f"  installed in Ollama:        {current.describe() if current else '-'}")
        typer.echo(f"  available in the registry:  {remote.describe()}")
        updates.append(model)
        download = download or not current or current.digest != remote.digest

    if not updates:
        typer.echo("\nEverything is up to date.")
        return
    if download:
        typer.echo(
            "\nThe new version replaces the installed one in Ollama, the old one can't be "
            "downloaded again.\nIf it fails the answer checks, models.lock.yaml stays unchanged "
            "and you'll see why."
        )
    question = "Download and test it?" if download else "Test the installed version and pin it?"
    if not yes and not typer.confirm(question):
        typer.echo("Nothing changed.")
        return

    for model in updates:
        for status in ollama.pull(model):
            if status.startswith("downloading"):
                typer.echo(f"  {status}")
    if not rerank.is_installed(settings):
        rerank.download(settings)
    cases = evaluate.load_cases(Path("eval/smoke.yaml"))
    with store.connect(settings.database_url) as conn:
        if full_name(settings.embed_model) in updates:
            # Vectors from the old and new embedding model can't be mixed
            typer.echo("the embedding model changed, rebuilding the index")
            run_ingest(conn, ollama, settings, only=source, reindex=True, log=typer.echo)
        else:
            # The checks need their docs indexed; usually nothing to do except on a fresh setup
            needed = sorted({e.split(":", 1)[0] for c in cases for e in c.expected})
            run_ingest(conn, ollama, settings, only=source or needed, log=typer.echo)
        typer.echo("checking answers with the new version (eval/smoke.yaml)")
        passed = evaluate.run_checks(
            conn, ollama, settings, cases, min_hit_rate=1.0, answers=True, log=typer.echo
        )
    if not passed:
        raise RuntimeError(
            "the new version failed the checks above, models.lock.yaml was not changed. "
            "Pick another model in .env (CHAT_MODEL) or fix the failing cases"
        )
    installed = ollama.installed()
    save_model_lock(settings.models_lock_file, {m: installed[m] for m in ollama.models})
    with store.connect(settings.database_url) as conn:
        store.set_meta(conn, "embedding_digest", installed[full_name(settings.embed_model)])
    typer.secho("Checks passed, the new version is pinned. Commit models.lock.yaml.", fg="green")


@app.command()
def stats() -> None:
    """Show what is in the index."""
    settings = get_settings()
    lock = load_lock(settings.sources_file)
    with store.connect(settings.database_url) as conn:
        typer.echo(f"embedding model: {store.get_meta(conn, 'embedding_model') or '-'}")
        for source, docs, chunks in store.stats(conn):
            indexed = store.get_meta(conn, f"version:{source}") or "?"
            locked = lock[source].version if source in lock else "?"
            note = "" if indexed == locked else f" (lock has {locked}, run `make update`)"
            typer.echo(f"{source}: {docs} documents, {chunks} chunks, version {indexed}{note}")


@app.command()
def search(
    query: str,
    mode: str = typer.Option("rerank", help="vector, fts, hybrid or rerank (hybrid + reranker)"),
    limit: int | None = typer.Option(None, help="How many chunks to show (default: TOP_K)"),
    source: str | None = typer.Option(None, help="Only search this source"),
) -> None:
    """Show which chunks retrieval finds for a query, without calling the LLM."""
    settings = get_settings()
    with store.connect(settings.database_url) as conn:
        hits = retrieval.search(
            conn, Ollama.from_settings(settings), settings, query, mode, limit, source
        )
    for i, hit in enumerate(hits, start=1):
        preview = " ".join(hit.content.split())[:200]
        typer.secho(f"[{i}] {hit.score:.4f}  {hit.source}: {hit.heading_path}", bold=True)
        typer.echo(f"    {hit.url}")
        typer.echo(f"    {preview}\n")


@app.command()
def ask(
    question: str,
    source: str | None = typer.Option(None, help="Only use this source"),
) -> None:
    """Answer a question from the documentation, with sources."""
    settings = get_settings()
    with store.connect(settings.database_url) as conn:
        for event in answer.ask(conn, Ollama.from_settings(settings), settings, question, source):
            if event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "token":
                typer.echo(event["text"], nl=False)
            elif event["type"] == "done":
                typer.echo("\n")
                for s in sources:
                    if s["n"] in event["cited"]:
                        typer.echo(f"[{s['n']}] {s['title']}\n    {s['url']}")
                t = event["timings"]
                typer.secho(
                    f"\nretrieve {t['retrieve']:.2f}s, rerank {t['rerank']:.2f}s, "
                    f"generate {t['generate']:.2f}s",
                    dim=True,
                )


@app.command(name="eval")
def evaluate_cmd(
    k: int | None = typer.Option(None, help="How many retrieved chunks count as a hit (TOP_K)"),
    answers: bool = typer.Option(False, help="Also check LLM answers (slow)"),
    questions: str = typer.Option("eval/questions.yaml", help="Question set"),
    min_hit_rate: float = typer.Option(0.0, help="Fail if reranked hit@k is lower (for CI)"),
) -> None:
    """Measure retrieval quality (and optionally answers). Exits with 1 when a check fails."""
    settings = get_settings()
    k = k or settings.top_k
    cases = evaluate.load_cases(Path(questions))
    with store.connect(settings.database_url) as conn:
        ollama = Ollama.from_settings(settings)
        passed = evaluate.run_checks(
            conn, ollama, settings, cases, k, min_hit_rate, answers, typer.echo
        )
    if not passed:
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Address to listen on"),
    port: int = typer.Option(8000, help="Port to listen on"),
) -> None:
    """Run the web UI and API."""
    import uvicorn

    uvicorn.run("opsrag.web:app", host=host, port=port)


@app.command(name="check-links")
def check_links(
    source: list[str] | None = typer.Option(None, help="Only check this source (repeatable)"),
) -> None:
    """Check that every source link shown with answers opens, including its #section."""
    settings = get_settings()
    lock = load_lock(settings.sources_file)
    failed = 0
    for src in load_sources(settings.sources_file):
        if source and src.name not in source:
            continue
        checkout = sync(src, lock[src.name], settings.data_dir)
        slug = SLUGGERS[src.anchors]
        links: dict[str, str] = {}  # link -> document path, for the report
        for doc in iter_documents(src, checkout):
            links[doc.url] = doc.path
            for chunk in chunk_markdown(doc.title, doc.body, slug, doc.heading_ids):
                links[with_anchor(doc.url, chunk.anchor)] = doc.path
        broken = linkcheck.check(links)
        pages = {link.split("#", 1)[0] for link in links}
        typer.echo(f"{src.name}: {len(pages)} pages, {len(links)} links, {len(broken)} broken")
        for link, problem in sorted(broken.items()):
            path = links.get(link) or next(p for u, p in links.items() if u.startswith(link))
            typer.echo(f"  {problem}  {link}  ({path})")
        failed += len(broken)
    if failed:
        raise typer.Exit(1)
