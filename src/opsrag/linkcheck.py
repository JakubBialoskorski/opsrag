import re
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import httpx

# Be polite to the docs sites: a few requests at a time
WORKERS = 8


def missing_ids(html: str, ids: Iterable[str]) -> list[str]:
    """Which #section ids don't exist on the page. Minified HTML may leave out the quotes."""
    return [i for i in ids if not re.search(rf"""\sid=["']?{re.escape(i)}["'\s>]""", html)]


def check(urls: Iterable[str]) -> dict[str, str]:
    """Return {url: problem} for every link that doesn't work: the page doesn't load, or the
    #section it points to doesn't exist on it."""
    pages: dict[str, set[str]] = {}
    for url in urls:
        page, _, fragment = url.partition("#")
        pages.setdefault(page, set())
        if fragment:
            pages[page].add(fragment)
    broken: dict[str, str] = {}

    with httpx.Client(follow_redirects=True, timeout=15) as http:

        def fetch(page: str, need_html: bool) -> httpx.Response:
            # Two attempts, so one network hiccup doesn't fail the check
            for attempt in (1, 2):
                try:
                    response = http.get(page) if need_html else http.head(page)
                    if response.status_code == 405:  # some servers don't allow HEAD
                        response = http.get(page)
                    if response.status_code < 400 or attempt == 2:
                        return response
                except httpx.HTTPError:
                    if attempt == 2:
                        raise
                time.sleep(2)
            raise AssertionError("unreachable")

        def probe(page: str) -> None:
            fragments = pages[page]
            try:
                response = fetch(page, need_html=bool(fragments))
            except httpx.HTTPError as e:
                broken[page] = type(e).__name__
                return
            if response.status_code >= 400:
                broken[page] = f"HTTP {response.status_code}"
                return
            for fragment in missing_ids(response.text, fragments):
                broken[f"{page}#{fragment}"] = "section not found"

        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(probe, pages))
    return broken
