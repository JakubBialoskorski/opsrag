import html
import re
from collections.abc import Callable
from dataclasses import dataclass

import yaml

MAX_CHARS = 1500  # roughly 375 tokens, well under nomic-embed-text's 2048 token limit
MIN_CHARS = 50
MAX_HEADING_LEVEL = 3  # H4 and deeper stay inside their parent chunk

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SHORTCODE_RE = re.compile(r"\{\{[<%]\s*(/?)\s*([\w-]+)(.*?)\s*[>%]\}\}", re.DOTALL)
ATTR_RE = re.compile(r'([\w-]+)="([^"]*)"')
MDX_IMPORT_RE = re.compile(r"^import\s+.+\s+from\s+['\"].+['\"];?\s*$", re.MULTILINE)
JSX_TAG_RE = re.compile(r"^\s*</?[A-Z][\w.]*[^>]*>\s*$", re.MULTILINE)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
ADMONITION_RE = re.compile(r"^:::\s*(\w+)?.*$", re.MULTILINE)
HEADING_ANCHOR_RE = re.compile(r"^(#{1,6}[ \t].*?)[ \t]*\{#[^}]*\}[ \t]*$", re.MULTILINE)
EXPLICIT_ID_RE = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*\{#([^}]+)\}[ \t]*$", re.MULTILINE)
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
BLANK_LINES_RE = re.compile(r"\n{3,}")

CALLOUTS = {"note", "caution", "warning", "alert"}

# What kubernetes.io shows for {{% heading "..." %}}, from i18n/en/en.toml in kubernetes/website.
# The real text matters: the section's #id on the page is generated from it.
HUGO_HEADINGS = {
    "cleanup": "Cleaning up",
    "envvars": "Environment variables",
    "examples": "Examples",
    "objectives": "Objectives",
    "options": "Options",
    "parentoptions": "Parent Options Inherited",
    "prerequisites": "Before you begin",
    "seealso": "See Also",
    "synopsis": "Synopsis",
    "whatsnext": "What's next",
}


@dataclass
class Chunk:
    heading_path: str
    content: str
    anchor: str | None = None  # id of the section on the docs page, None for the page top

    @property
    def token_estimate(self) -> int:
        return len(self.content) // 4


def split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    try:
        meta = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), text[end + 4 :].lstrip("\n")


def _render_shortcode(match: re.Match) -> str:
    closing, name, args = match.groups()
    attrs = dict(ATTR_RE.findall(args))
    if closing:
        return ""
    if name == "glossary_tooltip":
        return attrs.get("text") or attrs.get("term_id", "").replace("-", " ")
    if name == "heading":
        key = args.strip().strip('"')
        return HUGO_HEADINGS.get(key, key.replace("-", " ").capitalize())
    if name == "feature-state":
        return (
            f"(feature state: {attrs.get('state', '?')} since {attrs.get('for_k8s_version', '?')})"
        )
    if name in CALLOUTS:
        return f"**{name.capitalize()}:**"
    return ""


def clean_markdown(text: str) -> str:
    """Strip Hugo shortcodes and MDX/HTML noise that only make sense to a site generator."""
    text = HTML_COMMENT_RE.sub("", text)
    text = SHORTCODE_RE.sub(_render_shortcode, text)
    text = MDX_IMPORT_RE.sub("", text)
    text = JSX_TAG_RE.sub("", text)
    text = HEADING_ANCHOR_RE.sub(r"\1", text)
    text = ADMONITION_RE.sub(
        lambda m: f"**{m.group(1).capitalize()}:**" if m.group(1) else "", text
    )
    return BLANK_LINES_RE.sub("\n\n", text).strip()


# Docs sites turn heading text into the id used in #links. Each generator does it differently,
# so the style is set per source in sources.yaml.


def github_slug(text: str) -> str:
    """GitHub, Docusaurus: lowercase, drop punctuation, spaces become dashes."""
    text = html.unescape(MD_LINK_RE.sub(r"\1", text)).lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def blackfriday_slug(text: str) -> str:
    """Hugo with autoHeadingIDType = "blackfriday": runs of letters and digits joined by dashes."""
    text = html.unescape(MD_LINK_RE.sub(r"\1", text)).lower()
    return "-".join(re.findall(r"[^\W_]+", text))


SLUGGERS: dict[str, Callable[[str], str]] = {"github": github_slug, "blackfriday": blackfriday_slug}


def explicit_heading_ids(raw: str) -> dict[str, list[str | None]]:
    """Headings written as `## Title {#custom-id}` use that id instead of a generated one.
    Read from the raw Markdown, because clean_markdown removes these markers. Returns the id of
    every occurrence of a heading text, in order: the same text can appear twice, once with an
    explicit id and once without."""
    ids: dict[str, list[str | None]] = {}
    in_code = False
    for line in raw.splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code
        heading = None if in_code else HEADING_RE.match(line)
        if heading:
            explicit = EXPLICIT_ID_RE.match(line)
            text = explicit.group(1) if explicit else heading.group(2)
            ids.setdefault(text, []).append(explicit.group(2) if explicit else None)
    return ids


def _blocks(lines: list[str]) -> list[str]:
    """Group lines into paragraphs; a fenced code block is always a single block."""
    blocks, current, in_code = [], [], False
    for line in lines:
        if FENCE_RE.match(line):
            in_code = not in_code
        if not line.strip() and not in_code:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _pack(blocks: list[str]) -> list[str]:
    # A single block larger than MAX_CHARS (usually a long code sample) is kept whole
    parts, current = [], ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > MAX_CHARS:
            parts.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        parts.append(current)
    return parts


def chunk_markdown(
    title: str,
    body: str,
    slug: Callable[[str], str] = github_slug,
    explicit_ids: dict[str, list[str | None]] | None = None,
) -> list[Chunk]:
    """Split a document by H1-H3 headings, then by paragraphs if a section is too long."""
    sections: list[tuple[list[str], str | None, list[str]]] = []
    stack: list[tuple[int, str]] = []  # (level, text) of currently open headings
    headings: list[str] = []
    anchor: str | None = None
    lines: list[str] = []
    in_code = False
    seen: dict[str, int] = {}  # repeated headings get -1, -2, ... like on the docs sites
    occurrence: dict[str, int] = {}  # how many times each heading text was seen so far

    for line in body.splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code
        heading = None if in_code else HEADING_RE.match(line)
        if heading:
            text = heading.group(2)
            n = occurrence.get(text, 0)
            occurrence[text] = n + 1
            explicit = (explicit_ids or {}).get(text, [])
            heading_id = explicit[n] if n < len(explicit) else None
            # Generated ids get numbered when repeated, explicit ones are used as written
            if not heading_id:
                heading_id = slug(text)
                count = seen.get(heading_id, 0)
                seen[heading_id] = count + 1
                if count:
                    heading_id = f"{heading_id}-{count}"
        if heading and len(heading.group(1)) <= MAX_HEADING_LEVEL:
            sections.append((headings, anchor, lines))
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2)))
            headings = [text for _, text in stack]
            # An H1 is the page title, link to the top of the page instead
            anchor = heading_id if level > 1 else None
            lines = []
        else:
            lines.append(line)
    sections.append((headings, anchor, lines))

    chunks = []
    for section_headings, section_anchor, section_lines in sections:
        # Skip an H1 that just repeats the document title
        path = [title] + [h for h in section_headings if h and h != title]
        for part in _pack(_blocks(section_lines)):
            if len(part) >= MIN_CHARS:
                chunks.append(Chunk(" > ".join(path), part, section_anchor))
    return chunks
