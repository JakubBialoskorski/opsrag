from opsrag.chunking import (
    MAX_CHARS,
    blackfriday_slug,
    chunk_markdown,
    clean_markdown,
    explicit_heading_ids,
    github_slug,
    split_front_matter,
)


def test_front_matter_is_parsed_and_removed():
    meta, body = split_front_matter("---\ntitle: Deployments\nweight: 10\n---\n\nBody text")
    assert meta == {"title": "Deployments", "weight": 10}
    assert body == "Body text"


def test_no_front_matter():
    assert split_front_matter("# Hello") == ({}, "# Hello")


def test_hugo_shortcodes():
    text = (
        'Updates {{< glossary_tooltip text="Pods" term_id="pod" >}} and '
        '{{< glossary_tooltip term_id="replica-set" >}}.\n\n'
        "{{< note >}}\nBe careful.\n{{</ note >}}\n\n"
        '{{< feature-state for_k8s_version="v1.31" state="stable" >}}\n\n'
        '## {{% heading "prerequisites" %}}\n\n'
        '{{< code_sample file="pod.yaml" >}}'
    )
    cleaned = clean_markdown(text)
    assert "Updates Pods and replica set." in cleaned
    assert "**Note:**\nBe careful." in cleaned
    assert "(feature state: stable since v1.31)" in cleaned
    assert "## Before you begin" in cleaned
    assert "{{" not in cleaned


def test_mdx_noise_is_removed():
    text = (
        'import Helm4 from "/docs/_v4-in-progress.mdx"\n\n<Helm4/>\n\n'
        "We import files verbatim from somewhere.\n\n"
        ":::warning\nDanger\n:::\n<!-- hidden -->"
    )
    cleaned = clean_markdown(text)
    assert "Helm4" not in cleaned
    assert "We import files verbatim" in cleaned
    assert "**Warning:**\nDanger" in cleaned
    assert "hidden" not in cleaned


def test_chunks_follow_headings_with_breadcrumbs():
    body = (
        "Intro paragraph that is long enough to be kept as its own chunk.\n\n"
        "## Rolling back\n\nUse kubectl rollout undo to go back to the previous revision.\n\n"
        "### To a specific revision\n\nPass --to-revision with the revision number you want.\n\n"
        "#### Deep heading stays inside\n\nThis text belongs to the H3 chunk above it.\n\n"
        "## Scaling\n\nUse kubectl scale to change the number of replicas of a Deployment."
    )
    chunks = chunk_markdown("Deployments", body)
    assert [c.heading_path for c in chunks] == [
        "Deployments",
        "Deployments > Rolling back",
        "Deployments > Rolling back > To a specific revision",
        "Deployments > Scaling",
    ]
    assert "Deep heading stays inside" in chunks[2].content


def test_heading_inside_code_block_is_not_a_heading():
    body = (
        "## Config\n\nExample configuration file for the server:\n\n"
        "```yaml\n# this is a yaml comment\nkey: value\n```"
    )
    chunks = chunk_markdown("Doc", body)
    assert len(chunks) == 1
    assert "# this is a yaml comment" in chunks[0].content


def test_h1_equal_to_title_is_not_repeated():
    chunks = chunk_markdown("Doc", "# Doc\n\nSome introduction text that is long enough to keep.")
    assert chunks[0].heading_path == "Doc"


def test_long_section_is_split_by_paragraphs_and_code_stays_whole():
    paragraph = "word " * 100  # 500 chars
    code = "```\n" + "line\n" * 200 + "```"  # longer than MAX_CHARS
    body = "\n\n".join([paragraph] * 5 + [code])
    chunks = chunk_markdown("Doc", body)
    assert len(chunks) > 1
    assert all(len(c.content) <= MAX_CHARS for c in chunks[:-1])
    assert chunks[-1].content == code


def test_tiny_sections_are_dropped():
    chunks = chunk_markdown("Doc", "## Empty\n\nshort\n\n## Real\n\n" + "content " * 20)
    assert [c.heading_path for c in chunks] == ["Doc > Real"]


def test_heading_anchors_are_removed():
    assert clean_markdown("## Conditions {#condition}\n\nText") == "## Conditions\n\nText"


def test_slug_styles_match_the_docs_sites():
    # Checked against the live pages of helm.sh, prometheus.io and kubernetes.io
    assert github_slug("`absent_over_time()`") == "absent_over_time"
    assert github_slug("Hook deletion policies") == "hook-deletion-policies"
    assert blackfriday_slug("`absent_over_time()`") == "absent-over-time"
    assert blackfriday_slug("What's next") == "what-s-next"
    assert blackfriday_slug("Visualization &amp; Control") == "visualization-control"
    assert github_slug("[Ingress](/docs/ingress) controllers") == "ingress-controllers"


def test_chunks_point_to_their_section():
    body = (
        "# Doc\n\nIntro text that is long enough to be kept as its own chunk.\n\n"
        "## Node controller\n\nThe first node controller section with enough text.\n\n"
        "## Authorization\n\nAuthorization overview with enough text to be kept.\n\n"
        "### Node controller\n\nThe second node controller section, with an explicit id.\n\n"
        "## Examples\n\nFirst examples section with enough text to be kept.\n\n"
        "## Examples\n\nSecond examples section with enough text to be kept."
    )
    explicit = explicit_heading_ids(
        body.replace("### Node controller", "### Node controller {#authorization-node}")
    )
    chunks = chunk_markdown("Doc", body, blackfriday_slug, explicit)
    assert [c.anchor for c in chunks] == [
        None,  # under the H1 page title: link to the top of the page
        "node-controller",
        "authorization",
        "authorization-node",  # explicit id only for the second occurrence
        "examples",
        "examples-1",  # repeated generated ids are numbered
    ]


def test_explicit_ids_are_not_numbered():
    body = "## Phase {#cmd}\n\n" + "text " * 20 + "\n\n## Phase {#cmd}\n\n" + "more " * 20
    chunks = chunk_markdown("Doc", clean_markdown(body), github_slug, explicit_heading_ids(body))
    assert [c.anchor for c in chunks] == ["cmd", "cmd"]
