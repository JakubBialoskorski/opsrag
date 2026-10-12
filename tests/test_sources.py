from opsrag.sources import (
    Locked,
    Source,
    doc_url,
    is_excluded,
    is_published,
    load_lock,
    pick_latest_tag,
    save_lock,
    with_anchor,
)

K8S = Source(
    name="kubernetes",
    repo="https://github.com/kubernetes/website",
    track="main",
    root="content/en/docs",
    url="https://kubernetes.io/docs/{path}/",
    exclude=["reference/kubernetes-api/*", "test.md"],
)


def test_doc_url():
    assert (
        doc_url(K8S, "concepts/workloads/controllers/deployment.md")
        == "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/"
    )


def test_doc_url_for_index_pages():
    assert doc_url(K8S, "concepts/_index.md") == "https://kubernetes.io/docs/concepts/"
    assert doc_url(K8S, "_index.md") == "https://kubernetes.io/docs/"


def test_exclusions():
    assert is_excluded(K8S, "reference/kubernetes-api/workload-resources/pod-v1.md")
    assert is_excluded(K8S, "test.md")
    assert is_excluded(K8S, "_v4-in-progress.mdx")
    assert not is_excluded(K8S, "concepts/_index.md")
    assert not is_excluded(K8S, "concepts/overview.md")


def test_pick_latest_tag_ignores_prereleases():
    tags = ["v3.9.0", "v3.10.0-rc.0", "v3.9.1", "v3.2.0"]
    assert pick_latest_tag(tags) == "v3.9.1"
    assert pick_latest_tag(["v4.0.0-rc.1"]) is None


def test_lock_roundtrip(tmp_path):
    sources_file = tmp_path / "sources.yaml"
    lock = {"prometheus": Locked(ref="v3.9.1", version="v3.9.1")}
    save_lock(sources_file, lock)
    assert (tmp_path / "sources.lock.yaml").exists()
    assert load_lock(sources_file) == lock


def test_link_rule_uses_front_matter_fields():
    source = Source(
        name="kubernetes",
        repo="r",
        track="main",
        root="docs",
        url="https://kubernetes.io/docs/{path}/",
        links=[
            {
                "match": "reference/glossary/*",
                "url": "https://kubernetes.io/docs/reference/glossary/?all=true#term-{id}",
            }
        ],
    )
    assert (
        doc_url(source, "reference/glossary/istio.md", {"id": "istio"})
        == "https://kubernetes.io/docs/reference/glossary/?all=true#term-istio"
    )
    # The glossary page itself has no id, so it keeps the default URL
    assert (
        doc_url(source, "reference/glossary/index.md", {"id": ""})
        == "https://kubernetes.io/docs/reference/glossary/"
    )


def test_unpublished_hugo_pages_are_skipped():
    assert is_published({"title": "Deployments"})
    assert not is_published({"headless": True})
    assert not is_published({"_build": {"list": "never", "render": "never"}})


def test_with_anchor():
    assert (
        with_anchor("https://k8s.io/docs/x/", "pod-template")
        == "https://k8s.io/docs/x/#pod-template"
    )
    assert with_anchor("https://k8s.io/docs/x/", None) == "https://k8s.io/docs/x/"
    # Glossary entries already point to their place on the page
    assert (
        with_anchor("https://k8s.io/g/?all=true#term-istio", "x")
        == "https://k8s.io/g/?all=true#term-istio"
    )
