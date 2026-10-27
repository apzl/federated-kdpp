"""Offline document structure and teaching-math checks; no private fixtures read."""

from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "supplements/mpc_euclidean_explainer.html"


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.external_resources = []
        self.controls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a":
            self.links.append(attrs.get("href", ""))
        if tag in ("script", "img", "iframe", "link") and (attrs.get("src") or attrs.get("href")):
            self.external_resources.append(attrs)
        if tag in ("button", "input", "select"):
            self.controls.append((tag, attrs))


def test_explainer_is_self_contained_and_has_valid_internal_links():
    html = PAGE.read_text()
    document = Document()
    document.feed(html)
    assert len(document.ids) == len(set(document.ids))
    assert not document.external_resources
    assert "connect-src 'none'" in html
    for link in document.links:
        if link.startswith("#"):
            assert link[1:] in document.ids
        elif link and not link.startswith(("https:", "http:")):
            target = (PAGE.parent / unquote(link)).resolve()
            # Generated run records are optional in a fresh checkout.
            assert "/results/" in str(target) or target.exists(), link
    for tag, attrs in document.controls:
        if tag == "button":
            assert attrs.get("type") == "button"
    assert "prefers-reduced-motion" in html
    assert 'lang="en"' in html
    referenced_ids = set(re.findall(r'\$\("([\w-]+)"\)', html))
    assert referenced_ids - {"chart-cursor", "chart-dot"} <= set(document.ids)


def test_explainer_teaching_math_and_paired_result_snapshot():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; required only to check browser teaching math")
    result = subprocess.run([node, str(ROOT / "tests/check_mpc_explainer.mjs")],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Explainer checks passed" in result.stdout


def test_mermaid_setting_diagrams_are_separate_and_link_to_existing_sources():
    path = ROOT / "supplements/centralized_federated_overlap.md"
    markdown = path.read_text()
    assert len(re.findall(r"^```mermaid$", markdown, flags=re.MULTILINE)) == 2
    assert "flowchart TB" in markdown
    assert "PLAINTEXT DISCLOSURE" in markdown
    assert "no local suggestion pool" in markdown
    for target in re.findall(r"\]\(([^)]+)\)", markdown):
        if not target.startswith(("https:", "http:", "#")):
            resolved = (path.parent / unquote(target)).resolve()
            assert "/results/" in str(resolved) or resolved.exists(), target
    assert "```mermaid" not in PAGE.read_text()
