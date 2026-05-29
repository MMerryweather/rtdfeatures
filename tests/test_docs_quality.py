"""legacy milestone tests for docs delivery and markdown usability."""

from __future__ import annotations

import re
from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _python_blocks(markdown: str) -> list[tuple[int, str]]:
    pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    return [(match.start(), match.group(1)) for match in pattern.finditer(markdown)]


def _internal_links(markdown: str) -> list[str]:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    links: list[str] = []
    for link in link_pattern.findall(markdown):
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        links.append(link.split("#", maxsplit=1)[0])
    return links


def test_documentation_delivery_decision_is_explicit() -> None:
    text = _read("docs/13_documentation_delivery_decision.md")
    assert "Do not ship a documentation site for `v0.3`." in text
    assert "repository markdown" in text
    assert "docs-test:" in text


def test_readme_and_docs_navigation_include_decision_record() -> None:
    readme = _read("README.md")
    assert "docs/index.md" in readme


def test_marked_executable_examples_run() -> None:
    targets = (
        ("README.md", "docs-test:readme-primary"),
        ("README.md", "docs-test:readme-shared"),
        ("docs/10_examples_and_use_cases.md", "docs-test:examples-single-delay"),
        ("docs/10_examples_and_use_cases.md", "docs-test:examples-v05-minimal"),
    )
    for path, marker in targets:
        markdown = _read(path)
        blocks = _python_blocks(markdown)
        executed = False
        for start, code in blocks:
            prior = markdown[max(0, start - 220) : start]
            if marker in prior:
                exec(code, {})
                executed = True
                break
        assert executed, f"missing executable block for marker {marker} in {path}"


def test_non_executable_examples_are_marked() -> None:
    text = _read("docs/10_examples_and_use_cases.md")
    assert "Illustrative example (non-executable in docs tests" in text


def test_practical_internal_markdown_links_resolve() -> None:
    doc_roots = [Path("README.md")]
    doc_roots.extend(sorted(path for path in Path("docs").rglob("*.md")))
    for doc in doc_roots:
        markdown = doc.read_text(encoding="utf-8")
        for link in _internal_links(markdown):
            candidate = (doc.parent / link).resolve()
            assert candidate.exists(), f"broken link in {doc}: {link}"


def test_feature_evidence_interpretation_taxonomy_is_consistent() -> None:
    docs05 = _read("docs/05_feature_generation_design.md")
    docs07 = _read("docs/07_validation_and_diagnostics.md")
    docs08 = _read("docs/08_api_design.md")

    required_labels = (
        "material_memory",
        "process_response",
        "statistical_pattern",
        "unknown",
    )
    removed_labels = ("`rtd_like`", "`response_like`", "`statistical`", "`weak`")

    for label in required_labels:
        assert label in docs05
        assert label in docs07
        assert label in docs08

    for label in removed_labels:
        assert label not in docs05
        assert label not in docs07
        assert label not in docs08


def test_roadmap_v09_v095_split_wording_is_explicit() -> None:
    roadmap = _read("docs/12_development_roadmap.md")

    assert "feature evidence contract and evidence reporting only" in roadmap
    assert "no public OOF execution helper in this version" in roadmap
    assert "historical filename includes OOF wording" in roadmap
    assert "public OOF split contracts and `fit_transform_oof(...)`" in roadmap
    assert "OOF execution work is split here" in roadmap


def test_feature_plan_validation_and_manifest_are_version_scoped() -> None:
    docs07 = _read("docs/07_validation_and_diagnostics.md")

    assert "(`v1.5`/`v1.6`, provisional)" in docs07
    assert "FeaturePlanManifest` is planned for `v1.5`" in docs07
    assert "FeaturePlanValidationReport` is planned for `v1.6`" in docs07
    assert "shipped `v1.0` contract" in docs07


def test_scope_doc_does_not_claim_post_v1_surfaces_are_shipped_v1() -> None:
    docs03 = _read("docs/03_scope_and_non_goals.md")
    docs07 = _read("docs/07_validation_and_diagnostics.md")

    assert "execution surfaces are post-`v1.0`" in docs03
    assert "validation reports are post-`v1.0`" in docs03
    assert "FeaturePlanManifest` is planned for `v1.5`" in docs07
    assert "FeaturePlanValidationReport` is planned for `v1.6`" in docs07
    assert "shipped `v1.0` contract" in docs07


def test_package_architecture_current_stable_modules_are_not_incomplete() -> None:
    docs09 = _read("docs/09_package_architecture.md")

    required_current_packages = ("`candidates/`", "`bootstrap/`", "`oof/`")
    for package_dir in required_current_packages:
        assert package_dir in docs09

    required_support_modules = (
        "`baselines.py`",
        "`reporting.py`",
        "`synthetic.py`",
        "`utils.py`",
    )
    for module in required_support_modules:
        assert module in docs09

    assert "Stable Core Packages (Current `v1.0`)" in docs09
