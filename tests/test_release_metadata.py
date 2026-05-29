"""Tests for package metadata, dependencies, and install docs."""
from __future__ import annotations

import os
import re

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PYPROJECT_TOML = os.path.join(PROJECT_ROOT, "pyproject.toml")
INSTALL_DOCS = os.path.join(PROJECT_ROOT, "docs", "install.md")


def _read_pyproject() -> str:
    with open(PYPROJECT_TOML, encoding="utf-8") as f:
        return f.read()


def _read_version() -> str:
    content = _read_pyproject()
    m = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    assert m is not None, "version not found in pyproject.toml"
    return m.group(1)


def test_classifier_matches_version() -> None:
    """Version-aware classifier check.

    Rules:
    - version ending in ``rc1`` requires Beta classifier
    - version exactly ``1.0.0`` requires Production/Stable classifier
    - Alpha classifier is always rejected
    """
    content = _read_pyproject()
    version = _read_version()

    has_alpha = "Development Status :: 3 - Alpha" in content
    has_beta = "Development Status :: 4 - Beta" in content
    has_stable = "Development Status :: 5 - Production/Stable" in content

    assert not has_alpha, "Alpha classifier is not allowed in this release path"

    if version.endswith("rc1"):
        assert has_beta, (
            f"Version {version!r} ends with 'rc1' but Beta classifier "
            f"is missing from pyproject.toml"
        )
        assert not has_stable, (
            f"Version {version!r} is an RC but Production/Stable "
            f"classifier is present; use Beta until final release"
        )
    elif version == "1.0.0":
        assert has_stable, (
            f"Version {version!r} is the final release but "
            f"Production/Stable classifier is missing"
        )
        assert not has_beta, (
            f"Version {version!r} is the final release but Beta "
            f"classifier is still present"
        )
    else:
        raise AssertionError(
            f"Unrecognised version scheme: {version!r}. "
            f"Expected 1.0.0rc1 or 1.0.0."
        )


def test_torch_is_core_dependency() -> None:
    content = _read_pyproject()
    deps_section = re.search(
        r"^dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL | re.MULTILINE
    )
    assert deps_section is not None, "dependencies section not found"
    assert '"torch>=2.0"' in deps_section.group(
        1
    ), "torch>=2.0 not in dependencies"


def test_sklearn_extra_is_optional() -> None:
    content = _read_pyproject()

    deps_match = re.search(
        r"^dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL | re.MULTILINE
    )
    assert deps_match is not None, "dependencies section not found"
    deps = deps_match.group(1)

    assert "scikit-learn" not in deps, "scikit-learn should not be in core deps"
    assert "pandas" not in deps, "pandas should not be in core deps"

    headers = re.findall(r"^\[.+\]", content, re.MULTILINE)
    assert "[project.optional-dependencies]" in headers

    opt_start = content.index("[project.optional-dependencies]")
    next_section = -1
    for h in headers:
        idx = content.index(h)
        if idx > opt_start and (next_section == -1 or idx < next_section):
            next_section = idx
    opt_body = content[opt_start:next_section] if next_section != -1 else content[opt_start:]

    assert "sklearn" in opt_body, "sklearn extra not in optional-dependencies"
    assert "scikit-learn>=1.3" in opt_body
    assert "pandas>=2.0" in opt_body


def test_install_docs_mentions_sklearn_extra() -> None:
    with open(INSTALL_DOCS, encoding="utf-8") as f:
        content = f.read()
    assert "sklearn" in content, "docs/install.md should mention sklearn extra"
