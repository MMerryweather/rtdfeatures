"""Test documentation link integrity and user-facing language hygiene."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories that contain internal (non-user-facing) documentation.
_INTERNAL_DOC_DIRS = frozenset({
    "docs/benchmarks",
})

# Paths excluded from the marker-language scan (in addition to whole directories
# above).
_MARKER_SCAN_EXCLUDE = frozenset({
})


def _is_internal(path: Path) -> bool:
    """True if *path* lives under an internal-only directory."""
    rel = path.relative_to(ROOT).as_posix()
    for d in _INTERNAL_DOC_DIRS:
        if rel.startswith(d):
            return True
    return False


def _internal_links(markdown: str, doc_parent: Path) -> list[Path]:
    """Return resolved Paths for every relative markdown link in *markdown*."""
    links: list[Path] = []
    for text, link in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", markdown):
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        # Strip anchor fragment for file-existence check
        target = (doc_parent / link.split("#", 1)[0]).resolve()
        links.append(target)
    return links


# ── link-existence tests ────────────────────────────────────────────────


def test_readme_local_links_exist() -> None:
    """Every relative markdown link in README.md resolves to an existing file."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in _internal_links(readme, ROOT):
        assert target.exists(), f"README.md link -> {target} not found"


def test_docs_index_local_links_exist() -> None:
    """Every relative markdown link in docs/index.md resolves to an existing file."""
    index = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    for target in _internal_links(index, ROOT / "docs"):
        assert target.exists(), f"docs/index.md link -> {target} not found"


# ── marker-language scan ────────────────────────────────────────────────

_USER_FACING_DOC_GLOBS = (
    "README.md",
    "docs/index.md",
    "docs/quickstart.md",
    "docs/install.md",
    "docs/concepts/*.md",
    "docs/examples/*.md",
    "docs/user-guide/*.md",
    "docs/api/*.md",
    "docs/0[1-9]_*.md",
    "docs/16_*.md",
    "docs/17_*.md",
)

# Patterns that should not appear in user-facing documentation.
_INTERNAL_PATTERNS = re.compile(
    r"work package|work-package|milestone",
    re.IGNORECASE,
)


def _user_facing_doc_files() -> list[Path]:
    """Yield user-facing markdown files included in hygiene enforcement."""
    files: list[Path] = []
    for glob_expr in _USER_FACING_DOC_GLOBS:
        for path in sorted(ROOT.glob(glob_expr)):
            if path.name.startswith("."):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel in _MARKER_SCAN_EXCLUDE:
                continue
            if _is_internal(path):
                continue
            files.append(path)
    return files


def test_user_facing_docs_do_not_contain_rollout_markers() -> None:
    """Scan user-facing docs for rollout marker language.

    Excludes internal directories (docs/benchmarks/).
    """
    failures: list[str] = []
    for path in _user_facing_doc_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _INTERNAL_PATTERNS.search(line):
                rel = path.relative_to(ROOT)
                failures.append(f"{rel}:{lineno}: {line.strip()}")
    assert not failures, (
        "Rollout markers found in user-facing docs:\n"
        + "\n".join(failures)
    )


# ── example-file and generated-gallery existence tests ───────────────────


def test_examples_01_to_08_exist() -> None:
    """Every example script (01 through 08) exists under examples/."""
    for i in range(1, 9):
        p = Path(f"examples/{i:02d}_*.py")
        assert list(Path(ROOT).glob(str(p))), f"example {i:02d} missing"


def test_generated_parametric_empirical_gallery_exists() -> None:
    """The regenerated gallery markdown file exists."""
    p = ROOT / "docs/examples/parametric_empirical_fit_gallery.md"
    assert p.exists(), f"{p} missing"


def test_gallery_references_existing_png_files() -> None:
    """Every PNG referenced by the gallery exists on disk."""
    gallery = (ROOT / "docs/examples/parametric_empirical_fit_gallery.md").read_text()
    refs = re.findall(r'\[([^\]]*)\]\(([^)]+\.png)\)', gallery)
    gallery_dir = ROOT / "docs/examples"
    for text, link in refs:
        target = (gallery_dir / link).resolve()
        assert target.exists(), f"Gallery references missing PNG: {link}"
