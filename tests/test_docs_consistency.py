"""Docs consistency checks for contributor-facing paths."""

from __future__ import annotations

from pathlib import Path


def test_docs_do_not_reference_removed_synthetic_package_dir() -> None:
    stale_reference = "src/rtdfeatures/synthetic/"
    expected_reference = "src/rtdfeatures/synthetic.py"
    targets = (
        Path("README.md"),
        Path("CONTRIBUTING.md"),
        Path("docs"),
    )

    found_stale: list[str] = []
    for target in targets:
        if target.is_dir():
            files = sorted(path for path in target.rglob("*.md"))
        else:
            files = [target]
        for file in files:
            text = file.read_text(encoding="utf-8")
            if stale_reference in text:
                found_stale.append(str(file))

    assert not found_stale, f"stale synthetic path reference found in: {found_stale}"
    assert expected_reference in Path("CONTRIBUTING.md").read_text(encoding="utf-8")
