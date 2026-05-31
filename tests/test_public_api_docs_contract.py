from __future__ import annotations

from pathlib import Path

import rtdfeatures

ROOT = Path(__file__).resolve().parents[1]


def _section_between(text: str, start_heading: str, next_heading: str) -> str:
    assert start_heading in text, f"Missing heading: {start_heading}"
    assert next_heading in text, f"Missing heading: {next_heading}"
    start = text.index(start_heading)
    end = text.index(next_heading, start + len(start_heading))
    return text[start:end]


def test_stability_doc_lists_all_root_exports_in_stable_section() -> None:
    text = (ROOT / "docs" / "api" / "stability.md").read_text(encoding="utf-8")
    stable_section = _section_between(
        text,
        "## Stable V1 API",
        "## Provisional V1 API",
    )

    for name in rtdfeatures.__all__:
        assert f"`{name}`" in stable_section


def test_release_notes_lists_all_root_exports_in_stable_section() -> None:
    text = (ROOT / "docs" / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    stable_section = _section_between(
        text,
        "## Stable public API",
        "## Advanced and provisional APIs",
    )

    for name in rtdfeatures.__all__:
        assert f"`{name}`" in stable_section


def test_non_root_objects_are_not_listed_as_stable_root_api() -> None:
    non_root_names = [
        "ErlangKernel",
        "LogNormalKernel",
        "FixedDelayKernelLearner",
        "UniformKernelLearner",
        "DelayedExponentialKernelLearner",
        "ErlangKernelLearner",
        "LogNormalKernelLearner",
    ]

    stability = (ROOT / "docs" / "api" / "stability.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs" / "RELEASE_NOTES.md").read_text(encoding="utf-8")

    stability_stable = _section_between(
        stability,
        "## Stable V1 API",
        "## Provisional V1 API",
    )
    release_stable = _section_between(
        release_notes,
        "## Stable public API",
        "## Advanced and provisional APIs",
    )

    for name in non_root_names:
        assert f"`{name}`" not in stability_stable
        assert f"`{name}`" not in release_stable
