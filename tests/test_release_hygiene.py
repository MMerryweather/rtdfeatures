from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]


def test_dev_extra_includes_pytest_cov() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = data["project"]["optional-dependencies"]["dev"]
    assert any(dep.startswith("pytest-cov") for dep in dev)


def test_coverage_threshold_is_configured() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    report = data["tool"]["coverage"]["report"]
    fail_under = int(report["fail_under"])
    assert fail_under >= 85

    ci_text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert f"-cov-fail-under={fail_under}" in ci_text


def test_dependency_review_or_audit_is_configured() -> None:
    dep_review = ROOT / ".github" / "workflows" / "dependency-review.yml"
    ci = ROOT / ".github" / "workflows" / "ci.yml"
    has_pip_audit = ci.exists() and "pip-audit" in ci.read_text(encoding="utf-8")
    assert dep_review.exists() or has_pip_audit


def test_behavior_tests_avoid_rollout_markers_in_names_and_text() -> None:
    marker_pattern = re.compile(
        r"work package|work-package",
        re.IGNORECASE,
    )
    excluded_prefixes = (
        "tests/test_archive_inspection.py",
        "tests/test_benchmark_extraction.py",
    )

    failures: list[str] = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("tests/simulation_harness/"):
            continue
        if rel in excluded_prefixes:
            continue
        if marker_pattern.search(path.name):
            failures.append(f"{rel}:filename")
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(('"""', "#")) and marker_pattern.search(stripped):
                failures.append(f"{rel}:{lineno}: {stripped}")

    assert not failures, "Rollout markers found in behavior test surfaces:\n" + "\n".join(failures)
