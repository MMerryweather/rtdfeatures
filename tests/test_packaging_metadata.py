"""Release packaging metadata and typing marker contract tests."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_release_metadata_contract() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'setuptools>=77.0.3' in pyproject_text
    assert 'license-files = ["LICENSE"]' in pyproject_text
    assert (
        'Documentation = "https://github.com/<org-or-user>/rtdfeatures/tree/main/docs"'
        in pyproject_text
    )
    assert '"Changelog" = "https://github.com/<org-or-user>/rtdfeatures/releases"' in pyproject_text
    assert "\ndocs = []\n" not in pyproject_text


def test_py_typed_marker_exists() -> None:
    marker_path = REPO_ROOT / "src" / "rtdfeatures" / "py.typed"
    assert marker_path.exists()
    assert marker_path.is_file()
