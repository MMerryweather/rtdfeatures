"""Archive inspection tests."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "inspect_nrtd_archive.py"
ARCHIVE_PATH = REPO_ROOT / "test_data" / "nRTD-v1.0.0.zip"


def test_import_smoke() -> None:
    import numpy  # noqa: F401
    import polars  # noqa: F401
    import pyarrow  # noqa: F401


@pytest.mark.external_data
def test_script_inspection_and_metadata(tmp_path: Path) -> None:
    if not ARCHIVE_PATH.exists():
        pytest.skip(f"local nRTD archive not present at {ARCHIVE_PATH}")

    metadata_out = tmp_path / "inspection.json"
    table_out = tmp_path / "inventory.md"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--archive",
            str(ARCHIVE_PATH),
            "--metadata-out",
            str(metadata_out),
            "--table-out",
            str(table_out),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    metadata = json.loads(metadata_out.read_text(encoding="utf-8"))

    assert metadata["zenodo_doi"] == "10.5281/zenodo.15609214"
    assert metadata["zenodo_record_url"] == "https://zenodo.org/records/15609214"
    assert metadata["source_archive"] == "ChemRxnEngLab/nRTD-v1.0.0.zip"
    assert metadata["license"] == "MIT"

    summaries = metadata["array_summaries"]
    assert summaries

    hsa000_paths = [
        entry["source_archive_path"]
        for entry in summaries
        if "/HSA_000/" in entry["source_archive_path"]
    ]
    for case in ("adler", "cholette", "dispersion", "laminar_flow"):
        assert any(f"/data/{case}/" in path for path in hsa000_paths)

    assert any(
        entry["source_archive_path"].endswith("_E_nRTD.npz")
        and entry["fixture_kind"] == "learned_experimental_reference"
        for entry in summaries
    )
    npz_entries = [
        entry
        for entry in summaries
        if entry["source_archive_path"].endswith("_E_nRTD.npz")
    ]
    assert npz_entries
    for entry in npz_entries:
        assert entry["dtype"] == "npz"
        assert entry["min_value"] is None
        assert entry["max_value"] is None
        assert entry["npz_keys"]
        assert entry["npz_members"]
        assert len(entry["npz_keys"]) == len(entry["npz_members"])
        for member in entry["npz_members"]:
            assert set(member) == {"key", "shape", "dtype", "min_value", "max_value"}
            assert member["key"] in entry["npz_keys"]
            assert isinstance(member["shape"], list)
            assert isinstance(member["dtype"], str)
    assert any(entry["fixture_kind"] == "analytical_reference" for entry in summaries)

    table = table_out.read_text(encoding="utf-8")
    assert "| fixture_kind | case | source_archive_path |" in table


def test_pyproject_dependency_policy() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "numpy>=" in pyproject_text
    assert "polars>=" in pyproject_text
    assert "torch>=" in pyproject_text
    assert "dev = [" in pyproject_text
    assert "\"pytest\"" in pyproject_text
    assert "\"ruff\"" in pyproject_text
    assert "\"mypy\"" in pyproject_text
    assert "benchmark = [" in pyproject_text
    assert "\"pyarrow\"" in pyproject_text

    # extract the core [project] dependencies block only
    deps_start = pyproject_text.index("dependencies = [")
    deps_end = pyproject_text.index("\n[project.urls]")
    core_deps = pyproject_text[deps_start:deps_end].lower()

    banned = ["scipy", "lightning", "matplotlib", "seaborn", "notebook", "jupyter"]
    for token in banned:
        assert token not in core_deps
