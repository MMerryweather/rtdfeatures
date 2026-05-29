"""Benchmark extraction tests."""

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from rtdfeatures.benchmarks.extract_nrtd_benchmarks import (
    BENCHMARK_KERNEL_SUM_TOLERANCE,
    BENCHMARK_REGULAR_GRID_TOLERANCE,
    build_kernel_reference_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = REPO_ROOT / "test_data" / "nRTD-v1.0.0.zip"


def _npy_payload(values: list[float]) -> bytes:
    buffer = BytesIO()
    np.save(buffer, np.asarray(values, dtype=np.float64))
    return buffer.getvalue()


def _build_minimal_archive(archive_path: Path) -> None:
    base = "ChemRxnEngLab-nRTD-b7cdd47/Experiments/HSA_000/data"
    with ZipFile(archive_path, "w") as zf:
        for case in ("adler", "cholette", "dispersion", "laminar_flow"):
            case_base = f"{base}/{case}"
            zf.writestr(f"{case_base}/t_E_expected.npy", _npy_payload([0.0, 1.0, 2.0]))
            zf.writestr(f"{case_base}/E_expected.npy", _npy_payload([1.0, 2.0, 3.0]))
            zf.writestr(f"{case_base}/t_E_predicted.npy", _npy_payload([0.0, 1.0, 2.0]))
            zf.writestr(f"{case_base}/E_predicted.npy", _npy_payload([1.0, 1.0, 1.0]))

        laminar_base = f"{base}/laminar_flow"
        zf.writestr(f"{laminar_base}/time.npy", _npy_payload([0.0, 1.0, 2.0]))
        zf.writestr(f"{laminar_base}/c_conv_in.npy", _npy_payload([10.0, 11.0, 12.0]))
        zf.writestr(f"{laminar_base}/concentration.npy", _npy_payload([0.5, 0.7, 0.9]))


def test_array_to_polars_conversion_in_memory() -> None:
    df, predicted_aligned = build_kernel_reference_frame(
        case="toy",
        source_kind="analytical_reference",
        source_archive_path="archive/toy",
        lag_time_expected=np.array([0.0, 1.0, 2.0]),
        e_expected=np.array([1.0, 2.0, 3.0]),
        lag_time_predicted=np.array([0.0, 2.0]),
        e_predicted=np.array([0.5, 0.5]),
        sum_tolerance=BENCHMARK_KERNEL_SUM_TOLERANCE,
        regular_grid_tolerance=BENCHMARK_REGULAR_GRID_TOLERANCE,
    )

    assert predicted_aligned is False
    assert df.columns == [
        "case",
        "source_kind",
        "source_archive_path",
        "lag_time",
        "E_expected_raw",
        "E_expected_normalized",
        "E_predicted_raw",
        "E_predicted_normalized",
    ]
    assert df.height == 3
    assert float(df["E_expected_normalized"].drop_nulls().sum()) == pytest.approx(1.0, abs=1e-12)
    assert df["E_predicted_normalized"].drop_nulls().is_empty()
    assert df["E_predicted_raw"].drop_nulls().is_empty()


def test_predicted_raw_preserved_when_aligned_but_nonnegative_constraint_fails() -> None:
    df, predicted_aligned = build_kernel_reference_frame(
        case="toy",
        source_kind="analytical_reference",
        source_archive_path="archive/toy",
        lag_time_expected=np.array([0.0, 1.0, 2.0]),
        e_expected=np.array([1.0, 2.0, 3.0]),
        lag_time_predicted=np.array([0.0, 1.0, 2.0]),
        e_predicted=np.array([1.0, -0.25, 0.25]),
        sum_tolerance=BENCHMARK_KERNEL_SUM_TOLERANCE,
        regular_grid_tolerance=BENCHMARK_REGULAR_GRID_TOLERANCE,
    )

    assert predicted_aligned is False
    assert df["E_predicted_raw"].drop_nulls().to_list() == [1.0, -0.25, 0.25]
    assert df["E_predicted_normalized"].drop_nulls().is_empty()


def test_malformed_array_lengths_fail_clearly() -> None:
    with pytest.raises(ValueError, match="Mismatched lengths"):
        build_kernel_reference_frame(
            case="toy",
            source_kind="analytical_reference",
            source_archive_path="archive/toy",
            lag_time_expected=np.array([0.0, 1.0]),
            e_expected=np.array([1.0, 2.0, 3.0]),
            lag_time_predicted=np.array([0.0, 1.0]),
            e_predicted=np.array([1.0, 1.0]),
            sum_tolerance=BENCHMARK_KERNEL_SUM_TOLERANCE,
            regular_grid_tolerance=BENCHMARK_REGULAR_GRID_TOLERANCE,
        )


def test_manifest_and_notice_include_required_citations(tmp_path: Path) -> None:
    archive_path = tmp_path / "nRTD-v1.0.0.zip"
    _build_minimal_archive(archive_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "rtdfeatures.benchmarks.extract_nrtd_benchmarks",
            "--archive",
            str(archive_path),
            "--out-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    notice = (tmp_path / "NOTICE.md").read_text(encoding="utf-8")

    assert manifest["zenodo_doi"] == "10.5281/zenodo.15609214"
    assert manifest["zenodo_record_url"] == "https://zenodo.org/records/15609214"
    assert manifest["source_archive"] == "ChemRxnEngLab/nRTD-v1.0.0.zip"
    assert manifest["license"] == "MIT"

    assert "10.5281/zenodo.15609214" in notice
    assert "https://zenodo.org/records/15609214" in notice
    assert "ChemRxnEngLab/nRTD-v1.0.0.zip" in notice
    assert "MIT" in notice


@pytest.mark.external_data
def test_integration_script_outputs_when_zip_present(tmp_path: Path) -> None:
    if not ARCHIVE_PATH.exists():
        pytest.skip(f"local nRTD archive not present at {ARCHIVE_PATH}")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rtdfeatures.benchmarks.extract_nrtd_benchmarks",
            "--archive",
            str(ARCHIVE_PATH),
            "--out-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = manifest.get("assumption_events", [])
    cholette_events = [e for e in events if e.get("case") == "cholette"]
    assert cholette_events
    assert (
        cholette_events[0]["assumption"] == "cholette_expected_length_mismatch_trim_to_min"
    )

    first_kernel = tmp_path / "hsa_000_cholette_kernel_reference.parquet"
    assert first_kernel.exists()
    first_sha = first_kernel.read_bytes()
    second_out = tmp_path / "rerun"
    rerun = subprocess.run(
        [
            sys.executable,
            "-m",
            "rtdfeatures.benchmarks.extract_nrtd_benchmarks",
            "--archive",
            str(ARCHIVE_PATH),
            "--out-dir",
            str(second_out),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rerun.returncode == 0, rerun.stderr
    second_kernel = second_out / "hsa_000_cholette_kernel_reference.parquet"
    assert second_kernel.exists()
    assert first_sha == second_kernel.read_bytes()
