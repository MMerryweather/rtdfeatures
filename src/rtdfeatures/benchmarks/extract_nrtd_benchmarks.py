"""Extract selected nRTD HSA_000 benchmark fixtures directly from archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
import polars as pl

BENCHMARK_KERNEL_SUM_TOLERANCE = 1e-9
BENCHMARK_REGULAR_GRID_TOLERANCE = 1e-9

ZENODO_RECORD_URL = "https://zenodo.org/records/15609214"
ZENODO_DOI = "10.5281/zenodo.15609214"
SOURCE_ARCHIVE_NAME = "ChemRxnEngLab/nRTD-v1.0.0.zip"
LICENSE_NAME = "MIT"

HSA_CASES = ("adler", "cholette", "dispersion", "laminar_flow")
CASE_PREFIX = "ChemRxnEngLab-nRTD-b7cdd47/Experiments/HSA_000/data"


def _array_path(case: str, filename: str) -> str:
    return f"{CASE_PREFIX}/{case}/{filename}"


def _load_npy_from_zip(zip_file: ZipFile, archive_path: str) -> np.ndarray:
    try:
        payload = zip_file.read(archive_path)
    except KeyError as exc:
        raise ValueError(f"Missing required source array: {archive_path}") from exc
    with io.BytesIO(payload) as buffer:
        return cast(npt.NDArray[np.float64], np.load(buffer))


def _require_1d_finite(name: str, array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 1:
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 1:
            arr = squeezed
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D numeric array; got shape={arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _require_same_length(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    if a.size != b.size:
        raise ValueError(f"Mismatched lengths: {name_a}={a.size} vs {name_b}={b.size}")


def _validate_regular_grid(name: str, grid: np.ndarray, tolerance: float) -> None:
    if grid.size < 2:
        raise ValueError(f"{name} must contain at least 2 points")
    diffs = np.diff(grid)
    if np.any(diffs <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if not np.allclose(diffs, diffs[0], atol=tolerance, rtol=0.0):
        raise ValueError(f"{name} is irregular beyond tolerance={tolerance}")


def _normalized_weights(name: str, values: np.ndarray, sum_tolerance: float) -> np.ndarray:
    if np.any(values < 0.0):
        raise ValueError(f"{name} contains negative weights")
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have a positive finite sum")
    normalized = values / total
    if np.any(normalized < 0.0):
        raise ValueError(f"{name} normalization produced negative values")
    if abs(float(normalized.sum()) - 1.0) > sum_tolerance:
        raise ValueError(f"{name} normalization sum check failed for tolerance={sum_tolerance}")
    return normalized


def build_kernel_reference_frame(
    *,
    case: str,
    source_kind: str,
    lag_time_expected: np.ndarray,
    e_expected: np.ndarray,
    lag_time_predicted: np.ndarray,
    e_predicted: np.ndarray,
    source_archive_path: str,
    sum_tolerance: float,
    regular_grid_tolerance: float,
) -> tuple[pl.DataFrame, bool]:
    lag_time_expected = _require_1d_finite("lag_time_expected", lag_time_expected)
    e_expected = _require_1d_finite("E_expected_raw", e_expected)
    lag_time_predicted = _require_1d_finite("lag_time_predicted", lag_time_predicted)
    e_predicted = _require_1d_finite("E_predicted_raw", e_predicted)

    _require_same_length("lag_time_expected", lag_time_expected, "E_expected_raw", e_expected)
    _require_same_length("lag_time_predicted", lag_time_predicted, "E_predicted_raw", e_predicted)
    _validate_regular_grid("lag_time_expected", lag_time_expected, tolerance=regular_grid_tolerance)

    e_expected_norm = _normalized_weights("E_expected_raw", e_expected, sum_tolerance)
    predicted_nonnegative = bool(np.all(e_predicted >= 0.0))
    e_predicted_norm: np.ndarray | None = None
    if predicted_nonnegative:
        e_predicted_norm = _normalized_weights("E_predicted_raw", e_predicted, sum_tolerance)

    expected_df = pl.DataFrame(
        {
            "lag_time": lag_time_expected,
            "E_expected_raw": e_expected,
            "E_expected_normalized": e_expected_norm,
            "E_predicted_raw": [None] * lag_time_expected.size,
            "E_predicted_normalized": [None] * lag_time_expected.size,
        }
    )
    predicted_aligned = lag_time_expected.size == lag_time_predicted.size and np.allclose(
        lag_time_expected, lag_time_predicted, atol=regular_grid_tolerance, rtol=0.0
    )
    if predicted_aligned:
        expected_df = expected_df.with_columns(pl.Series("E_predicted_raw", e_predicted))
        if predicted_nonnegative and e_predicted_norm is not None:
            expected_df = expected_df.with_columns(
                pl.Series("E_predicted_normalized", e_predicted_norm),
            )

    frame = expected_df.with_columns(
        pl.lit(case).alias("case"),
        pl.lit(source_kind).alias("source_kind"),
        pl.lit(source_archive_path).alias("source_archive_path"),
    ).select(
        [
            "case",
            "source_kind",
            "source_archive_path",
            "lag_time",
            "E_expected_raw",
            "E_expected_normalized",
            "E_predicted_raw",
            "E_predicted_normalized",
        ]
    )
    return frame, predicted_aligned and predicted_nonnegative


def _coerce_expected_arrays_for_case(
    *,
    case: str,
    lag_time_expected: np.ndarray,
    e_expected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object] | None]:
    if lag_time_expected.shape[0] == e_expected.shape[0]:
        return lag_time_expected, e_expected, None

    if case == "cholette":
        trimmed_len = int(min(lag_time_expected.shape[0], e_expected.shape[0]))
        assumption = {
            "case": case,
            "assumption": "cholette_expected_length_mismatch_trim_to_min",
            "lag_time_expected_original_length": int(lag_time_expected.shape[0]),
            "e_expected_original_length": int(e_expected.shape[0]),
            "trimmed_length": trimmed_len,
            "deterministic_rule": "trim both arrays to shared min length from index 0",
        }
        return lag_time_expected[:trimmed_len], e_expected[:trimmed_len], assumption

    raise ValueError(
        f"Mismatched lengths for case={case}: "
        f"t_E_expected={lag_time_expected.shape[0]} vs E_expected={e_expected.shape[0]}"
    )


def build_signal_frame(
    *,
    case: str,
    source_archive_path: str,
    time: np.ndarray,
    input_signal: np.ndarray,
    target_signal: np.ndarray,
) -> pl.DataFrame:
    time = _require_1d_finite("time", time)
    input_signal = _require_1d_finite("input_signal", input_signal)
    target_signal = _require_1d_finite("target_signal", target_signal)

    _require_same_length("time", time, "input_signal", input_signal)
    _require_same_length("time", time, "target_signal", target_signal)
    _validate_regular_grid("time", time, tolerance=BENCHMARK_REGULAR_GRID_TOLERANCE)

    return pl.DataFrame(
        {
            "case": [case] * time.size,
            "source_archive_path": [source_archive_path] * time.size,
            "time": time,
            "input_signal": input_signal,
            "target_signal": target_signal,
        }
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_notice(out_dir: Path) -> None:
    notice = f"""# nRTD Benchmark Fixture Notice

Source benchmark arrays were extracted from nRTD v1.0.0.

- Zenodo DOI: {ZENODO_DOI}
- Zenodo record URL: {ZENODO_RECORD_URL}
- Source archive: {SOURCE_ARCHIVE_NAME}
- License: {LICENSE_NAME}

Please cite the nRTD Zenodo record and preserve the original MIT license terms.
"""
    (out_dir / "NOTICE.md").write_text(notice, encoding="utf-8")


def extract_benchmarks(*, archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[dict[str, str | list[str]]] = []
    skipped_selected_sources: list[dict[str, str]] = []
    assumption_events: list[dict[str, object]] = []

    with ZipFile(archive_path, "r") as zip_file:
        for case in HSA_CASES:
            t_expected_path = _array_path(case, "t_E_expected.npy")
            e_expected_path = _array_path(case, "E_expected.npy")
            t_predicted_path = _array_path(case, "t_E_predicted.npy")
            e_predicted_path = _array_path(case, "E_predicted.npy")

            t_expected = _load_npy_from_zip(zip_file, t_expected_path)
            e_expected = _load_npy_from_zip(zip_file, e_expected_path)
            t_predicted = _load_npy_from_zip(zip_file, t_predicted_path)
            e_predicted = _load_npy_from_zip(zip_file, e_predicted_path)
            t_expected, e_expected, assumption = _coerce_expected_arrays_for_case(
                case=case,
                lag_time_expected=t_expected,
                e_expected=e_expected,
            )
            if assumption is not None:
                assumption_events.append(assumption)
            predicted_grid_aligned = t_expected.shape[0] == t_predicted.shape[0] and np.allclose(
                t_expected,
                t_predicted,
                atol=BENCHMARK_REGULAR_GRID_TOLERANCE,
                rtol=0.0,
            )
            predicted_nonnegative = bool(np.all(e_predicted >= 0.0))

            kernel_df, predicted_aligned = build_kernel_reference_frame(
                case=case,
                source_kind="analytical_reference",
                lag_time_expected=t_expected,
                e_expected=e_expected,
                lag_time_predicted=t_predicted,
                e_predicted=e_predicted,
                source_archive_path=f"{CASE_PREFIX}/{case}",
                sum_tolerance=BENCHMARK_KERNEL_SUM_TOLERANCE,
                regular_grid_tolerance=BENCHMARK_REGULAR_GRID_TOLERANCE,
            )

            kernel_name = f"hsa_000_{case}_kernel_reference.parquet"
            kernel_out = out_dir / kernel_name
            kernel_df.write_parquet(kernel_out)
            if not predicted_aligned:
                if not predicted_grid_aligned:
                    skipped_selected_sources.extend(
                        [
                            {
                                "source_archive_path": t_predicted_path,
                                "reason": (
                                    "Predicted lag grid not aligned to expected lag grid; "
                                    "predicted columns omitted at "
                                    f"tolerance={BENCHMARK_REGULAR_GRID_TOLERANCE}"
                                ),
                            },
                            {
                                "source_archive_path": e_predicted_path,
                                "reason": (
                                    "Predicted lag grid not aligned to expected lag grid; "
                                    "predicted columns omitted at "
                                    f"tolerance={BENCHMARK_REGULAR_GRID_TOLERANCE}"
                                ),
                            },
                        ]
                    )
                elif not predicted_nonnegative:
                    skipped_selected_sources.append(
                        {
                            "source_archive_path": e_predicted_path,
                            "reason": (
                                "Predicted weights contain negatives; "
                                "E_predicted_normalized omitted while E_predicted_raw is retained"
                            ),
                        }
                    )
            generated_files.append(
                {
                    "output_path": str(kernel_out.relative_to(out_dir.parent.parent.parent)),
                    "kind": "kernel_reference",
                    "case": case,
                    "source_archive_paths": [
                        t_expected_path,
                        e_expected_path,
                        t_predicted_path,
                        e_predicted_path,
                    ],
                    "sha256": _sha256(kernel_out),
                }
            )

        laminar_time_path = _array_path("laminar_flow", "time.npy")
        laminar_input_path = _array_path("laminar_flow", "c_conv_in.npy")
        laminar_target_path = _array_path("laminar_flow", "concentration.npy")

        signal_df = build_signal_frame(
            case="laminar_flow",
            source_archive_path=f"{CASE_PREFIX}/laminar_flow",
            time=_load_npy_from_zip(zip_file, laminar_time_path),
            input_signal=_load_npy_from_zip(zip_file, laminar_input_path),
            target_signal=_load_npy_from_zip(zip_file, laminar_target_path),
        )
        signal_out = out_dir / "hsa_000_laminar_flow_signals.parquet"
        signal_df.write_parquet(signal_out)
        generated_files.append(
            {
                "output_path": str(signal_out.relative_to(out_dir.parent.parent.parent)),
                "kind": "signals",
                "case": "laminar_flow",
                "source_archive_paths": [
                    laminar_time_path,
                    laminar_input_path,
                    laminar_target_path,
                ],
                "sha256": _sha256(signal_out),
            }
        )

    for case in ("adler", "cholette", "dispersion"):
        skipped_selected_sources.extend(
            [
                {
                    "source_archive_path": _array_path(case, "time.npy"),
                    "reason": "No unambiguous input signal mapping for this case",
                },
                {
                    "source_archive_path": _array_path(case, "concentration.npy"),
                    "reason": "No unambiguous input signal mapping for this case",
                },
            ]
        )

    generated_files = sorted(generated_files, key=lambda row: str(row["output_path"]))
    skipped_selected_sources = sorted(
        skipped_selected_sources, key=lambda row: row["source_archive_path"]
    )

    manifest = {
        "zenodo_record_url": ZENODO_RECORD_URL,
        "zenodo_doi": ZENODO_DOI,
        "source_archive": SOURCE_ARCHIVE_NAME,
        "license": LICENSE_NAME,
        "extraction_script": "rtdfeatures.benchmarks.extract_nrtd_benchmarks",
        "conversion_assumptions": {
            "benchmark_kernel_sum_tolerance": BENCHMARK_KERNEL_SUM_TOLERANCE,
            "benchmark_regular_grid_tolerance": BENCHMARK_REGULAR_GRID_TOLERANCE,
            "kernel_weights_normalized_in_derived_columns_only": True,
            "direct_zip_reads": True,
        },
        "assumption_events": assumption_events,
        "generated_files": generated_files,
        "skipped_selected_source_files": skipped_selected_sources,
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_notice(out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("test_data/nRTD-v1.0.0.zip"),
        help="Path to test_data/nRTD-v1.0.0.zip",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("test_data/benchmarks/nrtd"),
        help="Output directory for extracted benchmark fixtures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_benchmarks(archive_path=args.archive, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

