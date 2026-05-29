#!/usr/bin/env python3
"""Inspect nRTD archive metadata used by benchmark planning."""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ZENODO_RECORD_URL = "https://zenodo.org/records/15609214"
ZENODO_DOI = "10.5281/zenodo.15609214"
SOURCE_ARCHIVE_NAME = "ChemRxnEngLab/nRTD-v1.0.0.zip"
LICENSE = "MIT"

KNOWN_HSA000_CASES = ("adler", "cholette", "dispersion", "laminar_flow")


@dataclass(frozen=True)
class ArraySummary:
    source_archive_path: str
    fixture_kind: str
    case: str
    file_type: str
    shape: list[int]
    dtype: str
    min_value: float | None
    max_value: float | None
    start_time: float | None
    end_time: float | None
    grid_is_regular: bool | None
    npz_keys: list[str] | None
    npz_members: list[dict[str, Any]] | None


def _regular_grid(values: np.ndarray, atol: float = 1e-12) -> bool:
    if values.size < 3:
        return True
    diffs = np.diff(values)
    return bool(np.allclose(diffs, diffs[0], atol=atol, rtol=0.0))


def _summarize_array(path: str, data: np.ndarray) -> ArraySummary:
    arr = np.asarray(data)
    arr_1d = arr.reshape(-1) if arr.size else arr

    file_name = Path(path).name
    case = "unknown"
    parts = Path(path).parts
    if "data" in parts:
        data_idx = parts.index("data")
        if data_idx + 1 < len(parts):
            case = parts[data_idx + 1]

    fixture_kind = "other"
    if any(f"/data/{name}/" in path for name in KNOWN_HSA000_CASES):
        fixture_kind = "analytical_reference"
    elif path.endswith("_E_nRTD.npz"):
        fixture_kind = "learned_experimental_reference"

    min_value = float(np.nanmin(arr_1d)) if arr_1d.size else None
    max_value = float(np.nanmax(arr_1d)) if arr_1d.size else None

    is_time_like = file_name in {"time.npy", "t_E_expected.npy", "t_E_predicted.npy"}
    start_time = float(arr_1d[0]) if is_time_like and arr_1d.size else None
    end_time = float(arr_1d[-1]) if is_time_like and arr_1d.size else None
    grid_is_regular = _regular_grid(arr_1d) if is_time_like and arr_1d.size else None

    return ArraySummary(
        source_archive_path=path,
        fixture_kind=fixture_kind,
        case=case,
        file_type=Path(path).suffix,
        shape=list(arr.shape),
        dtype=str(arr.dtype),
        min_value=min_value,
        max_value=max_value,
        start_time=start_time,
        end_time=end_time,
        grid_is_regular=grid_is_regular,
        npz_keys=None,
        npz_members=None,
    )


def _summarize_npz(path: str, npz: np.lib.npyio.NpzFile) -> ArraySummary:
    keys = sorted(npz.files)
    members: list[dict[str, Any]] = []
    for key in keys:
        arr = np.asarray(npz[key])
        arr_1d = arr.reshape(-1) if arr.size else arr
        min_value = float(np.nanmin(arr_1d)) if arr_1d.size else None
        max_value = float(np.nanmax(arr_1d)) if arr_1d.size else None
        members.append(
            {
                "key": key,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min_value": min_value,
                "max_value": max_value,
            }
        )

    summary = _summarize_array(path, np.array([], dtype=np.float64))
    return ArraySummary(
        source_archive_path=summary.source_archive_path,
        fixture_kind=summary.fixture_kind,
        case=summary.case,
        file_type=summary.file_type,
        shape=[],
        dtype="npz",
        min_value=None,
        max_value=None,
        start_time=None,
        end_time=None,
        grid_is_regular=None,
        npz_keys=keys,
        npz_members=members,
    )


def inspect_archive(archive_path: Path) -> dict[str, Any]:
    summaries: list[ArraySummary] = []

    with zipfile.ZipFile(archive_path) as zf:
        for name in sorted(zf.namelist()):
            if not (name.endswith(".npy") or name.endswith(".npz")):
                continue

            keep = any(f"/data/{case}/" in name for case in KNOWN_HSA000_CASES)
            keep = keep or name.endswith("_E_nRTD.npz")
            if not keep:
                continue

            with zf.open(name) as handle:
                if name.endswith(".npy"):
                    arr = np.load(handle, allow_pickle=False)
                    summaries.append(_summarize_array(name, arr))
                else:
                    with np.load(handle, allow_pickle=False) as npz:
                        summaries.append(_summarize_npz(name, npz))

    return {
        "zenodo_doi": ZENODO_DOI,
        "zenodo_record_url": ZENODO_RECORD_URL,
        "source_archive": SOURCE_ARCHIVE_NAME,
        "license": LICENSE,
        "archive_path": str(archive_path),
        "known_hsa000_cases": list(KNOWN_HSA000_CASES),
        "array_summaries": [s.__dict__ for s in summaries],
    }


def _render_markdown_table(metadata: dict[str, Any]) -> str:
    header = (
        "| fixture_kind | case | source_archive_path | shape | dtype | "
        "time_range | value_range | regular_grid |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for entry in metadata["array_summaries"]:
        time_range = (
            f"{entry['start_time']}..{entry['end_time']}"
            if entry["start_time"] is not None
            else "n/a"
        )
        value_range = f"{entry['min_value']}..{entry['max_value']}"
        regular = "n/a" if entry["grid_is_regular"] is None else str(entry["grid_is_regular"])
        rows.append(
            "| {fixture_kind} | {case} | {source_archive_path} | {shape} | {dtype} | "
            "{time_range} | {value_range} | {regular} |".format(
                fixture_kind=entry["fixture_kind"],
                case=entry["case"],
                source_archive_path=entry["source_archive_path"],
                shape=entry["shape"],
                dtype=entry["dtype"],
                time_range=time_range,
                value_range=value_range,
                regular=regular,
            )
        )
    return "\n".join([header, *rows])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        default="test_data/nRTD-v1.0.0.zip",
        type=Path,
        help="Path to local nRTD zip archive.",
    )
    parser.add_argument(
        "--metadata-out",
        default=Path("docs/benchmarks/wp0_archive_inspection.json"),
        type=Path,
    )
    parser.add_argument(
        "--table-out",
        default=Path("docs/benchmarks/wp0_nrtd_archive_inventory.md"),
        type=Path,
    )
    args = parser.parse_args()

    metadata = inspect_archive(args.archive)

    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.write_text(_render_markdown_table(metadata) + "\n", encoding="utf-8")

    print(f"Wrote {args.metadata_out}")
    print(f"Wrote {args.table_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
