"""Release-gate checks for benchmark and harness boundaries."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "test_data" / "benchmarks" / "nrtd"
MAX_TRACKED_FIXTURE_BYTES = 5 * 1024 * 1024
REQUIRED_TRACKED_FIXTURE_FILES = (
    "NOTICE.md",
    "manifest.json",
    "hsa_000_adler_kernel_reference.parquet",
    "hsa_000_cholette_kernel_reference.parquet",
    "hsa_000_dispersion_kernel_reference.parquet",
    "hsa_000_laminar_flow_kernel_reference.parquet",
    "hsa_000_laminar_flow_signals.parquet",
)


def test_benchmark_fixture_directory_is_under_size_threshold() -> None:
    assert BENCHMARK_DIR.exists(), f"Tracked benchmark directory missing: {BENCHMARK_DIR}"

    total_bytes = sum(path.stat().st_size for path in BENCHMARK_DIR.rglob("*") if path.is_file())
    assert total_bytes < MAX_TRACKED_FIXTURE_BYTES


def test_benchmark_manifest_and_notice_are_present_with_required_metadata() -> None:
    assert BENCHMARK_DIR.exists(), f"Tracked benchmark directory missing: {BENCHMARK_DIR}"

    manifest_path = BENCHMARK_DIR / "manifest.json"
    notice_path = BENCHMARK_DIR / "NOTICE.md"

    assert manifest_path.exists()
    assert notice_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    notice_text = notice_path.read_text(encoding="utf-8")

    assert manifest.get("source_archive") == "ChemRxnEngLab/nRTD-v1.0.0.zip"
    assert manifest.get("license") == "MIT"
    assert manifest.get("zenodo_doi") == "10.5281/zenodo.15609214"
    assert manifest.get("zenodo_record_url") == "https://zenodo.org/records/15609214"

    assert "ChemRxnEngLab/nRTD-v1.0.0.zip" in notice_text
    assert "MIT" in notice_text
    assert "10.5281/zenodo.15609214" in notice_text


def test_required_tracked_fixture_files_exist_and_manifest_outputs_are_present() -> None:
    assert BENCHMARK_DIR.exists(), f"Tracked benchmark directory missing: {BENCHMARK_DIR}"

    for relpath in REQUIRED_TRACKED_FIXTURE_FILES:
        fixture_path = BENCHMARK_DIR / relpath
        assert fixture_path.exists(), f"Tracked fixture file missing: {fixture_path}"

    manifest = json.loads((BENCHMARK_DIR / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest.get("generated_files", []):
        output_path = entry.get("output_path")
        assert isinstance(output_path, str) and output_path, (
            "manifest generated_files.output_path missing"
        )
        absolute = REPO_ROOT / output_path
        assert absolute.exists(), f"manifest output_path missing from repository: {absolute}"
