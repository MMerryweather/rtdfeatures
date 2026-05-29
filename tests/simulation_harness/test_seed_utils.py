from __future__ import annotations

from tests.simulation_harness.seed import make_rng


def test_seed_determinism_same_seed_same_sequence() -> None:
    seq_a = make_rng(42).normal(size=5)
    seq_b = make_rng(42).normal(size=5)
    assert (seq_a == seq_b).all()
