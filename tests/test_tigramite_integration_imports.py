"""Phase 2 tests for optional Tigramite integration module boundaries."""

from __future__ import annotations

import importlib
import sys

import pytest


def _clear_modules(prefix: str) -> None:
    module_names = [name for name in sys.modules if name == prefix or name.startswith(f"{prefix}.")]
    for name in module_names:
        del sys.modules[name]


def test_core_import_does_not_require_tigramite() -> None:
    _clear_modules("rtdfeatures")
    _clear_modules("tigramite")

    importlib.import_module("rtdfeatures")

    assert "tigramite" not in sys.modules


def test_integration_module_import_does_not_require_tigramite() -> None:
    _clear_modules("rtdfeatures")
    _clear_modules("tigramite")

    module = importlib.import_module("rtdfeatures.integrations.tigramite")

    assert module is not None
    assert "tigramite" not in sys.modules


def test_adapter_accepts_mock_payloads() -> None:
    from rtdfeatures.integrations.tigramite import (
        parse_mock_graph_payload,
        parse_mock_pvalue_matrix,
        parse_mock_value_matrix,
    )

    graph_payload = {"graph": [["", "->"]]}
    value_payload = {"val_matrix": [[0.0, 0.5], [1, 2.5]]}
    pvalue_payload = {"p_matrix": [[0.2, 0.05]]}

    assert parse_mock_graph_payload(graph_payload) == [[[""], ["->"]]]
    assert parse_mock_value_matrix(value_payload) == [[[0.0], [0.5]], [[1.0], [2.5]]]
    assert parse_mock_pvalue_matrix(pvalue_payload) == [[[0.2], [0.05]]]


def test_adapter_fails_clearly_on_malformed_payloads() -> None:
    from rtdfeatures.integrations.tigramite import TigramitePayloadError, parse_mock_value_matrix

    with pytest.raises(TigramitePayloadError, match="missing required key 'val_matrix'"):
        parse_mock_value_matrix({"value": [[1.0]]})

    with pytest.raises(TigramitePayloadError, match="must be numeric"):
        parse_mock_value_matrix({"val_matrix": [["bad"]]})
