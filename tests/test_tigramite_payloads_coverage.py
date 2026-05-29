from __future__ import annotations

import pytest


def test_parse_mock_pvalue_matrix_accepts_3d() -> None:
    from rtdfeatures.integrations.tigramite.payloads import parse_mock_pvalue_matrix
    result = parse_mock_pvalue_matrix({"p_matrix": [[[0.01, 0.02], [0.03, 0.04]]]})
    assert result == [[[0.01, 0.02], [0.03, 0.04]]]


def test_parse_mock_string_cube_3d() -> None:
    from rtdfeatures.integrations.tigramite.payloads import _parse_string_cube
    result = _parse_string_cube(
        value=[[[""], ["->"]]], key="graph",
    )
    assert result == [[[""], ["->"]]]


def test_parse_mock_value_matrix_3d() -> None:
    from rtdfeatures.integrations.tigramite.payloads import _parse_numeric_cube
    result = _parse_numeric_cube(
        value=[[[0.5, 0.3], [0.2, 0.1]]], key="val_matrix",
    )
    assert result == [[[0.5, 0.3], [0.2, 0.1]]]


def test_extract_required_mapping_value_not_mapping() -> None:
    from rtdfeatures.integrations.tigramite.payloads import (
        TigramitePayloadError,
        parse_mock_graph_payload,
    )
    with pytest.raises(TigramitePayloadError, match="Malformed payload"):
        parse_mock_graph_payload("not a mapping")


def test_parse_string_cube_not_2d_not_3d() -> None:
    from rtdfeatures.integrations.tigramite.payloads import (
        TigramitePayloadError,
        _parse_string_cube,
    )
    with pytest.raises(TigramitePayloadError, match="must be a 2D matrix or 3D cube"):
        _parse_string_cube(value=[1, 2, 3], key="graph")


def test_parse_numeric_cube_not_2d_not_3d() -> None:
    from rtdfeatures.integrations.tigramite.payloads import (
        TigramitePayloadError,
        _parse_numeric_cube,
    )
    with pytest.raises(TigramitePayloadError, match="must be a 2D matrix or 3D cube"):
        _parse_numeric_cube(value=[1, 2, 3], key="val_matrix")


def test_parse_string_cube_row_not_sequence() -> None:
    from rtdfeatures.integrations.tigramite.payloads import (
        TigramitePayloadError,
        _parse_string_cube,
    )
    with pytest.raises(TigramitePayloadError, match="must be a 2D matrix or 3D cube"):
        _parse_string_cube(value="not a sequence", key="graph")


def test_parse_numeric_cube_row_not_sequence() -> None:
    from rtdfeatures.integrations.tigramite.payloads import (
        TigramitePayloadError,
        _parse_numeric_cube,
    )
    with pytest.raises(TigramitePayloadError, match="must be a 2D matrix or 3D cube"):
        _parse_numeric_cube(value="not a sequence", key="val_matrix")


def test_parse_string_cube_inner_row_not_sequence() -> None:
    from rtdfeatures.integrations.tigramite.payloads import (
        TigramitePayloadError,
        _parse_string_cube,
    )
    with pytest.raises(TigramitePayloadError, match="must be a 2D matrix or 3D cube"):
        _parse_string_cube(value="bad", key="graph")


def test_parse_numeric_cube_inner_row_not_sequence() -> None:
    from rtdfeatures.integrations.tigramite.payloads import (
        TigramitePayloadError,
        _parse_numeric_cube,
    )
    with pytest.raises(TigramitePayloadError, match="must be a 2D matrix or 3D cube"):
        _parse_numeric_cube(value="bad", key="val_matrix")


def test_convert_2d_string_matrix_cell_not_string() -> None:
    from rtdfeatures.integrations.tigramite.payloads import (
        TigramitePayloadError,
        _convert_2d_string_matrix_to_cube,
    )
    with pytest.raises(TigramitePayloadError, match="must be a string"):
        _convert_2d_string_matrix_to_cube(value=[["a", 1]], key="graph")


def test_as_rows_not_sequence() -> None:
    from rtdfeatures.integrations.tigramite.payloads import TigramitePayloadError, _as_rows
    with pytest.raises(TigramitePayloadError, match="must be a sequence of rows"):
        _as_rows(value=42, key="test")


def test_as_rows_element_not_sequence() -> None:
    from rtdfeatures.integrations.tigramite.payloads import TigramitePayloadError, _as_rows
    with pytest.raises(TigramitePayloadError, match="must be a row sequence"):
        _as_rows(value=[1, 2, 3], key="test")


def test_is_2d_matrix_value_not_sequence() -> None:
    from rtdfeatures.integrations.tigramite.payloads import _is_2d_matrix
    assert not _is_2d_matrix(42)


def test_is_2d_matrix_row_not_sequence() -> None:
    from rtdfeatures.integrations.tigramite.payloads import _is_2d_matrix
    assert not _is_2d_matrix([1, 2, 3])


def test_is_3d_cube_value_not_sequence() -> None:
    from rtdfeatures.integrations.tigramite.payloads import _is_3d_cube
    assert not _is_3d_cube(42)


def test_is_3d_cube_row_not_sequence() -> None:
    from rtdfeatures.integrations.tigramite.payloads import _is_3d_cube
    assert not _is_3d_cube([1, 2, 3])


def test_is_3d_cube_inner_not_sequence() -> None:
    from rtdfeatures.integrations.tigramite.payloads import _is_3d_cube
    assert not _is_3d_cube([["a", "b"]])
