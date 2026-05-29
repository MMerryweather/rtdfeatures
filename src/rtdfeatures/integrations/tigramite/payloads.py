"""Mock-compatible payload parsing for optional Tigramite integration."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence


class TigramitePayloadError(ValueError):
    """Raised when an adapter payload is malformed."""


def parse_mock_graph_payload(payload: object) -> list[list[list[str]]]:
    """Return a normalized graph cube from a plain mapping payload."""
    graph = _extract_required_mapping_value(payload=payload, key="graph")
    return _parse_string_cube(value=graph, key="graph")


def parse_mock_value_matrix(payload: object) -> list[list[list[float]]]:
    """Return a normalized value cube from a plain mapping payload."""
    val_matrix = _extract_required_mapping_value(payload=payload, key="val_matrix")
    return _parse_numeric_cube(value=val_matrix, key="val_matrix")


def parse_mock_pvalue_matrix(payload: object) -> list[list[list[float]]]:
    """Return a normalized p-value cube from a plain mapping payload."""
    p_matrix = _extract_required_mapping_value(payload=payload, key="p_matrix")
    return _parse_numeric_cube(value=p_matrix, key="p_matrix")


def _extract_required_mapping_value(*, payload: object, key: str) -> object:
    if not isinstance(payload, Mapping):
        raise TigramitePayloadError(
            f"Malformed payload: expected mapping with key '{key}', got {type(payload).__name__}."
        )

    if key not in payload:
        raise TigramitePayloadError(f"Malformed payload: missing required key '{key}'.")

    return payload[key]


def _parse_string_cube(*, value: object, key: str) -> list[list[list[str]]]:
    if _is_2d_matrix(value):
        return _convert_2d_string_matrix_to_cube(value=value, key=key)
    if not _is_3d_cube(value):
        raise TigramitePayloadError(
            f"Malformed payload: '{key}' must be a 2D matrix or 3D cube."
        )

    layers = _as_rows(value=value, key=key)
    parsed: list[list[list[str]]] = []
    for layer_index, rows in enumerate(layers):
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TigramitePayloadError(
                f"Malformed payload: '{key}[{layer_index}]' must be a row sequence."
            )
        parsed_rows: list[list[str]] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TigramitePayloadError(
                    "Malformed payload: "
                    f"'{key}[{layer_index}][{row_index}]' must be a lag sequence."
                )
            parsed_row: list[str] = []
            for col_index, cell in enumerate(row):
                if not isinstance(cell, str):
                    raise TigramitePayloadError(
                        "Malformed payload: "
                        f"'{key}[{layer_index}][{row_index}][{col_index}]' must be a string."
                    )
                parsed_row.append(cell)
            parsed_rows.append(parsed_row)
        parsed.append(parsed_rows)
    return parsed


def _parse_numeric_cube(*, value: object, key: str) -> list[list[list[float]]]:
    if _is_2d_matrix(value):
        return _convert_2d_numeric_matrix_to_cube(value=value, key=key)
    if not _is_3d_cube(value):
        raise TigramitePayloadError(
            f"Malformed payload: '{key}' must be a 2D matrix or 3D cube."
        )

    layers = _as_rows(value=value, key=key)
    parsed: list[list[list[float]]] = []
    for layer_index, rows in enumerate(layers):
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TigramitePayloadError(
                f"Malformed payload: '{key}[{layer_index}]' must be a row sequence."
            )
        parsed_rows: list[list[float]] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TigramitePayloadError(
                    "Malformed payload: "
                    f"'{key}[{layer_index}][{row_index}]' must be a lag sequence."
                )
            parsed_row: list[float] = []
            for col_index, cell in enumerate(row):
                if not isinstance(cell, (int, float)):
                    raise TigramitePayloadError(
                        "Malformed payload: "
                        f"'{key}[{layer_index}][{row_index}][{col_index}]' must be numeric."
                    )
                parsed_row.append(float(cell))
            parsed_rows.append(parsed_row)
        parsed.append(parsed_rows)
    return parsed


def _convert_2d_string_matrix_to_cube(*, value: object, key: str) -> list[list[list[str]]]:
    warnings.warn(
        "2D Tigramite graph payload was provided without a lag axis; values were mapped to "
        "tau=0 (contemporaneous-only), which can drop lag evidence. "
        "Provide a 3D graph payload shaped graph[target][source][tau] for lag-safe extraction.",
        UserWarning,
        stacklevel=3,
    )
    rows = _as_rows(value=value, key=key)
    parsed: list[list[list[str]]] = []
    for row_index, row in enumerate(rows):
        parsed_row: list[list[str]] = []
        for col_index, cell in enumerate(row):
            if not isinstance(cell, str):
                raise TigramitePayloadError(
                    f"Malformed payload: '{key}[{row_index}][{col_index}]' must be a string."
                )
            parsed_row.append([cell])
        parsed.append(parsed_row)
    return parsed


def _convert_2d_numeric_matrix_to_cube(*, value: object, key: str) -> list[list[list[float]]]:
    rows = _as_rows(value=value, key=key)
    parsed: list[list[list[float]]] = []
    for row_index, row in enumerate(rows):
        parsed_row: list[list[float]] = []
        for col_index, cell in enumerate(row):
            if not isinstance(cell, (int, float)):
                raise TigramitePayloadError(
                    f"Malformed payload: '{key}[{row_index}][{col_index}]' must be numeric."
                )
            parsed_row.append([float(cell)])
        parsed.append(parsed_row)
    return parsed


def _as_rows(*, value: object, key: str) -> Sequence[Sequence[object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TigramitePayloadError(f"Malformed payload: '{key}' must be a sequence of rows.")

    rows: list[Sequence[object]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TigramitePayloadError(
                f"Malformed payload: '{key}[{row_index}]' must be a row sequence."
            )
        rows.append(row)

    return rows


def _is_2d_matrix(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return False
        for cell in row:
            if isinstance(cell, Sequence) and not isinstance(cell, (str, bytes)):
                return False
    return True


def _is_3d_cube(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    for rows in value:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            return False
        for row in rows:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                return False
    return True
