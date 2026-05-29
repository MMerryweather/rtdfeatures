"""Internal validation helpers for diagnostic and candidate contracts."""

from __future__ import annotations

import json
from typing import Any


def _require_non_empty_name(field_name: str, value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return cleaned


def _validate_json_serializable(field_name: str, value: Any) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be JSON-serializable and must not include live objects."
        ) from exc
