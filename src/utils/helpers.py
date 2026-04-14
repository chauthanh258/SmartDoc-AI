from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return a timezone-aware ISO timestamp for stable session records."""
    return datetime.now(timezone.utc).isoformat()


def ensure_dict(value: Any) -> dict[str, Any]:
    """Return a dictionary copy or an empty dict for unsupported values."""
    if isinstance(value, dict):
        return dict(value)
    return {}


def to_snippet(text: Any, max_length: int = 220) -> str:
    """Normalize text into a short citation-friendly snippet."""
    raw_text = str(text or "").strip().replace("\n", " ")
    compact = " ".join(raw_text.split())

    if len(compact) <= max_length:
        return compact

    return compact[: max_length - 3].rstrip() + "..."


def to_optional_int(value: Any) -> int | None:
    """Safely convert a value to int, or return None when unavailable."""
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None
