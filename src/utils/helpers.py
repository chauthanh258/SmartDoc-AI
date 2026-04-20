from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
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


def upsert_env_var(env_path: str, key: str, value: str) -> None:
    """Create or update a single key in a .env file."""
    path = Path(env_path)
    if not path.exists():
        path.write_text("", encoding="utf-8")

    lines = path.read_text(encoding="utf-8").splitlines()
    new_line = f"{key}={value}"
    replaced = False
    updated_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            updated_lines.append(line)
            continue

        existing_key = stripped.split("=", 1)[0].strip()
        if existing_key == key:
            updated_lines.append(new_line)
            replaced = True
        else:
            updated_lines.append(line)

    if not replaced:
        if updated_lines and updated_lines[-1].strip() != "":
            updated_lines.append("")
        updated_lines.append(new_line)

    path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def persist_ollama_runtime_settings(
    env_path: str,
    mode: str,
    local_base_url: str,
    local_model: str,
    cloud_base_url: str,
    cloud_model: str,
    api_key: str,
) -> None:
    """Persist Ollama runtime settings to .env so they survive app restart."""
    upsert_env_var(env_path, "OLLAMA_MODE", mode)
    upsert_env_var(env_path, "OLLAMA_LOCAL_BASE_URL", local_base_url)
    upsert_env_var(env_path, "OLLAMA_LOCAL_MODEL", local_model)
    upsert_env_var(env_path, "OLLAMA_CLOUD_BASE_URL", cloud_base_url)
    upsert_env_var(env_path, "OLLAMA_CLOUD_MODEL", cloud_model)
    upsert_env_var(env_path, "OLLAMA_API_KEY", api_key)

    # Backward-compatible defaults used by old code paths.
    if mode == "cloud":
        upsert_env_var(env_path, "OLLAMA_BASE_URL", cloud_base_url)
        upsert_env_var(env_path, "LLM_MODEL", cloud_model)
    else:
        upsert_env_var(env_path, "OLLAMA_BASE_URL", local_base_url)
        upsert_env_var(env_path, "LLM_MODEL", local_model)
