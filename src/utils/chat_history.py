from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping

import streamlit as st

from src.utils.helpers import ensure_dict, to_optional_int, to_snippet, utc_now_iso


CHAT_HISTORY_KEY = "messages"
REGISTERED_DOCUMENTS_KEY = "registered_documents"


def _resolve_state(
    session_state: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Resolve Streamlit session state, allowing injection for tests."""
    if session_state is not None:
        return session_state
    return st.session_state


def init_chat_history(
    session_state: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Initialize chat containers that persist for the active Streamlit session."""
    state = _resolve_state(session_state)

    if CHAT_HISTORY_KEY not in state:
        state[CHAT_HISTORY_KEY] = []

    if REGISTERED_DOCUMENTS_KEY not in state:
        state[REGISTERED_DOCUMENTS_KEY] = {}

    return state


def set_chat_history(
    history_data: list[dict[str, Any]],
    session_state: MutableMapping[str, Any] | None = None,
) -> None:
    """
    Ghi đè dữ liệu lịch sử từ Database vào Session State.
    Hàm này giải quyết lỗi click vào sidebar nhưng không hiện tin nhắn.
    """
    state = _resolve_state(session_state)
    state[CHAT_HISTORY_KEY] = history_data


def register_document(
    document_id: str,
    document_name: str,
    metadata: dict[str, Any] | None = None,
    session_state: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Register uploaded documents to support future multi-document chat."""
    state = init_chat_history(session_state)

    record = {
        "document_id": document_id,
        "document_name": document_name,
        "registered_at": utc_now_iso(),
        "metadata": ensure_dict(metadata),
    }
    state[REGISTERED_DOCUMENTS_KEY][document_id] = record
    return record


def get_registered_documents(
    session_state: MutableMapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return all registered documents in the current session."""
    state = init_chat_history(session_state)
    return dict(state[REGISTERED_DOCUMENTS_KEY])


def _normalize_source_item(
    source: Any,
    index: int,
    default_document_id: str | None = None,
    default_document_name: str | None = None,
) -> dict[str, Any]:
    """Normalize dict/Document-like source objects into one citation schema."""
    metadata: dict[str, Any]
    snippet_raw: Any

    if isinstance(source, dict):
        metadata = ensure_dict(source.get("metadata"))
        snippet_raw = source.get("snippet") or source.get("content") or ""
        source_id = source.get("source_id")
        document_id = source.get("document_id") or metadata.get("document_id")
        document_name = source.get("document_name") or source.get("file_name")
        page = source.get("page", metadata.get("page"))
        chunk_id = source.get("chunk_id", metadata.get("chunk_id"))
        score = source.get("score", metadata.get("score"))
    elif hasattr(source, "metadata") and hasattr(source, "page_content"):
        metadata = ensure_dict(getattr(source, "metadata", {}))
        snippet_raw = getattr(source, "page_content", "")
        source_id = metadata.get("source_id")
        document_id = metadata.get("document_id")
        document_name = metadata.get("document_name") or metadata.get("source")
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id")
        score = metadata.get("score")
    else:
        metadata = {}
        snippet_raw = source
        source_id = None
        document_id = None
        document_name = None
        page = None
        chunk_id = None
        score = None

    final_document_id = document_id or default_document_id
    final_document_name = document_name or default_document_name

    if final_document_name:
        final_file_name = Path(str(final_document_name)).name
    else:
        final_file_name = None

    normalized_source_id = str(source_id or f"src-{index + 1}")
    normalized_page = to_optional_int(page)
    snippet = to_snippet(snippet_raw)

    citation_label = final_file_name or "Unknown"
    citation_suffix = ""
    if normalized_page is not None:
        citation_suffix = f":p{normalized_page}"

    return {
        "source_id": normalized_source_id,
        "document_id": final_document_id,
        "document_name": final_document_name,
        "file_name": final_file_name,
        "page": normalized_page,
        "chunk_id": chunk_id,
        "score": score,
        "snippet": snippet,
        "metadata": metadata,
        "citation": {
            "label": f"[{citation_label}{citation_suffix}]",
            "file_name": final_file_name,
            "page": normalized_page,
            "chunk_id": chunk_id,
        },
    }


def normalize_sources(
    sources: list[Any] | None,
    default_document_id: str | None = None,
    default_document_name: str | None = None,
) -> list[dict[str, Any]]:
    """Convert source candidates into citation-ready records."""
    normalized: list[dict[str, Any]] = []

    for index, source in enumerate(sources or []):
        normalized.append(
            _normalize_source_item(
                source=source,
                index=index,
                default_document_id=default_document_id,
                default_document_name=default_document_name,
            )
        )

    return normalized


def add_chat_turn(
    question: str,
    answer: str,
    sources: list[Any] | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    session_state: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one Q/A entry with timestamp and citation-ready sources."""
    state = init_chat_history(session_state)

    normalized_sources = normalize_sources(
        sources=sources,
        default_document_id=document_id,
        default_document_name=document_name,
    )

    related_document_ids = sorted(
        {
            source["document_id"]
            for source in normalized_sources
            if source.get("document_id")
        }
    )

    entry = {
        "timestamp": utc_now_iso(),
        "question": question.strip(),
        "answer": answer.strip(),
        "sources": normalized_sources,
        "document_ids": related_document_ids,
        "metadata": ensure_dict(metadata),
    }

    state[CHAT_HISTORY_KEY].append(entry)
    return entry


def get_chat_history(
    session_state: MutableMapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Get a copy of the current session chat history."""
    state = init_chat_history(session_state)
    return list(state[CHAT_HISTORY_KEY])


def clear_chat_history(
    session_state: MutableMapping[str, Any] | None = None,
    clear_documents: bool = False,
) -> None:
    """Clear chat turns and optionally registered document records."""
    state = init_chat_history(session_state)
    state[CHAT_HISTORY_KEY] = []

    if clear_documents:
        state[REGISTERED_DOCUMENTS_KEY] = {}
