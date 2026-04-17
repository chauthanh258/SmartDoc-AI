from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from uuid import uuid4
from typing import Any, MutableMapping

import streamlit as st

from src.utils.helpers import ensure_dict, to_optional_int, to_snippet, utc_now_iso


CHAT_HISTORY_KEY = "messages"
REGISTERED_DOCUMENTS_KEY = "registered_documents"
CONVERSATIONS_KEY = "conversations"
ACTIVE_CONVERSATION_KEY = "active_conversation_id"


def _resolve_state(
    session_state: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Resolve Streamlit session state, allowing injection for tests."""
    if session_state is not None:
        return session_state
    return st.session_state


def _new_conversation_id() -> str:
    """Generate a compact unique identifier for each conversation."""
    return f"conv-{uuid4().hex[:8]}"


def _generate_conversation_name(question: str) -> str:
    """Create a readable title from the first user question."""
    normalized = " ".join(question.strip().split())
    if not normalized:
        timestamp = utc_now_iso().replace("T", " ")[:19]
        return f"Conversation {timestamp}"

    if len(normalized) > 60:
        return normalized[:57].rstrip() + "..."

    return normalized


def _sync_legacy_messages(state: MutableMapping[str, Any]) -> None:
    """Keep legacy messages key mirrored to active conversation history."""
    active_id = state.get(ACTIVE_CONVERSATION_KEY)
    conversations = state.get(CONVERSATIONS_KEY, {})

    if not active_id or active_id not in conversations:
        state[CHAT_HISTORY_KEY] = []
        return

    active_turns = conversations[active_id].get("turns", [])
    state[CHAT_HISTORY_KEY] = list(active_turns)


def init_chat_history(
    session_state: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Initialize chat containers that persist for the active Streamlit session."""
    state = _resolve_state(session_state)

    if CHAT_HISTORY_KEY not in state:
        state[CHAT_HISTORY_KEY] = []

    if REGISTERED_DOCUMENTS_KEY not in state:
        state[REGISTERED_DOCUMENTS_KEY] = {}

    if CONVERSATIONS_KEY not in state:
        state[CONVERSATIONS_KEY] = {}

    if ACTIVE_CONVERSATION_KEY not in state:
        state[ACTIVE_CONVERSATION_KEY] = None

    active_id = state.get(ACTIVE_CONVERSATION_KEY)
    if active_id and active_id not in state[CONVERSATIONS_KEY]:
        state[ACTIVE_CONVERSATION_KEY] = None

    if state[ACTIVE_CONVERSATION_KEY] is None and state[CONVERSATIONS_KEY]:
        state[ACTIVE_CONVERSATION_KEY] = max(
            state[CONVERSATIONS_KEY],
            key=lambda conv_id: state[CONVERSATIONS_KEY][conv_id].get("updated_at", ""),
        )

    _sync_legacy_messages(state)

    return state


def create_conversation(
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    session_state: MutableMapping[str, Any] | None = None,
) -> str:
    """Create a new conversation and set it as active."""
    state = init_chat_history(session_state)
    conversation_id = _new_conversation_id()
    timestamp = utc_now_iso()
    conversation_name = (name or "").strip() or f"Conversation {timestamp.replace('T', ' ')[:19]}"

    state[CONVERSATIONS_KEY][conversation_id] = {
        "id": conversation_id,
        "name": conversation_name,
        "created_at": timestamp,
        "updated_at": timestamp,
        "turns": [],
        "metadata": {
            "auto_generated_name": name is None,
            **ensure_dict(metadata),
        },
    }
    state[ACTIVE_CONVERSATION_KEY] = conversation_id
    _sync_legacy_messages(state)
    return conversation_id


def list_conversations(
    session_state: MutableMapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return conversations sorted by most recently updated."""
    state = init_chat_history(session_state)
    conversations = []

    for conversation in state[CONVERSATIONS_KEY].values():
        turns = conversation.get("turns", [])
        conversations.append(
            {
                "id": conversation.get("id"),
                "name": conversation.get("name", "Untitled"),
                "created_at": conversation.get("created_at"),
                "updated_at": conversation.get("updated_at"),
                "turn_count": len(turns),
            }
        )

    conversations.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return conversations


def get_active_conversation_id(
    session_state: MutableMapping[str, Any] | None = None,
) -> str | None:
    """Return the active conversation identifier."""
    state = init_chat_history(session_state)
    return state.get(ACTIVE_CONVERSATION_KEY)


def get_active_conversation(
    session_state: MutableMapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a copy of the active conversation record."""
    state = init_chat_history(session_state)
    active_id = state.get(ACTIVE_CONVERSATION_KEY)

    if not active_id:
        return None

    conversation = state[CONVERSATIONS_KEY].get(active_id)
    if not conversation:
        return None

    payload = deepcopy(conversation)
    payload["turn_count"] = len(payload.get("turns", []))
    return payload


def set_active_conversation(
    conversation_id: str,
    session_state: MutableMapping[str, Any] | None = None,
) -> bool:
    """Switch active conversation by identifier."""
    state = init_chat_history(session_state)

    if conversation_id not in state[CONVERSATIONS_KEY]:
        return False

    state[ACTIVE_CONVERSATION_KEY] = conversation_id
    _sync_legacy_messages(state)
    return True


def rename_conversation(
    conversation_id: str,
    new_name: str,
    session_state: MutableMapping[str, Any] | None = None,
) -> bool:
    """Rename one conversation if it exists and new name is valid."""
    state = init_chat_history(session_state)
    conversation = state[CONVERSATIONS_KEY].get(conversation_id)
    cleaned_name = " ".join(new_name.strip().split())

    if not conversation or not cleaned_name:
        return False

    conversation["name"] = cleaned_name
    conversation["updated_at"] = utc_now_iso()
    conversation.setdefault("metadata", {})["auto_generated_name"] = False
    _sync_legacy_messages(state)
    return True


def delete_conversation(
    conversation_id: str,
    session_state: MutableMapping[str, Any] | None = None,
) -> bool:
    """Delete a conversation and safely update active selection."""
    state = init_chat_history(session_state)

    if conversation_id not in state[CONVERSATIONS_KEY]:
        return False

    del state[CONVERSATIONS_KEY][conversation_id]

    if not state[CONVERSATIONS_KEY]:
        state[ACTIVE_CONVERSATION_KEY] = None
    elif state.get(ACTIVE_CONVERSATION_KEY) == conversation_id:
        state[ACTIVE_CONVERSATION_KEY] = max(
            state[CONVERSATIONS_KEY],
            key=lambda conv_id: state[CONVERSATIONS_KEY][conv_id].get("updated_at", ""),
        )

    _sync_legacy_messages(state)
    return True


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
        document_name = (
            source.get("document_name")
            or source.get("file_name")
            or source.get("filename")
            or source.get("file")
            or metadata.get("document_name")
            or metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("source")
        )
        page = source.get("page")
        if page is None:
            page = metadata.get("page_number", metadata.get("page"))
        chunk_id = source.get("chunk_id")
        if chunk_id is None:
            chunk_id = source.get(
                "chunk_index",
                metadata.get("chunk_id", metadata.get("chunk_index")),
            )
        score = source.get("score", metadata.get("score"))
    elif hasattr(source, "metadata") and hasattr(source, "page_content"):
        metadata = ensure_dict(getattr(source, "metadata", {}))
        snippet_raw = getattr(source, "page_content", "")
        source_id = metadata.get("source_id")
        document_id = metadata.get("document_id")
        document_name = (
            metadata.get("document_name")
            or metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("source")
        )
        page = metadata.get("page_number", metadata.get("page"))
        chunk_id = metadata.get("chunk_id", metadata.get("chunk_index"))
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

    if final_file_name and final_file_name.strip().lower() in {
        "none",
        "null",
        "nan",
        "na",
        "n/a",
    }:
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
    conversation_id: str | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    session_state: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one Q/A entry with timestamp and citation-ready sources."""
    state = init_chat_history(session_state)

    active_conversation_id = conversation_id or state.get(ACTIVE_CONVERSATION_KEY)
    if not active_conversation_id or active_conversation_id not in state[CONVERSATIONS_KEY]:
        active_conversation_id = create_conversation(session_state=state)

    conversation = state[CONVERSATIONS_KEY][active_conversation_id]
    conversation_metadata = conversation.setdefault("metadata", {})

    if not conversation.get("turns") and conversation_metadata.get("auto_generated_name", True):
        conversation["name"] = _generate_conversation_name(question)
        conversation_metadata["auto_generated_name"] = False

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

    conversation.setdefault("turns", []).append(entry)
    conversation["updated_at"] = utc_now_iso()
    conversation_metadata["document_ids"] = sorted(
        set(conversation_metadata.get("document_ids", [])) | set(related_document_ids)
    )

    state[ACTIVE_CONVERSATION_KEY] = active_conversation_id
    _sync_legacy_messages(state)
    return entry


def get_chat_history(
    conversation_id: str | None = None,
    session_state: MutableMapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Get turns from one conversation, defaulting to the active thread."""
    state = init_chat_history(session_state)
    selected_conversation_id = conversation_id or state.get(ACTIVE_CONVERSATION_KEY)

    if not selected_conversation_id:
        return []

    conversation = state[CONVERSATIONS_KEY].get(selected_conversation_id)
    if not conversation:
        return []

    return deepcopy(conversation.get("turns", []))


def clear_chat_history(
    session_state: MutableMapping[str, Any] | None = None,
    conversation_id: str | None = None,
    clear_documents: bool = False,
    clear_all_conversations: bool = True,
) -> None:
    """Clear active/all conversations and optionally registered document records."""
    state = init_chat_history(session_state)

    if clear_all_conversations:
        state[CONVERSATIONS_KEY] = {}
        state[ACTIVE_CONVERSATION_KEY] = None
        state[CHAT_HISTORY_KEY] = []
    else:
        selected_conversation_id = conversation_id or state.get(ACTIVE_CONVERSATION_KEY)
        conversation = state[CONVERSATIONS_KEY].get(selected_conversation_id)

        if conversation:
            conversation["turns"] = []
            conversation["updated_at"] = utc_now_iso()

        _sync_legacy_messages(state)

    if clear_documents:
        state[REGISTERED_DOCUMENTS_KEY] = {}
