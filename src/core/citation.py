from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_core.documents import Document


_INVALID_TEXT_VALUES = {"", "none", "null", "nan", "na", "n/a"}


def _normalize_text_or_none(value: Any) -> str | None:
    """Convert value to a meaningful non-empty text string when possible."""
    if value is None:
        return None

    text = str(value).strip()
    if text.lower() in _INVALID_TEXT_VALUES:
        return None
    return text


def _resolve_file_name(metadata: Mapping[str, Any]) -> str | None:
    """Resolve filename from multiple metadata conventions used in the project."""
    raw_name = (
        _normalize_text_or_none(metadata.get("file_name"))
        or _normalize_text_or_none(metadata.get("filename"))
        or _normalize_text_or_none(metadata.get("file"))
        or _normalize_text_or_none(metadata.get("document_name"))
        or _normalize_text_or_none(metadata.get("source"))
    )
    if not raw_name:
        return None
    return Path(raw_name).name


def _to_int_or_none(value: Any) -> int | None:
    """Best-effort conversion of metadata values to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_citation_data(document: Document) -> dict[str, Any]:
    """Extract normalized citation fields from a LangChain Document.

    Returned keys are stable across PDF and DOCX inputs:
    - file_name: source filename
    - page_number: PDF page number when available
    - section: DOCX section label when available
    - chunk_index: index assigned by text splitter
    """
    metadata = dict(document.metadata or {})

    file_name = _resolve_file_name(metadata)

    page_number = _to_int_or_none(metadata.get("page_number"))
    if page_number is None:
        page_number = _to_int_or_none(metadata.get("page"))

    section = metadata.get("section")
    chunk_index = _to_int_or_none(metadata.get("chunk_index"))

    return {
        "file_name": file_name,
        "page_number": page_number,
        "section": section,
        "chunk_index": chunk_index,
    }


def format_citation(metadata: Mapping[str, Any]) -> str:
    """Format a citation string.

    Preferred output for PDF follows the required style:
    [Trang 5 - sample.pdf]

    For DOCX sections (no page number), the format becomes:
    [Muc Gioi thieu - sample.docx]
    """
    file_name = _resolve_file_name(metadata) or "unknown_source"
    page_number = _to_int_or_none(metadata.get("page_number"))
    section = metadata.get("section")

    if page_number is not None:
        return f"[Trang {page_number} - {file_name}]"

    if section:
        return f"[Muc {section} - {file_name}]"

    return f"[{file_name}]"


def format_document_citation(document: Document) -> str:
    """Build a citation string directly from a Document instance."""
    citation_data = extract_citation_data(document)
    return format_citation(citation_data)


def extract_citations(
    documents: Sequence[Document],
    unique: bool = True,
) -> list[str]:
    """Extract citations from a list of documents.

    The default unique=True keeps first-seen order and removes duplicates.
    """
    citations: list[str] = []
    seen: set[str] = set()

    for doc in documents:
        citation = format_document_citation(doc)
        if unique:
            if citation in seen:
                continue
            seen.add(citation)
        citations.append(citation)

    return citations


def format_citation_block(documents: Sequence[Document]) -> str:
    """Return citations as a readable multi-line block for UI/LLM responses."""
    citations = extract_citations(documents, unique=True)
    if not citations:
        return ""
    return "\n".join(f"- {citation}" for citation in citations)
