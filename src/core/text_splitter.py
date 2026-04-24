from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Allowed values requested in Phase 2 so users can tune chunking safely.
ALLOWED_CHUNK_SIZES = (500, 600, 1000, 1500, 2000)
ALLOWED_CHUNK_OVERLAPS = (0, 50, 100, 200, 300, 400)


# Keep defaults aligned with current project behavior.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk parameters against supported user-selectable values."""
    if chunk_size not in ALLOWED_CHUNK_SIZES:
        raise ValueError(
            f"Invalid chunk_size={chunk_size}. "
            f"Supported values: {list(ALLOWED_CHUNK_SIZES)}"
        )

    if chunk_overlap not in ALLOWED_CHUNK_OVERLAPS:
        raise ValueError(
            f"Invalid chunk_overlap={chunk_overlap}. "
            f"Supported values: {list(ALLOWED_CHUNK_OVERLAPS)}"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size to avoid empty chunks."
        )


def _to_int_or_none(value: Any) -> int | None:
    """Convert a metadata value to int when possible; return None otherwise."""
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _enrich_chunk_metadata(chunk: Document, chunk_index: int) -> Document:
    """Ensure every chunk has consistent metadata used by retrieval and citation."""
    metadata = dict(chunk.metadata or {})

    source = metadata.get("source")
    if source and not metadata.get("file_name"):
        metadata["file_name"] = Path(str(source)).name

    # Normalize page metadata under one key used across the app.
    page_number = _to_int_or_none(metadata.get("page_number"))
    if page_number is None:
        page_number = _to_int_or_none(metadata.get("page"))
    if page_number is not None:
        metadata["page_number"] = page_number

    metadata["chunk_index"] = chunk_index
    chunk.metadata = metadata

    # Contextual Chunking: Attach document identity directly to the text
    # so that the vector embedding includes the doc context.
    file_name = metadata.get("file_name", "Unknown Document")
    location = ""
    if page_number is not None:
        location = f" - Trang {page_number}"
    elif metadata.get("section"):
        location = f" - Mục {metadata.get('section')}"
    
    context_header = f"[{file_name}{location}]"
    if not chunk.page_content.startswith(context_header):
        chunk.page_content = f"{context_header}\n{chunk.page_content}"

    return chunk


def get_chunking_options() -> dict[str, tuple[int, ...]]:
    """Expose supported chunk options for UI or API dropdowns."""
    return {
        "chunk_size": ALLOWED_CHUNK_SIZES,
        "chunk_overlap": ALLOWED_CHUNK_OVERLAPS,
    }


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split LangChain Document objects and attach chunk-level metadata.

    Every returned chunk keeps existing metadata and is enriched with:
    - chunk_index: global index in the split output
    - file_name: extracted from source path when available
    - page_number: normalized page field for PDF citation support
    """
    _validate_chunk_params(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
    )

    raw_chunks = text_splitter.split_documents(documents)
    return [_enrich_chunk_metadata(chunk, idx) for idx, chunk in enumerate(raw_chunks)]
