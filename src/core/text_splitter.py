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

    # Normalize page metadata
    page_number = _to_int_or_none(metadata.get("page_number"))
    if page_number is None:
        page_number = _to_int_or_none(metadata.get("page"))
    if page_number is not None:
        metadata["page_number"] = page_number

    metadata["chunk_index"] = chunk_index
    chunk.metadata = metadata

    # Contextual Chunking: Attach document identity directly to the text
    file_name = metadata.get("file_name", "Unknown Document")
    location = ""
    if page_number is not None:
        location = f" - Trang {page_number}"
    elif metadata.get("section"):
        location = f" - Mục {metadata.get('section')}"
    
    category = metadata.get("category", "")
    type_info = f" [{category}]" if category else ""
    
    context_header = f"[{file_name}{location}]{type_info}"
    
    if not chunk.page_content.startswith(context_header):
        chunk.page_content = f"{context_header}\n{chunk.page_content}"

    return chunk


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split LangChain Document objects.
    Hỗ trợ 3 chiến lược:
    1. Markdown Splitting: Dành cho pymupdf4llm (có format='markdown').
    2. Structural Chunking: Dành cho Unstructured (có category).
    3. Recursive Splitter: Fallback cho text thuần.
    """
    _validate_chunk_params(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    markdown_docs = [doc for doc in documents if doc.metadata.get("format") == "markdown"]
    plain_docs = [doc for doc in documents if doc.metadata.get("format") != "markdown"]

    all_chunks = []

    # ── 1. Xử lý Markdown Docs (PyMuPDF4LLM) ────────────────────────────────
    if markdown_docs:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        for doc in markdown_docs:
            header_splits = md_splitter.split_text(doc.page_content)
            for split in header_splits:
                new_metadata = dict(doc.metadata)
                new_metadata.update(split.metadata)
                if len(split.page_content) > chunk_size:
                    sub_chunks = recursive_splitter.split_text(split.page_content)
                    for sub in sub_chunks:
                        all_chunks.append(Document(page_content=sub, metadata=new_metadata))
                else:
                    all_chunks.append(Document(page_content=split.page_content, metadata=new_metadata))

    # ── 2. Xử lý Plain Docs ──────────────────────────────────────────────────
    if plain_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        all_chunks.extend(text_splitter.split_documents(plain_docs))

    return [_enrich_chunk_metadata(chunk, idx) for idx, chunk in enumerate(all_chunks)]
