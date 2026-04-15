from pathlib import Path
import re
from typing import Any

from langchain_community.document_loaders import Docx2txtLoader, PDFPlumberLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".docm"}


def _to_int_or_none(value: Any) -> int | None:
    """Convert value to int when possible, otherwise return None."""
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_heading_block(block: str) -> bool:
    """Heuristic to detect DOCX headings in plain extracted text.

    Docx2txtLoader does not expose structural nodes directly, so headings are
    inferred from short stand-alone lines that look like section titles.
    """
    text = block.strip()
    if not text:
        return False

    # Heading candidates are typically one short line.
    if "\n" in text:
        return False

    if len(text) > 100:
        return False

    if len(text.split()) > 12:
        return False

    # If it ends with sentence punctuation, it is likely normal paragraph text.
    if text.endswith((".", "!", "?", ";", ":")):
        return False

    # Common heading patterns: numbered title or mostly uppercase/title-style text.
    starts_with_number = bool(re.match(r"^\d+(?:[.)]|\s)", text))
    looks_like_title = text.istitle() or text.isupper()
    return starts_with_number or looks_like_title


def _split_docx_sections(page_content: str) -> list[tuple[str, str]]:
    """Split DOCX text into logical sections.

    Returns list of tuples: (section_name, section_text)
    """
    normalized = page_content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    blocks = [b.strip() for b in re.split(r"\n\s*\n+", normalized) if b.strip()]
    if not blocks:
        return []

    sections: list[tuple[str, str]] = []
    current_heading = "Section 1"
    current_body: list[str] = []

    for block in blocks:
        if _is_heading_block(block):
            if current_body:
                sections.append((current_heading, "\n\n".join(current_body).strip()))
                current_body = []
            current_heading = block
        else:
            current_body.append(block)

    if current_body:
        sections.append((current_heading, "\n\n".join(current_body).strip()))

    if not sections:
        sections.append(("Section 1", normalized))

    return sections


def _load_pdf(file_path: Path) -> list[Document]:
    """Load PDF and normalize metadata for citation-friendly retrieval."""
    loader = PDFPlumberLoader(str(file_path))
    documents = loader.load()

    if not documents:
        return []

    page_values = [_to_int_or_none(doc.metadata.get("page")) for doc in documents]
    page_values = [value for value in page_values if value is not None]
    zero_based_pages = bool(page_values) and min(page_values) == 0

    normalized_docs: list[Document] = []
    for index, doc in enumerate(documents):
        metadata = dict(doc.metadata or {})

        raw_page = _to_int_or_none(metadata.get("page"))
        if raw_page is None:
            page_number = index + 1
        else:
            page_number = raw_page + 1 if zero_based_pages else raw_page

        metadata["source"] = str(file_path)
        metadata["file_name"] = file_path.name
        metadata["page_number"] = page_number

        normalized_docs.append(Document(page_content=doc.page_content, metadata=metadata))

    return normalized_docs


def _load_docx(file_path: Path) -> list[Document]:
    """Load DOCX/DOCM and preserve logical section metadata."""
    loader = Docx2txtLoader(str(file_path))
    raw_documents = loader.load()

    if not raw_documents:
        return []

    enriched_documents: list[Document] = []
    for raw_doc in raw_documents:
        base_metadata = dict(raw_doc.metadata or {})
        base_metadata["source"] = str(file_path)
        base_metadata["file_name"] = file_path.name

        sections = _split_docx_sections(raw_doc.page_content)
        if not sections:
            sections = [("Section 1", raw_doc.page_content)]

        for section_index, (section_name, section_text) in enumerate(sections, start=1):
            metadata = dict(base_metadata)
            metadata["section"] = section_name
            metadata["section_index"] = section_index

            enriched_documents.append(
                Document(page_content=section_text, metadata=metadata)
            )

    return enriched_documents


def load_document(file_path: str) -> list[Document]:
    """Load PDF/DOCX and return LangChain Document objects.

    Extracted text is returned as-is so Vietnamese characters are preserved.
    """
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {path.suffix.lower()}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if path.suffix.lower() == ".pdf":
            documents = _load_pdf(path)
        else:
            documents = _load_docx(path)
    except Exception as exc:
        raise RuntimeError(
            "Unable to read document. The file may be corrupted, encrypted, "
            f"or not parseable: {file_path}"
        ) from exc

    if not documents:
        raise RuntimeError(f"No readable text content found in document: {file_path}")

    return documents


def load_pdf(file_path: str) -> list[Document]:
    """Backward-compatible PDF loader for existing call sites."""
    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError("load_pdf only accepts .pdf files.")

    return load_document(file_path)


