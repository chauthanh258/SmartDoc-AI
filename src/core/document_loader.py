from pathlib import Path
import re
from datetime import date, datetime, time, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from langchain_community.document_loaders import Docx2txtLoader, PDFPlumberLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".docm"}


def _new_doc_id() -> str:
    """Generate a globally unique identifier for each uploaded document."""
    return str(uuid4())


def _normalize_upload_date(value: str | datetime | date | None) -> str:
    """Normalize upload date into UTC ISO-8601 string.

    We keep one stable format in metadata so filtering by date is predictable.
    """
    if value is None:
        return datetime.now(timezone.utc).isoformat()

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time.min)
    else:
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            # Fallback: keep current UTC timestamp when user input is invalid.
            return datetime.now(timezone.utc).isoformat()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def detect_language(text: str) -> str:
    """Best-effort language detection without adding extra dependencies.

    Returns:
    - "vi" for Vietnamese
    - "en" for English
    - "unknown" when confidence is too low
    """
    normalized = text.lower().strip()
    if not normalized:
        return "unknown"

    # Vietnamese-specific unicode characters are a strong signal.
    vietnamese_chars = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ")
    if any(char in vietnamese_chars for char in normalized):
        return "vi"

    words = re.findall(r"[a-zA-ZÀ-ỹ]+", normalized)
    if not words:
        return "unknown"

    vi_keywords = {
        "va", "la", "cua", "cho", "voi", "nhung", "trong", "duoc", "nguoi", "tai",
        "theo", "khong", "mot", "co", "cac", "phan", "chuong", "muc", "noi", "dung",
    }
    en_keywords = {
        "the", "and", "for", "with", "this", "that", "from", "into", "about", "document",
        "section", "page", "content", "is", "are", "to", "of", "in", "on", "by",
    }

    vi_score = sum(1 for word in words if word in vi_keywords)
    en_score = sum(1 for word in words if word in en_keywords)

    if vi_score > en_score and vi_score >= 2:
        return "vi"
    if en_score > vi_score and en_score >= 2:
        return "en"
    return "unknown"


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


def _resolve_language(documents: Sequence[Document], language: str | None) -> str:
    """Use caller-provided language when available, otherwise infer from content."""
    if language:
        return language

    # Limit to a small sample for speed while still capturing language signal.
    sample_text = "\n".join(doc.page_content[:1500] for doc in documents[:3])
    return detect_language(sample_text)


def _attach_document_metadata(
    documents: list[Document],
    file_path: Path,
    *,
    doc_id: str,
    upload_date: str,
    language: str,
) -> list[Document]:
    """Attach standardized document-level metadata to every chunk/page.

    Metadata fields added for Phase 3:
    - doc_id: globally unique document ID
    - filename: original filename
    - upload_date: UTC ISO timestamp
    - language: detected or user-specified language

    Compatibility keys (already used by existing code) are preserved.
    """
    upload_date_only = upload_date.split("T", maxsplit=1)[0]
    for doc in documents:
        metadata = dict(doc.metadata or {})

        metadata["doc_id"] = doc_id
        metadata["filename"] = file_path.name
        metadata["file_name"] = file_path.name
        metadata["source"] = str(file_path)
        metadata["upload_date"] = upload_date
        metadata["upload_date_only"] = upload_date_only
        metadata["language"] = language

        doc.metadata = metadata

    return documents


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


def load_document(
    file_path: str,
    *,
    doc_id: str | None = None,
    upload_date: str | datetime | date | None = None,
    language: str | None = None,
) -> list[Document]:
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

    normalized_doc_id = doc_id or _new_doc_id()
    normalized_upload_date = _normalize_upload_date(upload_date)

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

    resolved_language = _resolve_language(documents, language=language)
    return _attach_document_metadata(
        documents,
        path,
        doc_id=normalized_doc_id,
        upload_date=normalized_upload_date,
        language=resolved_language,
    )


def load_multiple_documents(
    file_paths: Sequence[str],
    *,
    upload_date: str | datetime | date | None = None,
    language_map: Mapping[str, str] | None = None,
    skip_failed: bool = True,
) -> list[Document]:
    """Load multiple PDF/DOCX files and return one merged document list.

    Each input file receives its own unique `doc_id`, while all chunks/pages of
    that file share the same metadata group for filtering and citation.

    Args:
        file_paths: List of document paths.
        upload_date: Optional shared upload timestamp for batch ingest.
        language_map: Optional per-file language override.
            Supports key by full path or filename.
        skip_failed: If True, continue processing remaining files when one fails.
    """
    if not file_paths:
        return []

    all_documents: list[Document] = []
    errors: list[str] = []

    for file_path in file_paths:
        path = Path(file_path)
        override_language = None
        if language_map:
            override_language = language_map.get(str(path)) or language_map.get(path.name)

        try:
            docs = load_document(
                str(path),
                doc_id=_new_doc_id(),
                upload_date=upload_date,
                language=override_language,
            )
            all_documents.extend(docs)
        except Exception as exc:
            if not skip_failed:
                raise
            errors.append(f"{path.name}: {exc}")

    if not all_documents and errors:
        raise RuntimeError(
            "Unable to load any documents from batch input. "
            f"Errors: {' | '.join(errors)}"
        )

    return all_documents


def load_pdf(file_path: str) -> list[Document]:
    """Backward-compatible PDF loader for existing call sites."""
    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError("load_pdf only accepts .pdf files.")

    return load_document(file_path)


