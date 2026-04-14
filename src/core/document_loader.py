from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PDFPlumberLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".docm"}


def _get_loader(file_path: Path):
    """Return the suitable LangChain loader for the current extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return PDFPlumberLoader(str(file_path))
    if suffix in {".docx", ".docm"}:
        return Docx2txtLoader(str(file_path))

    raise ValueError(
        f"Unsupported file format: {suffix}. "
        f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


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
        loader = _get_loader(path)
        documents = loader.load()
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
