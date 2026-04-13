from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF file and return LangChain Document objects.

    This Phase 0 loader accepts PDF input only and keeps extracted text
    untouched so Vietnamese diacritics are preserved for embedding.
    """
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported in this phase.")

    try:
        loader = PDFPlumberLoader(str(path))
        documents = loader.load()
    except Exception as exc:
        # Normalize low-level parser failures to a readable domain error.
        raise RuntimeError(
            f"Unable to read PDF. The file may be corrupted or encrypted: {file_path}"
        ) from exc

    if not documents:
        raise RuntimeError(f"No readable text content found in PDF: {file_path}")

    return documents
