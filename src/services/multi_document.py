from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import config
from langchain_core.documents import Document

from src.core.document_loader import load_document
from src.core.text_splitter import split_documents
from src.core.vectorstore import VectorStoreManager, build_metadata_filter


@dataclass
class FileIngestError:
    """Stores non-fatal ingest errors for individual files in a batch."""

    file_name: str
    error: str


@dataclass
class MultiDocumentIngestResult:
    """Structured result returned after multi-document ingest."""

    saved_files: list[str] = field(default_factory=list)
    loaded_file_count: int = 0
    loaded_pages_or_sections: int = 0
    total_chunks: int = 0
    failed_files: list[FileIngestError] = field(default_factory=list)
    indexed_documents: list[dict[str, Any]] = field(default_factory=list)


class MultiDocumentRAGService:
    """High-level orchestration service for multi-document RAG.

    Responsibilities:
    1. Save and ingest multiple uploaded files in one batch.
    2. Ensure all chunks carry document-level metadata (doc_id, filename, date, language).
    3. Persist chunks into FAISS and expose metadata-aware retrievers.
    4. Provide filename/date filter logic consumable by Streamlit UI.
    """

    def __init__(self, embedding_model, folder_name: str = "faiss_index"):
        self.vectorstore_manager = VectorStoreManager(
            embedding_model=embedding_model,
            folder_name=folder_name,
        )

    def _resolve_unique_upload_path(self, file_name: str) -> Path:
        """Avoid overwriting by suffixing duplicate file names."""
        base_dir = Path(config.UPLOAD_DIR)
        base_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(file_name).name
        candidate = base_dir / safe_name
        if not candidate.exists():
            return candidate

        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1
        while True:
            deduplicated = base_dir / f"{stem}_{counter}{suffix}"
            if not deduplicated.exists():
                return deduplicated
            counter += 1

    def save_uploaded_files(self, uploaded_files: Sequence[Any]) -> list[str]:
        """Persist multiple Streamlit-uploaded files and return saved paths.

        Expected upload object shape is compatible with Streamlit's UploadedFile:
        - .name
        - .getbuffer()
        """
        saved_paths: list[str] = []

        for uploaded in uploaded_files:
            target_path = self._resolve_unique_upload_path(uploaded.name)
            with open(target_path, "wb") as destination:
                destination.write(uploaded.getbuffer())
            saved_paths.append(str(target_path))

        return saved_paths

    def ingest_file_paths(
        self,
        file_paths: Sequence[str],
        *,
        chunk_size: int,
        chunk_overlap: int,
        append_to_existing_index: bool = True,
    ) -> MultiDocumentIngestResult:
        """Load, split, and index multiple files from disk paths.

        This method is resilient: one broken file does not stop the entire batch.
        """
        result = MultiDocumentIngestResult(saved_files=[str(path) for path in file_paths])

        if not file_paths:
            return result

        all_documents: list[Document] = []

        for file_path in file_paths:
            path = Path(file_path)
            try:
                docs = load_document(str(path))
                all_documents.extend(docs)
                result.loaded_file_count += 1
                result.loaded_pages_or_sections += len(docs)
            except Exception as exc:
                result.failed_files.append(
                    FileIngestError(file_name=path.name, error=str(exc))
                )

        if not all_documents:
            # No successful documents means there is nothing to index.
            return result

        chunks = split_documents(
            all_documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        result.total_chunks = len(chunks)

        if append_to_existing_index:
            # Load previous index in case user uploads documents in multiple sessions.
            self.vectorstore_manager.load_vectorstore()
            self.vectorstore_manager.add_documents(chunks)
        else:
            self.vectorstore_manager.create_vectorstore(chunks, append=False)

        self.vectorstore_manager.save_vectorstore()
        result.indexed_documents = self.vectorstore_manager.get_document_registry()
        return result

    def ingest_uploaded_files(
        self,
        uploaded_files: Sequence[Any],
        *,
        chunk_size: int,
        chunk_overlap: int,
        append_to_existing_index: bool = True,
    ) -> MultiDocumentIngestResult:
        """Convenience API for UI: save uploads first, then ingest them."""
        saved_paths = self.save_uploaded_files(uploaded_files)
        return self.ingest_file_paths(
            saved_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            append_to_existing_index=append_to_existing_index,
        )

    def get_filter_options(self) -> dict[str, list[str]]:
        """Generate UI-ready options for metadata filtering controls.

        Returns:
            - filenames: unique file names in index
            - upload_dates: unique YYYY-MM-DD values
            - doc_ids: unique doc IDs (useful for advanced UI/debug)
        """
        registry = self.vectorstore_manager.get_document_registry()

        filenames = sorted(
            {
                item["filename"]
                for item in registry
                if item.get("filename")
            }
        )

        upload_dates = sorted(
            {
                (item.get("upload_date_only") or str(item.get("upload_date") or "")[:10])
                for item in registry
                if item.get("upload_date") or item.get("upload_date_only")
            },
            reverse=True,
        )

        doc_ids = sorted(
            {
                str(item["doc_id"])
                for item in registry
                if item.get("doc_id")
            }
        )

        return {
            "filenames": filenames,
            "upload_dates": upload_dates,
            "doc_ids": doc_ids,
        }

    def build_filter_state(
        self,
        *,
        selected_filename: str | None = None,
        selected_upload_date: str | None = None,
        selected_doc_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        """Create normalized filter payload for UI and retriever wiring.

        This function is the UI logic layer for filename/date filtering.
        It accepts raw widget values and returns:
        - metadata_filter: callable for FAISS retrieval
        - active_filters: compact values for status badges/tooltips
        """

        def _normalize_choice(value: str | None) -> str | None:
            if value is None:
                return None
            lowered = value.strip().lower()
            if lowered in {"", "all", "tat ca", "tất cả"}:
                return None
            return value.strip()

        normalized_filename = _normalize_choice(selected_filename)
        normalized_upload_date = _normalize_choice(selected_upload_date)
        normalized_doc_id = _normalize_choice(selected_doc_id)

        metadata_filter = build_metadata_filter(
            filename=normalized_filename,
            upload_date=normalized_upload_date,
            date_from=date_from,
            date_to=date_to,
            doc_id=normalized_doc_id,
        )

        return {
            "metadata_filter": metadata_filter,
            "active_filters": {
                "filename": normalized_filename,
                "upload_date": normalized_upload_date,
                "doc_id": normalized_doc_id,
                "date_from": date_from,
                "date_to": date_to,
            },
        }

    def get_filtered_retriever(
        self,
        *,
        k: int = 3,
        selected_filename: str | None = None,
        selected_upload_date: str | None = None,
        selected_doc_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        fetch_k: int | None = None,
    ):
        """Return a retriever that applies filename/date/doc filters."""
        filter_state = self.build_filter_state(
            selected_filename=selected_filename,
            selected_upload_date=selected_upload_date,
            selected_doc_id=selected_doc_id,
            date_from=date_from,
            date_to=date_to,
        )
        return self.vectorstore_manager.get_retriever(
            k=k,
            metadata_filter=filter_state["metadata_filter"],
            fetch_k=fetch_k,
        )

    def search(
        self,
        query: str,
        *,
        k: int = 3,
        selected_filename: str | None = None,
        selected_upload_date: str | None = None,
        selected_doc_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        fetch_k: int | None = None,
    ) -> list[Document]:
        """Run metadata-filtered semantic search directly (without chain wrapper)."""
        filter_state = self.build_filter_state(
            selected_filename=selected_filename,
            selected_upload_date=selected_upload_date,
            selected_doc_id=selected_doc_id,
            date_from=date_from,
            date_to=date_to,
        )
        return self.vectorstore_manager.similarity_search(
            query,
            k=k,
            metadata_filter=filter_state["metadata_filter"],
            fetch_k=fetch_k,
        )

    @staticmethod
    def build_rich_citation_payload(documents: Sequence[Document]) -> list[dict[str, Any]]:
        """Return citation-ready records with rich metadata for UI/LLM outputs."""
        payload: list[dict[str, Any]] = []

        for doc in documents:
            metadata = dict(doc.metadata or {})
            payload.append(
                {
                    "content": doc.page_content,
                    "snippet": doc.page_content[:300].strip(),
                    "doc_id": metadata.get("doc_id"),
                    "filename": metadata.get("filename") or metadata.get("file_name"),
                    "upload_date": metadata.get("upload_date"),
                    "language": metadata.get("language"),
                    "page_number": metadata.get("page_number") or metadata.get("page"),
                    "section": metadata.get("section"),
                    "chunk_index": metadata.get("chunk_index"),
                    "source": metadata.get("source"),
                }
            )

        return payload
