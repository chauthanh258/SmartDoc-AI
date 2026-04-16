import os
from datetime import date, datetime
from typing import Any, Callable

import config
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


def _to_date_or_none(value: Any) -> date | None:
    """Parse metadata date values into date objects for range filtering."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        # Handle ISO timestamps containing trailing "Z".
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"

        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            try:
                return datetime.strptime(raw, "%Y-%m-%d").date()
            except ValueError:
                return None

    return None


def build_metadata_filter(
    *,
    filename: str | None = None,
    upload_date: str | date | datetime | None = None,
    date_from: str | date | datetime | None = None,
    date_to: str | date | datetime | None = None,
    doc_id: str | None = None,
    language: str | None = None,
) -> Callable[[dict[str, Any]], bool] | None:
    """Build a FAISS-compatible metadata predicate.

    The FAISS integration supports passing a callable as `filter`, allowing
    flexible filtering logic (exact match + date range checks).
    """
    normalized_filename = filename.strip() if filename else None
    normalized_doc_id = doc_id.strip() if doc_id else None
    normalized_language = language.strip() if language else None
    normalized_upload_date = _to_date_or_none(upload_date)
    normalized_date_from = _to_date_or_none(date_from)
    normalized_date_to = _to_date_or_none(date_to)

    if not any(
        [
            normalized_filename,
            normalized_doc_id,
            normalized_language,
            normalized_upload_date,
            normalized_date_from,
            normalized_date_to,
        ]
    ):
        return None

    def _metadata_matches(metadata: dict[str, Any]) -> bool:
        current_filename = metadata.get("filename") or metadata.get("file_name")
        current_doc_id = metadata.get("doc_id")
        current_language = metadata.get("language")
        current_upload_date = _to_date_or_none(
            metadata.get("upload_date") or metadata.get("upload_date_only")
        )

        if normalized_filename and current_filename != normalized_filename:
            return False
        if normalized_doc_id and current_doc_id != normalized_doc_id:
            return False
        if normalized_language and current_language != normalized_language:
            return False

        if normalized_upload_date and current_upload_date != normalized_upload_date:
            return False
        if normalized_date_from and (
            current_upload_date is None or current_upload_date < normalized_date_from
        ):
            return False
        if normalized_date_to and (
            current_upload_date is None or current_upload_date > normalized_date_to
        ):
            return False

        return True

    return _metadata_matches


class VectorStoreManager:
    def __init__(self, embedding_model, folder_name="faiss_index"):
        self.embedding_model = embedding_model
        self.folder_name = folder_name
        self.path = os.path.join(config.VECTORSTORE_DIR, self.folder_name)
        self.vectorstore = None

    def create_vectorstore(self, chunks: list[Document], append: bool = True):
        """Tạo mới hoặc thêm vào vectorstore hiện có."""
        if append:
            # Sử dụng hàm add_documents để tự động xử lý việc load và cộng dồn
            return self.add_documents(chunks)
        
        # Nếu append=False, ghi đè hoàn toàn
        self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        return self.vectorstore

    def add_documents(self, chunks: list[Document]):
        """Cộng dồn tài liệu mới vào bộ nhớ hiện tại."""
        # 1. Nếu RAM chưa có, thử load từ ổ đĩa
        if self.vectorstore is None:
            self.load_vectorstore()

        # 2. Nếu sau khi load vẫn None (nghĩa là chưa từng có index nào)
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        else:
            # 3. Nếu đã có, thêm vào kiến thức hiện tại
            self.vectorstore.add_documents(chunks)
        
        return self.vectorstore

    def save_vectorstore(self):
        """Lưu toàn bộ bộ nhớ xuống ổ đĩa."""
        if self.vectorstore is None:
            return False
            
        os.makedirs(self.path, exist_ok=True)
        self.vectorstore.save_local(self.path)
        return True

    def load_vectorstore(self):
        """Tải dữ liệu từ ổ đĩa vào RAM."""
        index_file = os.path.join(self.path, "index.faiss")
        if os.path.exists(index_file):
            try:
                self.vectorstore = FAISS.load_local(
                    self.path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                return self.vectorstore
            except Exception as e:
                print(f"Lỗi khi nạp FAISS: {e}")
        return None

    def similarity_search(
        self,
        query: str,
        *,
        k: int = 3,
        metadata_filter: Callable[[dict[str, Any]], bool] | None = None,
        fetch_k: int | None = None,
    ) -> list[Document]:
        """Run semantic retrieval with optional metadata filtering."""
        if self.vectorstore is None:
            self.load_vectorstore()

        if self.vectorstore is None:
            return []

        effective_fetch_k = fetch_k if fetch_k is not None else max(k * 5, 20)
        return self.vectorstore.similarity_search(
            query,
            k=k,
            fetch_k=effective_fetch_k,
            filter=metadata_filter,
        )

    def get_retriever(
        self,
        *,
        k: int = 3,
        metadata_filter: Callable[[dict[str, Any]], bool] | None = None,
        fetch_k: int | None = None,
    ):
        """Return retriever object compatible with LangChain runnables.

        When a metadata filter is provided, we wrap retrieval in RunnableLambda
        because built-in retriever wrappers do not expose date-range filtering
        ergonomics for FAISS in this project.
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        if self.vectorstore is None:
            return None

        if metadata_filter is None:
            return self.vectorstore.as_retriever(search_kwargs={"k": k})

        return RunnableLambda(
            lambda query: self.similarity_search(
                query,
                k=k,
                metadata_filter=metadata_filter,
                fetch_k=fetch_k,
            )
        )

    def get_retriever_with_filters(
        self,
        *,
        k: int = 3,
        filename: str | None = None,
        upload_date: str | date | datetime | None = None,
        date_from: str | date | datetime | None = None,
        date_to: str | date | datetime | None = None,
        doc_id: str | None = None,
        language: str | None = None,
        fetch_k: int | None = None,
    ):
        """Convenience API to build a filtered retriever in one call."""
        metadata_filter = build_metadata_filter(
            filename=filename,
            upload_date=upload_date,
            date_from=date_from,
            date_to=date_to,
            doc_id=doc_id,
            language=language,
        )
        return self.get_retriever(
            k=k,
            metadata_filter=metadata_filter,
            fetch_k=fetch_k,
        )

    def get_document_registry(self) -> list[dict[str, Any]]:
        """Return unique documents currently indexed with rich metadata.

        This method scans FAISS docstore chunks and groups them by doc_id,
        producing a compact registry that UI code can use for filter widgets.
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        if self.vectorstore is None:
            return []

        docstore_dict = getattr(self.vectorstore.docstore, "_dict", {})
        grouped: dict[str, dict[str, Any]] = {}

        for doc in docstore_dict.values():
            if not isinstance(doc, Document):
                continue

            metadata = dict(doc.metadata or {})
            current_doc_id = str(metadata.get("doc_id") or "")
            if not current_doc_id:
                # Backward compatibility for old indexes without doc_id.
                source = str(metadata.get("source") or "unknown")
                current_doc_id = f"legacy::{source}"

            if current_doc_id not in grouped:
                grouped[current_doc_id] = {
                    "doc_id": current_doc_id,
                    "filename": metadata.get("filename") or metadata.get("file_name"),
                    "file_name": metadata.get("file_name") or metadata.get("filename"),
                    "upload_date": metadata.get("upload_date"),
                    "upload_date_only": metadata.get("upload_date_only"),
                    "language": metadata.get("language"),
                    "source": metadata.get("source"),
                    "chunk_count": 0,
                }

            grouped[current_doc_id]["chunk_count"] += 1

        return sorted(
            grouped.values(),
            key=lambda item: (
                str(item.get("upload_date") or ""),
                str(item.get("filename") or ""),
            ),
            reverse=True,
        )

    def get_hybrid_retriever(self, chunks, k=3, metadata_filter=None):
        """
        Thiết lập Hybrid Search:
        - BM25: Tìm kiếm theo từ khóa (Keyword)
        - FAISS: Tìm kiếm theo ý nghĩa (Semantic)
        - Ensemble: Kết hợp cả hai để tăng độ chính xác
        """
        if not self.vectorstore:
            print("Lỗi: Vectorstore chưa được khởi tạo.")
            return None

        # BM25 retriever does not natively support metadata constraints.
        # If a filter is requested, return semantic retriever with filter.
        if metadata_filter is not None:
            return self.get_retriever(k=k, metadata_filter=metadata_filter)

        # 1. Tạo FAISS retriever
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # 2. Tạo BM25 retriever từ các đoạn văn bản (chunks)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = k

        # 3. Kết hợp 2 bộ tìm kiếm với trọng số 50/50
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever