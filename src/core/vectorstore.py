from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
import os
import config

class VectorStoreManager:
    def __init__(self, embedding_model, folder_name="faiss_index"):
        self.embedding_model = embedding_model
        self.folder_name = folder_name
        self.path = os.path.join(config.VECTORSTORE_DIR, self.folder_name)
        self.vectorstore = None

    def create_vectorstore(self, chunks):
        """Tạo FAISS vectorstore từ các đoạn văn bản."""
        self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        return self.vectorstore

    def save_vectorstore(self):
        """Lưu vectorstore xuống ổ đĩa."""
        if self.vectorstore is None:
            print("Lỗi: Chưa có vectorstore để lưu.")
            return False
            
        os.makedirs(self.path, exist_ok=True)
        rel_path = os.path.relpath(self.path, start=os.getcwd())
        self.vectorstore.save_local(rel_path)
        return True

    def load_vectorstore(self):
        """Tải vectorstore từ ổ đĩa."""
        index_file = os.path.join(self.path, "index.faiss")
        if os.path.exists(index_file):
            rel_path = os.path.relpath(self.path, start=os.getcwd())
            self.vectorstore = FAISS.load_local(
                rel_path, 
                self.embedding_model, 
                allow_dangerous_deserialization=True
            )
            return self.vectorstore
        return None

    def get_hybrid_retriever(self, chunks, k=3):
        """
        Thiết lập Hybrid Search:
        - BM25: Tìm kiếm theo từ khóa (Keyword)
        - FAISS: Tìm kiếm theo ý nghĩa (Semantic)
        - Ensemble: Kết hợp cả hai để tăng độ chính xác
        """
        if not self.vectorstore:
            print("Lỗi: Vectorstore chưa được khởi tạo.")
            return None

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