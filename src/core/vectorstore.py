from langchain_community.vectorstores import FAISS
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
        # Sử dụng đường dẫn tương đối để tránh lỗi Unicode trên Windows
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

    def get_retriever(self, k=3):
        """Khởi tạo retriever từ vectorstore hiện có."""
        if self.vectorstore:
            return self.vectorstore.as_retriever(search_kwargs={"k": k})
        return None
