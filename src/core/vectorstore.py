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
        """Tạo mới hoàn toàn một FAISS vectorstore."""
        self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        self.save()
        return self.vectorstore

    def save(self):
        """Lưu vectorstore xuống đĩa."""
        if self.vectorstore:
            self.vectorstore.save_local(self.path)

    def load(self):
        """Tải vectorstore từ đĩa."""
        if os.path.exists(self.path):
            self.vectorstore = FAISS.load_local(
                self.path, 
                self.embedding_model, 
                allow_dangerous_deserialization=True
            )
            return self.vectorstore
        return None

    def add_documents(self, new_chunks):
        """Nạp thêm tài liệu mới vào index hiện có."""
        self.load() # Đảm bảo đã load index cũ trước khi add
        if self.vectorstore:
            self.vectorstore.add_documents(new_chunks)
            self.save()
            print(f"Đã cập nhật thêm {len(new_chunks)} chunks.")
        else:
            self.create_vectorstore(new_chunks)
        return self.vectorstore
    
