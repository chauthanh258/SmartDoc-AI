import os
import config
from src.core.embeddings import get_embedding_model
from src.core.vectorstore import VectorStoreManager
from src.core.chain import RAGChainManager
from src.core.llm import get_llm

# 1. Khởi tạo các thành phần
embedding_model = get_embedding_model()
llm = get_llm()

vsm = VectorStoreManager(embedding_model)
rag_chain = RAGChainManager(llm)

# 2. Load vectorstore đã tồn tại trên đĩa
# Hàm load() trả về đối tượng FAISS vectorstore nếu tìm thấy file trong thư mục config.VECTORSTORE_DIR
vectorstore = vsm.load()

if vectorstore:
    # 3. Cập nhật retriever vào chain (Bước này cực kỳ quan trọng để khởi tạo self.chain)
    rag_chain.update_retriever(vectorstore)
    
    # 4. Thực hiện query
    question = "Nội dung chính của tài liệu này là gì?" # Thay đổi câu hỏi của bạn tại đây
    answer = rag_chain.ask(question)
    
    print(f"Câu hỏi: {question}")
    print("-" * 30)
    print(f"Câu trả lời: {answer}")
else:
    print("Lỗi: Không tìm thấy file vectorstore tại đường dẫn đã cấu hình. Vui lòng kiểm tra lại thư mục lưu trữ.")