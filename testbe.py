import os
from src.core.embeddings import get_embedding_model
from src.core.vectorstore import VectorStoreManager
from src.core.chain import RAGChainManager
from src.core.llm import get_llm

# 1. Khởi tạo hệ thống
embedding_model = get_embedding_model()
llm = get_llm()
vsm = VectorStoreManager(embedding_model)
rag_chain = RAGChainManager(llm)

# 2. Load Vectorstore đã có sẵn RRTO.pdf
vectorstore = vsm.load_vectorstore()

if vectorstore:
    # Để test Hybrid Search, bạn cần truyền lại chunks (văn bản thô) 
    # Nếu chưa có cơ chế load chunks từ đĩa, tạm thời dùng retriever từ vectorstore
    # (Nhưng để test Hybrid thực thụ, hãy đảm bảo vsm.get_hybrid_retriever nhận được list documents)
    rag_chain.update_retriever(vectorstore)
    print("--- Hệ thống đã sẵn sàng với dữ liệu RRTO.pdf ---\n")

def test_ask(question, label):
    print(f"[{label}]")
    print(f"Hỏi: {question}")
    answer, sources = rag_chain.ask(question, conversational=True)
    print(f"Đáp: {answer}")
    if sources:
        print(f"Nguồn: {sources[0]['file']} - Trang: {sources[0]['page']}")
    print("-" * 50)

# --- KỊCH BẢN TEST CHI TIẾT ---

# Test 1: Khả năng truy xuất thông tin cụ thể (Kiểm tra Retrieval)
test_ask("Mr. BupBe là ai và chuyên mục này nói về vấn đề gì?", "KIỂM TRA THÔNG TIN CHUNG")

# Test 2: Kiểm tra Hybrid Search (Tìm từ khóa riêng biệt: "Mộng Bạch")
# BM25 sẽ giúp tìm chính xác tên người này trong hàng ngàn đoạn văn bản
test_ask("Bạn Mộng Bạch ở Lạng Sơn đã hỏi điều gì về anh bạn trai của mình?", "KIỂM TRA HYBRID SEARCH")

# Test 3: Kiểm tra Conversational (Khả năng nhớ lịch sử)
# Câu hỏi này không nhắc lại "Mộng Bạch", AI phải tự hiểu từ câu trước.
test_ask("Mr. BupBe đã đưa ra lời khuyên như thế nào cho bạn ấy?", "KIỂM TRA NGỮ CẢNH/LỊCH SỬ")

# Test 4: Kiểm tra khả năng xử lý thông tin không có trong tài liệu
test_ask("Công thức nấu phở bò có trong tài liệu không?", "KIỂM TRA ĐỘ CHÍNH XÁC/HALLUCINATION")