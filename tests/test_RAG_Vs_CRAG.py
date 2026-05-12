# test_RAG_Vs_CRAG.py
import os
import time
from typing import List, Dict

# LangChain Imports
from langchain_ollama import OllamaLLM
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Project Imports
import config
from src.core.vectorstore import VectorStoreManager

# --- Cấu hình màu sắc console ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def get_test_llm():
    """Khởi tạo LLM cho script test (không dùng cache của streamlit)"""
    return OllamaLLM(
        base_url=config.OLLAMA_CLOUD_BASE_URL,
        model=config.OLLAMA_CLOUD_MODEL,
        api_key=config.OLLAMA_API_KEY,
        temperature=0
    )

def get_test_embeddings():
    """Khởi tạo Embedding model cho script test"""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

# --- Các Prompt Templates ---

# 1. RAG Answer Prompt
RAG_PROMPT = ChatPromptTemplate.from_template("""
Bạn là một trợ lý thông minh. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.
Nếu không có thông tin trong ngữ cảnh, hãy nói bạn không biết.

Ngữ cảnh:
{context}

Câu hỏi: {question}
Trả lời:""")

# 2. CRAG Evaluator Prompt (Đánh giá mức độ liên quan)
CRAG_EVALUATOR_PROMPT = ChatPromptTemplate.from_template("""
Bạn là một người kiểm duyệt tài liệu thông minh. Nhiệm vụ của bạn là đánh giá xem đoạn tài liệu dưới đây có chứa thông tin (dù là trực tiếp hay gián tiếp) có thể giúp trả lời câu hỏi của người dùng hay không.

Câu hỏi: {question}
Tài liệu: {document}

Tiêu chí đánh giá:
- Trả về "RELEVANT" nếu tài liệu chứa câu trả lời trực tiếp hoặc thông tin cốt lõi để trả lời.
- Trả về "PARTIAL" nếu tài liệu nhắc đến các thực thể (tên người, địa danh, khái niệm) có trong câu hỏi hoặc cung cấp ngữ cảnh hữu ích.
- Trả về "IRRELEVANT" nếu tài liệu hoàn toàn không liên quan đến câu hỏi.

Chỉ trả về 1 từ duy nhất: RELEVANT, PARTIAL hoặc IRRELEVANT. Không giải thích gì thêm.
""")

class ComparisonRunner:
    def __init__(self):
        print(f"{Colors.HEADER}--- Đang khởi tạo hệ thống so sánh ---{Colors.ENDC}")
        self.llm = get_test_llm()
        self.embeddings = get_test_embeddings()
        self.vs_manager = VectorStoreManager(self.embeddings)
        self.vectorstore = self.vs_manager.load_vectorstore()
        
        if not self.vectorstore:
            print(f"{Colors.RED}Lỗi: Không tìm thấy Vectorstore. Vui lòng upload tài liệu trên giao diện web trước.{Colors.ENDC}")
            exit(1)
            
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 7})
        
        # Chains
        self.rag_chain = RAG_PROMPT | self.llm | StrOutputParser()
        self.evaluator_chain = CRAG_EVALUATOR_PROMPT | self.llm | StrOutputParser()

    def run_standard_rag(self, question: str) -> Dict:
        """Thực hiện RAG truyền thống"""
        start_time = time.time()
        # Retrieve
        docs = self.retriever.invoke(question)
        context = "\n".join([d.page_content for d in docs])
        
        # Generate
        answer = self.rag_chain.invoke({"context": context, "question": question})
        
        return {
            "answer": answer,
            "docs_count": len(docs),
            "time": time.time() - start_time
        }

    def run_crag(self, question: str) -> Dict:
        """Thực hiện Corrective RAG (CRAG)"""
        start_time = time.time()
        
        # 1. Retrieve
        docs = self.retriever.invoke(question)
        
        # 2. Evaluate (Corrective Step)
        relevant_docs = []
        evaluations = []
        
        for d in docs:
            raw_score = self.evaluator_chain.invoke({"question": question, "document": d.page_content}).strip().upper()
            score = raw_score.replace(".", "").strip() 
            evaluations.append(score)
            if score == "RELEVANT" or score == "PARTIAL":
                relevant_docs.append(d)
        
        # 3. Decision logic
        if not relevant_docs:
            # Fallback: Nếu không có tài liệu nào liên quan, báo cho người dùng
            decision = "Tài liệu không phù hợp (Fallback)"
            context = "Hệ thống nhận thấy tài liệu trong kho kiến thức không chứa thông tin chính xác cho câu hỏi này."
            # Trong thực tế CRAG có thể gọi thêm Web Search ở đây.
        else:
            decision = f"Sử dụng {len(relevant_docs)}/{len(docs)} tài liệu liên quan"
            context = "\n".join([d.page_content for d in relevant_docs])
            
        # 4. Generate
        answer = self.rag_chain.invoke({"context": context, "question": question})
        
        return {
            "answer": answer,
            "decision": decision,
            "evaluations": evaluations,
            "docs_count": len(relevant_docs),
            "time": time.time() - start_time
        }

def main():
    runner = ComparisonRunner()
    
    test_questions = [
        # NHÓM 1: CÂU HỎI TRỰC TIẾP (Kiểm tra khả năng truy xuất chính xác)
        "Mr. BupBe khuyên bạn Tòng Thị Xuân Th. ở Sơn La làm gì khi bạn của anh họ xin số điện thoại của mình thay vì cô bạn đi cùng?",
        "Lý do tại sao tác giả lại lấy bút danh là Mr. BupBe?",
        "Mr. BupBe giải thích thế nào về việc 'dao sắc không gọt được chuôi' trong chuyện tư vấn tình cảm?",
        "Theo tài liệu, khi nào thì Mr. BupBe cảm thấy mình bị 'bó tay' trước các câu hỏi của bạn đọc?",
        "Mr. BupBe nghĩ gì về việc một bạn gái bị bạn trai viết vào nhật ký là ăn mặc luộm thuộm?"

        # NHÓM 2: CÂU HỎI SUY LUẬN/TỔNG HỢP (Kiểm tra khả năng hiểu ngữ cảnh)
        "Phong cách tư vấn của Mr. BupBe đối với các vấn đề tình cảm của giới trẻ là hài hước hay nghiêm túc? Trích dẫn minh chứng.",
        "Tác giả phản ứng thế nào với những câu hỏi liên quan đến mối quan hệ với bố mẹ chồng hoặc bố mẹ vợ?",

        # NHÓM 3: CÂU HỎI GÂY NHIỄU (Kiểm tra khả năng lọc thông tin nhiễu)
        "Trong sách có nhắc đến sữa Milo không? Nếu có thì ngữ cảnh là gì?", 
        # (Lưu ý: RAG có thể nhầm đây là công thức nấu ăn, CRAG cần xác định đúng ngữ cảnh mỉa mai)

        # NHÓM 4: CÂU HỎI NGOÀI PHẠM VI (Kiểm tra khả năng từ chối/tìm kiếm ngoài của CRAG)
        "Làm thế nào để cài đặt card Wifi Intel BE200 trên Windows 11?",
        "Dự báo thời tiết tại Lạng Sơn vào ngày 24 tháng 4 năm 2026 như thế nào?",
        "Công thức toán học để tính độ phức tạp của thuật toán sắp xếp nhanh (Quick Sort) là gì?",
    ]
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}BẢNG SO SÁNH RAG VS CRAG{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    
    for i, q in enumerate(test_questions):
        print(f"{Colors.BLUE}Câu hỏi {i+1}: {q}{Colors.ENDC}")
        
        # Run Standard RAG
        print(f"  {Colors.YELLOW}[1] Đang chạy Standard RAG...{Colors.ENDC}")
        rag_res = runner.run_standard_rag(q)
        
        # Run CRAG
        print(f"  {Colors.YELLOW}[2] Đang chạy CRAG...{Colors.ENDC}")
        crag_res = runner.run_crag(q)
        
        # Output results
        print(f"\n  {Colors.BOLD}--- KẾT QUẢ ---{Colors.ENDC}")
        print(f"  {Colors.GREEN}Standard RAG:{Colors.ENDC}")
        print(f"    - Thời gian: {rag_res['time']:.2f}s")
        print(f"    - Trả lời: {rag_res['answer']}")
        
        print(f"\n  {Colors.GREEN}CRAG (Corrective RAG):{Colors.ENDC}")
        print(f"    - Thời gian: {crag_res['time']:.2f}s")
        print(f"    - Đánh giá: {crag_res['evaluations']}")
        print(f"    - Quyết định: {crag_res['decision']}")
        print(f"    - Trả lời: {crag_res['answer']}")
        
        print(f"\n{'-'*60}\n")

if __name__ == "__main__":
    main()
