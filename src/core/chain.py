# src/core/chain.py
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from src.core.citation import format_document_citation


class RAGChainManager:
    def __init__(self, llm):
        self.llm = llm
        self.retriever = None
        self.basic_chain = None       # Chain không nhớ lịch sử (Basic RAG)
        self.conv_chain = None        # Chain có nhớ lịch sử (Conversational RAG)
        self.condense_chain = None    # Chain tóm tắt câu hỏi
        self.chat_history = []

        # --- Template cho Basic RAG (mỗi câu hỏi độc lập) ---
        self.basic_answer_template = """Sử dụng thông tin dưới đây để trả lời câu hỏi.
            Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói thật thay vì bịa đặt.
            Trả lời bằng tiếng Việt, rõ ràng và đầy đủ.

            Ngữ cảnh từ tài liệu:
            {context}

            Câu hỏi: {question}

            Câu trả lời:"""
        self.basic_prompt = ChatPromptTemplate.from_template(self.basic_answer_template)

        # --- Template tóm tắt lịch sử để viết lại câu hỏi (Conversational RAG) ---
        self.condense_template = """Dựa trên lịch sử hội thoại và câu hỏi mới, 
            hãy viết lại câu hỏi thành một câu hoàn chỉnh, độc lập (không cần xem lại lịch sử).
            Nếu câu hỏi đã rõ ràng, hãy giữ nguyên.

            Lịch sử hội thoại:
            {chat_history}

            Câu hỏi mới: {question}
            Câu hỏi độc lập:"""

        self.condense_prompt = ChatPromptTemplate.from_template(self.condense_template)

        # --- Template trả lời cuối (Conversational RAG) ---
        self.conv_answer_template = """Sử dụng thông tin dưới đây để trả lời câu hỏi.
            Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói thật thay vì bịa đặt.
            Trả lời bằng tiếng Việt, rõ ràng và đầy đủ.

            Ngữ cảnh từ tài liệu:
            {context}

            Câu hỏi: {question}

            Câu trả lời:"""
        self.conv_answer_prompt = ChatPromptTemplate.from_template(self.conv_answer_template)

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _format_chat_history(self):
        """Chuyển đổi lịch sử sang chuỗi text cho prompt tóm tắt."""
        buffer = ""
        for message in self.chat_history:
            if isinstance(message, HumanMessage):
                buffer += f"Người dùng: {message.content}\n"
            elif isinstance(message, AIMessage):
                buffer += f"AI: {message.content}\n"
        return buffer

    def update_retriever(self, vectorstore, k: int = 3):
        """Cập nhật retriever và xây dựng lại cả 2 chains."""
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self._build_basic_chain()
        self._build_conv_chain()

    def update_retriever_direct(self, retriever):
        """Nhận thẳng retriever object (đã được filter/hybrid) từ bên ngoài."""
        self.retriever = retriever
        self._build_basic_chain() 
        self._build_conv_chain()

    def _build_basic_chain(self):
        """Xây dựng Basic RAG chain - không có memory."""
        if not self.retriever:
            return
        self.basic_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.basic_prompt
            | self.llm
            | StrOutputParser()
        )

    def _build_conv_chain(self):
        """Xây dựng Conversational RAG chain - có memory."""
        if not self.retriever:
            return

        self.condense_chain = (
            {
                "chat_history": lambda x: self._format_chat_history(),
                "question": RunnablePassthrough()
            }
            | self.condense_prompt
            | self.llm
            | StrOutputParser()
        )

        self.conv_chain = (
            {
                "context": self.condense_chain | self.retriever | self._format_docs,
                "question": self.condense_chain
            }
            | self.conv_answer_prompt
            | self.llm
            | StrOutputParser()
        )

    def _extract_sources(self, question: str) -> list:
        """Trích xuất danh sách nguồn (source documents) cho câu hỏi (Yêu cầu 5)."""
        if not self.retriever:
            return []
        try:
            docs = self.retriever.invoke(question)
            sources = []
            for doc in docs:
                meta = doc.metadata or {}
                sources.append({
                    "content": doc.page_content,
                    "file": meta.get("file_name", meta.get("source", "Tài liệu")),
                    "page": meta.get("page_number", meta.get("page", "?")),
                    "citation": format_document_citation(doc)
                })
            return sources
        except Exception:
            return []

    def ask(self, question: str, conversational: bool = True) -> tuple[str, list]:
        """
        Xử lý câu hỏi và trả về (answer, sources).
        - conversational=True: Dùng Conversational RAG (nhớ lịch sử)
        - conversational=False: Dùng Basic RAG (mỗi câu hỏi độc lập)
        """
        # Sử dụng stream_ask và gộp lại kết quả (để giữ tương thích ngược)
        full_answer = ""
        sources = []
        for packet in self.stream_ask(question, conversational):
            if packet["type"] == "chunk":
                full_answer += packet["content"]
            elif packet["type"] == "sources":
                sources = packet["content"]
            elif packet["type"] == "error":
                return packet["content"], []
        
        return full_answer, sources

    def stream_ask(self, question: str, conversational: bool = True):
        """
        Generator xử lý câu hỏi và yield các gói thông tin (status, analysis, sources, chunk).
        Giúp UI hiển thị quá trình xử lý và streaming văn bản.
        """
        if not self.chain:
            yield {"type": "error", "content": "Vui lòng tải tài liệu lên trước."}
            return

        query_for_retrieval = question
        
        # 1. Phân tích ngữ cảnh hội thoại (nếu có lịch sử)
        if conversational and self.chat_history:
            yield {"type": "status", "content": "Đang phân tích ngữ cảnh hội thoại..."}
            if self.condense_chain:
                query_for_retrieval = self.condense_chain.invoke(question)
                if query_for_retrieval.strip().lower() != question.strip().lower():
                    yield {"type": "analysis", "content": f"🔍 **Câu hỏi tối ưu:** *{query_for_retrieval}*"}

        # 2. Truy xuất tài liệu
        yield {"type": "status", "content": "Đang tìm kiếm thông tin trong tài liệu..."}
        docs = self.retriever.invoke(query_for_retrieval)
        sources = []
        for doc in docs:
            meta = doc.metadata or {}
            sources.append({
                "content": doc.page_content,
                "file": meta.get("file_name", meta.get("source", "Tài liệu")),
                "page": meta.get("page_number", meta.get("page", "?")),
                "citation": format_document_citation(doc)
            })
        
        yield {"type": "sources", "content": sources}

        # 3. Tổng hợp câu trả lời
        if not sources:
            yield {"type": "status", "content": "Không tìm thấy thông tin trực tiếp, đang trả lời dựa trên kiến thức chung..."}
        else:
            yield {"type": "status", "content": "Đang tổng hợp câu trả lời từ tài liệu..."}

        context = self._format_docs(docs)
        prompt_template = self.conv_answer_prompt if (conversational and self.conv_chain) else self.basic_prompt
        
        # Khởi tạo streaming chain
        streaming_chain = prompt_template | self.llm | StrOutputParser()
        
        full_answer = ""
        for chunk in streaming_chain.stream({"context": context, "question": query_for_retrieval}):
            full_answer += chunk
            yield {"type": "chunk", "content": chunk}

        # 4. Cập nhật lịch sử
        if conversational:
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=full_answer))
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]

    def invoke(self, question: str):
        """Compatibility wrapper: trả về chỉ answer (không trả sources)."""
        answer, _ = self.ask(question, conversational=True)
        return answer

    def clear_history(self):
        """Xóa lịch sử hội thoại."""
        self.chat_history = []

    @property
    def chain(self):
        """Trả về chain đang dùng (basic hoặc conv), dùng để kiểm tra chain đã sẵn sàng chưa."""
        return self.conv_chain or self.basic_chain