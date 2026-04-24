# src/core/chain.py
from pathlib import Path
from src.utils.logger import logger

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from src.core.citation import extract_citation_data, format_citation


class RAGChainManager:
    def __init__(self, llm):
        self.llm = llm
        self.retriever = None
        self.basic_chain = None       # Chain không nhớ lịch sử (Basic RAG)
        self.conv_chain = None        # Chain có nhớ lịch sử (Conversational RAG)
        self.condense_chain = None    # Chain tóm tắt câu hỏi
        self.chat_history = []

        # --- Template cho Basic RAG (mỗi câu hỏi độc lập) ---
        self.basic_answer_template = """Sử dụng thông tin từ các tài liệu được cung cấp dưới đây để trả lời câu hỏi.
            Mỗi đoạn thông tin sẽ được đánh dấu NGUỒN (ví dụ [NGUỒN: file.pdf | Trang: 1]). 
            - Hãy dùng các nhãn NGUỒN này để trích dẫn hoặc phân biệt thông tin nếu chúng đến từ các tài liệu khác nhau.
            - Nếu có sự mâu thuẫn hay khác biệt giữa các tài liệu, hãy chỉ rõ tài liệu nào nói gì.
            - Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói thật thay vì bịa đặt.
            - Trả lời bằng tiếng Việt, rõ ràng và đầy đủ.

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
        self.conv_answer_template = """Sử dụng thông tin từ các tài liệu được cung cấp dưới đây để trả lời câu hỏi.
            Mỗi đoạn thông tin sẽ được đánh dấu NGUỒN (ví dụ [NGUỒN: file.pdf | Trang: 1]). 
            - Hãy dùng các nhãn NGUỒN này để trích dẫn hoặc phân biệt thông tin nếu chúng đến từ các tài liệu khác nhau.
            - Nếu có sự mâu thuẫn hay khác biệt giữa các tài liệu, hãy chỉ rõ tài liệu nào nói gì.
            - Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói thật thay vì bịa đặt.
            - Trả lời bằng tiếng Việt, rõ ràng và đầy đủ.

            Ngữ cảnh từ tài liệu:
            {context}

            Câu hỏi: {question}

            Câu trả lời:"""
        self.conv_answer_prompt = ChatPromptTemplate.from_template(self.conv_answer_template)

        # --- Template cho CRAG Evaluator ---
        self.crag_evaluator_template = """Bạn là một chuyên gia đánh giá độ liên quan giữa tài liệu và câu hỏi.
            Nhiệm vụ của bạn là đánh giá xem đoạn tài liệu dưới đây có chứa thông tin để trả lời câu hỏi hay không.
            
            Các mức độ đánh giá:
            - RELEVANT: Tài liệu chứa thông tin trực tiếp hoặc gián tiếp để trả lời câu hỏi.
            - PARTIAL: Tài liệu có liên quan một phần nhưng chưa đủ để trả lời hoàn toàn.
            - IRRELEVANT: Tài liệu hoàn toàn không liên quan đến câu hỏi.

            Chỉ trả lời DUY NHẤT một từ: RELEVANT, PARTIAL hoặc IRRELEVANT.

            Tài liệu:
            {context}

            Câu hỏi: {question}

            Đánh giá của bạn:"""
        self.crag_evaluator_prompt = ChatPromptTemplate.from_template(self.crag_evaluator_template)

    def _format_docs(self, docs):
        formatted_docs = []
        for doc in docs:
            metadata = dict(doc.metadata or {})
            file_name = metadata.get("file_name", metadata.get("filename", "Unknown Document"))
            page_number = metadata.get("page_number", metadata.get("page"))
            section = metadata.get("section")
            
            location = ""
            if page_number is not None:
                location = f" | Trang: {page_number}"
            elif section:
                location = f" | Mục: {section}"
                
            header = f"\n--- [NGUỒN: {file_name}{location}] ---"
            formatted_docs.append(f"{header}\n{doc.page_content}")
            
        return "\n".join(formatted_docs)

    def _format_chat_history(self):
        """Chuyển đổi lịch sử sang chuỗi text cho prompt tóm tắt."""
        buffer = ""
        for message in self.chat_history:
            if isinstance(message, HumanMessage):
                buffer += f"Người dùng: {message.content}\n"
            elif isinstance(message, AIMessage):
                buffer += f"AI: {message.content}\n"
        return buffer

    def _build_sources_from_docs(self, docs: list, crag_decisions: list = None) -> list[dict]:
        """Build stable, deduplicated source records from retrieved documents."""
        sources: list[dict] = []
        seen_citations: set[str] = set()

        for i, doc in enumerate(docs):
            meta = dict(doc.metadata or {})
            citation_data = extract_citation_data(doc)

            raw_file_name = (
                citation_data.get("file_name")
                or meta.get("filename")
                or meta.get("file_name")
                or meta.get("file")
                or meta.get("document_name")
                or meta.get("source")
            )

            file_name = Path(str(raw_file_name)).name if raw_file_name else "unknown_source"

            # Prepare citation data with resilient fallback so UI never gets None filename.
            citation_payload = {
                **citation_data,
                "file_name": file_name,
                "page_number": citation_data.get("page_number"),
                "section": citation_data.get("section"),
            }
            citation_label = format_citation(citation_payload)

            # Remove duplicate sources by citation text (same page/section + file).
            if citation_label in seen_citations:
                continue
            seen_citations.add(citation_label)

            page_value = citation_data.get("page_number")
            if page_value is None:
                page_value = meta.get("page_number", meta.get("page", "?"))

            # Lấy điểm số từ Reranker (nếu có, ưu tiên trường score của Flashrank)
            relevance_score = meta.get("relevance_score")
            if relevance_score is None:
                relevance_score = meta.get("score")
            
            source_item = {
                "content": doc.page_content,
                "snippet": doc.page_content,
                "file": file_name,
                "file_name": file_name,
                "page": page_value,
                "chunk_index": citation_data.get("chunk_index"),
                "citation": citation_label,
                "metadata": meta,
                "score": relevance_score,
            }
            
            # Gán quyết định CRAG nếu có
            if crag_decisions and i < len(crag_decisions):
                source_item["crag_decision"] = crag_decisions[i]
                
            sources.append(source_item)

        return sources

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

    def update_llm(self, llm):
        """Cập nhật LLM runtime và build lại chains hiện có."""
        self.llm = llm
        if self.retriever:
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
            return self._build_sources_from_docs(docs)
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

    def _prepare_query(self, question: str, conversational: bool) -> tuple[str, bool]:
        """Phân tích ngữ cảnh và tối ưu câu hỏi. Trả về (optimized_query, is_changed)."""
        logger.info(f"Chuẩn bị câu hỏi: '{question}' (Conversational: {conversational})")
        if conversational and self.chat_history and self.condense_chain:
            try:
                optimized = self.condense_chain.invoke(question)
                changed = optimized.strip().lower() != question.strip().lower()
                logger.info(f"Câu hỏi tối ưu: '{optimized}'")
                return optimized, changed
            except Exception as e:
                logger.error("Lỗi khi tối ưu câu hỏi bằng condense_chain", exc_info=True)
        return question, False

    def _retrieve_documents(self, query: str) -> tuple[list, list]:
        """Thực hiện tìm kiếm tài liệu."""
        logger.info(f"Bắt đầu retrieve tài liệu cho: '{query}'")
        try:
            docs = self.retriever.invoke(query)
            sources = self._build_sources_from_docs(docs)
            logger.info(f"Hoàn tất retrieve. Số lượng tài liệu: {len(docs)}")
            return docs, sources
        except Exception as e:
            logger.error("Lỗi trong quá trình truy hồi tài liệu", exc_info=True)
            return [], []

    def _build_prompt_and_generate(self, query: str, docs: list, conversational: bool):
        """Xây dựng context và khởi tạo chuỗi streaming."""
        context = self._format_docs(docs)
        prompt_template = self.conv_answer_prompt if (conversational and self.conv_chain) else self.basic_prompt
        streaming_chain = prompt_template | self.llm | StrOutputParser()
        logger.info("Bắt đầu sinh câu trả lời")
        return streaming_chain.stream({"context": context, "question": query})

    def stream_ask(self, question: str, conversational: bool = True, use_crag: bool = False):
        """
        Generator xử lý câu hỏi và yield các gói thông tin (status, analysis, sources, chunk).
        Giúp UI hiển thị quá trình xử lý và streaming văn bản.
        """
        import time
        if not self.chain:
            yield {"type": "error", "content": "Vui lòng tải tài liệu lên trước."}
            return

        overall_start = time.time()
        
        # 1. Phân tích ngữ cảnh hội thoại (nếu có lịch sử)
        yield {"type": "status", "content": "Đang phân tích ngữ cảnh hội thoại..."}
        query_for_retrieval, is_changed = self._prepare_query(question, conversational)
        
        if is_changed:
            yield {"type": "analysis", "content": f"🔍 **Câu hỏi tối ưu:** *{query_for_retrieval}*"}

        # 2. Truy xuất tài liệu
        yield {"type": "status", "content": "Đang tìm kiếm thông tin trong tài liệu..."}
        retrieval_start = time.time()
        docs, sources = self._retrieve_documents(query_for_retrieval)
        retrieval_latency = time.time() - retrieval_start
        
        yield {"type": "analysis", "content": f"⏱️ **Thời gian truy hồi:** {retrieval_latency:.2f}s ({len(docs)} documents)"}

        # 2b. Corrective RAG (CRAG) logic
        final_docs = docs
        if use_crag and docs:
            yield {"type": "status", "content": "Đang kiểm duyệt độ liên quan của tài liệu (CRAG)..."}
            eval_start = time.time()
            
            relevant_docs = []
            relevant_decisions = []
            decisions = []
            
            evaluator_chain = self.crag_evaluator_prompt | self.llm | StrOutputParser()
            
            # Đánh giá từng tài liệu (có thể chạy song song nếu muốn nhanh, nhưng ở đây chạy tuần tự cho ổn định)
            for i, doc in enumerate(docs):
                res = evaluator_chain.invoke({"context": doc.page_content, "question": query_for_retrieval})
                decision = res.strip().upper()
                decisions.append(decision)
                
                if "RELEVANT" in decision or "PARTIAL" in decision:
                    relevant_docs.append(doc)
                    relevant_decisions.append(decision)
            
            eval_latency = time.time() - eval_start
            
            # Ghi nhật ký quyết định CRAG
            decision_counts = {d: decisions.count(d) for d in set(decisions)}
            decision_summary = ", ".join([f"{k}: {v}" for k, v in decision_counts.items()])
            yield {"type": "analysis", "content": f"⚖️ **CRAG Eval ({eval_latency:.2f}s):** {decision_summary}"}
            
            if not relevant_docs:
                yield {"type": "analysis", "content": "⚠️ **Cảnh báo:** Không có tài liệu nào đủ độ liên quan. Hệ thống sẽ trả lời dựa trên kiến thức chung hoặc từ chối."}
                final_docs = []
                sources = []
            else:
                final_docs = relevant_docs
                # Chỉ hiển thị những nguồn mà CRAG đánh dấu là RELEVANT/PARTIAL
                sources = self._build_sources_from_docs(relevant_docs, crag_decisions=relevant_decisions)

        yield {"type": "sources", "content": sources}

        # 3. Tổng hợp câu trả lời
        if not sources:
            if use_crag and docs:
                yield {"type": "status", "content": "Tài liệu không khớp, đang trả lời bằng kiến thức hệ thống..."}
            else:
                yield {"type": "status", "content": "Không tìm thấy thông tin trực tiếp, đang trả lời dựa trên kiến thức chung..."}
        else:
            yield {"type": "status", "content": "Đang tổng hợp câu trả lời từ tài liệu..."}

        gen_start = time.time()
        stream_generator = self._build_prompt_and_generate(query_for_retrieval, final_docs, conversational)
        
        full_answer = ""
        for chunk in stream_generator:
            full_answer += chunk
            yield {"type": "chunk", "content": chunk}
            
        gen_latency = time.time() - gen_start
        logger.info(f"Hoàn thành xử lý. Total time: {time.time() - overall_start:.2f}s. Generation: {gen_latency:.2f}s")
        yield {"type": "analysis", "content": f"⏱️ **Thời gian sinh trả lời:** {gen_latency:.2f}s"}

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