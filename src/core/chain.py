from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

class RAGChainManager:
    def __init__(self, llm):
        self.llm = llm
        self.retriever = None
        self.chain = None
        # Khởi tạo bộ nhớ lịch sử (Memory)
        self.chat_history = []
        
        # 1. Template để tóm tắt lịch sử và viết lại câu hỏi (Xử lý follow-up questions)
        self.condense_template = """Dựa trên lịch sử hội thoại và câu hỏi mới nhất, 
hãy tạo một câu hỏi độc lập có thể hiểu được mà không cần xem lại lịch sử.
Nếu câu hỏi đã rõ ràng, hãy giữ nguyên nó.

Lịch sử hội thoại:
{chat_history}

Câu hỏi mới: {question}
Câu hỏi độc lập:"""
        self.condense_prompt = ChatPromptTemplate.from_template(self.condense_template)

        # 2. Template trả lời cuối cùng (Theo yêu cầu Task 5 nhưng có thêm Memory)
        self.answer_template = """Sử dụng thông tin sau đây để trả lời câu hỏi của người dùng.  
Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố gắng bịa ra câu trả lời.
Trả lời bằng tiếng Việt.

Thông tin ngữ cảnh:
{context}

Câu hỏi: {question}

Câu trả lời hữu ích:"""
        self.answer_prompt = ChatPromptTemplate.from_template(self.answer_template)

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _format_chat_history(self, chat_history):
        """Chuyển đổi danh sách tin nhắn thành văn bản để đưa vào prompt tóm tắt."""
        buffer = ""
        for message in chat_history:
            if isinstance(message, HumanMessage):
                buffer += f"Người dùng: {message.content}\n"
            elif isinstance(message, AIMessage):
                buffer += f"AI: {message.content}\n"
        return buffer

    def update_retriever(self, vectorstore):
        """Cập nhật retriever và build lại chain hỗ trợ Memory (Task 3)"""
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Bước 1: Tạo câu hỏi độc lập từ lịch sử
        condense_chain = (
            {
                "chat_history": lambda x: self._format_chat_history(self.chat_history),
                "question": RunnablePassthrough()
            }
            | self.condense_prompt 
            | self.llm 
            | StrOutputParser()
        )

        # Bước 2: Dùng câu hỏi đã độc lập để tìm vector và trả lời
        self.chain = (
            {
                "context": condense_chain | self.retriever | self._format_docs,
                "question": condense_chain # Sử dụng câu hỏi đã được re-phrase
            }
            | self.answer_prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str):
        """Xử lý câu hỏi, trả lời và cập nhật Memory."""
        if not self.chain:
            return "Vui lòng tải tài liệu lên trước khi đặt câu hỏi."
        
        # Gọi chain
        response = self.chain.invoke(question)
        
        # Lưu vào lịch sử (Memory)
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response))
        
        # Giới hạn lịch sử để tránh quá tải (giữ 5 cặp câu hỏi-trả lời gần nhất)
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
            
        return response

    def invoke(self, question: str):
        return self.ask(question)

    def clear_history(self):
        """Xóa lịch sử khi cần (ví dụ khi đổi tài liệu mới)"""
        self.chat_history = []