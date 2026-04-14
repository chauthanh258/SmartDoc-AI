from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class RAGChainManager:
    def __init__(self, llm):
        self.llm = llm
        self.retriever = None
        self.chain = None
        
        # Template cố định theo yêu cầu Task 5
        self.template = """Sử dụng thông tin sau đây để trả lời câu hỏi của người dùng.  
Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố gắng bịa ra câu trả lời.
Trả lời bằng tiếng Việt.

Thông tin ngữ cảnh:
{context}

Câu hỏi: {question}

Câu trả lời hữu ích:"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def _format_docs(self, docs):
        """Helper để format context."""
        return "\n\n".join(doc.page_content for doc in docs)

    def update_retriever(self, vectorstore):
        """Cập nhật retriever mới khi có tài liệu mới được load (Task 3)"""
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Build lại chain với retriever mới
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str):
        """Hàm gọi chính để trả lời câu hỏi"""
        if not self.chain:
            return "Vui lòng tải tài liệu lên trước khi đặt câu hỏi."
        return self.chain.invoke(question)