from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def format_docs(docs):
    """Formats multiple documents into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(llm, retriever):
    """Creates a modern RAG chain using LCEL (LangChain Expression Language)."""
    template = """Sử dụng thông tin sau đây để trả lời câu hỏi của người dùng. 
Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố gắng bịa ra câu trả lời.
Trả lời bằng tiếng Việt.

Thông tin ngữ cảnh:
{context}

Câu hỏi: {question}

Câu trả lời hữu ích:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Building the LCEL chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

