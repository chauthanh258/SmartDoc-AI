import os
import config
from src.core.document_loader import load_pdf
from src.core.text_splitter import split_documents
from src.core.embeddings import get_embedding_model
from src.core.vectorstore import create_vectorstore
from src.core.llm import get_llm
from src.core.chain import create_rag_chain

def test_integration():
    print("--- Phase 0 Integration Test ---")
    
    # 1. Load PDF
    sample_path = os.path.join("data", "samples", "sample_vi.pdf")
    if not os.path.exists(sample_path):
        print(f"Error: {sample_path} not found.")
        return
    
    print(f"Loading {sample_path}...")
    docs = load_pdf(sample_path)
    print(f"Loaded {len(docs)} pages.")
    
    # 2. Split
    print("Splitting documents...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    
    # 3. Embed & Index
    print("Creating vectorstore...")
    embedding_model = get_embedding_model()
    vectorstore = create_vectorstore(chunks, embedding_model)
    
    # 4. Query
    print("Initializing RAG chain...")
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = create_rag_chain(llm, retriever)
    
    query = "Hệ thống này tên là gì?"
    print(f"Querying: {query}")
    
    answer = qa_chain.invoke(query)
    print("\n--- Response ---")
    print(answer)
    print("----------------")

if __name__ == "__main__":
    test_integration()

