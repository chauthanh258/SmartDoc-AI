# scratch/performance_test.py
import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.core.document_loader import load_document
from src.core.text_splitter import split_documents
from src.core.embeddings import get_embedding_model
from src.core.vectorstore import VectorStoreManager
from src.core.chain import RAGChainManager
from src.core.llm import get_llm

def run_performance_test():
    sample_path = r"e:\VSCode\PhatTrienPhanMemMaNguonMo\SmartDoc-AI\data\samples\sample_test.docx"
    if not os.path.exists(sample_path):
        print(f"Error: Sample file not found at {sample_path}")
        return

    questions = [
        "SmartDoc AI là gì?",
        "Làm thế nào để tải tài liệu lên?",
        "Hệ thống hỗ trợ những định dạng file nào?"
    ]
    
    chunk_sizes = [500, 1000, 1500]
    overlap = 100
    
    results = []
    
    print("Loading core models...")
    embedding_model = get_embedding_model()
    llm = get_llm()
    vs_manager = VectorStoreManager(embedding_model)
    rag_manager = RAGChainManager(llm)
    
    docs = load_document(sample_path)
    
    for size in chunk_sizes:
        print(f"\n--- Testing Chunk Size: {size} ---")
        
        # Measure processing time
        start_proc = time.time()
        chunks = split_documents(docs, chunk_size=size, chunk_overlap=overlap)
        vectorstore = vs_manager.create_vectorstore(chunks)
        proc_time = time.time() - start_proc
        
        rag_manager.update_retriever(vectorstore)
        
        qa_results = []
        total_qa_time = 0
        
        for q in questions:
            start_qa = time.time()
            answer, sources = rag_manager.ask(q, conversational=False)
            qa_time = time.time() - start_qa
            total_qa_time += qa_time
            
            qa_results.append({
                "question": q,
                "answer": answer[:100] + "...", # Truncate for report
                "time": qa_time,
                "num_sources": len(sources)
            })
            print(f"Q: {q} | Time: {qa_time:.2f}s")

        results.append({
            "chunk_size": size,
            "proc_time": proc_time,
            "avg_qa_time": total_qa_time / len(questions),
            "num_chunks": len(chunks),
            "qa_details": qa_results
        })

    # Save results
    output_file = Path("scratch/performance_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nResults saved to {output_file}")
    
    # Generate Markdown Table for easier reading
    print("\n| Chunk Size | Num Chunks | Proc Time (s) | Avg QA Time (s) |")
    print("|------------|------------|---------------|-----------------|")
    for r in results:
        print(f"| {r['chunk_size']} | {r['num_chunks']} | {r['proc_time']:.2f} | {r['avg_qa_time']:.2f} |")

if __name__ == "__main__":
    run_performance_test()
