import pytest
import os
import sys
from unittest.mock import MagicMock
from langchain_core.documents import Document

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.text_splitter import split_documents
from src.core.citation import format_document_citation
from src.core.chain import RAGChainManager

# --- Test Case 1: Dynamic Chunking ---
def test_dynamic_chunking():
    doc = Document(page_content="A " * 1000, metadata={"source": "test.txt"})
    
    # Test with size 500, overlap 0 (now allowed)
    chunks_500 = split_documents([doc], chunk_size=500, chunk_overlap=0)
    assert len(chunks_500) >= 4
    
    # Test with size 1500
    chunks_1500 = split_documents([doc], chunk_size=1500, chunk_overlap=0)
    assert len(chunks_1500) < len(chunks_500)

# --- Test Case 2: Citation Formatting ---
def test_citation_formatting():
    # PDF mock
    pdf_doc = Document(
        page_content="content", 
        metadata={"file_name": "manual.pdf", "page_number": 5}
    )
    assert format_document_citation(pdf_doc) == "[Trang 5 - manual.pdf]"
    
    # DOCX mock
    docx_doc = Document(
        page_content="content", 
        metadata={"file_name": "report.docx", "section": "Introduction"}
    )
    assert format_document_citation(docx_doc) == "[Muc Introduction - report.docx]"

# --- Test Case 3: Conversational Memory (Simplified Mock) ---
def test_conversational_memory():
    # Mock LLM and Chains to avoid Pydantic validation issues with real LangChain objects in mock state
    mock_llm = MagicMock()
    manager = RAGChainManager(mock_llm)
    manager.retriever = MagicMock()
    
    # Mock the internal chains directly to isolate history logic
    manager.conv_chain = MagicMock()
    manager.conv_chain.invoke.side_effect = ["AI Answer 1", "AI Answer 2"]
    
    manager.basic_chain = MagicMock()
    manager.basic_chain.invoke.return_value = "Basic Answer"
    
    # Mock _extract_sources to avoid retriever issues
    manager._extract_sources = MagicMock(return_value=[{"citation": "source1"}])
    
    # First question
    ans1, sources = manager.ask("Who is Einstein?", conversational=True)
    assert ans1 == "AI Answer 1"
    assert len(manager.chat_history) == 2
    
    # Second question
    ans2, sources = manager.ask("What did he do?", conversational=True)
    assert ans2 == "AI Answer 2"
    assert len(manager.chat_history) == 4
    
    assert manager.chat_history[0].content == "Who is Einstein?"
    assert manager.chat_history[1].content == "AI Answer 1"

