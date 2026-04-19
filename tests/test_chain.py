import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.core.chain import RAGChainManager

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    # For stream returns we often use generator
    def fake_stream(*args, **kwargs):
        yield "Fake "
        yield "answer"
    llm.stream.side_effect = fake_stream
    llm.invoke.return_value = "Optimized query"
    return llm

@pytest.fixture
def rag_manager(mock_llm):
    manager = RAGChainManager(mock_llm)
    
    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="context piece 1", metadata={"filename": "file1.pdf"}),
        Document(page_content="context piece 2", metadata={"filename": "file2.pdf"})
    ]
    manager.update_retriever_direct(mock_retriever)
    return manager

def test_format_docs(rag_manager):
    docs = [
        Document(page_content="text1", metadata={"filename": "a.pdf", "page": 1}),
        Document(page_content="text2", metadata={"filename": "b.pdf", "page_number": 5})
    ]
    formatted = rag_manager._format_docs(docs)
    assert "--- [NGUỒN: a.pdf | Trang: 1] ---" in formatted
    assert "text1" in formatted
    assert "--- [NGUỒN: b.pdf | Trang: 5] ---" in formatted
    assert "text2" in formatted

def test_build_sources_dedup(rag_manager):
    docs = [
        Document(page_content="chunk 0", metadata={"filename": "doc.pdf", "page_number": 1}),
        # duplicate citation label since it's same page & same file
        Document(page_content="chunk 1", metadata={"filename": "doc.pdf", "page_number": 1}),
        Document(page_content="chunk 2", metadata={"filename": "doc.pdf", "page_number": 2}),
    ]
    sources = rag_manager._build_sources_from_docs(docs)
    # Deduplication by exact same citation label (doc.pdf - Page 1) shouldn't produce duplicate source labels logically,
    # Actually wait: The dedup is purely by the label generated!
    # Let's count them
    labels = [s["citation"] for s in sources]
    assert len(set(labels)) == len(labels) # should be unique

def test_prepare_query_condensable(rag_manager):
    rag_manager.chat_history = [{"role": "user", "content": "hi"}] # Dummy
    rag_manager.condense_chain = MagicMock()
    rag_manager.condense_chain.invoke.return_value = "What is X?"
    
    query, changed = rag_manager._prepare_query("it", conversational=True)
    assert changed is True
    assert query == "What is X?"
