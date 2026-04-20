import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from src.core.vectorstore import VectorStoreManager, build_metadata_filter


class DummyEmbeddings(Embeddings):
    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed_documents(self, texts):
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.1] * self.dim

@pytest.fixture
def mock_embedding_model():
    return DummyEmbeddings(dim=16)

@pytest.fixture
def vector_manager(mock_embedding_model):
    manager = VectorStoreManager(mock_embedding_model, folder_name="dummy_test")
    docs = [
        Document(page_content="Content 1", metadata={"filename": "doc1.pdf", "doc_id": "1"}),
        Document(page_content="Content 2", metadata={"filename": "doc2.pdf", "doc_id": "2"}),
        Document(page_content="Content 3", metadata={"filename": "doc1.pdf", "doc_id": "1"})
    ]
    manager.create_vectorstore(docs)
    return manager

def test_metadata_filter_builder():
    filter_func = build_metadata_filter(filename="doc1.pdf")
    assert filter_func is not None
    assert filter_func({"filename": "doc1.pdf"}) == True
    assert filter_func({"filename": "doc2.pdf"}) == False

def test_retrieval_with_filter(vector_manager):
    # Retrieve only docs with filename doc1.pdf
    filter_func = build_metadata_filter(filename="doc1.pdf")
    results = vector_manager.similarity_search("content", k=5, metadata_filter=filter_func)
    assert len(results) == 2
    for doc in results:
        assert doc.metadata["filename"] == "doc1.pdf"

def test_hybrid_retriever_creation(vector_manager):
    docs = [
        Document(page_content="Content 1", metadata={"filename": "doc1.pdf"}),
        Document(page_content="Content 2", metadata={"filename": "doc2.pdf"})
    ]
    hybrid = vector_manager.get_hybrid_retriever(docs, k=2)
    assert hybrid is not None

def test_reranker_retriever_creation(vector_manager):
    base_retriever = vector_manager.get_retriever(k=2)
    reranker = vector_manager.get_reranker_retriever(base_retriever, k=1)
    assert reranker is not None
