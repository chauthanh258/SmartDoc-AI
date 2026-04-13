from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def split_documents(documents: list[Document]) -> list[Document]:
    """Split LangChain Document objects into FAISS-ready chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
    )
    return text_splitter.split_documents(documents)
