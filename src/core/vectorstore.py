from langchain_community.vectorstores import FAISS
import os
import config

def create_vectorstore(chunks, embedding_model):
    """Creates a FAISS vectorstore from text chunks."""
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore

def save_vectorstore(vectorstore, folder_name="faiss_index"):
    """Saves the vectorstore to disk."""
    path = os.path.join(config.VECTORSTORE_DIR, folder_name)
    os.makedirs(path, exist_ok=True)
    
    # Sử dụng đường dẫn tương đối để tránh lỗi FAISS C++ với ký tự tiếng Việt (Unicode) trên Windows
    rel_path = os.path.relpath(path, start=os.getcwd())
    vectorstore.save_local(rel_path)

def load_vectorstore(embedding_model, folder_name="faiss_index"):
    """Loads the vectorstore from disk."""
    path = os.path.join(config.VECTORSTORE_DIR, folder_name)
    # Check if the actual index file exists before attempting to load
    if os.path.exists(os.path.join(path, "index.faiss")):
        # Sử dụng đường dẫn tương đối để tránh lỗi FAISS C++ với ký tự tiếng Việt
        rel_path = os.path.relpath(path, start=os.getcwd())
        return FAISS.load_local(rel_path, embedding_model, allow_dangerous_deserialization=True)
    return None
