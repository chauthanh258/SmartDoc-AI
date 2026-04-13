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
    vectorstore.save_local(path)

def load_vectorstore(embedding_model, folder_name="faiss_index"):
    """Loads the vectorstore from disk."""
    path = os.path.join(config.VECTORSTORE_DIR, folder_name)
    if os.path.exists(path):
        return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    return None
