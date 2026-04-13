from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(file_path):
    """Loads a PDF document using PyMuPDF."""
    loader = PyMuPDFLoader(file_path)
    return loader.load()
