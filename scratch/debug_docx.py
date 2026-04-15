from langchain_community.document_loaders import Docx2txtLoader
import os

path = r"e:\VSCode\PhatTrienPhanMemMaNguonMo\SmartDoc-AI\data\samples\sample_test.docx"
try:
    loader = Docx2txtLoader(path)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    print(f"Content snippet: {docs[0].page_content[:100]}")
except Exception as e:
    import traceback
    traceback.print_exc()
