import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Model Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TEMPERATURE = 0.1

# Ensure directories exist
for directory in [UPLOAD_DIR, SAMPLES_DIR, VECTORSTORE_DIR, LOGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
