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
OLLAMA_PROXY_BASE_URL = os.getenv("OLLAMA_PROXY_BASE_URL", OLLAMA_BASE_URL)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:0.6b")
OLLAMA_MODE = os.getenv("OLLAMA_MODE", "local")
OLLAMA_LOCAL_BASE_URL = os.getenv("OLLAMA_LOCAL_BASE_URL", OLLAMA_PROXY_BASE_URL)
OLLAMA_LOCAL_MODEL = os.getenv("OLLAMA_LOCAL_MODEL", LLM_MODEL)
OLLAMA_CLOUD_BASE_URL = os.getenv("OLLAMA_CLOUD_BASE_URL", OLLAMA_PROXY_BASE_URL)
OLLAMA_CLOUD_MODEL = os.getenv("OLLAMA_CLOUD_MODEL", LLM_MODEL)
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# RAG Configuration
CHUNK_SIZE = 600
CHUNK_OVERLAP = 300
TEMPERATURE = 0.1

# Ensure directories exist
for directory in [UPLOAD_DIR, SAMPLES_DIR, VECTORSTORE_DIR, LOGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
