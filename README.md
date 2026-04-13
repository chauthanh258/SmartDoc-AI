# SmartDoc AI - RAG Solution

SmartDoc AI is a powerful Retrieval-Augmented Generation (RAG) system built with Streamlit, LangChain, and Ollama. It allows users to upload PDF documents and ask questions in natural language, receiving accurate answers supported by the document's content.

## Features
- **PDF Document Loading**: Supports multiple PDF uploads.
- **Efficient Chunking**: Advanced text splitting strategies.
- **Local Embedding & Vector Search**: Fast and secure local retrieval using FAISS.
- **Ollama Integration**: Powered by state-of-the-art local LLMs (default: `qwen2.5:7b`).
- **Conversational UI**: User-friendly chat interface with Streamlit.

## Installation Steps

### 1. Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com/).

### 2. Pull the LLM Model
Open your terminal and run:
```bash
ollama pull qwen2.5:7b
```

### 3. Setup Project Environment
Clone the repository and install the dependencies:
```bash
# Create a virtual environment (optional)
#On Linux / macOS:
python -m venv myenv 

#On Windows:
py -m venv myenv

# Activate the virtual environment
# On Linux / macOS:
source myenv/bin/activate

# On Windows:
myenv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

#### HF Hub token (optional but recommended)
If you use Hugging Face-hosted models or `sentence-transformers`, setting a `HF_TOKEN` avoids unauthenticated rate limits and speeds up downloads. Example commands:

PowerShell:
```powershell
setx HF_TOKEN "your_token_here"
$env:HF_TOKEN = "your_token_here"
```

Windows CMD:
```cmd
setx HF_TOKEN "your_token_here"
```

Linux / macOS:
```bash
export HF_TOKEN="your_token_here"
```

If you see a LangChain deprecation warning mentioning `HuggingFaceEmbeddings`, run:
```bash
pip install -U langchain-huggingface
```

### 4. Run the Application
Start the Streamlit app:
```bash
streamlit run app.py
```

## Directory Structure
- `app.py`: Main entry point.
- `config.py`: Central configuration.
- `src/`: Source code.
  - `core/`: RAG logic (Document Loader, Embeddings, LLM, etc.).
  - `ui/`: Streamlit UI components.
  - `services/`: High-level business logic.
  - `utils/`: Helper functions and logging.
- `data/`: Sample and uploaded documents.
- `vectorstore/`: Local FAISS index storage.

## License
MIT License