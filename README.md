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
```bash
smartdoc-ai-rag/
├── app.py                          # ← File chính chạy Streamlit (entry point)
├── config.py                       # Cấu hình chung (model name, chunk_size, temperature...)
├── requirements.txt                # Dependencies
├── .env                            # (tùy chọn) lưu API key nếu sau này cần
├── .gitignore
├── README.md                       # Hướng dẫn chạy project
├── LICENSE                         # (tùy chọn)

├── data/                           # Tài liệu mẫu + tài liệu user upload
│   ├── samples/                    # PDF mẫu để test (gutenberg.pdf, test_vi.pdf...)
│   └── uploaded/                   # (gitignored) thư mục lưu file user upload

├── vectorstore/                    # FAISS index (gitignored)
│   └── faiss_index/                # Thư mục lưu index tự động

├── src/                            # ← Tất cả code nguồn (quan trọng nhất)
│   ├── __init__.py
│   │
│   ├── core/                       # Logic cốt lõi RAG
│   │   ├── __init__.py
│   │   ├── document_loader.py      # PDF + DOCX loader (Yêu cầu 1)
│   │   ├── text_splitter.py        # Chunk strategy + tùy chỉnh (Yêu cầu 4)
│   │   ├── embeddings.py           # HuggingFace embeddings
│   │   ├── vectorstore.py          # FAISS wrapper
│   │   ├── retriever.py            # Hybrid search + Re-ranking (Yêu cầu 7,9)
│   │   ├── llm.py                  # Ollama + Prompt templates
│   │   ├── chain.py                # RAG Chain (basic + conversational)
│   │   ├── memory.py               # Conversational memory (Yêu cầu 6)
│   │   ├── citation.py             # Citation & source tracking (Yêu cầu 5)
│   │   └── self_rag.py             # Self-RAG (Yêu cầu 10 - optional)
│   │
│   ├── ui/                         # Giao diện Streamlit
│   │   ├── __init__.py
│   │   ├── sidebar.py              # Sidebar, history, settings
│   │   ├── chat_interface.py       # Khu vực chat + hiển thị response
│   │   ├── components.py           # Các component tái sử dụng (upload, button...)
│   │   └── styles.py               # CSS custom
│   │
│   ├── services/                   # Service layer (cao cấp)
│   │   ├── __init__.py
│   │   ├── rag_service.py          # Orchestrator chính (xử lý upload + query)
│   │   └── multi_document.py       # Multi-document + metadata filtering (Yêu cầu 8)
│   │
│   └── utils/                      # Tiện ích chung
│       ├── __init__.py
│       ├── helpers.py
│       └── logger.py

├── tests/                          # Test cases (thành viên D phụ trách)
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_chunking.py
│   ├── test_rag.py
│   └── test_performance.py

├── docs/                           # Tài liệu dự án
│   ├── project_report.pdf          # Báo cáo cuối cùng
│   ├── project_report.tex          # LaTeX (nếu dùng)
│   ├── screenshots/                # Ảnh chụp màn hình
│   ├── demo_video.mp4              # Video demo
│   └── architecture.md             # Mô tả kiến trúc hệ thống

├── notebooks/                      # Jupyter notebooks (dùng để thử nghiệm)
│   ├── 01_chunk_experiment.ipynb   # Thử chunk_size & overlap
│   └── 02_reranking_test.ipynb

└── logs/                           # (gitignored) file log nếu cần
```
## License
MIT License

## Phase 2 Completed Features

- **Dynamic Chunking Strategy**: 
    - Sidebar support for selecting `chunk_size` (500, 1000, 1500, 2000) and `chunk_overlap` (0, 50, 100, 200).
    - Validation whitelist ensuring safe and efficient processing.
- **Professional Citations**:
    - Automatic formatting of sources: `[Trang X - file.pdf]` or `[Muc X - file.docx]`.
    - Integrated with `RAGChainManager` for consistent reporting.
- **Conversational RAG (Multi-turn)**:
    - Context-aware answers using conversational memory (keeps last 5 turns).
    - Toggle-able via UI settings.
- **Performance Optimization**:
    - Benchmark results collected for different configurations to ensure optimal defaults.

## Performance Benchmark

Test Document: `sample_test.docx` (~2.5k characters)
Model: `qwen2.5:7b`

| Chunk Size | Num Chunks | Proc Time (s) | Avg QA Time (s) |
|------------|------------|---------------|-----------------|
| 500        | 3          | 0.08          | 41.68           |
| 1000       | 3          | 0.07          | 38.34           |
| 1500       | 3          | 0.03          | 38.51           |

*Note: QA Time includes retrieval + LLM synthesis. Processing time (Proc) is faster for larger chunks as expected.*

## How to Test

### 1. Automated Tests (Unit Tests)
Ensure all dependencies are installed, then run the test suites:

```bash
# Install additional dependencies
pip install pytest python-docx docx2txt

# Run Phase 1 Tests
pytest tests/test_phase1.py -v

# Run Phase 2 Tests (Dynamic Chunking, Citations, Conversational Logic)
pytest tests/test_phase2.py -v
```

### 2. Manual Verification
1. Run: `streamlit run app.py`.
2. Upload a document and open "⚙️ Cài đặt nâng cao".
3. Change **Chunk Size** to **500** and observe processing status.
4. Ask "Ai là người sáng lập ra SmartDoc AI?".
5. Ask a follow-up: "Họ làm điều đó khi nào?" (Verify it understands "Họ" refers to the founders).
6. Click "Xem nguồn trích dẫn" and verify the citation format (e.g., `[Muc ... - file.docx]`).
