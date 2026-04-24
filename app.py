# app.py
import os
import hashlib
import warnings
# ── TẮT WARNING __path__ từ transformers (rất phổ biến) ─────────────────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings(
    "ignore",
    message=r"Accessing `__path__` from .*",
    category=FutureWarning,
)
import streamlit as st
import config
from src.core.document_loader import load_multiple_documents
from src.core.text_splitter import split_documents, ALLOWED_CHUNK_SIZES, ALLOWED_CHUNK_OVERLAPS
from src.core.embeddings import get_embedding_model
from src.core.vectorstore import VectorStoreManager, build_metadata_filter
from src.core.chain import RAGChainManager
from src.core.llm import get_llm
from src.ui.sidebar import render_sidebar
from src.ui.chat_interface import render_chat_interface

# ── Cấu hình trang ──────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

# ── CSS tùy chỉnh (Primary: #007BFF, Secondary: #FFC107) ────────────────────
st.markdown("""
<style>
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 5px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FFC107;
        color: black;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stExpander {
        border-left: 4px solid #FFC107;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 SmartDoc AI - Hỗ trợ tài liệu thông minh")


def _init_runtime_llm_state():
    """Initialize runtime Ollama state once per Streamlit session."""
    shared_base_url = config.OLLAMA_PROXY_BASE_URL
    defaults = {
        "ollama_mode_effective": config.OLLAMA_MODE if config.OLLAMA_MODE in {"local", "cloud"} else "local",
        "ollama_local_base_url_effective": shared_base_url,
        "ollama_local_model_effective": config.OLLAMA_LOCAL_MODEL,
        "ollama_cloud_base_url_effective": shared_base_url,
        "ollama_cloud_model_effective": config.OLLAMA_CLOUD_MODEL,
        "ollama_api_key_effective": config.OLLAMA_API_KEY,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _resolve_runtime_llm_config() -> dict:
    """Return active runtime config for Ollama local/cloud mode."""
    shared_base_url = config.OLLAMA_PROXY_BASE_URL
    mode = st.session_state.get("ollama_mode_effective", "local")
    if mode == "cloud":
        base_url = shared_base_url
        model = st.session_state.get("ollama_cloud_model_effective", config.OLLAMA_CLOUD_MODEL)
        api_key = st.session_state.get("ollama_api_key_effective", config.OLLAMA_API_KEY)
    else:
        base_url = shared_base_url
        model = st.session_state.get("ollama_local_model_effective", config.OLLAMA_LOCAL_MODEL)
        api_key = ""

    return {
        "mode": mode,
        "base_url": (base_url or "").strip(),
        "model": (model or "").strip(),
        "api_key": (api_key or "").strip(),
        "temperature": config.TEMPERATURE,
    }


def _llm_signature(runtime_cfg: dict) -> str:
    """Generate a stable signature to detect runtime LLM config changes."""
    api_key_hash = ""
    if runtime_cfg["api_key"]:
        api_key_hash = hashlib.sha256(runtime_cfg["api_key"].encode("utf-8")).hexdigest()[:12]
    return "|".join(
        [
            runtime_cfg["mode"],
            runtime_cfg["base_url"],
            runtime_cfg["model"],
            str(runtime_cfg["temperature"]),
            api_key_hash,
        ]
    )

# ── Khởi tạo Session State ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_manager" not in st.session_state:
    st.session_state.rag_manager = None

_init_runtime_llm_state()
runtime_llm_cfg = _resolve_runtime_llm_config()
current_llm_signature = _llm_signature(runtime_llm_cfg)

# ── Khởi tạo các thành phần Core ────────────────────────────────────────────
embedding_model = get_embedding_model()
llm = get_llm(
    base_url=runtime_llm_cfg["base_url"],
    model=runtime_llm_cfg["model"],
    temperature=runtime_llm_cfg["temperature"],
    api_key=runtime_llm_cfg["api_key"],
)
vs_manager = VectorStoreManager(embedding_model)

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
#     # Tự động đồng bộ khóa tri thức cũ tránh xử lý lại
#     if vs_manager.vectorstore is None:
#         vs_manager.load_vectorstore()
#     if vs_manager.vectorstore is not None:
#         for doc in vs_manager.get_document_registry():
#             fname = doc.get("file_name") or doc.get("filename")
#             if fname:
#                 st.session_state.indexed_files.add(fname)

if st.session_state.rag_manager is None:
    st.session_state.rag_manager = RAGChainManager(llm)
    st.session_state.active_llm_signature = current_llm_signature
else:
    previous_signature = st.session_state.get("active_llm_signature", "")
    if previous_signature != current_llm_signature:
        st.session_state.rag_manager.update_llm(llm)
        st.session_state.active_llm_signature = current_llm_signature
        st.toast("Đã cập nhật model Ollama live", icon="✅")

vectorstore = vs_manager.load_vectorstore()
if vectorstore:
    st.session_state.rag_manager.update_retriever(vectorstore)

# ── Render Sidebar (truyền vs_manager để render document registry) ───────────
uploaded_files, selected_docs = render_sidebar(vs_manager)

# ── Xử lý các file được upload (Phase 3: hỗ trợ nhiều file) ─────────────────
if uploaded_files:
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    # Kiểm tra xem file này đã được index chưa
    existing_docs = vs_manager.get_document_registry()
    existing_docs = existing_docs or []
    uploaded_names = [f.name for f in uploaded_files]

    # Build set of already indexed filenames from registry
    indexed_names = set()
    for d in existing_docs:
        if d.get("filename"):
            indexed_names.add(d.get("filename"))
        if d.get("file_name"):
            indexed_names.add(d.get("file_name"))

    # Keep only files that are not yet indexed
    new_uploaded_files = [f for f in uploaded_files if f.name not in indexed_names]

    if new_uploaded_files:
        # Lưu tất cả file mới xuống disk
        saved_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)

        with st.status(f"Đang xử lý {len(uploaded_files)} tài liệu...", expanded=True) as status:
            # Thử tải vectorstore sẵn có từ disk
            vectorstore = vs_manager.load_vectorstore()

            # Đọc tất cả file mới cùng lúc bằng load_multiple_documents
            st.write(f"📖 Đọc {len(uploaded_files)} tài liệu...")
            docs = load_multiple_documents(saved_paths, skip_failed=True)
            st.write(f"✅ Đọc xong: {len(docs)} trang/section")

            # Lấy chunk settings từ session_state (clamp về whitelist)
            chunk_size = st.session_state.get("chunk_size", config.CHUNK_SIZE)
            chunk_overlap = st.session_state.get("chunk_overlap", config.CHUNK_OVERLAP)
            if chunk_size not in ALLOWED_CHUNK_SIZES:
                chunk_size = min(ALLOWED_CHUNK_SIZES, key=lambda x: abs(x - chunk_size))
            if chunk_overlap not in ALLOWED_CHUNK_OVERLAPS:
                chunk_overlap = min(ALLOWED_CHUNK_OVERLAPS, key=lambda x: abs(x - chunk_overlap))
            if chunk_overlap >= chunk_size:
                chunk_overlap = max(o for o in ALLOWED_CHUNK_OVERLAPS if o < chunk_size)

            st.write(f"✂️ Chunk size={chunk_size}, overlap={chunk_overlap}")
            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.write(f"✅ Tạo ra {len(chunks)} đoạn văn bản")

            # Thêm vào vectorstore (append nếu đã có, tạo mới nếu chưa có)
            st.write("🗂️ Cập nhật Vector Index...")
            if vectorstore is None:
                vectorstore = vs_manager.create_vectorstore(chunks)
            else:
                vs_manager.vectorstore = vectorstore
                vectorstore = vs_manager.add_documents(chunks)
            vs_manager.save_vectorstore()

            # Ensure manager has the in-memory reference and update the RAG retriever
            vs_manager.vectorstore = vectorstore
            if st.session_state.get("rag_manager"):
                try:
                    st.session_state.rag_manager.update_retriever(vectorstore)
                except Exception:
                    # Fallback to direct retriever update if necessary
                    try:
                        retr = vs_manager.get_retriever()
                        st.session_state.rag_manager.update_retriever_direct(retr)
                    except Exception:
                        pass

            # Đánh dấu các file đã index
            for f in uploaded_files:
                st.session_state.indexed_files.add(f.name)

            status.update(
                label=f"✅ Đã xử lý {len(uploaded_files)} tài liệu, {len(chunks)} đoạn!",
                state="complete",
                expanded=False,
            )
        st.rerun()
    else:
        for f in uploaded_files:
            st.info(f"Tài liệu `{f.name}` đã có sẵn trong kho tri thức.")

    # Đảm bảo vs_manager luôn có vectorstore (kể cả khi không có file mới)
    if vs_manager.vectorstore is None:
        vs_manager.load_vectorstore()

# ── Cập nhật Retriever theo filter và settings ───────────────────────────────
if vs_manager.vectorstore is not None:
    k = 7
    use_hybrid = st.session_state.get("hybrid_search", False)

    # Xây dựng metadata filter nếu người dùng đã chọn tài liệu cụ thể (Phase 3)
    metadata_filter = None
    if selected_docs:
        # Tạo filter OR: khớp bất kỳ file nào trong danh sách selected_docs
        def _make_multi_filter(filenames):
            def _filter(meta):
                fname = meta.get("filename") or meta.get("file_name") or ""
                return fname in filenames
            return _filter
        metadata_filter = _make_multi_filter(set(selected_docs))

    # Hỗ trợ metadata filter + hybrid search
    use_hybrid = st.session_state.get("hybrid_search", False)
    use_reranker = st.session_state.get("reranking", False)

    # Nếu dùng hybrid, ta truyền cả chunks và filter. 
    if use_hybrid:
        docstore_dict = getattr(vs_manager.vectorstore.docstore, "_dict", {})
        from langchain_core.documents import Document as LCDoc
        all_chunks = [
            doc for doc in docstore_dict.values() if isinstance(doc, LCDoc)
        ]
        # Gọi BM25 retriever được tối ưu (đang được fix bên trong get_hybrid_retriever)
        retriever = vs_manager.get_hybrid_retriever(all_chunks, k=k*2 if use_reranker else k, metadata_filter=metadata_filter)
    else:
        retriever = vs_manager.get_retriever(k=k*2 if use_reranker else k, metadata_filter=metadata_filter)

    # Apply reranker if enabled
    if use_reranker and retriever:
        retriever = vs_manager.get_reranker_retriever(retriever, k=k)

    if retriever:
        st.session_state.rag_manager.update_retriever_direct(retriever)

# ── Giao diện Chat ───────────────────────────────────────────────────────────
if st.session_state.rag_manager and st.session_state.rag_manager.chain:
    render_chat_interface(st.session_state.rag_manager)
else:
    st.info("Vui lòng tải lên tài liệu PDF hoặc DOCX ở thanh bên để bắt đầu trò chuyện.")
