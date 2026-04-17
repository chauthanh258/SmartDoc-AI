# app.py
import streamlit as st
import os
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

# ── Khởi tạo Session State ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_manager" not in st.session_state:
    st.session_state.rag_manager = None
if "indexed_files" not in st.session_state:
    # Theo dõi những file đã được index để tránh tạo lại index khi rerun
    st.session_state.indexed_files = set()

# ── Khởi tạo các thành phần Core ────────────────────────────────────────────
embedding_model = get_embedding_model()
llm = get_llm()
vs_manager = VectorStoreManager(embedding_model)

if st.session_state.rag_manager is None:
    st.session_state.rag_manager = RAGChainManager(llm)

# ── Render Sidebar (truyền vs_manager để render document registry) ───────────
uploaded_files, selected_docs = render_sidebar(vs_manager=vs_manager)

# ── Xử lý các file được upload (Phase 3: hỗ trợ nhiều file) ─────────────────
if uploaded_files:
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    # Chỉ xử lý các file chưa được index trong phiên này
    new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]

    if new_files:
        # Lưu tất cả file mới xuống disk
        saved_paths = []
        for uploaded_file in new_files:
            file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)

        with st.status(f"Đang xử lý {len(new_files)} tài liệu...", expanded=True) as status:
            # Thử tải vectorstore sẵn có từ disk
            vectorstore = vs_manager.load_vectorstore()

            # Đọc tất cả file mới cùng lúc bằng load_multiple_documents
            st.write(f"📖 Đọc {len(new_files)} tài liệu...")
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

            # Đánh dấu các file đã index
            for f in new_files:
                st.session_state.indexed_files.add(f.name)

            status.update(
                label=f"✅ Đã xử lý {len(new_files)} tài liệu, {len(chunks)} đoạn!",
                state="complete",
                expanded=False,
            )

    # Đảm bảo vs_manager luôn có vectorstore (kể cả khi không có file mới)
    if vs_manager.vectorstore is None:
        vs_manager.load_vectorstore()

# ── Cập nhật Retriever theo filter và settings ───────────────────────────────
if vs_manager.vectorstore is not None:
    k = 3
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

    # Chọn retriever: hybrid search hoặc standard, áp dụng filter
    if use_hybrid:
        # Lấy chunks để BM25 (cần text đã được chunk)
        # Trường hợp hybrid với filter: fallback về semantic + filter
        if metadata_filter:
            retriever = vs_manager.get_retriever(k=k, metadata_filter=metadata_filter)
        else:
            # Để hybrid search, cần có chunks - lấy từ docstore
            docstore_dict = getattr(vs_manager.vectorstore.docstore, "_dict", {})
            from langchain_core.documents import Document as LCDoc
            all_chunks = [
                doc for doc in docstore_dict.values() if isinstance(doc, LCDoc)
            ]
            retriever = vs_manager.get_hybrid_retriever(all_chunks, k=k)
    else:
        retriever = vs_manager.get_retriever(k=k, metadata_filter=metadata_filter)

    if retriever:
        st.session_state.rag_manager.update_retriever_direct(retriever)

# ── Giao diện Chat ───────────────────────────────────────────────────────────
if st.session_state.rag_manager and st.session_state.rag_manager.chain:
    render_chat_interface(st.session_state.rag_manager)
else:
    st.info("Vui lòng tải lên tài liệu PDF hoặc DOCX ở thanh bên để bắt đầu trò chuyện.")
