# app.py
import streamlit as st
import os
import config
from src.core.document_loader import load_document
from src.core.text_splitter import split_documents
from src.core.embeddings import get_embedding_model
from src.core.vectorstore import VectorStoreManager
from src.core.chain import RAGChainManager
from src.core.llm import get_llm
from src.ui.sidebar import render_sidebar
from src.ui.chat_interface import render_chat_interface
from src.utils.chat_history import init_chat_history

# --- Cấu hình trang ---
st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

# --- CSS tùy chỉnh (Primary: #007BFF, Secondary: #FFC107) ---
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

# --- Khởi tạo Session State & Chat History ---
init_chat_history()
if "rag_manager" not in st.session_state:
    st.session_state.rag_manager = None

# --- Khởi tạo các thành phần Core ---
embedding_model = get_embedding_model()
llm = get_llm()
vs_manager = VectorStoreManager(embedding_model)

if st.session_state.rag_manager is None:
    st.session_state.rag_manager = RAGChainManager(llm)

# --- Render Sidebar (bao gồm file uploader + settings) ---
uploaded_file = render_sidebar()

# --- Xử lý tài liệu được upload ---
if uploaded_file is not None:
    file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Thử tải vectorstore sẵn có từ disk
    vectorstore = vs_manager.load_vectorstore()

    # Nếu chưa có, xây dựng mới từ tài liệu vừa upload
    if vectorstore is None:
        with st.status("Đang xử lý tài liệu...", expanded=True) as status:
            # Đọc tài liệu sử dụng bộ nạp tập trung
            try:
                docs = load_document(file_path)
            except Exception as e:
                st.error(f"Lỗi tải tài liệu: {str(e)}")
                docs = []

            if docs:
                # Lấy chunk settings từ session_state (được set bởi components.py)
                chunk_size = st.session_state.get("chunk_size", config.CHUNK_SIZE)
                chunk_overlap = st.session_state.get("chunk_overlap", config.CHUNK_OVERLAP)

                st.caption(f"Chunk size: **{chunk_size}** | Overlap: **{chunk_overlap}**")

                chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                vectorstore = vs_manager.create_vectorstore(chunks)
                vs_manager.save_vectorstore()
                status.update(label=f"✅ Đã xử lý {len(chunks)} đoạn văn bản!", state="complete", expanded=False)


    # Cập nhật retriever cho RAG chain
    if vectorstore:
        st.session_state.rag_manager.update_retriever(vectorstore)

# --- Giao diện Chat ---
if st.session_state.rag_manager and st.session_state.rag_manager.chain:
    render_chat_interface(st.session_state.rag_manager)
else:
    st.info("Vui lòng tải lên tài liệu PDF hoặc DOCX ở thanh bên để bắt đầu trò chuyện.")

