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

# --- Luôn đồng bộ Vector Store sẵn có vào RAG Manager ---
vectorstore = vs_manager.load_vectorstore()
if vectorstore:
    st.session_state.rag_manager.update_retriever(vectorstore)

# --- Render Sidebar (bao gồm file uploader + settings + document management) ---
uploaded_file = render_sidebar(vs_manager)

# --- Xử lý tài liệu được upload ---
if uploaded_file is not None:
    file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Kiểm tra xem file này đã được index chưa
    existing_docs = vs_manager.get_document_registry()
    is_already_indexed = any(d.get("filename") == uploaded_file.name for d in existing_docs)

    if not is_already_indexed:
        with st.status("Đang phân tích tài liệu mới...", expanded=True) as status:
            try:
                docs = load_document(file_path)
                if docs:
                    chunk_size = st.session_state.get("chunk_size", config.CHUNK_SIZE)
                    chunk_overlap = st.session_state.get("chunk_overlap", config.CHUNK_OVERLAP)
                    
                    st.caption(f"Đang chia nhỏ tài liệu: Chunk size {chunk_size}, Overlap {chunk_overlap}")
                    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    
                    # Thêm vào vectorstore hiện có (hoặc tạo mới nếu chưa có)
                    vectorstore = vs_manager.add_documents(chunks)
                    vs_manager.save_vectorstore()
                    
                    # Cập nhật lại retriever cho RAG
                    st.session_state.rag_manager.update_retriever(vectorstore)
                    status.update(label=f"✅ Đã thêm {uploaded_file.name} vào kho tri thức!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Lỗi khi xử lý tài liệu: {str(e)}")
    else:
        st.info(f"ℹ️ Tài liệu `{uploaded_file.name}` đã có sẵn trong kho tri thức.")

# --- Giao diện Chat ---
if st.session_state.rag_manager and st.session_state.rag_manager.chain:
    render_chat_interface(st.session_state.rag_manager)
else:
    st.info("Vui lòng tải lên tài liệu PDF hoặc DOCX ở thanh bên để bắt đầu trò chuyện.")

