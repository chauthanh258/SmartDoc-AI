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
from src.utils.chat_history import init_chat_history, get_chat_history
from src.core.database import DatabaseManager
import time

# --- Cấu hình trang ---
st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

# --- CSS tùy chỉnh (Giữ nguyên CSS của bạn) ---
st.markdown("""
<style>
    .stButton>button { background-color: #007BFF; color: white; border-radius: 5px; border: none; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #FFC107; color: black; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .stExpander { border-left: 4px solid #FFC107; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("📄 SmartDoc AI - Hỗ trợ tài liệu thông minh")

# --- 1. Khởi tạo Database & Session ID ---
if "db" not in st.session_state:
    st.session_state.db = DatabaseManager()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"session_{int(time.time())}"

# --- 2. Khởi tạo Chat History ---
init_chat_history()

# --- 3. Khởi tạo Core Components ---
embedding_model = get_embedding_model()
llm = get_llm()
vs_manager = VectorStoreManager(embedding_model)

if st.session_state.get("rag_manager") is None:
    st.session_state.rag_manager = RAGChainManager(llm)

# --- 4. TỰ ĐỘNG NẠP KIẾN THỨC CŨ (Nếu có) ---
# Kiểm tra nếu RAG chưa có chain nhưng ổ đĩa có FAISS thì nạp luôn
if st.session_state.rag_manager.chain is None:
    vectorstore = vs_manager.load_vectorstore()
    if vectorstore:
        st.session_state.rag_manager.update_retriever(vectorstore)

# --- 5. Render Sidebar ---
uploaded_file = render_sidebar()

# --- 6. Xử lý tài liệu được upload (Cộng dồn kiến thức) ---
if uploaded_file is not None:
    file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.status("Đang cập nhật kiến thức mới...", expanded=True) as status:
        try:
            docs = load_document(file_path)
            if docs:
                chunk_size = st.session_state.get("chunk_size", config.CHUNK_SIZE)
                chunk_overlap = st.session_state.get("chunk_overlap", config.CHUNK_OVERLAP)
                
                chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                # Sử dụng add_documents để cộng dồn vào FAISS hiện tại
                vectorstore = vs_manager.add_documents(chunks)
                vs_manager.save_vectorstore()
                
                # Cập nhật lại bộ não AI
                st.session_state.rag_manager.update_retriever(vectorstore)
                status.update(label=f"✅ Đã thêm {len(chunks)} đoạn kiến thức mới!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Lỗi xử lý tài liệu: {str(e)}")

# --- 7. Giao diện Chat ---
history = get_chat_history()
has_history = len(history) > 0
rag_ready = (
    st.session_state.rag_manager is not None and 
    st.session_state.rag_manager.chain is not None
)

if rag_ready or has_history:
    render_chat_interface(st.session_state.rag_manager)
    
    if not rag_ready:
        st.warning("⚠️ Thư viện kiến thức đang trống. Hãy tải lên tài liệu để AI có thể trả lời câu hỏi mới.")
else:
    st.info("Chào mừng bạn! Hãy tải lên tài liệu ở thanh bên để bắt đầu huấn luyện AI của riêng bạn.")