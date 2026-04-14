import streamlit as st
import os
import config
from src.core.document_loader import load_pdf
from src.core.text_splitter import split_documents
from src.core.embeddings import get_embedding_model
from src.core.vectorstore import VectorStoreManager
from src.core.chain import RAGChainManager
from src.core.llm import get_llm
from src.ui.sidebar import render_sidebar
from src.ui.chat_interface import render_chat_interface

# Cấu hình trang
st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

# CSS tùy chỉnh
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
</style>
""", unsafe_allow_html=True)

st.title("📄 SmartDoc AI - Hỗ trợ tài liệu thông minh")

# Khởi tạo Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_manager" not in st.session_state:
    st.session_state.rag_manager = None

# Khởi tạo các thành phần Core
embedding_model = get_embedding_model()
llm = get_llm()

# Khởi tạo Managers
vs_manager = VectorStoreManager(embedding_model)

if st.session_state.rag_manager is None:
    st.session_state.rag_manager = RAGChainManager(llm)

# 1. Render Sidebar
uploaded_file = render_sidebar()

# 2. Xử lý tài liệu
if uploaded_file is not None:
    file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Thử load vectorstore cũ
    vectorstore = vs_manager.load()
    
    # Nếu chưa có index, tiến hành tạo mới
    if vectorstore is None:
        with st.status("Đang xử lý tài liệu...", expanded=True) as status:
            if file_path.lower().endswith('.pdf'):
                docs = load_pdf(file_path)
            elif file_path.lower().endswith('.docx'):
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            else:
                docs = []
                st.error("Định dạng không hỗ trợ.")
            
            if docs:
                chunks = split_documents(docs)
                vectorstore = vs_manager.create_vectorstore(chunks)
                status.update(label="Đã sẵn sàng!", state="complete", expanded=False)
    
    # Cập nhật Retriever cho Chain ngay khi có vectorstore
    if vectorstore:
        st.session_state.rag_manager.update_retriever(vectorstore)

# 3. Giao diện Chat
if st.session_state.rag_manager and st.session_state.rag_manager.chain:
    render_chat_interface(st.session_state.rag_manager)
else:
    st.info("Vui lòng tải lên tài liệu để bắt đầu trò chuyện.")