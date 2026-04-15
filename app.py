import streamlit as st
import os
import config
from src.core.document_loader import load_pdf
from src.core.text_splitter import split_documents
from src.core.embeddings import get_embedding_model
from src.core.vectorstore import VectorStoreManager
from src.core.llm import get_llm
# Import class RAGChainManager mới của bạn
from src.core.chain import RAGChainManager
from src.ui.sidebar import render_sidebar
from src.ui.chat_interface import render_chat_interface

# Set page config
st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

# CSS giữ nguyên
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

# 1. Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Khởi tạo Chain Manager trong session state nếu chưa có
if "chain_manager" not in st.session_state:
    llm = get_llm()
    st.session_state.chain_manager = RAGChainManager(llm)

uploaded_file = render_sidebar()
embedding_model = get_embedding_model()
vs_manager = VectorStoreManager(embedding_model)

# 2. Xử lý tài liệu
if uploaded_file is not None:
    file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
    
    # Lưu file tạm thời
    if not os.path.exists(config.UPLOAD_DIR):
        os.makedirs(config.UPLOAD_DIR)
        
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Thử load vectorstore cũ hoặc tạo mới
    vectorstore = vs_manager.load_vectorstore()
    
    if vectorstore is None:
        with st.status("Thiết lập hệ thống RAG...", expanded=True) as status:
            with st.spinner(f"Đang đọc tài liệu {uploaded_file.name}..."):
                if file_path.lower().endswith('.pdf'):
                    docs = load_pdf(file_path)
                elif file_path.lower().endswith('.docx'):
                    from langchain_community.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                else:
                    docs = []
                    st.error("Định dạng chưa được hỗ trợ.")
            
            if docs:
                with st.spinner("Đang phân tách văn bản..."):
                    chunks = split_documents(docs)
                with st.spinner("Đang tạo Vector Index..."):
                    vectorstore = vs_manager.create_vectorstore(chunks)
                    vs_manager.save_vectorstore()
                status.update(label="Xử lý tài liệu hoàn tất!", state="complete", expanded=False)
    
    # 3. Cập nhật Retriever cho Chain Manager
    if vectorstore is not None:
        # Gọi method update_retriever của bạn để build chain LCEL
        st.session_state.chain_manager.update_retriever(vectorstore)

# 4. Render Chat Interface
# Kiểm tra xem chain đã được build thành công chưa thông qua thuộc tính .chain của class
if st.session_state.chain_manager.chain is not None:
    # Truyền toàn bộ manager vào chat interface
    render_chat_interface(st.session_state.chain_manager)
else:
    st.info("Vui lòng tải lên một file PDF hoặc DOCX ở thanh bên để bắt đầu.")