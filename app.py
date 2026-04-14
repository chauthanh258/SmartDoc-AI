# app.py
import streamlit as st
import os
import config
from src.core.document_loader import load_pdf
from src.core.text_splitter import split_documents
from src.core.embeddings import get_embedding_model
from src.core.vectorstore import create_vectorstore, load_vectorstore, save_vectorstore
from src.core.llm import get_llm
from src.core.chain import create_rag_chain
from src.ui.sidebar import render_sidebar
from src.ui.chat_interface import render_chat_interface

# Set page config
st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

# Apply custom colors (Primary: #007BFF, Secondary: #FFC107)
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
        background-color: #f8f9fa;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 SmartDoc AI - Hỗ trợ tài liệu thông minh")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# 1. Render Sidebar
uploaded_file = render_sidebar()

# 2. Process Document
embedding_model = get_embedding_model()

if uploaded_file is not None:
    file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Try to load existing local vectorstore
    vectorstore = load_vectorstore(embedding_model)
    
    if vectorstore is None:
        with st.status("Thiết lập hệ thống RAG...", expanded=True) as status:
            with st.spinner(f"Đang đọc tài liệu {uploaded_file.name}..."):
                # Integrate with new DOCX loader alongside PDF
                if file_path.lower().endswith('.pdf'):
                    docs = load_pdf(file_path)
                elif file_path.lower().endswith('.docx'):
                    # Using standard langchain loader for docx
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
                    vectorstore = create_vectorstore(chunks, embedding_model)
                    save_vectorstore(vectorstore)
                status.update(label="Xử lý tài liệu hoàn tất!", state="complete", expanded=False)
            
    st.session_state.vectorstore = vectorstore
    
    if st.session_state.qa_chain is None and vectorstore is not None:
        llm = get_llm()
        # Request returning source documents for citation phase
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = create_rag_chain(llm, retriever)
        st.session_state.qa_chain = qa_chain

# 3. Render Chat Interface
if st.session_state.qa_chain is not None:
    render_chat_interface(st.session_state.qa_chain)
else:
    st.info("Vui lòng tải lên một file PDF hoặc DOCX ở thanh bên (Sidebar) để bắt đầu sử dụng.")
