import streamlit as st
import os
import config
from src.core.document_loader import load_pdf
from src.core.text_splitter import split_documents
from src.core.embeddings import get_embedding_model
from src.core.vectorstore import create_vectorstore, load_vectorstore, save_vectorstore
from src.core.llm import get_llm
from src.core.chain import create_rag_chain

st.set_page_config(page_title="SmartDoc AI", layout="wide")

st.title("📄 SmartDoc AI - Hỗ trợ tài liệu thông minh (RAG)")

# Sidebar for configuration and status
with st.sidebar:
    st.header("Cấu hình & Trạng thái")
    st.info(f"LLM: {config.LLM_MODEL}")
    st.info(f"Embeddings: {config.EMBEDDING_MODEL.split('/')[-1]}")
    
    uploaded_file = st.file_uploader("Tải lên tài liệu PDF", type=["pdf"])
    
    if st.button("Xóa dữ liệu Vectorstore"):
        if os.path.exists(os.path.join(config.VECTORSTORE_DIR, "faiss_index")):
            import shutil
            shutil.rmtree(os.path.join(config.VECTORSTORE_DIR, "faiss_index"))
            st.success("Đã xóa dữ liệu cũ!")
            st.rerun()

# Logic to process and query
embedding_model = get_embedding_model()

if uploaded_file is not None:
    # Save uploaded file
    file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Check if we need to rebuild vectorstore
    vectorstore = load_vectorstore(embedding_model)
    
    if vectorstore is None:
        with st.status("Đang xử lý tài liệu...", expanded=True) as status:
            st.write("Đang đọc PDF...")
            docs = load_pdf(file_path)
            st.write(f"Đã đọc {len(docs)} trang.")
            
            st.write("Đang chia nhỏ văn bản...")
            chunks = split_documents(docs)
            st.write(f"Tạo ra {len(chunks)} đoạn hội thoại.")
            
            st.write("Đang tạo vector index...")
            vectorstore = create_vectorstore(chunks, embedding_model)
            save_vectorstore(vectorstore)
            status.update(label="Xử lý tài liệu hoàn tất!", state="complete", expanded=False)
    
    # Initialize LLM and RAG Chain
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = create_rag_chain(llm, retriever)
    
    # Chat interface
    st.header("Trò chuyện với tài liệu")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hỏi gì đó về tài liệu này?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                answer = qa_chain.invoke(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.write("Vui lòng tải lên một file PDF ở thanh bên để bắt đầu.")
