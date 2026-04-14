# src/ui/sidebar.py
import streamlit as st
import os
import config
import shutil

def render_sidebar():
    with st.sidebar:
        st.header("Cấu hình & Hướng dẫn")
        st.markdown(
            """
            **Hướng dẫn sử dụng:**
            1. Tải lên một tệp PDF hoặc DOCX.
            2. Hệ thống sẽ tự động xử lý.
            3. Trò chuyện ở cửa sổ bên cạnh.
            """
        )
        st.divider()
        
        st.subheader("Trạng thái")
        st.info(f"LLM: {config.LLM_MODEL}")
        st.info(f"Embeddings: {config.EMBEDDING_MODEL.split('/')[-1]}")
        
        st.divider()
        # Updated uploader to handle docx
        uploaded_file = st.file_uploader("Tải lên tài liệu mới", type=["pdf", "docx"])
        
        st.divider()
        st.subheader("Bộ nhớ hội thoại")
        
        # Display history preview
        if not st.session_state.messages:
            st.caption("Chưa có lịch sử hội thoại.")
        else:
            with st.container(height=200):
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f"👤 **Bạn:** {msg['content'][:40]}...", help=msg["content"])
        
        st.divider()
        st.subheader("Trình Quản lý")
        
        # 1. Clear History Button with Confirmation
        if "confirm_clear_chat" not in st.session_state:
            st.session_state.confirm_clear_chat = False
            
        if st.button("🗑️ Xóa Lịch sử Hội thoại", use_container_width=True):
            st.session_state.confirm_clear_chat = True
            
        if st.session_state.confirm_clear_chat:
            st.warning("Bạn có chắc muốn xóa sạch lịch sử chat hiện tại?")
            col1, col2 = st.columns(2)
            if col1.button("Có, Xóa", key="yes_chat"):
                st.session_state.messages = []
                st.session_state.confirm_clear_chat = False
                st.rerun()
            if col2.button("Hủy", key="no_chat"):
                st.session_state.confirm_clear_chat = False
                st.rerun()

        # 2. Clear Vector Store Button with Confirmation
        if "confirm_clear_vs" not in st.session_state:
            st.session_state.confirm_clear_vs = False
            
        if st.button("⚡ Xóa Vector Store", use_container_width=True):
            st.session_state.confirm_clear_vs = True
            
        if st.session_state.confirm_clear_vs:
            st.warning("Chắc chắn xóa tài liệu đã nhúng? Bạn sẽ phải tải lên lại.")
            c1, c2 = st.columns(2)
            if c1.button("Đồng ý", key="yes_vs"):
                vs_path = os.path.join(config.VECTORSTORE_DIR, "faiss_index")
                if os.path.exists(vs_path):
                    shutil.rmtree(vs_path)
                st.session_state.vectorstore = None
                st.session_state.qa_chain = None
                st.session_state.messages = []
                st.session_state.confirm_clear_vs = False
                st.success("Đã xóa dữ liệu nền!")
                st.rerun()
            if c2.button("Hủy bỏ", key="no_vs"):
                st.session_state.confirm_clear_vs = False
                st.rerun()
                
        return uploaded_file
