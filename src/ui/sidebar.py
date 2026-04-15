# src/ui/sidebar.py
import streamlit as st
import os
import config
import shutil
from src.ui.components import render_settings_panel
from src.utils.chat_history import get_chat_history, clear_chat_history

def render_sidebar():
    with st.sidebar:
        st.header("📄 SmartDoc AI")
        st.markdown(
            """
            **Hướng dẫn sử dụng:**
            1. Tải lên một tệp PDF hoặc DOCX.
            2. Hệ thống sẽ tự động xử lý tài liệu.
            3. Trò chuyện ở cửa sổ bên cạnh.
            """
        )
        st.divider()

        # --- Trạng thái hệ thống ---
        st.subheader("🖥️ Trạng thái")
        st.info(f"LLM: `{config.LLM_MODEL}`")
        st.info(f"Embeddings: `{config.EMBEDDING_MODEL.split('/')[-1]}`")

        st.divider()

        # --- Tải tài liệu ---
        st.subheader("📂 Tài liệu")
        uploaded_file = st.file_uploader(
            "Tải lên tài liệu mới",
            type=["pdf", "docx"],
            help="Chỉ hỗ trợ định dạng PDF và DOCX."
        )

        st.divider()

        # --- Panel cài đặt nâng cao (Phase 2 - Yêu cầu 4) ---
        render_settings_panel()

        st.divider()

        # --- Bộ nhớ hội thoại ---
        st.subheader("💬 Lịch sử hội thoại")
        history = get_chat_history()
        if not history:
            st.caption("Chưa có lịch sử hội thoại.")
        else:
            num_msgs = len(history)
            st.caption(f"Có **{num_msgs}** câu hỏi trong phiên này.")
            with st.container(height=180):
                for i, entry in enumerate(history):
                    label = entry["question"][:45] + "..." if len(entry["question"]) > 45 else entry["question"]
                    st.markdown(f"👤 {label}", help=entry["question"])

        st.divider()

        # --- Trình quản lý ---
        st.subheader("🛠️ Trình quản lý")

        # 1. Xóa lịch sử hội thoại (có confirm)
        if "confirm_clear_chat" not in st.session_state:
            st.session_state.confirm_clear_chat = False

        if st.button("🗑️ Xóa Lịch sử Hội thoại", use_container_width=True):
            st.session_state.confirm_clear_chat = True

        if st.session_state.confirm_clear_chat:
            st.warning("Bạn có chắc muốn xóa toàn bộ lịch sử chat?")
            col1, col2 = st.columns(2)
            if col1.button("✅ Có, Xóa", key="yes_chat"):
                clear_chat_history()
                # Xóa cả chat_history trong rag_manager nếu có
                if st.session_state.get("rag_manager"):
                    st.session_state.rag_manager.clear_history()
                st.session_state.confirm_clear_chat = False
                st.rerun()
            if col2.button("❌ Hủy", key="no_chat"):
                st.session_state.confirm_clear_chat = False
                st.rerun()

        # 2. Xóa Vector Store (có confirm)
        if "confirm_clear_vs" not in st.session_state:
            st.session_state.confirm_clear_vs = False

        if st.button("⚡ Xóa Vector Store", use_container_width=True):
            st.session_state.confirm_clear_vs = True

        if st.session_state.confirm_clear_vs:
            st.warning("Xóa dữ liệu đã nhúng? Bạn phải tải tài liệu lên lại.")
            c1, c2 = st.columns(2)
            if c1.button("✅ Đồng ý", key="yes_vs"):
                vs_path = os.path.join(config.VECTORSTORE_DIR, "faiss_index")
                if os.path.exists(vs_path):
                    shutil.rmtree(vs_path)
                st.session_state.rag_manager = None
                clear_chat_history() # Xóa luôn chat khi xóa VS
                st.session_state.confirm_clear_vs = False
                st.success("Đã xóa dữ liệu Vector Store!")
                st.rerun()
            if c2.button("❌ Hủy bỏ", key="no_vs"):
                st.session_state.confirm_clear_vs = False
                st.rerun()

        return uploaded_file

