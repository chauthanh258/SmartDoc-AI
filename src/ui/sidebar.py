# src/ui/sidebar.py
import streamlit as st
import os
import config
import shutil
from src.ui.components import render_settings_panel, render_document_filter


def render_sidebar(vs_manager=None):
    """
    Render toàn bộ sidebar.
    Phase 1: Instructions, Status, File uploader, History, Manager.
    Phase 2: Settings panel (chunk, conversational mode).
    Phase 3: Multi-file uploader, Document filter.
    Trả về (uploaded_files, selected_docs).
    """
    with st.sidebar:
        st.header("📄 SmartDoc AI")
        st.markdown(
            """
            **Hướng dẫn sử dụng:**
            1. Tải lên một hoặc nhiều tệp PDF/DOCX.
            2. Hệ thống tự động xử lý và lập chỉ mục.
            3. Chọn tài liệu cần truy vấn (hoặc để tìm tất cả).
            4. Đặt câu hỏi ở cửa sổ bên cạnh.
            """
        )
        st.divider()

        # ── Trạng thái hệ thống ────────────────────────────────────────────
        st.subheader("🖥️ Trạng thái")
        st.info(f"LLM: `{config.LLM_MODEL}`")
        st.info(f"Embeddings: `{config.EMBEDDING_MODEL.split('/')[-1]}`")

        st.divider()

        # ── Tải tài liệu (Phase 3: multiple=True) ─────────────────────────
        st.subheader("📂 Tài liệu")
        uploaded_files = st.file_uploader(
            "Tải lên tài liệu (PDF/DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Hỗ trợ nhiều file cùng lúc. Tối đa 200MB mỗi file.",
            key="file_uploader",
        )

        if uploaded_files:
            st.caption(f"Đã chọn **{len(uploaded_files)}** tệp:")
            for f in uploaded_files:
                size_mb = f.size / (1024 * 1024)
                st.caption(f"  📄 {f.name} ({size_mb:.1f} MB)")

        st.divider()

        # ── Cài đặt nâng cao (Phase 2 + 3) ────────────────────────────────
        render_settings_panel()

        st.divider()

        # ── Bộ lọc tài liệu (Phase 3) ─────────────────────────────────────
        selected_docs = render_document_filter(vs_manager)

        st.divider()

        # ── Lịch sử hội thoại ─────────────────────────────────────────────
        st.subheader("💬 Lịch sử hội thoại")
        user_msgs = [m for m in st.session_state.get("messages", []) if m["role"] == "user"]
        if not user_msgs:
            st.caption("Chưa có lịch sử hội thoại.")
        else:
            st.caption(f"Có **{len(user_msgs)}** câu hỏi trong phiên này.")
            with st.container(height=180):
                for msg in user_msgs:
                    label = msg["content"][:45] + "..." if len(msg["content"]) > 45 else msg["content"]
                    st.markdown(f"👤 {label}", help=msg["content"])

        st.divider()

        # ── Trình quản lý ──────────────────────────────────────────────────
        st.subheader("🛠️ Trình quản lý")

        # Nút 1: Xóa lịch sử hội thoại (có confirm)
        if "confirm_clear_chat" not in st.session_state:
            st.session_state.confirm_clear_chat = False

        if st.button("🗑️ Xóa Lịch sử Hội thoại", use_container_width=True):
            st.session_state.confirm_clear_chat = True

        if st.session_state.confirm_clear_chat:
            st.warning("Bạn có chắc muốn xóa toàn bộ lịch sử chat?")
            col1, col2 = st.columns(2)
            if col1.button("✅ Có, Xóa", key="yes_chat"):
                st.session_state.messages = []
                if st.session_state.get("rag_manager"):
                    st.session_state.rag_manager.clear_history()
                st.session_state.confirm_clear_chat = False
                st.rerun()
            if col2.button("❌ Hủy", key="no_chat"):
                st.session_state.confirm_clear_chat = False
                st.rerun()

        # Nút 2: Xóa Vector Store (có confirm)
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
                st.session_state.messages = []
                st.session_state.confirm_clear_vs = False
                st.success("Đã xóa dữ liệu Vector Store!")
                st.rerun()
            if c2.button("❌ Hủy bỏ", key="no_vs"):
                st.session_state.confirm_clear_vs = False
                st.rerun()

        return uploaded_files, selected_docs
