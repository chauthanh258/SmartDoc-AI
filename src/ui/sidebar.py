# src/ui/sidebar.py
import streamlit as st
import os
import config
import shutil
from src.ui.components import render_settings_panel
from src.utils.chat_history import (
    clear_chat_history,
    create_conversation,
    delete_conversation,
    get_active_conversation_id,
    get_chat_history,
    list_conversations,
    rename_conversation,
    set_active_conversation,
)


def _trim_label(text: str, max_len: int = 30) -> str:
    """Trim long labels so conversation list remains compact in sidebar."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _format_timestamp(iso_timestamp: str) -> str:
    """Format stored ISO timestamp for quick history preview."""
    if not iso_timestamp:
        return "-"
    return iso_timestamp.replace("T", " ")[:19]

def render_sidebar(vs_manager):
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

        # --- Quản lý tài liệu (Kho tri thức) ---
        st.subheader("📚 Kho tri thức")
        docs_registry = vs_manager.get_document_registry()
        if not docs_registry:
            st.info("Chưa có tài liệu nào trong bộ nhớ.")
        else:
            st.caption(f"Đang lưu trữ **{len(docs_registry)}** tài liệu.")
            for doc in docs_registry:
                # Tạo label gọn gàng
                fname = doc['filename']
                if len(fname) > 25:
                    fname = fname[:22] + "..."
                
                with st.expander(f"📄 {fname}", expanded=False):
                    st.write(f"**Số đoạn:** `{doc['chunk_count']}`")
                    st.write(f"**Ngôn ngữ:** `{doc.get('language', 'Không rõ')}`")
                    st.write(f"**Ngày tải:** `{doc.get('upload_date_only', 'N/A')}`")

        st.divider()

        # --- Panel cài đặt nâng cao (Phase 2 - Yêu cầu 4) ---
        render_settings_panel()

        st.divider()

        # --- Bộ nhớ hội thoại theo conversation ---
        st.subheader("💬 Cuộc trò chuyện")

        if st.button("➕ Trò chuyện mới", use_container_width=True):
            create_conversation()
            if st.session_state.get("rag_manager"):
                st.session_state.rag_manager.clear_history()
            st.rerun()

        conversations = list_conversations()
        active_conversation_id = get_active_conversation_id()

        if not conversations:
            st.caption("Chưa có conversation nào.")
        else:
            with st.container(height=220):
                for conversation in conversations:
                    conversation_id = conversation["id"]
                    conversation_name = conversation.get("name", "Untitled")
                    is_active = conversation_id == active_conversation_id

                    col1, col2, col3 = st.columns([0.68, 0.16, 0.16])

                    prefix = "●" if is_active else "○"
                    label = f"{prefix} {_trim_label(conversation_name)}"

                    if col1.button(
                        label,
                        key=f"select_conversation_{conversation_id}",
                        use_container_width=True,
                        help=conversation_name,
                    ):
                        if set_active_conversation(conversation_id):
                            if st.session_state.get("rag_manager"):
                                st.session_state.rag_manager.clear_history()
                        st.rerun()

                    if col2.button("✏️", key=f"rename_conversation_btn_{conversation_id}", use_container_width=True):
                        st.session_state["rename_conversation_id"] = conversation_id
                        st.session_state["rename_conversation_value"] = conversation_name

                    if col3.button("🗑️", key=f"delete_conversation_btn_{conversation_id}", use_container_width=True):
                        st.session_state["delete_conversation_id"] = conversation_id

                    st.caption(f"{conversation.get('turn_count', 0)} lượt hỏi")

        rename_conversation_id = st.session_state.get("rename_conversation_id")
        if rename_conversation_id:
            st.markdown("**Đổi tên conversation**")
            st.text_input("Tên mới", key="rename_conversation_value")

            c1, c2 = st.columns(2)
            if c1.button("✅ Lưu", key="rename_conversation_confirm"):
                new_name = st.session_state.get("rename_conversation_value", "")
                if rename_conversation(rename_conversation_id, new_name):
                    st.session_state.pop("rename_conversation_id", None)
                    st.session_state.pop("rename_conversation_value", None)
                    st.rerun()
                else:
                    st.error("Tên conversation không hợp lệ.")

            if c2.button("❌ Hủy", key="rename_conversation_cancel"):
                st.session_state.pop("rename_conversation_id", None)
                st.session_state.pop("rename_conversation_value", None)
                st.rerun()

        delete_conversation_id = st.session_state.get("delete_conversation_id")
        if delete_conversation_id:
            conversation_name = next(
                (
                    conversation.get("name", "Conversation")
                    for conversation in conversations
                    if conversation.get("id") == delete_conversation_id
                ),
                "Conversation",
            )
            st.warning(f"Xóa '{conversation_name}'?")
            d1, d2 = st.columns(2)
            if d1.button("✅ Xóa", key="delete_conversation_confirm"):
                delete_conversation(delete_conversation_id)
                st.session_state.pop("delete_conversation_id", None)
                if st.session_state.get("rag_manager"):
                    st.session_state.rag_manager.clear_history()
                st.rerun()
            if d2.button("❌ Hủy", key="delete_conversation_cancel"):
                st.session_state.pop("delete_conversation_id", None)
                st.rerun()

        st.divider()

        st.subheader("🕒 Lịch sử conversation hiện tại")
        history = get_chat_history()
        if not history:
            st.caption("Conversation hiện tại chưa có câu hỏi.")
        else:
            st.caption(f"Có **{len(history)}** câu hỏi trong conversation đang chọn.")
            with st.container(height=180):
                for entry in reversed(history[-10:]):
                    timestamp = _format_timestamp(entry.get("timestamp", ""))
                    question = entry.get("question", "")
                    st.markdown(f"`{timestamp}`  👤 {_trim_label(question, max_len=45)}", help=question)

        st.divider()

        # --- Trình quản lý ---
        st.subheader("🛠️ Trình quản lý")

        # 1. Xóa lịch sử hội thoại (có confirm)
        if "confirm_clear_chat" not in st.session_state:
            st.session_state.confirm_clear_chat = False

        if st.button("🗑️ Xóa Tất cả Hội thoại", use_container_width=True):
            st.session_state.confirm_clear_chat = True

        if st.session_state.confirm_clear_chat:
            st.warning("Bạn có chắc muốn xóa toàn bộ conversations?")
            col1, col2 = st.columns(2)
            if col1.button("✅ Có, Xóa", key="yes_chat"):
                clear_chat_history(clear_all_conversations=True)
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
                clear_chat_history(clear_all_conversations=True) # Xóa luôn chat khi xóa VS
                st.session_state.confirm_clear_vs = False
                st.success("Đã xóa dữ liệu Vector Store!")
                st.rerun()
            if c2.button("❌ Hủy bỏ", key="no_vs"):
                st.session_state.confirm_clear_vs = False
                st.rerun()

        return uploaded_file

