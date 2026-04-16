# src/ui/sidebar.py
import streamlit as st
import os
import config
import shutil
import time
from src.ui.components import render_settings_panel
from src.utils.chat_history import get_chat_history, clear_chat_history
from langchain_core.messages import HumanMessage, AIMessage

def render_sidebar():
    with st.sidebar:
        st.header("📄 SmartDoc AI")
        
        # --- Nút tạo Chat mới (Quan trọng để tách biệt các phiên) ---
        if st.button("➕ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
            st.session_state.current_session_id = f"session_{int(time.time())}"
            clear_chat_history(clear_documents=False)
            if st.session_state.get("rag_manager"):
                st.session_state.rag_manager.clear_history()
            st.rerun()

        st.divider()

        # --- Tải tài liệu ---
        st.subheader("📂 Tài liệu")
        uploaded_file = st.file_uploader(
            "Tải lên tài liệu mới",
            type=["pdf", "docx"],
            help="Chỉ hỗ trợ định dạng PDF và DOCX."
        )

        st.divider()

        # --- Lịch sử hội thoại (SQLite) ---
        st.subheader("💬 Lịch sử hội thoại")
        
        if "db" in st.session_state:
            sessions = st.session_state.db.get_all_sessions()
            
            if not sessions:
                st.caption("Chưa có cuộc hội thoại nào.")
            else:
                with st.container(height=300):
                    for s in sessions:
                        is_active = (s['session_id'] == st.session_state.get("current_session_id"))
                        
                        # Dùng dấu hiệu trực quan để biết mình đang ở session nào
                        label = f"💬 {s['title'][:25]}"
                        if is_active:
                            label = f"▶️ {s['title'][:25]}"

                        if st.button(label, key=f"select_{s['session_id']}", use_container_width=True):
                            # Gán ID mới
                            st.session_state.current_session_id = s['session_id']
                            
                            # Lấy data từ DB
                            history_from_db = st.session_state.db.get_session_history(s['session_id'])
                            
                            # Cập nhật trực tiếp vào CHAT_HISTORY_KEY (thường là "messages")
                            from src.utils.chat_history import set_chat_history, CHAT_HISTORY_KEY
                            set_chat_history(history_from_db)
                            
                            # Nạp lại bộ nhớ cho AI
                            if st.session_state.get("rag_manager"):
                                rag = st.session_state.rag_manager
                                rag.clear_history() # Hàm này phải xóa sạch list nội bộ của class RAGChainManager
                                for turn in history_from_db:
                                    rag.chat_history.append(HumanMessage(content=turn["question"]))
                                    rag.chat_history.append(AIMessage(content=turn["answer"]))
                            
                            st.rerun()


        st.divider()

        # --- Cấu hình & Trạng thái (Gom nhóm lại cho gọn) ---
        with st.expander("⚙️ Cài đặt & Hệ thống"):
            render_settings_panel()
            st.divider()
            st.caption(f"LLM: `{config.LLM_MODEL}`")
            st.caption(f"Embeddings: `{config.EMBEDDING_MODEL.split('/')[-1]}`")

        st.divider()

        # --- Trình quản lý dữ liệu ---
        st.subheader("🛠️ Quản trị")

        # Xóa lịch sử (Chỉ xóa session hiện tại hoặc toàn bộ tùy bạn, ở đây giữ logic cũ là clear UI)
        if st.button("🗑️ Xóa hội thoại hiện tại", use_container_width=True):
            clear_chat_history()
            if st.session_state.get("rag_manager"):
                st.session_state.rag_manager.clear_history()
            st.rerun()

        # Xóa Vector Store
        if st.button("⚡ Reset Vector Store", use_container_width=True):
            vs_path = os.path.join(config.VECTORSTORE_DIR, "faiss_index")
            if os.path.exists(vs_path):
                shutil.rmtree(vs_path)
            st.session_state.rag_manager = None
            st.success("Đã xóa dữ liệu nhúng!")
            st.rerun()

        return uploaded_file