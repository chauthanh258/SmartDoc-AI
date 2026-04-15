# src/ui/chat_interface.py
import streamlit as st
from src.utils.chat_history import get_chat_history, add_chat_turn

def _render_sources(sources: list):
    """Hiển thị danh sách nguồn trích dẫn bên dưới câu trả lời."""
    if not sources:
        return
    with st.expander(f"📄 Xem nguồn trích dẫn ({len(sources)} đoạn)"):
        for idx, source in enumerate(sources):
            page = source.get("page", "?")
            file = source.get("file_name", source.get("file", "Tài liệu"))
            content = source.get("snippet", source.get("content", ""))
            st.markdown(
                f"**Nguồn {idx + 1}:** `{file}` — Trang {page}",
            )
            st.info(content[:500] + ("..." if len(content) > 500 else ""))

def render_chat_interface(rag_manager):
    """Giao diện chat chính, hiển thị lịch sử và nhận câu hỏi."""

    # Tiêu đề động theo chế độ
    conversational_mode = st.session_state.get("conversational_mode", True)
    mode_label = "💬 Conversational RAG" if conversational_mode else "⚡ Basic RAG"
    st.header(f"Trò chuyện  —  {mode_label}")

    # --- Hiển thị lịch sử hội thoại (phiên bản mới sử dụng chat_history) ---
    history = get_chat_history()
    for entry in history:
        # Hiển thị câu hỏi
        with st.chat_message("user"):
            st.markdown(entry["question"])
        
        # Hiển thị câu trả lời
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("sources"):
                _render_sources(entry["sources"])

    # --- Ô nhập liệu ---
    if prompt := st.chat_input("Hỏi về nội dung tài liệu..."):

        # 1. Hiển thị câu hỏi của người dùng ngay lập tức
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Gọi RAG và hiển thị câu trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang trích xuất thông tin..."):
                answer, sources = rag_manager.ask(
                    prompt,
                    conversational=conversational_mode
                )
                st.markdown(answer)
                _render_sources(sources)

        # 3. Lưu vào lịch sử tập trung
        add_chat_turn(
            question=prompt,
            answer=answer,
            sources=sources
        )
        # Tự động reload để cập nhật sidebar nếu cần
        st.rerun()