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

        # 2. Gọi RAG và hiển thị câu trả lời dạng streaming
        with st.chat_message("assistant"):
            full_answer = ""
            sources = []
            
            # Sử dụng st.status để hiển thị quá trình phân tích và tìm kiếm
            with st.status("Đang khởi tạo...", expanded=True) as status:
                # Placeholder cho các thông tin phân tích
                analysis_area = st.empty()
                
                # Generator wrapper để st.write_stream có thể tiêu thụ chunks
                def stream_generator():
                    nonlocal full_answer, sources
                    for packet in rag_manager.stream_ask(prompt, conversational=conversational_mode):
                        if packet["type"] == "status":
                            status.update(label=packet["content"])
                        elif packet["type"] == "analysis":
                            analysis_area.markdown(packet["content"])
                        elif packet["type"] == "sources":
                            sources = packet["content"]
                        elif packet["type"] == "chunk":
                            # Khi nhận được chunk đầu tiên, đóng status lại để tập trung vào câu trả lời
                            status.update(label="✅ Đã xử lý xong", state="complete", expanded=False)
                            full_answer += packet["content"]
                            yield packet["content"]
                        elif packet["type"] == "error":
                            status.update(label="❌ Lỗi", state="error")
                            st.error(packet["content"])
                            return

                # Hiển thị câu trả lời với hiệu ứng gõ chữ
                full_answer = st.write_stream(stream_generator())
            
            # Hiển thị nguồn trích dẫn sau khi stream xong
            if sources:
                _render_sources(sources)

        # 3. Lưu vào lịch sử tập trung
        if full_answer:
            add_chat_turn(
                question=prompt,
                answer=full_answer,
                sources=sources
            )
        # Tự động reload để cập nhật sidebar nếu cần
        st.rerun()