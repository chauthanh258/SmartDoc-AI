# src/ui/chat_interface.py
import streamlit as st

def _render_sources(sources: list):
    """Hiển thị danh sách nguồn trích dẫn bên dưới câu trả lời (Yêu cầu 5)."""
    if not sources:
        return
    with st.expander(f"📄 Xem nguồn trích dẫn ({len(sources)} đoạn)"):
        for idx, source in enumerate(sources):
            page = source.get("page", "?")
            file = source.get("file", "Tài liệu")
            content = source.get("content", "")
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

    # --- Hiển thị lịch sử tin nhắn ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Hiển thị nguồn trích dẫn dưới mỗi câu trả lời (Yêu cầu 5)
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []))

    # --- Ô nhập liệu ---
    if prompt := st.chat_input("Hỏi về nội dung tài liệu..."):

        # 1. Hiển thị câu hỏi của người dùng
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Gọi RAG và hiển thị câu trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang trích xuất thông tin..."):
                # Chọn chế độ hội thoại hay basic theo toggle (Yêu cầu 4)
                answer, sources = rag_manager.ask(
                    prompt,
                    conversational=conversational_mode
                )
                st.markdown(answer)
                _render_sources(sources)

        # 3. Lưu vào lịch sử (kèm sources để render lại khi rerun)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })