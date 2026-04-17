# src/ui/chat_interface.py
import streamlit as st


def _render_sources(sources: list):
    """
    Hiển thị nguồn trích dẫn kèm tên file và số trang (Yêu cầu 5 & 8).
    Dùng st.expander để gọn gàng.
    """
    if not sources:
        return

    with st.expander(f"📄 Xem nguồn trích dẫn ({len(sources)} đoạn)"):
        for idx, source in enumerate(sources):
            page = source.get("page", "?")
            file = source.get("file", "Tài liệu")
            content = source.get("content", "")

            # Label hiển thị tên file + trang
            st.markdown(f"**Nguồn {idx + 1}:** `{file}` — Trang {page}")
            st.info(content[:500] + ("..." if len(content) > 500 else ""))


def _render_doc_badge(sources: list):
    """
    Hiển thị badge nhỏ phía trên câu trả lời, cho người dùng biết
    câu trả lời đến từ tài liệu nào (Phase 3 - Yêu cầu 8).
    """
    if not sources:
        return

    # Gom tên file duy nhất từ danh sách sources
    seen_files = {}
    for s in sources:
        fname = s.get("file", "")
        if fname and fname not in seen_files:
            page = s.get("page", "?")
            seen_files[fname] = page

    if not seen_files:
        return

    parts = [f"📎 `{fname}` (trang {page})" for fname, page in seen_files.items()]
    st.caption("Trả lời dựa trên: " + " · ".join(parts))


def render_chat_interface(rag_manager):
    """Giao diện chat chính — hiển thị lịch sử, nhận câu hỏi, render nguồn."""

    # Tiêu đề động theo chế độ
    conversational_mode = st.session_state.get("conversational_mode", True)
    hybrid = st.session_state.get("hybrid_search", False)
    rerank = st.session_state.get("reranking", False)

    mode_parts = ["💬 Conv. RAG" if conversational_mode else "⚡ Basic RAG"]
    if hybrid:
        mode_parts.append("🔀 Hybrid")
    if rerank:
        mode_parts.append("📊 Re-ranked")

    st.header(f"Trò chuyện  —  {' · '.join(mode_parts)}")

    # ── Hiển thị lịch sử tin nhắn ──────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_doc_badge(message.get("sources", []))
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []))

    # ── Ô nhập liệu ────────────────────────────────────────────────────────
    if prompt := st.chat_input("Hỏi về nội dung tài liệu..."):

        # 1. Hiển thị câu hỏi người dùng
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Gọi RAG và hiển thị câu trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang trích xuất thông tin..."):
                answer, sources = rag_manager.ask(
                    prompt,
                    conversational=conversational_mode,
                )
                _render_doc_badge(sources)
                st.markdown(answer)
                _render_sources(sources)

        # 3. Lưu vào lịch sử (kèm sources để render lại khi rerun)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })