# src/ui/chat_interface.py
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.utils.chat_history import (
    add_chat_turn,
    create_conversation,
    get_active_conversation,
    get_chat_history,
)


def _render_sources(sources: list):
    """
    Hiển thị nguồn trích dẫn kèm tên file và số trang (Yêu cầu 5 & 8).
    Dùng st.expander để gọn gàng.
    """
    if not sources:
        return

    with st.expander(f"📄 Xem nguồn trích dẫn ({len(sources)} đoạn)"):
        for idx, source in enumerate(sources):
            page = source.get("page") or source.get("page_number") or "?"
            file = (
                source.get("file_name")
                or source.get("file")
                or source.get("document_name")
                or "Tài liệu"
            )
            content = source.get("snippet", source.get("content", ""))
            
            # Metadata bổ sung: Score và CRAG Decision
            score = source.get("score")
            crag_decision = source.get("crag_decision")
            
            meta_info = []
            if score is not None:
                meta_info.append(f"📊 Score: **{score:.2f}**")
            if crag_decision:
                emoji = "✅" if crag_decision == "RELEVANT" else "⚠️"
                meta_info.append(f"{emoji} CRAG: **{crag_decision}**")
                
            meta_str = " | ".join(meta_info)
            if meta_str:
                meta_str = f" — {meta_str}"

            st.markdown(
                f"**Nguồn {idx + 1}:** `{file}` — Trang {page}{meta_str}",
            )
            st.info(content[:500] + ("..." if len(content) > 500 else ""))


def _sync_rag_history(rag_manager, history: list[dict]):
    """Sync active conversation turns into RAG conversational memory."""
    rag_manager.clear_history()

    for entry in history[-5:]:
        question = str(entry.get("question", "")).strip()
        answer = str(entry.get("answer", "")).strip()

        if question:
            rag_manager.chat_history.append(HumanMessage(content=question))
        if answer:
            rag_manager.chat_history.append(AIMessage(content=answer))


def render_chat_interface(rag_manager):
    """Giao diện chat chính, hiển thị lịch sử và nhận câu hỏi."""
    conversational_mode = st.session_state.get("conversational_mode", True)
    crag_mode = st.session_state.get("crag_mode", False)
    hybrid = st.session_state.get("hybrid_search", False)
    rerank = st.session_state.get("reranking", False)

    mode_parts = ["💬 Conv. RAG" if conversational_mode else "⚡ Basic RAG"]
    if crag_mode:
        mode_parts.append("⚖️ CRAG")
    if hybrid:
        mode_parts.append("🔀 Hybrid")
    if rerank:
        mode_parts.append("📊 Re-ranked")

    st.header(f"Trò chuyện  —  {' · '.join(mode_parts)}")

    active_conversation = get_active_conversation()
    history = get_chat_history()

    if active_conversation:
        conversation_name = active_conversation.get("name", "Untitled")
        turn_count = active_conversation.get("turn_count", len(history))
        st.caption(f"Conversation: **{conversation_name}** | Turns: **{turn_count}**")
    else:
        st.caption("No active conversation yet. Your first question will create one.")

    _sync_rag_history(rag_manager, history)

    for entry in history:
        with st.chat_message("user"):
            st.markdown(entry["question"])

        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("sources"):
                _render_sources(entry["sources"])

    if prompt := st.chat_input("Hỏi về nội dung tài liệu..."):
        if not active_conversation:
            create_conversation()
            active_conversation = get_active_conversation()

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            full_answer = ""
            sources = []
            
            # Sử dụng st.status để hiển thị quá trình phân tích và tìm kiếm
            with st.status("Đang khởi tạo...", expanded=True) as status:
                # Placeholder cho các thông tin phân tích
                analysis_area = st.empty()
                accumulated_analysis = []
                
                # Generator wrapper để st.write_stream có thể tiêu thụ chunks
                def stream_generator():
                    nonlocal full_answer, sources
                    for packet in rag_manager.stream_ask(prompt, conversational=conversational_mode, use_crag=crag_mode):
                        if packet["type"] == "status":
                            status.update(label=packet["content"])
                        elif packet["type"] == "analysis":
                            accumulated_analysis.append(packet["content"])
                            analysis_area.markdown("\n".join(accumulated_analysis))
                        elif packet["type"] == "sources":
                            sources = packet["content"]
                        elif packet["type"] == "chunk":
                            # Khi nhận được chunk đầu tiên, đóng status lại để tập trung vào câu trả lời
                            status.update(label="✅ Đã xử lý xong", state="complete", expanded=True)
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
