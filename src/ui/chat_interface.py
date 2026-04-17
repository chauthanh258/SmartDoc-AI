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
    mode_label = "💬 Conversational RAG" if conversational_mode else "⚡ Basic RAG"
    st.header(f"Trò chuyện  —  {mode_label}")

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

            with st.status("Đang khởi tạo...", expanded=True) as status:
                analysis_area = st.empty()

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
                            status.update(label="✅ Đã xử lý xong", state="complete", expanded=False)
                            full_answer += packet["content"]
                            yield packet["content"]
                        elif packet["type"] == "error":
                            status.update(label="❌ Lỗi", state="error")
                            st.error(packet["content"])
                            return

                full_answer = st.write_stream(stream_generator())

            if sources:
                _render_sources(sources)

        if full_answer:
            add_chat_turn(
                question=prompt,
                answer=full_answer,
                sources=sources,
                conversation_id=active_conversation.get("id") if active_conversation else None,
                metadata={"conversational_mode": conversational_mode},
            )

        st.rerun()
