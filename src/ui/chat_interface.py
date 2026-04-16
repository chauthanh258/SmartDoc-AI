# src/ui/chat_interface.py
import streamlit as st
from src.utils.chat_history import get_chat_history, add_chat_turn

def _render_sources(sources: list):
    """Hiển thị danh sách nguồn trích dẫn bên dưới câu trả lời."""
    if not sources:
        return
    with st.expander(f"📄 Xem nguồn trích dẫn ({len(sources)} đoạn)"):
        for idx, source in enumerate(sources):
            # Lấy thông tin từ source (đã được normalize ở utils)
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

    # --- Hiển thị lịch sử hội thoại (Lấy từ session_state đã được sync với DB) ---
    history = get_chat_history()
    for entry in history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("sources"):
                _render_sources(entry["sources"])

    # --- Ô nhập liệu ---
    if prompt := st.chat_input("Hỏi về nội dung tài liệu..."):

        # 1. Hiển thị câu hỏi ngay lập tức
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Gọi RAG và hiển thị câu trả lời dạng streaming
        with st.chat_message("assistant"):
            full_answer = ""
            sources = []
            
            with st.status("Đang khởi tạo...", expanded=True) as status:
                analysis_area = st.empty()
                
                def stream_generator():
                    nonlocal full_answer, sources
                    # Thực hiện stream từ RAG Manager
                    for packet in rag_manager.stream_ask(prompt, conversational=conversational_mode):
                        if packet["type"] == "status":
                            status.update(label=packet["content"])
                        elif packet["type"] == "analysis":
                            analysis_area.markdown(packet["content"])
                        elif packet["type"] == "sources":
                            sources = packet["content"]
                        elif packet["type"] == "chunk":
                            status.update(label="✅ Đã xử lý xong", state="complete", expanded=False)
                            yield packet["content"]
                        elif packet["type"] == "error":
                            status.update(label="❌ Lỗi", state="error")
                            st.error(packet["content"])
                            return

                # write_stream sẽ trả về nội dung text đầy đủ sau khi stream kết thúc
                full_answer = st.write_stream(stream_generator())
            
            if sources:
                _render_sources(sources)

        # 3. LƯU DỮ LIỆU VÀO CẢ SESSION VÀ SQLITE
        if full_answer:
            # Lưu vào bộ nhớ tạm (st.session_state["messages"])
            entry = add_chat_turn(
                question=prompt,
                answer=full_answer,
                sources=sources
            )
            
            # Lưu vĩnh viễn vào SQLite
            if "db" in st.session_state and "current_session_id" in st.session_state:
                session_id = st.session_state.current_session_id
                try:
                    # A. Nếu đây là tin nhắn đầu tiên của session, tạo mới bản ghi trong bảng 'sessions'
                    # Điều này giúp sidebar có tiêu đề để hiển thị
                    current_history = get_chat_history()
                    if len(current_history) == 1:
                        # Lấy 40 ký tự đầu của prompt làm tiêu đề chat
                        chat_title = prompt[:40] + ("..." if len(prompt) > 40 else "")
                        st.session_state.db.create_session(session_id, chat_title)
                    
                    # B. Lưu chi tiết tin nhắn vào bảng 'chat_turns'
                    st.session_state.db.save_chat_turn(session_id, entry)
                    
                except Exception as e:
                    st.warning(f"Lưu vào DB thất bại: {e}")

        # Rerun để đồng bộ trạng thái Sidebar và lịch sử mới
        st.rerun()