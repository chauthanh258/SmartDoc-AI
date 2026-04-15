# src/ui/components.py
import streamlit as st
import config

def render_settings_panel():
    """
    Hiển thị panel cài đặt nâng cao trong sidebar.
    Lưu settings vào session_state và trả về dict các giá trị.
    """
    # Khởi tạo giá trị mặc định trong session_state lần đầu
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = config.CHUNK_SIZE
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = config.CHUNK_OVERLAP
    if "conversational_mode" not in st.session_state:
        st.session_state.conversational_mode = True

    with st.expander("⚙️ Cài đặt nâng cao", expanded=False):
        st.caption("Thay đổi cài đặt sẽ có hiệu lực khi tải tài liệu mới.")

        # --- Cài đặt Chunking ---
        st.markdown("**Chunking**")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=200,
                max_value=4000,
                value=st.session_state.chunk_size,
                step=100,
                help="Số ký tự tối đa trong mỗi đoạn văn bản.",
                key="input_chunk_size"
            )
        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=1000,
                value=st.session_state.chunk_overlap,
                step=50,
                help="Số ký tự trùng lặp giữa các đoạn liên tiếp.",
                key="input_chunk_overlap"
            )

        # Lưu vào session_state khi giá trị thay đổi
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

        st.divider()

        # --- Chế độ hội thoại ---
        st.markdown("**Chế độ trả lời**")
        conversational_mode = st.toggle(
            "Chế độ hội thoại (Conversational RAG)",
            value=st.session_state.conversational_mode,
            help=(
                "**Bật:** Hệ thống nhớ lịch sử câu hỏi trước, phù hợp cho hội thoại liên tục.\n\n"
                "**Tắt:** Mỗi câu hỏi được xử lý độc lập, phù hợp cho tra cứu nhanh."
            ),
            key="toggle_conversational"
        )
        st.session_state.conversational_mode = conversational_mode

        if conversational_mode:
            st.success("✅ Đang dùng Conversational RAG")
        else:
            st.info("⚡ Đang dùng Basic RAG (không nhớ lịch sử)")

    return {
        "chunk_size": st.session_state.chunk_size,
        "chunk_overlap": st.session_state.chunk_overlap,
        "conversational_mode": st.session_state.conversational_mode,
    }
