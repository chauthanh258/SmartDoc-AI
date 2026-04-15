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
        from src.core.text_splitter import ALLOWED_CHUNK_SIZES, ALLOWED_CHUNK_OVERLAPS
        
        st.markdown("**Chunking**")
        col1, col2 = st.columns(2)
        with col1:
            # Tìm index của giá trị hiện tại trong whitelist để set index mặc định
            current_size = st.session_state.chunk_size
            size_index = ALLOWED_CHUNK_SIZES.index(current_size) if current_size in ALLOWED_CHUNK_SIZES else 1 # 1000 là mặc định
            
            chunk_size = st.selectbox(
                "Chunk Size",
                options=ALLOWED_CHUNK_SIZES,
                index=size_index,
                help="Số ký tự tối đa trong mỗi đoạn văn bản.",
                key="input_chunk_size"
            )
        with col2:
            current_overlap = st.session_state.chunk_overlap
            overlap_index = ALLOWED_CHUNK_OVERLAPS.index(current_overlap) if current_overlap in ALLOWED_CHUNK_OVERLAPS else 1 # 100 là mặc định
            
            chunk_overlap = st.selectbox(
                "Chunk Overlap",
                options=ALLOWED_CHUNK_OVERLAPS,
                index=overlap_index,
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
