# src/ui/components.py
import streamlit as st
import config
from src.core.text_splitter import ALLOWED_CHUNK_SIZES, ALLOWED_CHUNK_OVERLAPS


def render_settings_panel():
    """
    Panel cài đặt nâng cao trong sidebar.
    Phase 2: chunk_size, chunk_overlap, Conversational toggle.
    Phase 3: Hybrid Search toggle, Re-ranking toggle.
    Lưu tất cả settings vào session_state và trả về dict.
    """
    # ── Khởi tạo giá trị mặc định ──────────────────────────────────────────
    defaults = {
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "conversational_mode": True,
        "hybrid_search": False,
        "reranking": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    with st.expander("⚙️ Cài đặt nâng cao", expanded=False):
        st.caption("Thay đổi cài đặt sẽ có hiệu lực khi tải tài liệu mới.")

        # ── Chunking ───────────────────────────────────────────────────────
        st.markdown("**Chunking**")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.selectbox(
                "Chunk Size",
                options=list(ALLOWED_CHUNK_SIZES),
                index=list(ALLOWED_CHUNK_SIZES).index(
                    st.session_state.chunk_size
                    if st.session_state.chunk_size in ALLOWED_CHUNK_SIZES
                    else ALLOWED_CHUNK_SIZES[1]
                ),
                help="Số ký tự tối đa trong mỗi đoạn văn bản.",
                key="select_chunk_size",
            )
        with col2:
            valid_overlaps = [o for o in ALLOWED_CHUNK_OVERLAPS if o < chunk_size]
            if not valid_overlaps:
                valid_overlaps = [ALLOWED_CHUNK_OVERLAPS[0]]
            current_overlap = st.session_state.chunk_overlap
            if current_overlap not in valid_overlaps:
                current_overlap = valid_overlaps[-1]
            chunk_overlap = st.selectbox(
                "Chunk Overlap",
                options=valid_overlaps,
                index=valid_overlaps.index(current_overlap),
                help="Số ký tự trùng lặp giữa các đoạn liên tiếp.",
                key="select_chunk_overlap",
            )

        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

        st.divider()

        # ── Chế độ hội thoại ───────────────────────────────────────────────
        st.markdown("**Chế độ trả lời**")
        conversational_mode = st.toggle(
            "Chế độ hội thoại (Conversational RAG)",
            value=st.session_state.conversational_mode,
            help=(
                "**Bật:** Hệ thống nhớ lịch sử câu hỏi, phù hợp cho hội thoại liên tục.\n\n"
                "**Tắt:** Mỗi câu hỏi độc lập, phù hợp cho tra cứu nhanh."
            ),
            key="toggle_conversational",
        )
        st.session_state.conversational_mode = conversational_mode
        if conversational_mode:
            st.success("✅ Đang dùng Conversational RAG")
        else:
            st.info("⚡ Đang dùng Basic RAG (không nhớ lịch sử)")

        st.divider()

        # ── Tìm kiếm nâng cao (Phase 3) ────────────────────────────────────
        st.markdown("**Tìm kiếm nâng cao**")
        hybrid_search = st.toggle(
            "Hybrid Search (BM25 + Semantic)",
            value=st.session_state.hybrid_search,
            help=(
                "**Bật:** Kết hợp tìm theo từ khóa (BM25) và ngữ nghĩa (FAISS) — "
                "chính xác hơn với câu hỏi cụ thể.\n\n"
                "**Tắt:** Chỉ dùng Semantic Search (FAISS) — nhanh hơn."
            ),
            key="toggle_hybrid",
        )
        st.session_state.hybrid_search = hybrid_search

        reranking = st.toggle(
            "Re-ranking kết quả",
            value=st.session_state.reranking,
            help=(
                "**Bật:** Sắp xếp lại các đoạn trích xuất theo độ liên quan trước khi "
                "đưa vào LLM — cải thiện chất lượng câu trả lời.\n\n"
                "**Tắt:** Dùng thứ tự kết quả mặc định từ retriever."
            ),
            key="toggle_reranking",
        )
        st.session_state.reranking = reranking

        if hybrid_search:
            st.success("✅ Hybrid Search đang bật")
        if reranking:
            st.success("✅ Re-ranking đang bật")

    return {
        "chunk_size": st.session_state.chunk_size,
        "chunk_overlap": st.session_state.chunk_overlap,
        "conversational_mode": st.session_state.conversational_mode,
        "hybrid_search": st.session_state.hybrid_search,
        "reranking": st.session_state.reranking,
    }


def render_document_filter(vs_manager):
    """
    Phase 3 — Bộ lọc tài liệu multi-select.
    Đọc danh sách file từ vectorstore registry.
    Trả về list[str] filenames được chọn (rỗng = tất cả).
    """
    st.subheader("🔍 Lọc tài liệu")

    registry = []
    if vs_manager and vs_manager.vectorstore:
        try:
            registry = vs_manager.get_document_registry()
        except Exception:
            registry = []

    if not registry:
        st.caption("Chưa có tài liệu nào trong hệ thống.")
        return []

    # Lấy danh sách tên file (loại trùng lặp)
    filenames = sorted(
        {r.get("filename") or r.get("file_name") or "Không rõ" for r in registry}
    )

    selected = st.multiselect(
        "Chọn tài liệu để truy vấn",
        options=filenames,
        default=[],
        placeholder="Để trống = tất cả tài liệu",
        help="Chọn một hoặc nhiều tài liệu. Để trống để tìm trên toàn bộ kho.",
        key="doc_filter_select",
    )

    if selected:
        st.info(f"Đang lọc: **{', '.join(selected)}**")
    else:
        st.caption("Đang tìm kiếm trên **tất cả tài liệu**.")

    return selected
