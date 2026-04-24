# src/ui/components.py
import os
import json
import urllib.request
import streamlit as st
import config
from src.core.text_splitter import ALLOWED_CHUNK_SIZES, ALLOWED_CHUNK_OVERLAPS
from src.utils.helpers import persist_ollama_runtime_settings


@st.cache_data(ttl=20)
def _fetch_ollama_models(base_url: str, api_key: str = "") -> list[str]:
    """Fetch available Ollama models from /api/tags."""
    endpoint = f"{base_url.rstrip('/')}/api/tags"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(endpoint, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=4) as response:
        payload = json.loads(response.read().decode("utf-8"))

    models = payload.get("models", [])
    names = [m.get("name", "").strip() for m in models if isinstance(m, dict)]
    return sorted({name for name in names if name})


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
        "ollama_mode_effective": config.OLLAMA_MODE if config.OLLAMA_MODE in {"local", "cloud"} else "local",
        "ollama_local_base_url_effective": config.OLLAMA_PROXY_BASE_URL,
        "ollama_local_model_effective": config.OLLAMA_LOCAL_MODEL,
        "ollama_cloud_base_url_effective": config.OLLAMA_PROXY_BASE_URL,
        "ollama_cloud_model_effective": config.OLLAMA_CLOUD_MODEL,
        "ollama_api_key_effective": config.OLLAMA_API_KEY,
        "input_ollama_mode": config.OLLAMA_MODE if config.OLLAMA_MODE in {"local", "cloud"} else "local",
        "input_ollama_local_model": config.OLLAMA_LOCAL_MODEL,
        "input_ollama_cloud_model": config.OLLAMA_CLOUD_MODEL,
        "input_ollama_api_key": config.OLLAMA_API_KEY,
        "crag_mode": False,
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

        crag_mode = st.toggle(
            "Chế độ CRAG (Corrective RAG)",
            value=st.session_state.crag_mode,
            help=(
                "**Bật:** Hệ thống sẽ đánh giá độ liên quan của tài liệu trước khi trả lời. "
                "Nếu tài liệu không liên quan, nó sẽ cảnh báo hoặc tìm cách xử lý khác (giảm thiểu hallucination).\n\n"
                "**Tắt:** RAG truyền thống."
            ),
            key="toggle_crag",
        )
        st.session_state.crag_mode = crag_mode
        if crag_mode:
            st.success("🎯 Đang bật CRAG (Kiểm soát chất lượng)")

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

        st.divider()

        # ── Ollama runtime (Local/Cloud) ─────────────────────────────────
        st.markdown("**LLM Ollama (Live)**")
        shared_base_url = config.OLLAMA_PROXY_BASE_URL

        # Force both modes to use local Ollama proxy endpoint.
        st.session_state.ollama_local_base_url_effective = shared_base_url
        st.session_state.ollama_cloud_base_url_effective = shared_base_url

        active_mode = st.session_state.ollama_mode_effective
        active_base_url = shared_base_url
        active_model = (
            st.session_state.ollama_cloud_model_effective
            if active_mode == "cloud"
            else st.session_state.ollama_local_model_effective
        )

        st.caption(
            f"Đang dùng: **{active_mode.upper()}** | model `{active_model}` | endpoint `{active_base_url}`"
        )

        selected_mode = st.selectbox(
            "Chế độ Ollama",
            options=["local", "cloud"],
            index=0 if st.session_state.input_ollama_mode == "local" else 1,
            key="select_ollama_mode",
            help="Local: Ollama chạy trong LAN/máy cá nhân. Cloud: endpoint Ollama hosted.",
        )
        st.session_state.input_ollama_mode = selected_mode

        st.caption(f"Endpoint proxy dùng chung: `{shared_base_url}`")

        if st.button("Làm mới danh sách model", key="refresh_ollama_models"):
            _fetch_ollama_models.clear()
            st.rerun()

        current_api_key = st.session_state.input_ollama_api_key.strip()
        cloud_api_key = current_api_key

        if selected_mode == "cloud":
            st.text_input(
                "Cloud API key",
                key="input_ollama_api_key",
                type="password",
                help="API key dùng cho cloud model qua Ollama local proxy.",
            )
            cloud_api_key = st.session_state.input_ollama_api_key.strip()

        try:
            model_options = _fetch_ollama_models(shared_base_url, cloud_api_key if selected_mode == "cloud" else "")
        except Exception:
            model_options = []

        # Local mode hides cloud-tagged models; cloud mode only shows cloud-tagged models.
        if selected_mode == "cloud":
            mode_filtered_options = [m for m in model_options if "cloud" in m.lower()]
        else:
            mode_filtered_options = [m for m in model_options if "cloud" not in m.lower()]

        fallback_models = [
            st.session_state.input_ollama_local_model,
            st.session_state.input_ollama_cloud_model,
            st.session_state.ollama_local_model_effective,
            st.session_state.ollama_cloud_model_effective,
        ]
        for model_name in fallback_models:
            if not model_name:
                continue

            lowered = model_name.lower()
            if selected_mode == "cloud" and "cloud" not in lowered:
                continue
            if selected_mode == "local" and "cloud" in lowered:
                continue
            if model_name not in mode_filtered_options:
                mode_filtered_options.append(model_name)

        model_options = sorted({m.strip() for m in mode_filtered_options if m and m.strip()})
        if not model_options:
            model_options = [config.LLM_MODEL]

        if selected_mode == "local":
            current_local = st.session_state.input_ollama_local_model
            if current_local not in model_options:
                current_local = model_options[0]
            local_model = st.selectbox(
                "Local model",
                options=model_options,
                index=model_options.index(current_local),
                key="select_ollama_local_model",
            )
            st.session_state.input_ollama_local_model = local_model
        else:
            current_cloud = st.session_state.input_ollama_cloud_model
            if current_cloud not in model_options:
                current_cloud = model_options[0]
            cloud_model = st.selectbox(
                "Cloud model",
                options=model_options,
                index=model_options.index(current_cloud),
                key="select_ollama_cloud_model",
            )
            st.session_state.input_ollama_cloud_model = cloud_model

        if not model_options:
            st.warning("Không lấy được danh sách model từ Ollama.")

        if st.button("Áp dụng model/endpoint Ollama", key="apply_ollama_runtime"):
            selected_mode = st.session_state.input_ollama_mode
            local_base = shared_base_url
            local_model = st.session_state.input_ollama_local_model.strip()
            cloud_base = shared_base_url
            cloud_model = st.session_state.input_ollama_cloud_model.strip()
            api_key = st.session_state.input_ollama_api_key.strip()

            if selected_mode == "local" and not local_model:
                st.error("Vui lòng chọn Local model.")
            elif selected_mode == "cloud" and not cloud_model:
                st.error("Vui lòng chọn Cloud model.")
            else:
                st.session_state.ollama_mode_effective = selected_mode
                st.session_state.ollama_local_base_url_effective = local_base
                st.session_state.ollama_local_model_effective = local_model
                st.session_state.ollama_cloud_base_url_effective = cloud_base
                st.session_state.ollama_cloud_model_effective = cloud_model
                st.session_state.ollama_api_key_effective = api_key

                env_path = os.path.join(config.BASE_DIR, ".env")
                persist_ollama_runtime_settings(
                    env_path=env_path,
                    mode=selected_mode,
                    local_base_url=local_base,
                    local_model=local_model,
                    cloud_base_url=cloud_base,
                    cloud_model=cloud_model,
                    api_key=api_key,
                )

                st.success("Đã áp dụng cấu hình Ollama mới. Hệ thống đang chuyển model live.")
                st.rerun()

    return {
        "chunk_size": st.session_state.chunk_size,
        "chunk_overlap": st.session_state.chunk_overlap,
        "conversational_mode": st.session_state.conversational_mode,
        "crag_mode": st.session_state.crag_mode,
        "hybrid_search": st.session_state.hybrid_search,
        "reranking": st.session_state.reranking,
        "ollama_mode": st.session_state.ollama_mode_effective,
        "ollama_base_url": (
            st.session_state.ollama_cloud_base_url_effective
            if st.session_state.ollama_mode_effective == "cloud"
            else st.session_state.ollama_local_base_url_effective
        ),
        "ollama_model": (
            st.session_state.ollama_cloud_model_effective
            if st.session_state.ollama_mode_effective == "cloud"
            else st.session_state.ollama_local_model_effective
        ),
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
