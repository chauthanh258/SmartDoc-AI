import streamlit as st

def render_chat_interface(rag_manager):
    st.header("Trò chuyện")
    
    # Hiển thị lịch sử tin nhắn
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ô nhập liệu
    if prompt := st.chat_input("Hỏi về nội dung tài liệu..."):
        
        # 1. Hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Assistant trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang trích xuất thông tin..."):
                # Gọi phương thức ask() từ class RAGChainManager
                answer = rag_manager.ask(prompt)
                st.markdown(answer)
        
        # 3. Lưu vào lịch sử
        st.session_state.messages.append({"role": "assistant", "content": answer})