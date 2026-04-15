# src/ui/chat_interface.py
import streamlit as st

def render_chat_interface(qa_chain):
    st.header("Trò chuyện")
    
    # Render previous messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display source documents in expander if they exist
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 Nhấn để xem nguồn trích xuất bổ sung"):
                    for idx, doc in enumerate(message["sources"]):
                        st.markdown(f"**Nguồn {idx+1}:**")
                        st.info(doc)

    # Chat input area
    if prompt := st.chat_input("Hỏi câu hỏi dựa trên nội dung tài liệu..."):
        
        # 1. Store and show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Process and show assistant message
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                response = qa_chain.ask(prompt)
                
                # Handling LCEL/RetrievalQA dictionary response
                if isinstance(response, dict):
                    answer = response.get("answer", response.get("result", ""))
                    source_docs = response.get("source_documents", [])
                    extracted_sources = [doc.page_content for doc in source_docs]
                else:
                    answer = str(response)
                    extracted_sources = []
                
                # Output main answer
                st.markdown(answer)
                
                # Automatically append and show cited sources using expander
                if extracted_sources:
                    with st.expander("📄 Nhấn để xem trích đoạn hệ thống đã dùng"):
                        for idx, content in enumerate(extracted_sources):
                            st.markdown(f"**Nguồn {idx+1}:**")
                            st.info(content)
                
                # Store back into session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": extracted_sources
                })
