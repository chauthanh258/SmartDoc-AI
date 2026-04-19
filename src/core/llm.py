from langchain_ollama import OllamaLLM
import config

import streamlit as st

@st.cache_resource
def get_llm():
    """Returns the Ollama LLM instance."""
    return OllamaLLM(
        base_url=config.OLLAMA_BASE_URL,
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE
    )
