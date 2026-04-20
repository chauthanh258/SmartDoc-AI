from langchain_ollama import OllamaLLM
import config

import streamlit as st

@st.cache_resource
def get_llm(
    base_url: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
):
    """Return an Ollama LLM instance bound to runtime settings."""
    resolved_base_url = (base_url or config.OLLAMA_BASE_URL).strip()
    resolved_model = (model or config.LLM_MODEL).strip()
    resolved_temperature = config.TEMPERATURE if temperature is None else temperature

    resolved_api_key = api_key if api_key is not None else config.OLLAMA_API_KEY
    client_kwargs = None
    if resolved_api_key:
        client_kwargs = {
            "headers": {
                "Authorization": f"Bearer {resolved_api_key}",
            }
        }

    return OllamaLLM(
        base_url=resolved_base_url,
        model=resolved_model,
        temperature=resolved_temperature,
        client_kwargs=client_kwargs,
    )
