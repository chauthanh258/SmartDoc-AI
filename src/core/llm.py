from langchain_ollama import OllamaLLM
import config

def get_llm():
    """Returns the Ollama LLM instance."""
    return OllamaLLM(
        base_url=config.OLLAMA_BASE_URL,
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE
    )
