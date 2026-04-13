try:
    # Preferred package (new dedicated integration)
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        # Older community integration (fallback)
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        # Final fallback to older langchain location (very old installs)
        from langchain.embeddings import HuggingFaceEmbeddings

import config

def get_embedding_model():
    """Returns the HuggingFace embedding model.

    Tries the new `langchain_huggingface` package first (recommended).
    Falls back to community / legacy imports when necessary so the
    repository remains compatible with different environments.
    """
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
