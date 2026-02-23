"""Embedding provider - local (sentence-transformers) or OpenAI."""
from app.config import settings


def get_embedding_model():
    """Return the configured embedding model (same interface for local/Bytez)."""
    if settings.EMBEDDING_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        # dimensions must match Pinecone index (e.g. 1024); OpenAI can reduce output size
        return OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL,
            dimensions=settings.PINECONE_INDEX_DIMENSION,
        )
    # Default: local sentence-transformers
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
