"""Externalized configuration via environment variables."""
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings - all externalized for easy switching."""

    # LLM provider: "openai" (normal cloud), "bytez" (Bytez/custom API), or "local" (Ollama)
    LLM_PROVIDER: Literal["openai", "bytez", "local"] = "openai"

    # OpenAI (normal cloud LLM – no Ollama needed)
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"

    # Local LLM (Ollama – optional)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # Bytez: use their OpenAI-compatible endpoint and prefix model with provider (e.g. openai/gpt-4o-mini)
    BYTEZ_API_KEY: str = ""
    BYTEZ_BASE_URL: str = "https://api.bytez.com/models/v2/openai/v1"
    BYTEZ_MODEL: str = "openai/gpt-4o-mini"

    # Embeddings (local or external)
    EMBEDDING_PROVIDER: Literal["local", "openai"] = "openai"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    OPENAI_API_KEY: str = ""

    # Pinecone vector store
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "assessment2026"
    PINECONE_INDEX_DIMENSION: int = 1024  # must match your Pinecone index (e.g. 1024, 1536, 3072)
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # CORS (comma-separated origins; set on Render to your Vercel URL)
    CORS_ORIGINS: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
