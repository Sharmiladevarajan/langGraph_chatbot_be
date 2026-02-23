"""
LLM factory: switch between OpenAI (normal cloud), Bytez/custom API, or local Ollama.
All expose the same LangChain ChatModel interface – no workflow changes needed.
Supports runtime override from UI (dropdown) via set_llm_provider().
"""
from typing import Literal, Optional

from app.config import settings

# Runtime override: when set by UI, this takes precedence over env LLM_PROVIDER
_runtime_provider: Optional[Literal["openai", "bytez", "local"]] = None


def get_effective_provider() -> str:
    """Return the provider currently in use (runtime override or env)."""
    if _runtime_provider is not None:
        return _runtime_provider
    return settings.LLM_PROVIDER


def set_llm_provider(provider: str) -> None:
    """Set LLM provider at runtime (e.g. from UI dropdown). No restart needed."""
    global _runtime_provider
    if provider in ("openai", "bytez", "local"):
        _runtime_provider = provider


def get_llm():
    """
    Return the configured LLM. Uses runtime override from UI if set,
    else env LLM_PROVIDER.
    """
    provider = get_effective_provider()
    if provider == "openai" and settings.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_LLM_MODEL,
            temperature=0.2,
        )
    if provider == "bytez" and settings.BYTEZ_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=settings.BYTEZ_BASE_URL,
            api_key=settings.BYTEZ_API_KEY,
            model=settings.BYTEZ_MODEL,
            temperature=0.2,
        )
    # Local Ollama (when provider is local)
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=0.2,
    )
