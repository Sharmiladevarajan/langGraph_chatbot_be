"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.routes import documents, chat
from app.config import settings
from app.core.llm_factory import get_effective_provider, set_llm_provider

app = FastAPI(
    title="LangGraph Document Chatbot",
    description="Conversation + attachment-based chatbot with document RAG",
    version="1.0.0",
)

_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


class LLMProviderUpdate(BaseModel):
    provider: str  # "openai" | "bytez" | "local"


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": get_effective_provider()}


@app.get("/config/llm")
def get_llm_config():
    """Return current LLM provider (runtime override or env). Used by UI dropdown."""
    return {"provider": get_effective_provider()}


@app.post("/config/llm")
def update_llm_config(body: LLMProviderUpdate):
    """Set LLM provider at runtime from UI dropdown. No restart needed."""
    allowed = ("openai", "bytez", "local")
    if body.provider not in allowed:
        return {"provider": get_effective_provider(), "error": f"Provider must be one of: {allowed}"}
    set_llm_provider(body.provider)
    return {"provider": get_effective_provider()}
