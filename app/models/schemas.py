"""Pydantic schemas for API and internal state."""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_documents: bool = True
    subject_filter: Optional[str] = None  # e.g. "science" to restrict to that subject


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    sources: Optional[List[dict]] = None


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    subject: Optional[str] = None
    chunks_stored: int
    message: str


class DocumentChunkMetadata(BaseModel):
    """Metadata stored with each chunk for filtering and attribution."""
    doc_id: str
    filename: str
    subject: Optional[str] = None  # e.g. "science", "maths", "biology"
    page: Optional[int] = None
    chunk_index: int
