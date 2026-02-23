"""
Full pipeline: document upload -> parse -> chunk -> store in vector memory.
Handles PDF and TXT; supports subject metadata for multi-document separation.
"""
import uuid
from pathlib import Path
from typing import Optional, BinaryIO

from pypdf import PdfReader

from app.core.chunking import chunk_text
from app.services.vector_store import add_chunks
from app.models.schemas import DocumentUploadResponse


def extract_text_from_file(filename: str, content: bytes) -> str:
    """Extract raw text from PDF or TXT file."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return content.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        import io
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported format: {suffix}. Use .pdf or .txt")


def process_and_store_document(
    filename: str,
    content: bytes,
    subject: Optional[str] = None,
) -> DocumentUploadResponse:
    """
    Pipeline: upload -> extract text -> chunk -> store in vector DB.
    Subject is stored as metadata so retrieval can filter by subject later.
    """
    text = extract_text_from_file(filename, content)
    if not text.strip():
        return DocumentUploadResponse(
            doc_id="",
            filename=filename,
            subject=subject,
            chunks_stored=0,
            message="No text content could be extracted.",
        )
    doc_id = str(uuid.uuid4())
    chunks_with_meta = chunk_text(text, doc_id=doc_id, filename=filename, subject=subject)
    add_chunks(chunks_with_meta)
    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=filename,
        subject=subject,
        chunks_stored=len(chunks_with_meta),
        message=f"Stored {len(chunks_with_meta)} chunks.",
    )
