"""Document chunking for RAG - splits large files into overlapping chunks."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional
from app.config import settings
from app.models.schemas import DocumentChunkMetadata


def chunk_text(
    text: str,
    doc_id: str,
    filename: str,
    subject: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[tuple[str, dict]]:
    """
    Split text into overlapping chunks with metadata.
    Returns list of (chunk_text, metadata_dict) for vector store.
    """
    size = chunk_size or settings.CHUNK_SIZE
    overlap = chunk_overlap or settings.CHUNK_OVERLAP
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    result = []
    for i, content in enumerate(chunks):
        meta = {
            "doc_id": doc_id,
            "filename": filename,
            "subject": subject or "general",
            "chunk_index": i,
        }
        result.append((content, meta))
    return result
