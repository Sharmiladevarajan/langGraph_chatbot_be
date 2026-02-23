"""Vector store using Pinecone for document chunks with optional subject filtering."""
import uuid
from typing import List, Optional

from app.config import settings
from app.core.embeddings import get_embedding_model

_vector_store = None


def _ensure_index():
    """Get or create Pinecone index with the configured dimension."""
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    name = settings.PINECONE_INDEX_NAME
    existing = [i["name"] for i in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=settings.PINECONE_INDEX_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION,
            ),
        )
        import time
        while not pc.describe_index(name).status.get("ready"):
            time.sleep(1)
    return pc.Index(name)


def get_vector_store():
    """Get or create the Pinecone vector store."""
    global _vector_store
    if _vector_store is None:
        if not settings.PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is required. Set it in .env or environment."
            )
        from langchain_pinecone import PineconeVectorStore

        index = _ensure_index()
        embedding = get_embedding_model()
        _vector_store = PineconeVectorStore(
            index=index,
            embedding=embedding,
        )
    return _vector_store


def add_chunks(chunks_with_meta: List[tuple[str, dict]]) -> None:
    """Add text chunks and metadata to Pinecone."""
    vs = get_vector_store()
    texts = [c[0] for c in chunks_with_meta]
    metadatas = [c[1] for c in chunks_with_meta]
    ids = [str(uuid.uuid4()) for _ in texts]
    # Pinecone metadata: only str, number, bool, list[str]
    safe_metadatas = []
    for m in metadatas:
        safe = {}
        for k, v in m.items():
            if v is None:
                continue
            if isinstance(v, (str, bool)):
                safe[k] = v
            elif isinstance(v, (int, float)):
                safe[k] = v
            else:
                safe[k] = str(v)
        safe_metadatas.append(safe)
    vs.add_texts(texts=texts, metadatas=safe_metadatas, ids=ids)


def similarity_search(
    query: str,
    k: int = 4,
    subject_filter: Optional[str] = None,
) -> List[dict]:
    """
    Retrieve relevant chunks. If subject_filter is set, only chunks
    with matching subject metadata are returned (prevents unrelated docs).
    """
    vs = get_vector_store()
    if subject_filter:
        filter_dict = {"subject": subject_filter}
        docs = vs.similarity_search(query, k=k, filter=filter_dict)
    else:
        docs = vs.similarity_search(query, k=k)
    return [
        {"content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]
