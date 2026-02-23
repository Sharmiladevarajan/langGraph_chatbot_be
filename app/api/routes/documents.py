"""Document upload API: PDF and TXT."""
from fastapi import APIRouter, File, UploadFile, Form
from typing import Optional

from app.services.document_service import process_and_store_document
from app.models.schemas import DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])


ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    subject: Optional[str] = Form(None),
) -> DocumentUploadResponse:
    """Accept PDF or TXT file; chunk and store in vector database."""
    suffix = file.filename and file.filename[file.filename.rfind(".") :].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return DocumentUploadResponse(
            doc_id="",
            filename=file.filename or "unknown",
            subject=subject,
            chunks_stored=0,
            message=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    content = await file.read()
    return process_and_store_document(
        filename=file.filename or "document",
        content=content,
        subject=subject,
    )
