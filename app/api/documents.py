from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.models.schemas import DocumentSummary, DocumentUploadResponse


router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, file: UploadFile = File(...)) -> DocumentUploadResponse:
    runtime = request.app.state.runtime
    content = await file.read(runtime.settings.max_upload_size_bytes + 1)
    await file.close()
    if len(content) > runtime.settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB upload limit.")
    try:
        return await run_in_threadpool(runtime.documents.ingest, file.filename or "", file.content_type, content)
    except OverflowError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("", response_model=list[DocumentSummary])
def list_documents(request: Request) -> list[DocumentSummary]:
    return request.app.state.runtime.documents.list_documents()


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, request: Request) -> None:
    deleted = request.app.state.runtime.documents.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
