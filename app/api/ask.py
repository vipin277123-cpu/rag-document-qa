from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.models.schemas import AskRequest, AskResponse, Source
from app.services.generator import GenerationError


router = APIRouter(prefix="/api", tags=["questions"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank.")
    runtime = request.app.state.runtime
    try:
        chunks = await run_in_threadpool(
            runtime.retriever.retrieve,
            question,
            min(runtime.settings.retrieval_top_k, 5),
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {error}") from error
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents were found. Upload documents or ask a question covered by them.",
        )
    try:
        answer = await run_in_threadpool(runtime.generator.generate, question, chunks)
    except GenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    sources = [
        Source(
            document=chunk["filename"],
            filename=chunk["filename"],
            page=chunk.get("page_number"),
            page_number=chunk.get("page_number"),
            chunk_index=chunk["chunk_index"],
            relevance_score=chunk["relevance_score"],
            score=chunk["relevance_score"],
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
        )
        for chunk in chunks
    ]
    return AskResponse(answer=answer, sources=sources)
