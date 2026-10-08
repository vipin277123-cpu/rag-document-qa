from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class Source(BaseModel):
    document: str
    filename: str
    page: int | None
    page_number: int | None
    chunk_index: int
    relevance_score: float
    score: float
    chunk_id: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentSummary(BaseModel):
    id: str
    document_id: str
    filename: str
    status: Literal["ready"]
    size_bytes: int
    chunk_count: int
    uploaded_at: datetime


class DocumentUploadResponse(DocumentSummary):
    page_count: int


class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"
