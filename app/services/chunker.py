from typing import Any
from uuid import uuid4

import tiktoken

from app.services.document_processor import DocumentPage


class Chunker:
    """Token-aware chunking that keeps each chunk's source metadata intact."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_document(
        self,
        document_id: str,
        filename: str,
        pages: list[DocumentPage],
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for page in pages:
            tokens = self._tokenizer.encode(page.text)
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                text = self._tokenizer.decode(tokens[start:end]).strip()
                while end > start and len(self._tokenizer.encode(text)) > self.chunk_size:
                    end -= 1
                    text = self._tokenizer.decode(tokens[start:end]).strip()
                if not text:
                    break
                chunks.append(
                    {
                        "chunk_id": str(uuid4()),
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page.page_number,
                        "chunk_index": len(chunks),
                        "text": text,
                        "document_metadata": page.metadata,
                    }
                )
                if end >= len(tokens):
                    break
                start = max(start + 1, end - self.overlap)
        return [chunk for chunk in chunks if chunk["text"]]
