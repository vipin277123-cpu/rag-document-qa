import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import Settings
from app.models.schemas import DocumentSummary, DocumentUploadResponse
from app.services.chunker import Chunker
from app.services.document_processor import DocumentProcessor
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.utils.file_utils import validate_upload


class DocumentManager:
    """Coordinates document validation, extraction, indexing, and persistence."""

    def __init__(
        self,
        settings: Settings,
        processor: DocumentProcessor,
        chunker: Chunker,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.settings = settings
        self.processor = processor
        self.chunker = chunker
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self.settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.settings.vectorstore_dir / "documents.json"
        self._lock = threading.RLock()
        self._documents: dict[str, dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        if not self.registry_path.exists():
            return {}
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        documents: dict[str, dict[str, Any]] = {}
        for item in data:
            document_id = item["id"]
            stored_filename = item.get("stored_filename", "")
            stored_path = self.settings.upload_dir / stored_filename
            item.setdefault("document_id", document_id)
            item.setdefault("size_bytes", stored_path.stat().st_size if stored_path.is_file() else 0)
            documents[document_id] = item
        return documents

    def _persist_registry(self) -> None:
        temp_path = self.registry_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(list(self._documents.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, self.registry_path)

    def ingest(self, filename: str, content_type: str | None, content: bytes) -> DocumentUploadResponse:
        if not content:
            raise ValueError("The uploaded document is empty.")
        if len(content) > self.settings.max_upload_size_bytes:
            raise OverflowError("File exceeds the 10 MB upload limit.")
        safe_filename = validate_upload(filename, content_type, content)
        document_id = str(uuid4())
        stored_filename = f"{document_id}_{safe_filename}"
        file_path = self.settings.upload_dir / stored_filename
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        temp_path.write_bytes(content)
        os.replace(temp_path, file_path)

        indexed = False
        try:
            pages = self.processor.extract(file_path)
            chunks = self.chunker.chunk_document(document_id, safe_filename, pages)
            if not chunks:
                raise ValueError("No text could be chunked from the uploaded document.")
            vectors = self.embeddings.embed_texts([chunk["text"] for chunk in chunks])
            with self._lock:
                self.vector_store.add(vectors, chunks)
                indexed = True
                record = {
                    "id": document_id,
                    "document_id": document_id,
                    "filename": safe_filename,
                    "status": "ready",
                    "size_bytes": len(content),
                    "chunk_count": len(chunks),
                    "uploaded_at": datetime.now(UTC).isoformat(),
                    "page_count": len(pages),
                    "stored_filename": stored_filename,
                }
                self._documents[document_id] = record
                self._persist_registry()
            return DocumentUploadResponse(**record)
        except Exception as error:
            if indexed:
                self.vector_store.delete_document(document_id)
            file_path.unlink(missing_ok=True)
            if isinstance(error, (ValueError, OverflowError)):
                raise
            raise ValueError(f"Failed to process the uploaded document: {error}") from error

    def list_documents(self) -> list[DocumentSummary]:
        with self._lock:
            return [
                DocumentSummary(**{key: value for key, value in record.items() if key != "stored_filename" and key != "page_count"})
                for record in sorted(self._documents.values(), key=lambda item: item["uploaded_at"], reverse=True)
            ]

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            record = self._documents.get(document_id)
            if record is None:
                return False
            file_path = self.settings.upload_dir / record["stored_filename"]
            file_path.unlink(missing_ok=True)
            self.vector_store.delete_document(document_id)
            del self._documents[document_id]
            self._persist_registry()
            return True
