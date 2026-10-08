import json
import os
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class VectorStore:
    """Persistent cosine-similarity index with a separate vector-ID metadata map."""

    def __init__(self, directory: Path, dimension: int = 384) -> None:
        self.directory = directory
        self.dimension = dimension
        self.index_path = directory / "index.faiss"
        self.metadata_path = directory / "metadata.json"
        self._lock = threading.RLock()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()
        self._metadata = self._load_metadata()
        self._next_id = max(self._metadata, default=-1) + 1
        if self._index.ntotal != len(self._metadata):
            raise ValueError("FAISS index and metadata mapping are inconsistent.")

    def _load_index(self) -> Any:
        if not self.index_path.exists():
            return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))
        index = faiss.read_index(str(self.index_path))
        if index.d != self.dimension:
            raise ValueError(f"Stored index dimension {index.d} does not match {self.dimension}.")
        if not isinstance(index, faiss.IndexIDMap2):
            raise ValueError("Stored FAISS index has an unsupported format.")
        return index

    def _load_metadata(self) -> dict[int, dict[str, Any]]:
        if not self.metadata_path.exists():
            return {}
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return {int(vector_id): value for vector_id, value in data.items()}

    def _persist(self) -> None:
        index_temp = self.index_path.with_suffix(".faiss.tmp")
        metadata_temp = self.metadata_path.with_suffix(".json.tmp")
        faiss.write_index(self._index, str(index_temp))
        metadata_temp.write_text(
            json.dumps({str(key): value for key, value in self._metadata.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(index_temp, self.index_path)
        os.replace(metadata_temp, self.metadata_path)

    def add(self, vectors: np.ndarray, chunks: list[dict[str, Any]]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("The number of vectors must match the number of chunks.")
        if not chunks:
            return
        normalized = np.ascontiguousarray(vectors, dtype=np.float32)
        if normalized.shape != (len(chunks), self.dimension):
            raise ValueError(f"Expected vectors with shape ({len(chunks)}, {self.dimension}).")
        with self._lock:
            ids = np.arange(self._next_id, self._next_id + len(chunks), dtype=np.int64)
            self._index.add_with_ids(normalized, ids)
            self._metadata.update({int(vector_id): chunk for vector_id, chunk in zip(ids, chunks, strict=True)})
            self._next_id += len(chunks)
            self._persist()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        if query.shape[1] != self.dimension:
            raise ValueError(f"Expected query vector dimension {self.dimension}.")
        with self._lock:
            if self._index.ntotal == 0:
                return []
            scores, ids = self._index.search(query, min(top_k, int(self._index.ntotal)))
            results = [
                {**self._metadata[int(vector_id)], "relevance_score": float(score)}
                for score, vector_id in zip(scores[0], ids[0], strict=True)
                if vector_id >= 0 and int(vector_id) in self._metadata
            ]
        return sorted(results, key=lambda item: item["relevance_score"], reverse=True)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            ids = [key for key, value in self._metadata.items() if value.get("document_id") == document_id]
            if not ids:
                return 0
            self._index.remove_ids(np.asarray(ids, dtype=np.int64))
            for vector_id in ids:
                del self._metadata[vector_id]
            self._persist()
            return len(ids)

    def count_document_chunks(self, document_id: str) -> int:
        with self._lock:
            return sum(value.get("document_id") == document_id for value in self._metadata.values())
