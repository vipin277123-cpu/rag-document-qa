import threading
from typing import Any, ClassVar

import numpy as np


class EmbeddingService:
    """Lazily load the sentence-transformer model and return normalized vectors."""

    _models: ClassVar[dict[str, Any]] = {}
    _model_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_name: str, dimension: int = 384, model: Any | None = None) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model = model

    def _load_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self.model_name not in self._models:
                    from sentence_transformers import SentenceTransformer

                    self._models[self.model_name] = SentenceTransformer(self.model_name)
                self._model = self._models[self.model_name]
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        raw = self._load_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError(
                f"Embedding model returned shape {vectors.shape}; expected ({len(texts)}, {self.dimension})."
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding model returned a zero vector.")
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def embed_query(self, question: str) -> np.ndarray:
        return self.embed_texts([question])[0]
