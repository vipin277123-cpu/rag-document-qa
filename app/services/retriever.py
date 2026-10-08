from typing import Any

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        min_relevance_score: float = 0.15,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.min_relevance_score = min_relevance_score

    def retrieve(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vector = self.embeddings.embed_query(question)
        results = self.vector_store.search(query_vector, top_k=top_k)
        results = [item for item in results if item["relevance_score"] >= self.min_relevance_score]
        return sorted(results, key=lambda item: item["relevance_score"], reverse=True)
