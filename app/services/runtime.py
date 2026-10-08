from dataclasses import dataclass

from app.config import Settings, get_settings
from app.services.chunker import Chunker
from app.services.document_manager import DocumentManager
from app.services.document_processor import DocumentProcessor
from app.services.embeddings import EmbeddingService
from app.services.generator import AnswerGenerator
from app.services.retriever import Retriever
from app.services.vector_store import VectorStore


@dataclass
class Runtime:
    settings: Settings
    embeddings: EmbeddingService
    vector_store: VectorStore
    retriever: Retriever
    generator: AnswerGenerator
    documents: DocumentManager

    @classmethod
    def create(cls, settings: Settings | None = None) -> "Runtime":
        settings = settings or get_settings()
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        embeddings = EmbeddingService(settings.embedding_model, settings.embedding_dimension)
        vector_store = VectorStore(settings.vectorstore_dir, settings.embedding_dimension)
        processor = DocumentProcessor()
        chunker = Chunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens)
        return cls(
            settings=settings,
            embeddings=embeddings,
            vector_store=vector_store,
            retriever=Retriever(embeddings, vector_store, settings.min_relevance_score),
            generator=AnswerGenerator(settings),
            documents=DocumentManager(settings, processor, chunker, embeddings, vector_store),
        )
