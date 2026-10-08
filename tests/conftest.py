import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.chunker import Chunker
from app.services.document_manager import DocumentManager
from app.services.document_processor import DocumentProcessor
from app.services.embeddings import EmbeddingService
from app.services.generator import AnswerGenerator
from app.services.retriever import Retriever
from app.services.runtime import Runtime
from app.services.vector_store import VectorStore


@pytest.fixture
def settings(tmp_path):
    return Settings(
        upload_dir=tmp_path / "uploads",
        sample_documents_dir=tmp_path / "samples",
        vectorstore_dir=tmp_path / "vectorstore",
        generation_model="google/flan-t5-base",
        generation_device=-1,
        max_upload_size_bytes=10 * 1024 * 1024,
        chunk_size_tokens=20,
        chunk_overlap_tokens=3,
        min_relevance_score=0.0,
        retrieval_top_k=5,
    )


@pytest.fixture
def runtime(settings):
    embeddings = EmbeddingService(settings.embedding_model, settings.embedding_dimension)
    vector_store = VectorStore(settings.vectorstore_dir, settings.embedding_dimension)
    processor = DocumentProcessor()
    chunker = Chunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    documents = DocumentManager(settings, processor, chunker, embeddings, vector_store)
    return Runtime(
        settings=settings,
        embeddings=embeddings,
        vector_store=vector_store,
        retriever=Retriever(embeddings, vector_store, settings.min_relevance_score),
        generator=AnswerGenerator(settings),
        documents=documents,
    )


@pytest.fixture
def client(runtime):
    with TestClient(create_app(runtime=runtime, settings=runtime.settings)) as test_client:
        yield test_client
