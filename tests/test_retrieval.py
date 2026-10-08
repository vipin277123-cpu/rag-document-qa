import numpy as np

from app.services.chunker import Chunker
from app.services.document_processor import DocumentPage
from app.services.embeddings import EmbeddingService


def test_chunking_respects_token_limit_and_keeps_source_metadata():
    chunker = Chunker(chunk_size=50, overlap=5)
    page = DocumentPage(text="retrieval " * 120, page_number=7, metadata={"title": "Study"})
    chunks = chunker.chunk_document("doc-1", "study.txt", [page])
    assert len(chunks) > 1
    assert all(chunk["document_id"] == "doc-1" for chunk in chunks)
    assert all(chunk["filename"] == "study.txt" for chunk in chunks)
    assert all(chunk["page_number"] == 7 for chunk in chunks)
    assert all(chunk["document_metadata"]["title"] == "Study" for chunk in chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(len(chunker._tokenizer.encode(chunk["text"])) <= 50 for chunk in chunks)


def test_chunking_preserves_overlap_between_adjacent_chunks():
    chunker = Chunker(chunk_size=50, overlap=10)
    page = DocumentPage(text="alpha beta gamma delta epsilon " * 100, page_number=1)
    chunks = chunker.chunk_document("doc-overlap", "overlap.txt", [page])
    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()
    shared_word_count = max(
        (
            length
            for length in range(1, min(len(first_words), len(second_words), 10) + 1)
            if first_words[-length:] == second_words[:length]
        ),
        default=0,
    )
    assert shared_word_count >= 2


def test_embedding_generation_normalizes_vectors(runtime):
    vectors = runtime.embeddings.embed_texts(["first text", "second text"])
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(runtime.embeddings.embed_query("query")), 1.0, atol=1e-6)


def test_configured_sentence_transformer_produces_384_normalized_values():
    embedding_service = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", dimension=384)
    vector = embedding_service.embed_query("A real document passage for retrieval.")
    assert vector.shape == (384,)
    assert vector.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_faiss_search_ranks_and_deletes_document_chunks(runtime):
    text = "the exact phrase to retrieve"
    chunk = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "filename": "guide.txt",
        "page_number": 1,
        "chunk_index": 0,
        "text": text,
    }
    vector = runtime.embeddings.embed_texts([text])
    runtime.vector_store.add(vector, [chunk])

    results = runtime.retriever.retrieve(text, top_k=5)
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["relevance_score"] > 0.99
    assert runtime.vector_store.delete_document("doc-1") == 1
    assert runtime.retriever.retrieve(text, top_k=5) == []


def test_faiss_returns_top_five_sorted_chunks_across_documents(runtime):
    texts = [f"document passage number {index} about retrieval" for index in range(8)]
    chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "document_id": f"doc-{index}",
            "filename": f"file-{index}.txt",
            "page_number": 1,
            "chunk_index": 0,
            "text": text,
        }
        for index, text in enumerate(texts)
    ]
    runtime.vector_store.add(runtime.embeddings.embed_texts(texts), chunks)

    results = runtime.retriever.retrieve(texts[0], top_k=5)

    assert len(results) == 5
    assert results[0]["chunk_id"] == "chunk-0"
    assert [item["relevance_score"] for item in results] == sorted(
        (item["relevance_score"] for item in results), reverse=True
    )
    assert all({"filename", "page_number", "chunk_index", "relevance_score", "text"} <= item.keys() for item in results)
