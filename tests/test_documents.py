import json
from io import BytesIO

from docx import Document

from app.services.document_processor import DocumentProcessor


def _make_pdf(page_texts: list[str]) -> bytes:
    page_ids = [4 + index * 2 for index in range(len(page_texts))]
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] /Count {len(page_ids)} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for index, text in enumerate(page_texts):
        page_id = page_ids[index]
        content_id = page_id + 1
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 72 Td ({escaped}) Tj ET".encode("ascii")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        objects[content_id] = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id in range(1, max(objects) + 1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode() + objects[object_id] + b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(output)


def _make_docx(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def test_valid_upload_and_document_listing(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", b"A small document about reliable retrieval.", "text/plain")},
    )
    assert response.status_code == 201
    uploaded = response.json()
    assert uploaded["filename"] == "notes.txt"
    assert uploaded["chunk_count"] == 1
    assert uploaded["page_count"] == 1
    assert uploaded["status"] == "ready"
    assert uploaded["document_id"] == uploaded["id"]
    assert uploaded["size_bytes"] == len(b"A small document about reliable retrieval.")

    listing = client.get("/api/documents")
    assert listing.status_code == 200
    assert listing.json()[0]["id"] == uploaded["id"]
    assert listing.json()[0]["document_id"] == uploaded["id"]
    assert listing.json()[0]["size_bytes"] == uploaded["size_bytes"]
    assert listing.json()[0]["chunk_count"] == 1


def test_pdf_upload_extracts_each_page_and_preserves_metadata(client, runtime):
    content = _make_pdf(["First page retrieval evidence.", "Second page citation evidence."])
    response = client.post(
        "/api/documents/upload",
        files={"file": ("pages.pdf", content, "application/pdf")},
    )

    assert response.status_code == 201, response.text
    result = response.json()
    assert result["page_count"] == 2
    chunks = [chunk for chunk in runtime.vector_store._metadata.values() if chunk["document_id"] == result["id"]]
    assert {chunk["page_number"] for chunk in chunks} == {1, 2}
    assert all(chunk["filename"] == "pages.pdf" for chunk in chunks)
    assert all(isinstance(chunk["chunk_index"], int) for chunk in chunks)


def test_two_pdf_documents_retrieve_independently_and_delete_one(client, runtime):
    first_text = "The Orion telescope uses a segmented primary mirror to collect distant starlight."
    second_text = "The Cedar greenhouse uses drip irrigation to conserve water for seedlings."
    first_upload = client.post(
        "/api/documents/upload",
        files={"file": ("document1.pdf", _make_pdf([first_text]), "application/pdf")},
    )
    second_upload = client.post(
        "/api/documents/upload",
        files={"file": ("document2.pdf", _make_pdf([second_text]), "application/pdf")},
    )
    assert first_upload.status_code == second_upload.status_code == 201
    first_id = first_upload.json()["document_id"]
    second_id = second_upload.json()["document_id"]

    first_results = runtime.retriever.retrieve("What collects distant starlight?", top_k=5)
    second_results = runtime.retriever.retrieve("How does the greenhouse conserve water?", top_k=5)
    assert first_results[0]["filename"] == "document1.pdf"
    assert second_results[0]["filename"] == "document2.pdf"

    assert client.delete(f"/api/documents/{first_id}").status_code == 204
    assert runtime.vector_store.count_document_chunks(first_id) == 0
    assert runtime.vector_store.count_document_chunks(second_id) == 1
    assert runtime.retriever.retrieve("How does the greenhouse conserve water?", top_k=5)[0]["filename"] == "document2.pdf"


def test_docx_upload_extracts_real_document(client, runtime):
    content = _make_docx("DOCX extraction includes this paragraph.")
    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "notes.docx",
                content,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 201, response.text
    result = response.json()
    chunk = next(
        item for item in runtime.vector_store._metadata.values() if item["document_id"] == result["id"]
    )
    assert chunk["page_number"] == 1
    assert "DOCX extraction includes this paragraph." in chunk["text"]


def test_invalid_extension(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("payload.exe", b"not executable", "application/octet-stream")},
    )
    assert response.status_code == 422
    assert "Unsupported file type" in response.json()["detail"]


def test_file_size_validation(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("large.txt", b"x" * (10 * 1024 * 1024 + 1), "text/plain")},
    )
    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]


def test_empty_text_and_mismatched_mime_are_rejected(client):
    empty = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    mismatch = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", b"valid text", "application/pdf")},
    )
    assert empty.status_code == 422
    assert "empty" in empty.json()["detail"].lower()
    assert mismatch.status_code == 422
    assert "does not match" in mismatch.json()["detail"]


def test_invalid_pdf_is_rejected(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("broken.pdf", b"%PDF-not-a-real-file", "application/pdf")},
    )
    assert response.status_code == 422


def test_pdf_with_no_extractable_text_is_rejected(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("blank.pdf", _make_pdf([""]), "application/pdf")},
    )
    assert response.status_code == 422
    assert "extractable text" in response.json()["detail"]


def test_delete_removes_file_and_vectors(client, runtime):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("remove.txt", b"Document to remove from the index.", "text/plain")},
    )
    document_id = response.json()["id"]
    record = runtime.documents._documents[document_id]
    stored_file = runtime.settings.upload_dir / record["stored_filename"]
    assert stored_file.exists()
    assert runtime.vector_store.count_document_chunks(document_id) == 1

    deleted = client.delete(f"/api/documents/{document_id}")
    assert deleted.status_code == 204
    assert not stored_file.exists()
    assert runtime.vector_store.count_document_chunks(document_id) == 0
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{document_id}").status_code == 404


def test_deleting_one_document_preserves_other_document_and_persistence(client, runtime):
    first = client.post(
        "/api/documents/upload",
        files={"file": ("first.txt", b"First document survives only until selected deletion.", "text/plain")},
    ).json()
    second = client.post(
        "/api/documents/upload",
        files={"file": ("second.txt", b"Second document must remain indexed after deletion.", "text/plain")},
    ).json()

    assert client.delete(f"/api/documents/{first['id']}").status_code == 204
    assert runtime.vector_store.count_document_chunks(first["id"]) == 0
    assert runtime.vector_store.count_document_chunks(second["id"]) == second["chunk_count"]

    from app.services.vector_store import VectorStore

    reloaded = VectorStore(runtime.settings.vectorstore_dir, runtime.settings.embedding_dimension)
    assert reloaded.count_document_chunks(first["id"]) == 0
    assert reloaded.count_document_chunks(second["id"]) == second["chunk_count"]
    from app.services.runtime import Runtime

    restarted = Runtime.create(runtime.settings)
    assert [item.id for item in restarted.documents.list_documents()] == [second["id"]]
    remaining = client.get("/api/documents").json()
    assert [item["id"] for item in remaining] == [second["id"]]


def test_legacy_document_registry_gets_new_listing_fields(settings, runtime):
    document_id = "legacy-document"
    stored_filename = f"{document_id}_legacy.txt"
    stored_path = settings.upload_dir / stored_filename
    stored_path.write_bytes(b"older indexed file")
    registry = [
        {
            "id": document_id,
            "filename": "legacy.txt",
            "status": "ready",
            "chunk_count": 1,
            "uploaded_at": "2026-01-01T00:00:00+00:00",
            "page_count": 1,
            "stored_filename": stored_filename,
        }
    ]
    (settings.vectorstore_dir / "documents.json").write_text(json.dumps(registry), encoding="utf-8")

    from app.services.document_manager import DocumentManager

    manager = DocumentManager(
        settings,
        DocumentProcessor(),
        runtime.documents.chunker,
        runtime.embeddings,
        runtime.vector_store,
    )

    document = manager.list_documents()[0]
    assert document.id == document_id
    assert document.document_id == document_id
    assert document.size_bytes == len(b"older indexed file")


