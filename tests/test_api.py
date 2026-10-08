from app.services.generator import AnswerGenerator


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_endpoint_returns_answer_and_citations(client):
    text = "Supervised learning uses labeled examples to learn a mapping from inputs to outputs."
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("learning.txt", text.encode(), "text/plain")},
    )
    assert upload.status_code == 201

    response = client.post("/api/ask", json={"question": text})
    assert response.status_code == 200
    result = response.json()
    assert result["answer"].strip() not in {"", "[Source 1]"}
    assert result["sources"][0]["document"] == "learning.txt"
    assert result["sources"][0]["filename"] == "learning.txt"
    assert result["sources"][0]["page"] == 1
    assert result["sources"][0]["page_number"] == 1
    assert result["sources"][0]["chunk_index"] == 0
    assert result["sources"][0]["relevance_score"] > 0.99
    assert result["sources"][0]["score"] == result["sources"][0]["relevance_score"]
    assert result["sources"][0]["chunk_id"]
    assert "Supervised learning" in result["sources"][0]["text"]


def test_ask_rejects_blank_question(client):
    response = client.post("/api/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_without_documents_returns_not_found(client):
    response = client.post("/api/ask", json={"question": "Anything in the files?"})
    assert response.status_code == 404


def test_ask_generates_locally_without_api_key(client, runtime):
    assert not hasattr(runtime.settings, "openai_api_key")
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("source.txt", b"A local generator can answer from this retrieved source.", "text/plain")},
    )
    assert upload.status_code == 201

    response = client.post("/api/ask", json={"question": "What can answer from the retrieved source?"})

    assert response.status_code == 200, response.text
    assert response.json()["answer"].strip()
    assert response.json()["sources"][0]["filename"] == "source.txt"


def test_ask_rejects_question_over_limit(client):
    response = client.post("/api/ask", json={"question": "x" * 4001})
    assert response.status_code == 422


def test_ask_limits_response_to_five_sources(client):
    for index in range(7):
        response = client.post(
            "/api/documents/upload",
            files={"file": (f"source-{index}.txt", f"Unique retrieval passage {index}.".encode(), "text/plain")},
        )
        assert response.status_code == 201

    result = client.post("/api/ask", json={"question": "Unique retrieval passage 0."})

    assert result.status_code == 200
    assert len(result.json()["sources"]) == 5


def test_generation_prompt_contains_only_retrieved_context_and_instructions():
    chunks = [
        {
            "filename": "learning.txt",
            "page_number": 2,
            "text": "Supervised learning uses labeled examples.",
        }
    ]

    prompt = AnswerGenerator.build_prompt("What does supervised learning use?", chunks)

    assert "Supervised learning uses labeled examples." in prompt
    assert "Use only the context to answer" in prompt
    assert "Do not invent facts" in prompt
    assert "not available in the uploaded documents" in prompt
