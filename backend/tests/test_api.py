"""
API tests using FastAPI TestClient with mocked dependencies.
No live database or external services required.
Run: pytest backend/tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from src.main import app
from src.main import get_db, get_embedder, get_generator
from src.inference.llm_adapters import LLMAuthError


# ── Shared mocks ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    db = MagicMock()
    db.ping.return_value = True
    db.list_uploads.return_value = [
        {"filename": "doc.pdf", "blob_url": "", "page_count": 2,
         "chunk_count": 4, "uploaded_at": "2026-01-01T00:00:00", "status": "indexed", "embedded": True}
    ]
    db.search_chunks.return_value = [
        {"filename": "doc.pdf", "page": 1, "chunk_index": 0, "content": "hello world", "score": 0.9}
    ]
    db.get_messages.return_value = [
        {"role": "user",      "content": "hi",    "chunks": [], "created_at": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "hello", "chunks": [], "created_at": "2026-01-01T00:00:01"},
    ]
    return db


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 384]
    return embedder


@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.generate.return_value = "The answer is 42."
    return gen


@pytest.fixture
def client(mock_db, mock_embedder, mock_generator):
    app.dependency_overrides[get_db]        = lambda: mock_db
    app.dependency_overrides[get_embedder]  = lambda: mock_embedder
    app.dependency_overrides[get_generator] = lambda: mock_generator
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Health ───────────────────────────────────────────────────────────────────

def test_health_ok(client, mock_db):
    mock_db.ping.return_value = True
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_degraded(client, mock_db):
    mock_db.ping.return_value = False
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"


# ── Documents ────────────────────────────────────────────────────────────────

def test_list_documents(client, mock_db):
    r = client.get("/documents")
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "doc.pdf"


def test_delete_file_success(client, mock_db):
    mock_db.remove_upload.return_value = None
    r = client.delete("/files/doc.pdf")
    assert r.status_code == 200
    assert r.json()["deleted"] == "doc.pdf"


def test_delete_file_not_found(client, mock_db):
    mock_db.remove_upload.side_effect = FileNotFoundError("missing.pdf")
    r = client.delete("/files/missing.pdf")
    assert r.status_code == 404


# ── History ──────────────────────────────────────────────────────────────────

def test_get_history(client, mock_db):
    r = client.get("/history")
    assert r.status_code == 200
    messages = r.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


# ── /query ───────────────────────────────────────────────────────────────────

def test_query_success(client, mock_db, mock_embedder, mock_generator):
    r = client.post("/query", json={"question": "What is this about?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "The answer is 42."
    assert len(body["chunks"]) == 1
    assert "latency" in body
    mock_embedder.embed.assert_called_once()
    mock_db.search_chunks.assert_called_once()
    mock_generator.generate.assert_called_once()


def test_query_empty_question(client):
    r = client.post("/query", json={"question": "   "})
    assert r.status_code == 400


def test_query_top_k_clamped(client, mock_db):
    client.post("/query", json={"question": "test", "top_k": 999})
    call_kwargs = mock_db.search_chunks.call_args
    assert call_kwargs.kwargs.get("k", call_kwargs.args[2] if len(call_kwargs.args) > 2 else None) <= 20


def test_query_search_modes(client):
    for mode in ("hybrid", "semantic", "keyword"):
        r = client.post("/query", json={"question": "test", "search_mode": mode})
        assert r.status_code == 200


# ── /chat ────────────────────────────────────────────────────────────────────

def test_chat_success(client, mock_db, mock_embedder, mock_generator):
    r = client.post("/chat", json={"question": "What is this about?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "The answer is 42."
    assert len(body["chunks"]) == 1
    assert "latency" in body
    mock_generator.generate.assert_called_once()


def test_chat_saves_messages(client, mock_db):
    client.post("/chat", json={"question": "Save me"})
    calls = [c.args[0] for c in mock_db.add_message.call_args_list]
    assert "user" in calls
    assert "assistant" in calls


def test_chat_empty_question(client):
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 400


def test_chat_embedding_failure(client, mock_embedder):
    mock_embedder.embed.side_effect = RuntimeError("embed service down")
    r = client.post("/chat", json={"question": "test"})
    assert r.status_code == 503


def test_chat_llm_auth_failure(client, mock_generator):
    mock_generator.generate = AsyncMock(side_effect=LLMAuthError("LLM API key is invalid or unauthorized"))
    r = client.post("/chat", json={"question": "test"})
    assert r.status_code == 401
    assert r.json()["detail"] == "LLM API key is invalid or unauthorized"


def test_query_llm_auth_failure(client, mock_generator):
    mock_generator.generate = AsyncMock(side_effect=LLMAuthError("LLM API key is invalid or unauthorized"))
    r = client.post("/query", json={"question": "test"})
    assert r.status_code == 401
    assert r.json()["detail"] == "LLM API key is invalid or unauthorized"


def test_chat_llm_auth_failure_does_not_save_messages(client, mock_db, mock_generator):
    mock_generator.generate = AsyncMock(side_effect=LLMAuthError("LLM API key is invalid or unauthorized"))
    client.post("/chat", json={"question": "test"})
    mock_db.add_message.assert_not_called()


def test_chat_uses_history(client, mock_db, mock_generator):
    mock_db.get_messages.return_value = [
        {"role": "user", "content": "prior question", "chunks": [], "created_at": "2026-01-01T00:00:00"},
    ]
    client.post("/chat", json={"question": "follow-up"})
    call_args = mock_generator.generate.call_args
    history = call_args.args[2] if len(call_args.args) > 2 else call_args.kwargs.get("history", [])
    assert len(history) == 1
    assert history[0]["role"] == "user"
