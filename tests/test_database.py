"""
Integration tests for PostgreSQLStorageManager.
Requires a live DATABASE_URL — skipped automatically if unavailable.
Run: pytest tests/test_database.py -v
"""
import os
import pytest
from src.storage.database import DBManager


@pytest.fixture(scope="module")
def db():
    url = os.getenv("DATABASE_URL", "")
    if not url:
        pytest.skip("DATABASE_URL not set")
    try:
        manager = DBManager.create()
    except Exception as e:
        pytest.skip(f"Cannot connect to database: {e}")
    return manager


# ── Messages ────────────────────────────────────────────────────────────────

def test_add_and_get_messages(db):
    before = len(db.get_messages(limit=10000))
    db.add_message("user", "hello test")
    db.add_message("assistant", "hi there", chunks=[{"filename": "a.pdf", "page": 1}])
    messages = db.get_messages(limit=10000)
    assert len(messages) == before + 2
    last_two = messages[-2:]
    assert last_two[0]["role"] == "user"
    assert last_two[0]["content"] == "hello test"
    assert last_two[1]["role"] == "assistant"
    assert last_two[1]["content"] == "hi there"
    assert last_two[1]["chunks"][0]["filename"] == "a.pdf"


def test_get_messages_limit(db):
    # Insert enough to exceed limit
    for i in range(3):
        db.add_message("user", f"limit test {i}")
    messages = db.get_messages(limit=2)
    assert len(messages) <= 2


def test_messages_ordered_asc(db):
    messages = db.get_messages()
    timestamps = [m["created_at"] for m in messages if m["created_at"]]
    assert timestamps == sorted(timestamps)


def test_add_message_no_chunks(db):
    before = len(db.get_messages(limit=10000))
    db.add_message("user", "no chunks here")
    messages = db.get_messages(limit=10000)
    assert len(messages) == before + 1
    last = messages[-1]
    assert last["chunks"] == [] or last["chunks"] is None or last["chunks"] == {}


# ── Uploads ──────────────────────────────────────────────────────────────────

_TEST_FILENAME = "__pytest_test_upload__.pdf"


@pytest.fixture(autouse=False)
def cleanup_test_upload(db):
    yield
    try:
        db.remove_upload(_TEST_FILENAME)
    except FileNotFoundError:
        pass


def test_add_upload_appears_in_list(db, cleanup_test_upload):
    db.add_upload(_TEST_FILENAME, "https://example.com/test.pdf")
    docs = db.list_uploads()
    filenames = [d["filename"] for d in docs]
    assert _TEST_FILENAME in filenames


def test_add_upload_sets_pending_status(db, cleanup_test_upload):
    db.add_upload(_TEST_FILENAME, "https://example.com/test.pdf")
    docs = db.list_uploads()
    doc = next(d for d in docs if d["filename"] == _TEST_FILENAME)
    assert doc["status"] == "pending"


def test_set_status(db, cleanup_test_upload):
    db.add_upload(_TEST_FILENAME, "https://example.com/test.pdf")
    db.set_status(_TEST_FILENAME, "indexed", page_count=5)
    docs = db.list_uploads()
    doc = next(d for d in docs if d["filename"] == _TEST_FILENAME)
    assert doc["status"] == "indexed"
    assert doc["page_count"] == 5


def test_remove_upload(db):
    db.add_upload(_TEST_FILENAME, "https://example.com/test.pdf")
    db.remove_upload(_TEST_FILENAME)
    docs = db.list_uploads()
    assert _TEST_FILENAME not in [d["filename"] for d in docs]


def test_remove_upload_raises_on_missing(db):
    with pytest.raises(FileNotFoundError):
        db.remove_upload("__nonexistent_file__.pdf")


# ── Ping ────────────────────────────────────────────────────────────────────

def test_ping(db):
    assert db.ping() is True
