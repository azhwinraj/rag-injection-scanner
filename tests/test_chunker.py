"""
test_chunker.py
Pytest test suite for the chunker module.
"""

import pytest
from rag_scanner.chunker import chunk_text, make_chunk

# ── make_chunk tests ──────────────────────────────────────────────────────────

def test_make_chunk_fields():
    """make_chunk returns all required fields."""
    chunk = make_chunk(0, "hello world", "test.txt", 0, 11)
    assert chunk["chunk_index"] == 0
    assert chunk["text"] == "hello world"
    assert chunk["source"] == "test.txt"
    assert chunk["start_char"] == 0
    assert chunk["end_char"] == 11
    assert chunk["token_count"] == 11


# ── chunk_text basic tests ────────────────────────────────────────────────────

def test_empty_text_returns_empty_list():
    """Empty text produces no chunks."""
    assert chunk_text("", source="test.txt") == []


def test_whitespace_only_returns_empty_list():
    """Whitespace-only text produces no chunks."""
    assert chunk_text("   \n\t  ", source="test.txt") == []


def test_short_text_produces_one_chunk():
    """Text shorter than chunk_size produces exactly one chunk."""
    result = chunk_text("Hello world", source="test.txt")
    assert len(result) == 1
    assert result[0]["text"] == "Hello world"
    assert result[0]["chunk_index"] == 0
    assert result[0]["source"] == "test.txt"


def test_chunk_indices_are_sequential():
    """Chunk indices start at 0 and increment by 1."""
    text = "A" * 1000
    chunks = chunk_text(text, source="test.txt")
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_source_preserved_in_all_chunks():
    """Source field is present and correct in every chunk."""
    text = "A" * 1000
    chunks = chunk_text(text, source="my_document.pdf")
    for chunk in chunks:
        assert chunk["source"] == "my_document.pdf"


# ── overlap tests ─────────────────────────────────────────────────────────────

def test_overlap_creates_shared_content():
    """Consecutive chunks share content equal to overlap size."""
    text = "A" * 1000
    chunks = chunk_text(text, source="test.txt", chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    # Second chunk starts at chunk_size - overlap = 80
    assert chunks[1]["start_char"] == 80


def test_no_overlap_zero():
    """With overlap=0, chunks do not share content."""
    text = "A" * 200
    chunks = chunk_text(text, source="test.txt", chunk_size=100, overlap=0)
    assert chunks[1]["start_char"] == 100


def test_last_chunk_ends_at_text_length():
    """Last chunk end_char equals length of original text."""
    text = "A" * 550
    chunks = chunk_text(text, source="test.txt", chunk_size=512, overlap=50)
    assert chunks[-1]["end_char"] == len(text)


# ── guard clause tests ────────────────────────────────────────────────────────

def test_invalid_chunk_size_raises():
    """chunk_size <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("some text", source="test.txt", chunk_size=0)


def test_overlap_gte_chunk_size_raises():
    """overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", source="test.txt", chunk_size=100, overlap=100)


def test_overlap_gt_chunk_size_raises():
    """overlap > chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", source="test.txt", chunk_size=100, overlap=150)


# ── token count tests ─────────────────────────────────────────────────────────

def test_token_count_matches_text_length():
    """token_count equals length of chunk text."""
    chunks = chunk_text("Hello world this is a test", source="test.txt")
    for chunk in chunks:
        assert chunk["token_count"] == len(chunk["text"])


def test_full_text_covered():
    """All characters in the original text appear in at least one chunk."""
    text = "ABCDEFGHIJ" * 100
    chunks = chunk_text(text, source="test.txt", chunk_size=50, overlap=10)
    # First chunk starts at 0
    assert chunks[0]["start_char"] == 0
    # Last chunk ends at or beyond the last character
    assert chunks[-1]["end_char"] == len(text)