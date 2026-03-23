"""
chunker.py

Splits raw text into overlapping chunks with metadata.
This is the first stage of the RAG injection scanning pipeline.
Every downstream layer (regex, heuristics, LLM judge) operates on
the output of this module.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE: int = 512
DEFAULT_OVERLAP: int = 50


# ── Core Data Structure ──────────────────────────────────────────────────────

def make_chunk(
    chunk_index: int,
    text: str,
    source: str,
    start_char: int,
    end_char: int,
) -> dict[str, Any]:
    """
    Build a single chunk dictionary.

    Args:
        chunk_index: Position of this chunk in the document sequence.
        text:        The raw text content of this chunk.
        source:      Filename or identifier of the originating document.
        start_char:  Character offset where this chunk begins in the original text.
        end_char:    Character offset where this chunk ends in the original text.

    Returns:
        A dictionary representing one chunk with all metadata attached.
    """
    return {
        "chunk_index": chunk_index,
        "text": text,
        "source": source,
        "start_char": start_char,
        "end_char": end_char,
        "token_count": len(text),
    }


# ── Main Chunking Function ───────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split a raw text string into overlapping chunks with metadata.

    Args:
        text:       The raw text to chunk.
        source:     Filename or identifier of the originating document.
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        A list of chunk dictionaries. Returns an empty list if text is empty.

    Raises:
        ValueError: If chunk_size <= 0 or overlap >= chunk_size.
    """
    # ── Guard clauses ────────────────────────────────────────────────────────

    if not text or not text.strip():
        logger.warning("chunk_text received empty text from source: %s", source)
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    # ── Chunking loop ────────────────────────────────────────────────────────

    chunks: list[dict[str, Any]] = []
    chunk_index: int = 0
    start: int = 0
    step: int = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_slice = text[start:end]

        chunk = make_chunk(
            chunk_index=chunk_index,
            text=chunk_text_slice,
            source=source,
            start_char=start,
            end_char=end,
        )

        chunks.append(chunk)
        logger.debug(
            "Created chunk %d | source=%s | chars %d-%d",
            chunk_index, source, start, end,
        )

        chunk_index += 1
        start += step

    logger.info(
        "Chunking complete | source=%s | total_chunks=%d", source, len(chunks)
    )
    return chunks