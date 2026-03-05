"""
Text chunking for Mars Science RAG.

Splits documents into overlapping chunks suitable for embedding and retrieval.
Implements recursive character splitting without external dependencies.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default separators ordered by priority (paragraph → sentence → word)
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

CHUNK_SIZE = 512       # characters per chunk
CHUNK_OVERLAP = 64     # overlap between consecutive chunks


@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""
    text: str
    chunk_id: str
    doc_id: str
    source: str               # file path or URL
    title: str = ""
    chunk_index: int = 0      # position within document
    total_chunks: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            **self.metadata,
        }


def _make_chunk_id(doc_id: str, index: int) -> str:
    """Deterministic chunk ID from doc_id + index."""
    raw = f"{doc_id}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_doc_id(source: str, title: str = "") -> str:
    """Deterministic document ID from source path."""
    raw = f"{source}::{title}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Recursively split text by separators, keeping chunks under chunk_size.

    Tries the highest-priority separator first.  If a resulting piece is
    still too long, recurse with the next separator.
    """
    if separators is None:
        separators = list(DEFAULT_SEPARATORS)

    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # Last resort: hard split at chunk_size boundary
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    rest = separators[1:]

    parts = text.split(sep)
    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If the single part itself exceeds chunk_size, recurse deeper
            if len(part) > chunk_size:
                chunks.extend(_split_text(part, chunk_size, rest))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks


def chunk_text(
    text: str,
    source: str,
    title: str = "",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    metadata: Optional[dict] = None,
) -> List[Chunk]:
    """
    Split a document into overlapping Chunk objects.

    Parameters
    ----------
    text : str
        Full document text.
    source : str
        Origin path or URL for citation tracking.
    title : str
        Document title.
    chunk_size : int
        Target characters per chunk.
    chunk_overlap : int
        Characters of overlap between consecutive chunks.
    metadata : dict, optional
        Extra metadata attached to every chunk.

    Returns
    -------
    List[Chunk]
        Ordered list of chunks with IDs and provenance.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    raw_splits = _split_text(text.strip(), chunk_size)

    # Apply overlap: prepend tail of previous chunk to current
    overlapped: List[str] = []
    for i, segment in enumerate(raw_splits):
        if i > 0 and chunk_overlap > 0:
            prev_tail = raw_splits[i - 1][-chunk_overlap:]
            segment = prev_tail + segment
        overlapped.append(segment.strip())

    doc_id = _make_doc_id(source, title)
    extra = metadata or {}

    chunks = []
    for idx, segment in enumerate(overlapped):
        if not segment:
            continue
        chunks.append(
            Chunk(
                text=segment,
                chunk_id=_make_chunk_id(doc_id, idx),
                doc_id=doc_id,
                source=source,
                title=title,
                chunk_index=idx,
                total_chunks=len(overlapped),
                metadata=extra,
            )
        )

    logger.info(f"Chunked '{title or source}' → {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
