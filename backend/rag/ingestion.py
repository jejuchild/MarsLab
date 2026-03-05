"""
Document ingestion pipeline for Mars Science RAG.

Handles text files, Markdown, and JSON documents.
Orchestrates: read → chunk → embed → store.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from .chunker import Chunk, chunk_text
from .embedder import embed_texts
from .vector_store import upsert_chunks

logger = logging.getLogger(__name__)


def ingest_text(
    text: str,
    source: str,
    title: str = "",
    collection: str = "mars_science",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: Optional[dict] = None,
) -> Dict:
    """
    Ingest a single text document into the RAG vector store.

    Parameters
    ----------
    text : str
        Full document text.
    source : str
        Origin identifier (file path, URL, etc.).
    title : str
        Document title for citation.
    collection : str
        Target vector store collection.
    chunk_size, chunk_overlap : int
        Chunking parameters.
    metadata : dict, optional
        Extra metadata for all chunks.

    Returns
    -------
    Dict with ingestion statistics.
    """
    t0 = time.time()

    if not text or not text.strip():
        return {"status": "skipped", "reason": "empty document", "chunks": 0}

    # Step 1: Chunk
    chunks = chunk_text(
        text=text,
        source=source,
        title=title,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata=metadata,
    )

    if not chunks:
        return {"status": "skipped", "reason": "no chunks produced", "chunks": 0}

    # Step 2: Embed
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    # Step 3: Store
    chunk_ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "source": c.source,
            "title": c.title,
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "total_chunks": c.total_chunks,
            **c.metadata,
        }
        for c in chunks
    ]

    upserted = upsert_chunks(
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        texts=texts,
        metadatas=metadatas,
        collection=collection,
    )

    elapsed = time.time() - t0
    result = {
        "status": "ok",
        "source": source,
        "title": title,
        "chunks": upserted,
        "characters": len(text),
        "elapsed_s": round(elapsed, 2),
        "collection": collection,
    }
    logger.info(f"Ingested '{title or source}': {upserted} chunks in {elapsed:.1f}s")
    return result


def ingest_file(
    path: str,
    collection: str = "mars_science",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: Optional[dict] = None,
) -> Dict:
    """
    Ingest a file from disk. Supports .txt, .md, .json.
    """
    p = Path(path)

    if not p.exists():
        return {"status": "error", "reason": f"File not found: {path}"}

    if not p.is_file():
        return {"status": "error", "reason": f"Not a file: {path}"}

    suffix = p.suffix.lower()
    title = p.stem

    try:
        if suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Handle common JSON shapes
            if isinstance(data, str):
                text = data
            elif isinstance(data, dict):
                text = _json_to_text(data)
            elif isinstance(data, list):
                text = "\n\n".join(_json_to_text(item) if isinstance(item, dict) else str(item) for item in data)
            else:
                text = str(data)
        else:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
    except Exception as e:
        return {"status": "error", "reason": f"Read failed: {e}"}

    return ingest_text(
        text=text,
        source=str(p),
        title=title,
        collection=collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata=metadata,
    )


def ingest_directory(
    dir_path: str,
    collection: str = "mars_science",
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Dict:
    """
    Ingest all matching files from a directory.

    Parameters
    ----------
    dir_path : str
        Directory to scan.
    extensions : list, optional
        File extensions to include (default: .txt, .md, .json).
    recursive : bool
        Scan subdirectories.

    Returns
    -------
    Dict with aggregated ingestion stats.
    """
    if extensions is None:
        extensions = [".txt", ".md", ".json"]

    base = Path(dir_path)
    if not base.is_dir():
        return {"status": "error", "reason": f"Not a directory: {dir_path}"}

    pattern = "**/*" if recursive else "*"
    files = [
        f for f in base.glob(pattern)
        if f.is_file() and f.suffix.lower() in extensions
    ]

    results = []
    total_chunks = 0

    for fpath in sorted(files):
        result = ingest_file(
            path=str(fpath),
            collection=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        results.append(result)
        total_chunks += result.get("chunks", 0)

    return {
        "status": "ok",
        "directory": dir_path,
        "files_processed": len(results),
        "total_chunks": total_chunks,
        "collection": collection,
        "details": results,
    }


def _json_to_text(obj: dict) -> str:
    """Convert a JSON object into readable text for embedding."""
    parts = []
    for key, value in obj.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, indent=None)
        parts.append(f"{key}: {value}")
    return "\n".join(parts)
