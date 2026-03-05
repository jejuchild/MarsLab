"""
ChromaDB vector store wrapper for Mars Science RAG.

Handles collection management, upsert, and similarity search.
Falls back to a simple in-memory NumPy store if ChromaDB is unavailable.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "rag_vectordb",
)
_DEFAULT_COLLECTION = "mars_science"

# ---------------------------------------------------------------------------
# ChromaDB client singleton
# ---------------------------------------------------------------------------
_chroma_client = None
_fallback_mode = False


def _get_client():
    """Lazy-init ChromaDB persistent client."""
    global _chroma_client, _fallback_mode

    if _chroma_client is not None or _fallback_mode:
        return _chroma_client

    try:
        import chromadb
        os.makedirs(_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=_PERSIST_DIR)
        logger.info(f"ChromaDB initialized at {_PERSIST_DIR}")
        return _chroma_client
    except Exception as e:
        logger.warning(f"ChromaDB unavailable ({e}), using in-memory fallback")
        _fallback_mode = True
        return None


# ---------------------------------------------------------------------------
# In-memory fallback store
# ---------------------------------------------------------------------------
class _InMemoryStore:
    """Minimal NumPy-based vector store for when ChromaDB isn't installed."""

    def __init__(self):
        self._collections: Dict[str, Dict[str, Any]] = {}

    def _ensure_col(self, name: str) -> Dict[str, Any]:
        if name not in self._collections:
            self._collections[name] = {
                "ids": [],
                "embeddings": [],  # list of np arrays
                "documents": [],
                "metadatas": [],
            }
        return self._collections[name]

    def upsert(
        self,
        collection: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[dict],
    ):
        col = self._ensure_col(collection)
        existing_ids = set(col["ids"])

        for i, doc_id in enumerate(ids):
            if doc_id in existing_ids:
                idx = col["ids"].index(doc_id)
                col["embeddings"][idx] = np.array(embeddings[i], dtype=np.float32)
                col["documents"][idx] = documents[i]
                col["metadatas"][idx] = metadatas[i]
            else:
                col["ids"].append(doc_id)
                col["embeddings"].append(np.array(embeddings[i], dtype=np.float32))
                col["documents"].append(documents[i])
                col["metadatas"].append(metadatas[i])

    def query(
        self,
        collection: str,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        col = self._ensure_col(collection)
        if not col["ids"]:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q_vec = np.array(query_embedding, dtype=np.float32)
        emb_matrix = np.array(col["embeddings"])

        # Cosine similarity (embeddings are L2-normalized)
        similarities = emb_matrix @ q_vec
        top_k = min(n_results, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return {
            "ids": [[col["ids"][i] for i in top_indices]],
            "documents": [[col["documents"][i] for i in top_indices]],
            "metadatas": [[col["metadatas"][i] for i in top_indices]],
            "distances": [[float(1 - similarities[i]) for i in top_indices]],
        }

    def count(self, collection: str) -> int:
        col = self._ensure_col(collection)
        return len(col["ids"])

    def list_collections(self) -> List[str]:
        return list(self._collections.keys())

    def delete_collection(self, collection: str) -> bool:
        if collection in self._collections:
            del self._collections[collection]
            return True
        return False


_inmemory_store = _InMemoryStore()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert_chunks(
    chunk_ids: List[str],
    embeddings: np.ndarray,
    texts: List[str],
    metadatas: List[dict],
    collection: str = _DEFAULT_COLLECTION,
) -> int:
    """
    Upsert chunks into the vector store.

    Returns the number of chunks upserted.
    """
    client = _get_client()
    emb_lists = embeddings.tolist()

    if client is not None:
        col = client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        # ChromaDB has a batch limit; split into batches of 5000
        batch_size = 5000
        for start in range(0, len(chunk_ids), batch_size):
            end = start + batch_size
            col.upsert(
                ids=chunk_ids[start:end],
                embeddings=emb_lists[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
    else:
        _inmemory_store.upsert(collection, chunk_ids, emb_lists, texts, metadatas)

    logger.info(f"Upserted {len(chunk_ids)} chunks into '{collection}'")
    return len(chunk_ids)


def search(
    query_embedding: np.ndarray,
    n_results: int = 5,
    collection: str = _DEFAULT_COLLECTION,
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks.

    Returns list of dicts with keys: text, chunk_id, source, title, score, metadata.
    """
    client = _get_client()
    q_list = query_embedding.tolist()

    if client is not None:
        try:
            col = client.get_collection(name=collection)
        except Exception:
            logger.warning(f"Collection '{collection}' not found")
            return []
        results = col.query(
            query_embeddings=[q_list],
            n_results=n_results,
        )
    else:
        results = _inmemory_store.query(collection, q_list, n_results)

    # Flatten ChromaDB's nested structure
    hits = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        hits.append({
            "text": docs[i] if i < len(docs) else "",
            "chunk_id": ids[i],
            "source": meta.get("source", ""),
            "title": meta.get("title", ""),
            "score": round(1.0 - dists[i], 4) if i < len(dists) else 0.0,
            "doc_id": meta.get("doc_id", ""),
            "chunk_index": meta.get("chunk_index", 0),
        })

    return hits


def get_collection_count(collection: str = _DEFAULT_COLLECTION) -> int:
    """Return number of vectors in a collection."""
    client = _get_client()
    if client is not None:
        try:
            col = client.get_collection(name=collection)
            return col.count()
        except Exception:
            return 0
    return _inmemory_store.count(collection)


def list_collections() -> List[Dict[str, Any]]:
    """List all collections with their sizes."""
    client = _get_client()
    if client is not None:
        try:
            cols = client.list_collections()
            return [
                {"name": c.name, "count": c.count()}
                for c in cols
            ]
        except Exception as e:
            logger.warning(f"Failed to list collections: {e}")
            return []

    return [
        {"name": name, "count": _inmemory_store.count(name)}
        for name in _inmemory_store.list_collections()
    ]


def delete_collection(collection: str) -> bool:
    """Delete a collection. Returns True if it existed."""
    client = _get_client()
    if client is not None:
        try:
            client.delete_collection(name=collection)
            logger.info(f"Deleted collection '{collection}'")
            return True
        except Exception:
            return False
    return _inmemory_store.delete_collection(collection)


def get_store_info() -> dict:
    """Return store backend status."""
    client = _get_client()
    return {
        "backend": "chromadb" if client is not None else "in_memory",
        "persist_dir": _PERSIST_DIR if client is not None else None,
        "collections": list_collections(),
    }
