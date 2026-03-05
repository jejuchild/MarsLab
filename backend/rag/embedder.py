"""
Embedding generation for Mars Science RAG.

Primary: sentence-transformers (all-MiniLM-L6-v2, 384-dim).
Fallback: TF-IDF sparse vectors when the model is unavailable.
"""

import logging
import hashlib
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton model holder
# ---------------------------------------------------------------------------
_model = None
_model_name = "all-MiniLM-L6-v2"
_embedding_dim = 384
_fallback_mode = False

# TF-IDF fallback state
_tfidf_vectorizer = None


def _load_model():
    """Lazy-load sentence-transformer model. Falls back to TF-IDF on failure."""
    global _model, _fallback_mode

    if _model is not None or _fallback_mode:
        return

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {_model_name} ...")
        _model = SentenceTransformer(_model_name)
        logger.info(f"Embedding model loaded ({_embedding_dim}-dim)")
    except Exception as e:
        logger.warning(f"sentence-transformers unavailable ({e}), using TF-IDF fallback")
        _fallback_mode = True


def _tfidf_embed(texts: List[str], dim: int = 384) -> np.ndarray:
    """
    TF-IDF fallback embedding.

    Produces deterministic sparse-ish vectors by hashing term frequencies
    into a fixed-dimension space.  Not as good as transformer embeddings,
    but works without GPU or large model downloads.
    """
    vectors = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Parameters
    ----------
    texts : List[str]
        Input texts to embed.
    batch_size : int
        Batch size for model inference.

    Returns
    -------
    np.ndarray
        (N, dim) array of float32 embeddings.
    """
    _load_model()

    if not texts:
        return np.zeros((0, _embedding_dim), dtype=np.float32)

    if _fallback_mode:
        return _tfidf_embed(texts, _embedding_dim)

    try:
        embeddings = _model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding failed: {e}, falling back to TF-IDF")
        return _tfidf_embed(texts, _embedding_dim)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns (dim,) vector."""
    result = embed_texts([query])
    return result[0]


def get_embedding_dim() -> int:
    """Return the embedding dimensionality."""
    return _embedding_dim


def get_model_info() -> dict:
    """Return model status information."""
    _load_model()
    return {
        "model_name": _model_name,
        "embedding_dim": _embedding_dim,
        "fallback_mode": _fallback_mode,
        "status": "tfidf_fallback" if _fallback_mode else "transformer",
    }
