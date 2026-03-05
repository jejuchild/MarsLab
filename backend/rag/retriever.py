"""
Retrieval + reranking for Mars Science RAG.

Handles query embedding, vector search, and score-based reranking
with Mars science domain boosting.
"""

import logging
import re
from typing import Dict, List, Optional

from .embedder import embed_query
from .vector_store import search, get_collection_count

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain-specific boosting terms
# ---------------------------------------------------------------------------
MARS_BOOST_TERMS = {
    # Instruments
    "crism": 0.05, "hirise": 0.05, "sharad": 0.05, "mola": 0.05,
    "pixl": 0.05, "sherloc": 0.05, "libs": 0.05, "mastcam": 0.05,
    "supercam": 0.05, "ctx": 0.05, "themis": 0.05,
    # Missions
    "perseverance": 0.03, "curiosity": 0.03, "insight": 0.03,
    "maven": 0.03, "mro": 0.03, "odyssey": 0.03,
    # Geology
    "olivine": 0.04, "phyllosilicate": 0.04, "sulfate": 0.04,
    "carbonate": 0.04, "hematite": 0.04, "clay": 0.04,
    "basalt": 0.04, "pyroxene": 0.04, "feldspar": 0.04,
    # Features
    "crater": 0.03, "jezero": 0.04, "gale": 0.04, "valles marineris": 0.04,
    "olympus": 0.03, "hellas": 0.03, "arcadia": 0.04,
    # Science concepts
    "biosignature": 0.05, "habitability": 0.05, "water ice": 0.04,
    "dust storm": 0.04, "marsquake": 0.04, "subsurface": 0.03,
}


def _compute_domain_boost(query: str, doc_text: str) -> float:
    """
    Boost score when query and document share Mars domain terms.
    Returns a small additive bonus (0.0 to ~0.15).
    """
    query_lower = query.lower()
    doc_lower = doc_text.lower()
    boost = 0.0

    for term, weight in MARS_BOOST_TERMS.items():
        if term in query_lower and term in doc_lower:
            boost += weight

    return min(boost, 0.15)  # Cap boost


def retrieve(
    query: str,
    n_results: int = 5,
    collection: str = "mars_science",
    min_score: float = 0.15,
    rerank: bool = True,
) -> List[Dict]:
    """
    Retrieve relevant chunks for a query.

    Parameters
    ----------
    query : str
        Natural language query.
    n_results : int
        Max results to return.
    collection : str
        Vector store collection name.
    min_score : float
        Minimum similarity score threshold.
    rerank : bool
        Apply domain-specific reranking.

    Returns
    -------
    List[Dict]
        Ranked list of relevant chunks with scores.
    """
    count = get_collection_count(collection)
    if count == 0:
        logger.info(f"Collection '{collection}' is empty, no results")
        return []

    # Fetch more candidates for reranking
    fetch_k = min(n_results * 3, count)
    q_emb = embed_query(query)
    candidates = search(q_emb, n_results=fetch_k, collection=collection)

    if not candidates:
        return []

    if rerank:
        for hit in candidates:
            domain_boost = _compute_domain_boost(query, hit["text"])
            hit["score"] = hit["score"] + domain_boost
            hit["domain_boost"] = round(domain_boost, 4)

        candidates.sort(key=lambda x: x["score"], reverse=True)

    # Apply min_score filter and limit
    results = [
        hit for hit in candidates
        if hit["score"] >= min_score
    ][:n_results]

    logger.info(
        f"Retrieved {len(results)}/{len(candidates)} chunks for query "
        f"(top_score={results[0]['score']:.3f})" if results else
        f"Retrieved 0 chunks for query"
    )
    return results


def format_context(chunks: List[Dict], max_chars: int = 6000) -> str:
    """
    Format retrieved chunks into a context string for LLM prompting.

    Includes source citations for grounding.
    """
    if not chunks:
        return ""

    parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        title = chunk.get("title", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        header = f"[Source {i}] {title}" if title else f"[Source {i}] {source}"
        entry = f"{header} (relevance: {score:.2f})\n{text}"

        if total_chars + len(entry) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                entry = entry[:remaining] + "..."
                parts.append(entry)
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n\n---\n\n".join(parts)


def format_citations(chunks: List[Dict]) -> List[Dict]:
    """
    Extract citation metadata from chunks for the frontend.
    """
    citations = []
    seen_sources = set()

    for chunk in chunks:
        source = chunk.get("source", "")
        if source in seen_sources:
            continue
        seen_sources.add(source)

        citations.append({
            "source": source,
            "title": chunk.get("title", ""),
            "relevance": chunk.get("score", 0),
            "doc_id": chunk.get("doc_id", ""),
        })

    return citations
