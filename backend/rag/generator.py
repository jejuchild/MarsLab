"""
LLM answer generation for Mars Science RAG.

Uses Groq API (Llama 3.1/3.3) as primary, Gemini as fallback.
Generates grounded answers with citation references.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import requests

from .retriever import retrieve, format_context, format_citations

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Model selection: use 70B for complex science, 8B for simple queries
MODEL_EASY = "llama-3.1-8b-instant"
MODEL_HARD = "llama-3.3-70b-versatile"

RAG_SYSTEM_PROMPT = """\
You are a Mars science research assistant with deep expertise in planetary science, \
mineralogy, spectroscopy, geomorphology, and astrobiology.

You are given RETRIEVED CONTEXT from a knowledge base of Mars research documents. \
Your task is to answer the user's question using ONLY the provided context.

RULES:
1. Answer based ONLY on the retrieved context. Do NOT use prior knowledge.
2. If the context does not contain sufficient information, say so explicitly.
3. Reference sources using [Source N] notation matching the context headers.
4. Use precise scientific language suitable for a planetary science researcher.
5. Separate observations from interpretations.
6. When uncertain, express uncertainty clearly.
7. Keep answers concise but thorough.
8. If the question is in Korean, answer in Korean. If in English, answer in English.
"""

NO_CONTEXT_PROMPT = """\
You are a Mars science research assistant. The knowledge base has no relevant \
documents for this query. Provide a helpful response using your general knowledge \
but clearly state that this answer is NOT grounded in the local knowledge base. \
Recommend the user ingest relevant documents for more accurate answers.
"""


def _call_groq(
    messages: List[Dict[str, str]],
    model: str = MODEL_EASY,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Optional[str]:
    """Call Groq API and return the response text."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set, cannot generate answer")
        return None

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return None


def _select_model(query: str) -> str:
    """Select model based on query complexity heuristics."""
    complex_signals = [
        "compare", "contrast", "analyze", "mechanism", "hypothesis",
        "relationship between", "evidence for", "why does", "how does",
        "implications", "분석", "비교", "메커니즘", "가설", "증거",
    ]
    query_lower = query.lower()
    if any(signal in query_lower for signal in complex_signals):
        return MODEL_HARD
    return MODEL_EASY


def generate_answer(
    query: str,
    n_results: int = 5,
    collection: str = "mars_science",
    min_score: float = 0.15,
) -> Dict:
    """
    Full RAG pipeline: retrieve → format context → generate answer.

    Parameters
    ----------
    query : str
        User's natural language question.
    n_results : int
        Number of chunks to retrieve.
    collection : str
        Vector store collection to search.
    min_score : float
        Minimum relevance threshold.

    Returns
    -------
    Dict with keys:
        answer: str — Generated answer text
        citations: List[Dict] — Source citations
        chunks_used: int — Number of chunks in context
        model: str — LLM model used
        grounded: bool — Whether answer is based on retrieved context
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(
        query,
        n_results=n_results,
        collection=collection,
        min_score=min_score,
    )

    # Step 2: Format context and select model
    model = _select_model(query)
    grounded = len(chunks) > 0

    if grounded:
        context = format_context(chunks)
        system_prompt = RAG_SYSTEM_PROMPT
        user_message = f"RETRIEVED CONTEXT:\n{context}\n\n---\n\nQUESTION: {query}"
    else:
        system_prompt = NO_CONTEXT_PROMPT
        user_message = query

    # Step 3: Generate answer via LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    answer = _call_groq(messages, model=model)

    if answer is None:
        answer = (
            "RAG 시스템이 관련 문서를 찾았으나 LLM 응답 생성에 실패했습니다. "
            "GROQ_API_KEY 환경변수를 확인해주세요."
            if grounded else
            "지식 베이스에 관련 문서가 없고 LLM 연결도 실패했습니다. "
            "문서를 인제스트하고 API 키를 설정해주세요."
        )

    # Step 4: Build response
    citations = format_citations(chunks) if grounded else []

    return {
        "answer": answer,
        "citations": citations,
        "chunks_used": len(chunks),
        "model": model,
        "grounded": grounded,
        "collection": collection,
    }
