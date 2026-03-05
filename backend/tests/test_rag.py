"""
Tests for Mars Science RAG system.

Covers: chunking, embedding, vector store, ingestion, retrieval, and knowledge seeding.
"""

import os
import sys
import pytest

# Ensure backend is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Chunker tests ─────────────────────────────────────────────────────

class TestChunker:
    def test_chunk_short_text(self):
        from rag.chunker import chunk_text
        chunks = chunk_text("Hello Mars.", source="test.txt", title="Test")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello Mars."
        assert chunks[0].source == "test.txt"
        assert chunks[0].title == "Test"
        assert chunks[0].chunk_id  # non-empty

    def test_chunk_long_text(self):
        from rag.chunker import chunk_text
        text = "Mars is fascinating. " * 100  # ~2100 chars
        chunks = chunk_text(text, source="long.txt", chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 300  # with overlap tolerance

    def test_chunk_empty(self):
        from rag.chunker import chunk_text
        assert chunk_text("", source="x") == []
        assert chunk_text("   ", source="x") == []

    def test_chunk_ids_deterministic(self):
        from rag.chunker import chunk_text
        c1 = chunk_text("Same text", source="same.txt")
        c2 = chunk_text("Same text", source="same.txt")
        assert c1[0].chunk_id == c2[0].chunk_id

    def test_chunk_metadata(self):
        from rag.chunker import chunk_text
        chunks = chunk_text("Data", source="s", metadata={"type": "test"})
        assert chunks[0].metadata["type"] == "test"


# ── Embedder tests ────────────────────────────────────────────────────

class TestEmbedder:
    def test_embed_texts(self):
        from rag.embedder import embed_texts, get_embedding_dim
        embs = embed_texts(["Hello Mars", "Red planet"])
        assert embs.shape == (2, get_embedding_dim())

    def test_embed_empty(self):
        from rag.embedder import embed_texts
        embs = embed_texts([])
        assert embs.shape[0] == 0

    def test_embed_query(self):
        from rag.embedder import embed_query, get_embedding_dim
        emb = embed_query("What is CRISM?")
        assert emb.shape == (get_embedding_dim(),)

    def test_model_info(self):
        from rag.embedder import get_model_info
        info = get_model_info()
        assert "model_name" in info
        assert "embedding_dim" in info
        assert info["status"] in ("transformer", "tfidf_fallback")


# ── Vector Store tests ────────────────────────────────────────────────

class TestVectorStore:
    TEST_COLLECTION = "test.rag.collection"

    def setup_method(self):
        from rag.vector_store import delete_collection
        delete_collection(self.TEST_COLLECTION)

    def teardown_method(self):
        from rag.vector_store import delete_collection
        delete_collection(self.TEST_COLLECTION)

    def test_upsert_and_search(self):
        import numpy as np
        from rag.vector_store import upsert_chunks, search, get_collection_count

        # Create test embeddings (normalized)
        emb1 = np.random.randn(384).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        emb2 = np.random.randn(384).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)

        upsert_chunks(
            chunk_ids=["c1", "c2"],
            embeddings=np.array([emb1, emb2]),
            texts=["Mars has water ice", "Jupiter is a gas giant"],
            metadatas=[
                {"source": "a.txt", "title": "Mars"},
                {"source": "b.txt", "title": "Jupiter"},
            ],
            collection=self.TEST_COLLECTION,
        )

        assert get_collection_count(self.TEST_COLLECTION) == 2

        # Search with emb1 should return Mars first
        results = search(emb1, n_results=2, collection=self.TEST_COLLECTION)
        assert len(results) > 0
        assert results[0]["chunk_id"] == "c1"

    def test_empty_collection(self):
        from rag.vector_store import search, get_collection_count
        import numpy as np

        q = np.random.randn(384).astype(np.float32)
        results = search(q, collection=self.TEST_COLLECTION)
        assert results == []
        assert get_collection_count(self.TEST_COLLECTION) == 0

    def test_store_info(self):
        from rag.vector_store import get_store_info
        info = get_store_info()
        assert "backend" in info
        assert info["backend"] in ("chromadb", "in_memory")


# ── Ingestion tests ───────────────────────────────────────────────────

class TestIngestion:
    TEST_COLLECTION = "test.rag.ingest"

    def teardown_method(self):
        from rag.vector_store import delete_collection
        delete_collection(self.TEST_COLLECTION)

    def test_ingest_text(self):
        from rag.ingestion import ingest_text

        result = ingest_text(
            text="Mars has a thin CO2 atmosphere with surface pressure around 636 Pa.",
            source="test_doc",
            title="Mars Atmosphere",
            collection=self.TEST_COLLECTION,
        )
        assert result["status"] == "ok"
        assert result["chunks"] > 0

    def test_ingest_empty(self):
        from rag.ingestion import ingest_text

        result = ingest_text(text="", source="empty", collection=self.TEST_COLLECTION)
        assert result["status"] == "skipped"


# ── Retriever tests ───────────────────────────────────────────────────

class TestRetriever:
    TEST_COLLECTION = "test.rag.retrieve"

    def setup_method(self):
        from rag.ingestion import ingest_text
        ingest_text(
            text="CRISM detects phyllosilicates through absorption features at 2.3 micrometers. "
                 "These clay minerals indicate past water interaction on Mars.",
            source="crism_doc",
            title="CRISM Mineralogy",
            collection=self.TEST_COLLECTION,
        )
        ingest_text(
            text="SHARAD penetrating radar can detect subsurface water ice deposits "
                 "up to 1 km deep in the Martian regolith.",
            source="sharad_doc",
            title="SHARAD Capabilities",
            collection=self.TEST_COLLECTION,
        )

    def teardown_method(self):
        from rag.vector_store import delete_collection
        delete_collection(self.TEST_COLLECTION)

    def test_retrieve_relevant(self):
        from rag.retriever import retrieve
        results = retrieve(
            "What minerals does CRISM detect?",
            collection=self.TEST_COLLECTION,
            min_score=0.0,
        )
        assert len(results) > 0
        # CRISM doc should rank higher
        assert any("CRISM" in r.get("title", "") or "crism" in r.get("text", "").lower() for r in results)

    def test_format_context(self):
        from rag.retriever import retrieve, format_context
        chunks = retrieve("CRISM minerals", collection=self.TEST_COLLECTION, min_score=0.0)
        ctx = format_context(chunks)
        assert "[Source 1]" in ctx
        assert len(ctx) > 0

    def test_format_citations(self):
        from rag.retriever import retrieve, format_citations
        chunks = retrieve("SHARAD ice", collection=self.TEST_COLLECTION, min_score=0.0)
        citations = format_citations(chunks)
        assert len(citations) > 0
        assert "source" in citations[0]


# ── Knowledge Seeding tests ──────────────────────────────────────────

class TestKnowledgeSeeding:
    TEST_COLLECTION = "test.rag.seed"

    def teardown_method(self):
        from rag.vector_store import delete_collection
        delete_collection(self.TEST_COLLECTION)

    def test_seed_knowledge(self):
        from rag.mars_knowledge import seed_knowledge, _SEEDED
        import rag.mars_knowledge as mk

        # Reset seeded flag
        mk._SEEDED = False

        result = seed_knowledge(collection=self.TEST_COLLECTION)
        assert result["status"] == "ok"
        assert result["documents"] > 0
        assert result["total_chunks"] > 0

    def test_seed_idempotent(self):
        from rag.mars_knowledge import seed_knowledge
        import rag.mars_knowledge as mk

        mk._SEEDED = False
        r1 = seed_knowledge(collection=self.TEST_COLLECTION)
        r2 = seed_knowledge(collection=self.TEST_COLLECTION)  # Should skip
        assert r2["status"] == "already_seeded"

    def test_seed_then_query(self):
        """End-to-end: seed knowledge, then retrieve."""
        from rag.mars_knowledge import seed_knowledge
        from rag.retriever import retrieve
        import rag.mars_knowledge as mk

        mk._SEEDED = False
        seed_knowledge(collection=self.TEST_COLLECTION)

        results = retrieve(
            "What instruments does Perseverance have?",
            collection=self.TEST_COLLECTION,
            min_score=0.0,
        )
        assert len(results) > 0
        # Should find Perseverance mission overview
        texts = " ".join(r["text"] for r in results)
        assert "perseverance" in texts.lower() or "pixl" in texts.lower()
