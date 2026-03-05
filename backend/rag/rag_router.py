"""
FastAPI router for Mars Science RAG endpoints.

POST /api/rag/query          — RAG query (question → grounded answer)
POST /api/rag/ingest         — Ingest text document
POST /api/rag/ingest/file    — Ingest file from disk
POST /api/rag/ingest/dir     — Ingest directory
POST /api/rag/seed           — Seed built-in Mars knowledge
GET  /api/rag/stats          — DB statistics
GET  /api/rag/collections    — List collections
DELETE /api/rag/collection   — Delete a collection
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["RAG"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    n_results: int = Field(5, ge=1, le=20, description="Max chunks to retrieve")
    collection: str = Field("mars_science", description="Collection to search")
    min_score: float = Field(0.15, ge=0.0, le=1.0, description="Minimum relevance threshold")


class RAGQueryResponse(BaseModel):
    answer: str
    citations: list
    chunks_used: int
    model: str
    grounded: bool
    collection: str


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Document text")
    source: str = Field("user_upload", description="Source identifier")
    title: str = Field("", description="Document title")
    collection: str = Field("mars_science", description="Target collection")
    chunk_size: int = Field(512, ge=100, le=2000)
    chunk_overlap: int = Field(64, ge=0, le=500)


class IngestFileRequest(BaseModel):
    path: str = Field(..., description="File path on server")
    collection: str = Field("mars_science", description="Target collection")
    chunk_size: int = Field(512, ge=100, le=2000)
    chunk_overlap: int = Field(64, ge=0, le=500)


class IngestDirRequest(BaseModel):
    path: str = Field(..., description="Directory path on server")
    collection: str = Field("mars_science", description="Target collection")
    extensions: Optional[List[str]] = Field(None, description="File extensions filter")
    recursive: bool = Field(True, description="Scan subdirectories")


class DeleteCollectionRequest(BaseModel):
    collection: str = Field(..., min_length=1, description="Collection name to delete")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    """
    Query the Mars Science RAG system.

    Retrieves relevant chunks from the vector store and generates
    a grounded answer with citations.
    """
    from .generator import generate_answer

    try:
        result = generate_answer(
            query=req.query,
            n_results=req.n_results,
            collection=req.collection,
            min_score=req.min_score,
        )
        return RAGQueryResponse(**result)
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.post("/ingest")
async def ingest_text_endpoint(req: IngestTextRequest):
    """Ingest a text document into the RAG vector store."""
    from .ingestion import ingest_text

    try:
        result = ingest_text(
            text=req.text,
            source=req.source,
            title=req.title,
            collection=req.collection,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
        return result
    except Exception as e:
        logger.error(f"Ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/file")
async def ingest_file_endpoint(req: IngestFileRequest):
    """Ingest a file from the server filesystem."""
    from .ingestion import ingest_file

    try:
        result = ingest_file(
            path=req.path,
            collection=req.collection,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result["reason"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")


@router.post("/ingest/dir")
async def ingest_dir_endpoint(req: IngestDirRequest):
    """Ingest all matching files from a directory."""
    from .ingestion import ingest_directory

    try:
        result = ingest_directory(
            dir_path=req.path,
            collection=req.collection,
            extensions=req.extensions,
            recursive=req.recursive,
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result["reason"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Directory ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seed")
async def seed_knowledge_endpoint(force: bool = False):
    """
    Seed the vector store with built-in Mars science knowledge.

    Also auto-ingests documents from knowledge/, mars_research/,
    and agent_reports/ directories.
    """
    from .mars_knowledge import seed_knowledge, auto_ingest_local_knowledge

    try:
        seed_result = seed_knowledge(force=force)
        local_result = auto_ingest_local_knowledge()

        return {
            "built_in": seed_result,
            "local_files": local_result,
        }
    except Exception as e:
        logger.error(f"Seeding failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Seeding failed: {str(e)}")


@router.get("/stats")
async def rag_stats():
    """Return RAG system statistics."""
    from .vector_store import get_store_info
    from .embedder import get_model_info

    return {
        "store": get_store_info(),
        "embedder": get_model_info(),
    }


@router.get("/collections")
async def list_collections_endpoint():
    """List all vector store collections."""
    from .vector_store import list_collections
    return {"collections": list_collections()}


@router.delete("/collection")
async def delete_collection_endpoint(req: DeleteCollectionRequest):
    """Delete a vector store collection."""
    from .vector_store import delete_collection

    deleted = delete_collection(req.collection)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Collection '{req.collection}' not found")
    return {"status": "deleted", "collection": req.collection}
