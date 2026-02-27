from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.marslandform_v2.config import CLASS_NAMES, RAGConfig, get_config


@dataclass
class RetrievalResult:
    text: str
    source: str
    section: str
    class_tags: list[str]
    similarity_score: float


def _load_embedding_function(config: RAGConfig) -> tuple[SentenceTransformerEmbeddingFunction, str]:
    primary_model = config.embedding_model
    fallback_model = "all-MiniLM-L6-v2"

    for model_name in (primary_model, fallback_model):
        try:
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
            _ = embedding_fn(["mars landform query"])
            if model_name != primary_model:
                print(
                    f"[WARN] Could not load embedding model '{primary_model}'. Using fallback '{fallback_model}'."
                )
            return embedding_fn, model_name
        except Exception as exc:
            print(f"[WARN] Failed loading embedding model '{model_name}': {exc}")

    raise RuntimeError("No usable sentence-transformers embedding model found.")


class MarsRAG:
    def __init__(self, config: RAGConfig | None = None, db_dir: str | Path | None = None) -> None:
        pipeline_cfg = get_config()
        self.config: RAGConfig = config or pipeline_cfg.rag
        if db_dir is not None:
            self.config.db_path = str(db_dir)

        embedding_fn, model_name = _load_embedding_function(self.config)
        self.embedding_model: str = model_name

        client = chromadb.PersistentClient(path=self.config.db_path)
        self.collection: Any = client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=cast(Any, embedding_fn),
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        class_filter: str | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            return []

        normalized_top_k = max(1, int(top_k))
        where: dict[str, bool] | None = None
        if class_filter is not None:
            normalized_filter = class_filter.strip().upper()
            if normalized_filter not in CLASS_NAMES:
                raise ValueError(f"Unknown class_filter '{class_filter}'. Valid: {', '.join(CLASS_NAMES)}")
            where = {f"tag_{normalized_filter}": True}

        search_k = normalized_top_k if where else max(normalized_top_k * 4, normalized_top_k)
        response = self.collection.query(
            query_texts=[query],
            n_results=search_k,
            include=["documents", "metadatas", "distances"],
            where=cast(Any, where),
        )

        docs = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: list[RetrievalResult] = []
        for idx, text in enumerate(docs):
            metadata_raw = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            metadata = cast(dict[str, Any], metadata_raw)
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            similarity = max(0.0, min(1.0, 1.0 - distance))
            serialized_tags = str(metadata.get("class_tags", "")).strip()
            tags = [tag for tag in serialized_tags.split("|") if tag]

            results.append(
                RetrievalResult(
                    text=text,
                    source=str(metadata.get("source_file", "unknown")),
                    section=str(metadata.get("section_title", "General")),
                    class_tags=tags,
                    similarity_score=similarity,
                )
            )

            if len(results) >= normalized_top_k:
                break

        return results

    def format_context(self, results: list[RetrievalResult]) -> str:
        if not results:
            return "No relevant Mars landform context retrieved."

        blocks: list[str] = []
        for idx, result in enumerate(results, start=1):
            tags = ", ".join(result.class_tags) if result.class_tags else "UNSPECIFIED"
            block = (
                f"[{idx}] Source: {result.source} | Section: {result.section} | "
                f"Classes: {tags} | Similarity: {result.similarity_score:.3f}\n"
                f"{result.text}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)
