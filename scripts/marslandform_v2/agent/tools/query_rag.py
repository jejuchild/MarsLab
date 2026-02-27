from __future__ import annotations

from typing import Any, Dict, List, Optional


class QueryRAGTool:
    def __init__(self, rag: Any) -> None:
        self.rag = rag

    def run(self, query: str, class_filter: Optional[str] = None) -> Dict[str, Any]:
        if not self.rag:
            return {
                "relevant_excerpts": [],
                "sources": [],
                "class_tags": [],
                "error": "RAG retriever is not initialized.",
            }

        try:
            docs = self._retrieve(query=query, class_filter=class_filter)
        except Exception as exc:
            return {
                "relevant_excerpts": [],
                "sources": [],
                "class_tags": [],
                "error": str(exc),
            }

        excerpts: List[str] = []
        sources: List[str] = []
        class_tags: List[str] = []

        for doc in docs[:5]:
            if isinstance(doc, str):
                excerpts.append(doc)
                continue

            if not isinstance(doc, dict):
                excerpts.append(str(doc))
                continue

            text = str(
                doc.get("excerpt")
                or doc.get("text")
                or doc.get("content")
                or doc.get("page_content")
                or ""
            ).strip()
            if text:
                excerpts.append(text)

            source = str(doc.get("source") or doc.get("paper") or doc.get("id") or "").strip()
            if source:
                sources.append(source)

            tags = doc.get("class_tags") or doc.get("tags") or doc.get("classes") or []
            if isinstance(tags, str):
                class_tags.append(tags.upper())
            elif isinstance(tags, list):
                class_tags.extend([str(tag).upper() for tag in tags])

        unique_sources = sorted(set(sources))
        unique_class_tags = sorted(set(class_tags))

        return {
            "relevant_excerpts": excerpts[:5],
            "sources": unique_sources,
            "class_tags": unique_class_tags,
        }

    def _retrieve(self, query: str, class_filter: Optional[str]) -> List[Any]:
        kwargs = {"query": query}
        if class_filter:
            kwargs["class_filter"] = class_filter

        if hasattr(self.rag, "retrieve"):
            result = self.rag.retrieve(**kwargs)
        elif hasattr(self.rag, "query"):
            result = self.rag.query(**kwargs)
        elif hasattr(self.rag, "search"):
            result = self.rag.search(**kwargs)
        elif callable(self.rag):
            result = self.rag(**kwargs)
        else:
            raise RuntimeError("RAG object has no retrieve/query/search interface.")

        if isinstance(result, dict):
            docs = result.get("documents") or result.get("results") or result.get("items") or []
            return docs if isinstance(docs, list) else [docs]
        if isinstance(result, list):
            return result
        return [result]
