from __future__ import annotations

import argparse
import hashlib
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, cast

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.marslandform_v2.config import CLASS_NAMES, RAG_CORPUS_DIR, RAGConfig, get_config


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

CLASS_PATTERNS: dict[str, Sequence[str]] = {
    "LDA": (
        r"\bLDA\b",
        r"\blobate debris apron(?:s)?\b",
        r"\blobate apron(?:s)?\b",
    ),
    "LVF": (
        r"\bLVF\b",
        r"\blineated valley fill\b",
    ),
    "CCF": (
        r"\bCCF\b",
        r"\bconcentric crater fill\b",
    ),
    "GLF": (
        r"\bGLF\b",
        r"\bglacier-like form(?:s)?\b",
        r"\bglacier like form(?:s)?\b",
    ),
}

HEADER_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"\S+")
PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    source_file: str
    section_title: str
    class_tags: list[str]
    chunk_index: int
    content_hash: str


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()


def _split_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "General"
    current_lines: list[str] = []

    for line in lines:
        header_match = HEADER_RE.match(line.strip())
        if header_match:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = header_match.group(2).strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    return [(title, _normalize_text("\n".join(sec_lines))) for title, sec_lines in sections if _normalize_text("\n".join(sec_lines))]


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []
    if _token_count(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in PARA_SPLIT_RE.split(text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = _token_count(paragraph)
        if para_tokens > chunk_size:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_tokens = 0
            chunks.extend(_split_oversized_paragraph(paragraph, chunk_size, chunk_overlap))
            continue

        if current_tokens + para_tokens > chunk_size and current_parts:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = _overlap_tail(current_parts, chunk_overlap)
            current_tokens = _token_count("\n\n".join(current_parts)) if current_parts else 0

        current_parts.append(paragraph)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [c for c in chunks if c]


def _split_oversized_paragraph(paragraph: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    if not sentences:
        words = paragraph.split()
        return _word_window_chunks(words, chunk_size, chunk_overlap)

    sentence_chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sent_tokens = _token_count(sentence)
        if sent_tokens > chunk_size:
            if current:
                sentence_chunks.append(" ".join(current).strip())
                current = []
                current_tokens = 0
            sentence_chunks.extend(_word_window_chunks(sentence.split(), chunk_size, chunk_overlap))
            continue

        if current_tokens + sent_tokens > chunk_size and current:
            sentence_chunks.append(" ".join(current).strip())
            current = _overlap_tail(current, chunk_overlap)
            current_tokens = _token_count(" ".join(current)) if current else 0

        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        sentence_chunks.append(" ".join(current).strip())

    return [c for c in sentence_chunks if c]


def _word_window_chunks(words: Sequence[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    step = max(1, chunk_size - max(0, chunk_overlap))
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words).strip())
        if start + chunk_size >= len(words):
            break
    return chunks


def _overlap_tail(parts: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    joined = "\n\n".join(parts)
    words = joined.split()
    if not words:
        return []
    tail = words[-overlap_tokens:]
    return [" ".join(tail)]


def _detect_class_tags(text: str) -> list[str]:
    tags: list[str] = []
    for class_name in CLASS_NAMES:
        patterns = CLASS_PATTERNS.get(class_name, (rf"\\b{class_name}\\b",))
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            tags.append(class_name)
    return tags


def _compute_hash(text: str, source_file: str, section_title: str) -> str:
    payload = f"{source_file}\n{section_title}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_plain_text(path: Path) -> str:
    return _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))


def _read_pdf(path: Path) -> str:
    try:
        fitz = importlib.import_module("fitz")

        doc = fitz.open(str(path))
        pages = [str(page.get_text("text")) for page in doc]
        return _normalize_text("\n".join(pages))
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        print(f"[WARN] Failed extracting PDF with pymupdf for {path.name}: {exc}")
        return ""

    try:
        pdfplumber = importlib.import_module("pdfplumber")

        pages_text: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                pages_text.append(page.extract_text() or "")
        return _normalize_text("\n".join(pages_text))
    except ModuleNotFoundError:
        print(
            f"[WARN] PDF dependencies missing for {path.name}. Install pymupdf or pdfplumber to ingest PDFs."
        )
        return ""
    except Exception as exc:
        print(f"[WARN] Failed extracting PDF with pdfplumber for {path.name}: {exc}")
        return ""


def _read_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".md", ".txt"}:
        return _read_plain_text(path)
    if ext == ".pdf":
        return _read_pdf(path)
    return ""


def load_corpus_files(corpus_dir: Path) -> list[Path]:
    all_files = sorted(
        [
            p
            for p in corpus_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )
    return all_files


def chunk_document(path: Path, text: str, config: RAGConfig) -> list[Chunk]:
    if not text:
        return []

    sections = _split_sections(text)
    if not sections:
        sections = [("General", text)]

    chunks: list[Chunk] = []
    chunk_index = 0
    for section_title, section_text in sections:
        pieces = _split_long_text(section_text, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        for piece in pieces:
            class_tags = _detect_class_tags(piece)
            content_hash = _compute_hash(piece, source_file=path.name, section_title=section_title)
            chunks.append(
                Chunk(
                    text=piece,
                    source_file=path.name,
                    section_title=section_title,
                    class_tags=class_tags,
                    chunk_index=chunk_index,
                    content_hash=content_hash,
                )
            )
            chunk_index += 1

    return chunks


def _load_embedding_function(config: RAGConfig) -> tuple[SentenceTransformerEmbeddingFunction, str]:
    primary_model = config.embedding_model
    fallback_model = "all-MiniLM-L6-v2"

    for model_name in (primary_model, fallback_model):
        try:
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
            _ = embedding_fn(["mars landform diagnostic chunk"])
            if model_name != primary_model:
                print(
                    f"[WARN] Could not load embedding model '{primary_model}'. Using fallback '{fallback_model}'."
                )
            return embedding_fn, model_name
        except Exception as exc:
            print(f"[WARN] Failed loading embedding model '{model_name}': {exc}")

    raise RuntimeError("No usable sentence-transformers embedding model found.")


def _get_collection(
    db_path: str,
    config: RAGConfig,
    embedding_fn: SentenceTransformerEmbeddingFunction,
    reset: bool,
) -> Any:
    client = chromadb.PersistentClient(path=db_path)
    if reset:
        try:
            client.delete_collection(config.collection_name)
            print(f"[INFO] Reset collection '{config.collection_name}'.")
        except Exception:
            pass

    return client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=cast(Any, embedding_fn),
        metadata={"hnsw:space": "cosine"},
    )


def _existing_hashes(collection: Any) -> set[str]:
    hashes: set[str] = set()
    data = collection.get(include=["metadatas"])
    for metadata in data.get("metadatas") or []:
        if metadata and metadata.get("content_hash"):
            hashes.add(str(metadata["content_hash"]))
    return hashes


def ingest_corpus(corpus_dir: Path, db_dir: Path | None, reset: bool = False) -> None:
    pipeline_cfg = get_config()
    rag_cfg = pipeline_cfg.rag
    if db_dir is not None:
        rag_cfg.db_path = str(db_dir)

    embedding_fn, active_model = _load_embedding_function(rag_cfg)
    collection = _get_collection(rag_cfg.db_path, rag_cfg, embedding_fn, reset=reset)
    known_hashes = _existing_hashes(collection)

    files = load_corpus_files(corpus_dir)
    if not files:
        print(f"[WARN] No .md/.txt/.pdf files found in {corpus_dir}")
        return

    to_add_ids: list[str] = []
    to_add_docs: list[str] = []
    to_add_metas: list[dict[str, str | int | float | bool]] = []

    total_chunks = 0
    ingested_chunks = 0
    duplicate_chunks = 0
    per_class_counts: dict[str, int] = {class_name: 0 for class_name in CLASS_NAMES}

    for file_path in files:
        text = _read_document(file_path)
        chunks = chunk_document(file_path, text, rag_cfg)
        total_chunks += len(chunks)

        for chunk in chunks:
            if chunk.content_hash in known_hashes:
                duplicate_chunks += 1
                continue

            known_hashes.add(chunk.content_hash)
            tags_serialized = "|".join(chunk.class_tags)
            metadata: dict[str, str | int | float | bool] = {
                "source_file": chunk.source_file,
                "section_title": chunk.section_title,
                "class_tags": tags_serialized,
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
            }
            for class_name in CLASS_NAMES:
                metadata[f"tag_{class_name}"] = class_name in chunk.class_tags

            to_add_ids.append(chunk.content_hash)
            to_add_docs.append(chunk.text)
            to_add_metas.append(metadata)

            for class_name in chunk.class_tags:
                per_class_counts[class_name] += 1
            ingested_chunks += 1

    if to_add_ids:
        collection.add(ids=to_add_ids, documents=to_add_docs, metadatas=cast(Any, to_add_metas))

    collection_size = collection.count()
    print(f"[INFO] Embedding model: {active_model}")
    print(f"[INFO] Corpus files scanned: {len(files)}")
    print(f"[INFO] Total chunks generated: {total_chunks}")
    print(f"[INFO] Chunks ingested: {ingested_chunks}")
    print(f"[INFO] Duplicate chunks skipped: {duplicate_chunks}")
    for class_name in CLASS_NAMES:
        print(f"[INFO] {class_name} chunks: {per_class_counts[class_name]}")
    print(f"[INFO] Collection size: {collection_size}")


def parse_args() -> argparse.Namespace:
    default_cfg = get_config().rag
    parser = argparse.ArgumentParser(description="Ingest Mars landform corpus into ChromaDB.")
    _ = parser.add_argument(
        "--corpus_dir",
        type=str,
        default=str(RAG_CORPUS_DIR),
        help="Directory containing .md/.txt/.pdf corpus files.",
    )
    _ = parser.add_argument(
        "--db_dir",
        type=str,
        default=default_cfg.db_path,
        help="Persistent ChromaDB directory.",
    )
    _ = parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the existing collection before ingesting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest_corpus(
        corpus_dir=Path(args.corpus_dir),
        db_dir=Path(args.db_dir),
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
