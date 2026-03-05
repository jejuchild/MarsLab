"""
Mars Science Document Crawl & Ingest Runner.

Usage:
    python -m rag.crawl_and_ingest                    # full crawl
    python -m rag.crawl_and_ingest --arxiv-only        # arXiv only
    python -m rag.crawl_and_ingest --nasa-only         # NASA only
    python -m rag.crawl_and_ingest --max-papers 20     # limit arXiv per query
    python -m rag.crawl_and_ingest --dry-run           # crawl without ingesting
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

from .crawler import (
    ArxivCrawler,
    CrawlerIngestionPipeline,
    MarsDocument,
    NasaPdsCrawler,
    NasaTechDocsCrawler,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Crawl Mars science documents and ingest into RAG"
    )
    parser.add_argument("--arxiv-only", action="store_true",
                        help="Only crawl arXiv papers")
    parser.add_argument("--nasa-only", action="store_true",
                        help="Only crawl NASA PDS / tech docs")
    parser.add_argument("--max-papers", type=int, default=50,
                        help="Max papers per arXiv query (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Crawl without ingesting into RAG")
    parser.add_argument("--collection", default="mars_science",
                        help="ChromaDB collection name (default: mars_science)")
    return parser.parse_args(argv)


def run(args):
    t0 = time.time()
    all_docs: list = []
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "sources": {},
    }

    # ── arXiv ──────────────────────────────────────────────────────────
    if not args.nasa_only:
        logger.info("=" * 60)
        logger.info("PHASE 1: arXiv Mars Science Papers")
        logger.info("=" * 60)
        arxiv = ArxivCrawler(max_per_query=args.max_papers)
        arxiv_docs = arxiv.crawl()
        all_docs.extend(arxiv_docs)
        report["sources"]["arxiv"] = {
            "documents": len(arxiv_docs),
            "unique_ids": len(arxiv._seen_ids),
        }
        logger.info("arXiv: %d unique papers collected", len(arxiv_docs))

    # ── NASA PDS ───────────────────────────────────────────────────────
    if not args.arxiv_only:
        logger.info("=" * 60)
        logger.info("PHASE 2: NASA PDS Instrument Documentation")
        logger.info("=" * 60)
        pds = NasaPdsCrawler()
        pds_docs = pds.crawl()
        all_docs.extend(pds_docs)
        report["sources"]["nasa_pds"] = {"documents": len(pds_docs)}
        logger.info("NASA PDS: %d documents collected", len(pds_docs))

        logger.info("=" * 60)
        logger.info("PHASE 3: NASA Mars Mission Technical Pages")
        logger.info("=" * 60)
        tech = NasaTechDocsCrawler()
        tech_docs = tech.crawl()
        all_docs.extend(tech_docs)
        report["sources"]["nasa_tech"] = {"documents": len(tech_docs)}
        logger.info("NASA Tech: %d documents collected", len(tech_docs))

    # ── Deduplicate across sources ─────────────────────────────────────
    seen_keys = set()
    unique_docs = []
    for doc in all_docs:
        if doc.dedupe_key not in seen_keys:
            seen_keys.add(doc.dedupe_key)
            unique_docs.append(doc)
    logger.info("Total unique documents: %d (from %d raw)", len(unique_docs), len(all_docs))

    # ── Ingest ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("DRY RUN — skipping actual ingestion")
    else:
        logger.info("INGESTION into collection '%s'", args.collection)
    logger.info("=" * 60)

    pipeline = CrawlerIngestionPipeline(
        collection=args.collection,
        dry_run=args.dry_run,
    )
    stats = pipeline.ingest(unique_docs)

    elapsed = time.time() - t0
    report["ingestion"] = stats
    report["total_unique_docs"] = len(unique_docs)
    report["elapsed_s"] = round(elapsed, 1)

    # ── Vector count ───────────────────────────────────────────────────
    if not args.dry_run:
        try:
            import chromadb
            db_path = os.path.join(os.path.dirname(__file__), "..", "data", "rag_vectordb")
            client = chromadb.PersistentClient(path=db_path)
            col = client.get_collection(args.collection)
            report["final_vector_count"] = col.count()
            logger.info("Final vector count: %d", col.count())
        except Exception as exc:
            logger.warning("Could not read vector count: %s", exc)

    # ── Report ─────────────────────────────────────────────────────────
    report_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"crawl_report_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("CRAWL COMPLETE")
    logger.info("=" * 60)
    logger.info("Documents crawled:  %d", len(unique_docs))
    logger.info("Documents ingested: %d", stats["ingested"])
    logger.info("Documents skipped:  %d", stats["skipped"])
    logger.info("Errors:             %d", stats["errors"])
    logger.info("Total chunks:       %d", stats["total_chunks"])
    logger.info("Elapsed:            %.1fs", elapsed)
    logger.info("Report saved:       %s", report_path)
    if report.get("final_vector_count"):
        logger.info("Final vectors:      %d", report["final_vector_count"])

    return report


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
