"""
Mars Science Document Crawler.

Crawls arXiv papers, NASA PDS instrument docs, and NASA Mars mission
technical pages. Produces MarsDocument objects ready for RAG ingestion.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MarsDocument:
    """A single document ready for ingestion."""
    title: str
    text: str
    source: str          # origin URL or identifier
    dedupe_key: str      # unique key for deduplication
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse whitespace and strip."""
    return _WS.sub(" ", text).strip()


def extract_html_text(html: str, *, min_length: int = 100) -> str:
    """Extract readable text from HTML, stripping boilerplate."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                              "form", "noscript", "aside", "iframe"]):
        tag.decompose()

    # Prefer main/article content
    main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|main|body", re.I))
    target = main if main else soup.body if soup.body else soup

    text = _normalize(target.get_text(separator=" "))
    return text if len(text) >= min_length else ""


def _safe_get(url: str, *, timeout: int = 30, retries: int = 2,
              params: Optional[dict] = None) -> Optional[requests.Response]:
    """GET with retries and exponential backoff."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, params=params, headers={
                "User-Agent": "MarsLab-Crawler/1.0 (academic research)"
            })
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.warning("Rate limited on %s – waiting %ds", url, wait)
                time.sleep(wait)
                continue
            logger.warning("HTTP %d for %s", resp.status_code, url)
            return None
        except requests.RequestException as exc:
            logger.warning("Request error (attempt %d) for %s: %s", attempt + 1, url, exc)
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# arXiv crawler
# ---------------------------------------------------------------------------

ARXIV_API = "https://export.arxiv.org/api/query"

ARXIV_SEARCH_TERMS = [
    "Mars atmosphere",
    "Mars geology",
    "Mars climate",
    "Mars GCM",
    "Mars interior",
    "Mars seismology",
    "InSight SEIS",
    "Mars water ice",
    "Mars mineralogy",
    "HiRISE",
    "CRISM",
    "SHARAD",
    "Mars Reconnaissance Orbiter",
    "Curiosity rover",
    "Perseverance rover",
    "Mars habitability",
    "Mars radiation",
    "Mars dust storm",
    "Olympus Mons",
    "Valles Marineris",
    "Mars polar caps",
]

ARXIV_CATEGORIES = ["astro-ph.EP", "physics.geo-ph", "physics.ao-ph"]


class ArxivCrawler:
    """Fetch Mars-related papers from arXiv Atom API."""

    def __init__(self, *, max_per_query: int = 50, rate_delay: float = 3.0):
        self.max_per_query = max_per_query
        self.rate_delay = rate_delay
        self._seen_ids: set = set()

    def crawl(self, terms: Optional[List[str]] = None) -> List[MarsDocument]:
        terms = terms or ARXIV_SEARCH_TERMS
        docs: List[MarsDocument] = []

        for term in terms:
            logger.info("[arXiv] Searching: %s", term)
            batch = self._search_term(term)
            new = [d for d in batch if d.dedupe_key not in self._seen_ids]
            for d in new:
                self._seen_ids.add(d.dedupe_key)
            docs.extend(new)
            logger.info("[arXiv]   +%d new (total unique: %d)", len(new), len(self._seen_ids))

        logger.info("[arXiv] Crawl complete: %d unique papers", len(docs))
        return docs

    def _search_term(self, term: str) -> List[MarsDocument]:
        """Search a single term across categories with pagination."""
        cat_query = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
        search_query = f"all:{term} AND ({cat_query})"

        results: List[MarsDocument] = []
        start = 0
        page_size = min(self.max_per_query, 100)  # arXiv max 100 per request

        while start < self.max_per_query:
            fetch = min(page_size, self.max_per_query - start)
            params = {
                "search_query": search_query,
                "start": start,
                "max_results": fetch,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

            time.sleep(self.rate_delay)
            resp = _safe_get(ARXIV_API, timeout=60, params=params)
            if not resp:
                break

            feed = feedparser.parse(resp.text)
            entries = feed.get("entries", [])
            if not entries:
                break

            for entry in entries:
                doc = self._parse_entry(entry, term)
                if doc:
                    results.append(doc)

            start += len(entries)
            if len(entries) < fetch:
                break  # no more pages

        return results

    @staticmethod
    def _parse_entry(entry: dict, search_term: str) -> Optional[MarsDocument]:
        """Parse a single arXiv Atom entry."""
        arxiv_id = entry.get("id", "")
        if not arxiv_id:
            return None

        # Clean ID: http://arxiv.org/abs/XXXX.XXXXX -> XXXX.XXXXX
        clean_id = arxiv_id.split("/abs/")[-1] if "/abs/" in arxiv_id else arxiv_id

        title = _normalize(entry.get("title", ""))
        summary = _normalize(entry.get("summary", ""))
        if not title or not summary:
            return None

        authors = ", ".join(a.get("name", "") for a in entry.get("authors", []))
        published = entry.get("published", "")
        year = published[:4] if len(published) >= 4 else "unknown"

        # Build document text: title + authors + abstract
        text_parts = [
            f"Title: {title}",
            f"Authors: {authors}",
            f"Year: {year}",
            f"arXiv ID: {clean_id}",
            "",
            f"Abstract: {summary}",
        ]
        text = "\n".join(text_parts)

        categories = ", ".join(t.get("term", "") for t in entry.get("tags", []))

        return MarsDocument(
            title=title,
            text=text,
            source=f"arXiv:{clean_id}",
            dedupe_key=clean_id,
            metadata={
                "type": "arxiv_paper",
                "arxiv_id": clean_id,
                "authors": authors[:500],  # truncate very long author lists
                "year": year,
                "categories": categories,
                "search_term": search_term,
            },
        )


# ---------------------------------------------------------------------------
# NASA PDS crawler
# ---------------------------------------------------------------------------

NASA_PDS_URLS = [
    # PDS Geosciences Node – Mars instruments
    "https://pds-geosciences.wustl.edu/missions/mro/crism.htm",
    "https://pds-geosciences.wustl.edu/missions/mro/sharad.htm",
    "https://pds-geosciences.wustl.edu/missions/mro/mcs.htm",
    "https://pds-geosciences.wustl.edu/missions/msl/chemcam.htm",
    "https://pds-geosciences.wustl.edu/missions/msl/apxs.htm",
    "https://pds-geosciences.wustl.edu/missions/msl/chemin.htm",
    "https://pds-geosciences.wustl.edu/missions/msl/sam.htm",
    "https://pds-geosciences.wustl.edu/missions/mars2020/sherloc.htm",
    "https://pds-geosciences.wustl.edu/missions/mars2020/pixl.htm",
    "https://pds-geosciences.wustl.edu/missions/mars2020/supercam.htm",
    "https://pds-geosciences.wustl.edu/missions/mars2020/rimfax.htm",
    "https://pds-geosciences.wustl.edu/missions/mgs/tes.htm",
    "https://pds-geosciences.wustl.edu/missions/odyssey/grs.htm",
    "https://pds-geosciences.wustl.edu/missions/odyssey/ns.htm",
    "https://pds-geosciences.wustl.edu/missions/phoenix/tega.htm",
    "https://pds-geosciences.wustl.edu/missions/phoenix/meca.htm",
    # PDS Imaging Node
    "https://pds-imaging.jpl.nasa.gov/portal/mro_mission.html",
    "https://pds-imaging.jpl.nasa.gov/portal/msl_mission.html",
    "https://pds-imaging.jpl.nasa.gov/portal/mars2020_mission.html",
]


class NasaPdsCrawler:
    """Fetch NASA PDS instrument documentation pages."""

    def __init__(self, *, rate_delay: float = 2.0):
        self.rate_delay = rate_delay
        self._seen: set = set()

    def crawl(self, urls: Optional[List[str]] = None) -> List[MarsDocument]:
        urls = urls or NASA_PDS_URLS
        docs: List[MarsDocument] = []

        for url in urls:
            logger.info("[NASA PDS] Fetching: %s", url)
            time.sleep(self.rate_delay)

            resp = _safe_get(url)
            if not resp:
                continue

            text = extract_html_text(resp.text)
            if not text or len(text) < 100:
                logger.info("[NASA PDS]   Skipped (too short)")
                continue

            parsed = urlparse(url)
            dedupe = parsed.netloc + parsed.path
            if dedupe in self._seen:
                continue
            self._seen.add(dedupe)

            # Infer mission/instrument from URL path
            mission, instrument = self._infer_mission_instrument(url)
            title = self._extract_title(resp.text, url)

            docs.append(MarsDocument(
                title=title,
                text=text,
                source=url,
                dedupe_key=dedupe,
                metadata={
                    "type": "nasa_pds",
                    "mission": mission,
                    "instrument": instrument,
                    "url": url,
                },
            ))
            logger.info("[NASA PDS]   OK: %s (%d chars)", title, len(text))

        logger.info("[NASA PDS] Crawl complete: %d documents", len(docs))
        return docs

    @staticmethod
    def _infer_mission_instrument(url: str) -> tuple:
        path = urlparse(url).path.lower()
        missions = {
            "mro": "Mars Reconnaissance Orbiter",
            "msl": "Mars Science Laboratory (Curiosity)",
            "mars2020": "Mars 2020 (Perseverance)",
            "mgs": "Mars Global Surveyor",
            "odyssey": "Mars Odyssey",
            "phoenix": "Phoenix",
            "insight": "InSight",
        }
        mission = "Mars"
        instrument = ""
        for key, name in missions.items():
            if key in path:
                mission = name
                break
        # instrument is usually the last part of the path
        parts = [p for p in path.rstrip("/").split("/") if p]
        if parts:
            inst = parts[-1].replace(".htm", "").replace(".html", "")
            if inst not in missions:
                instrument = inst.upper()
        return mission, instrument

    @staticmethod
    def _extract_title(html: str, url: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return _normalize(title_tag.string)
        h1 = soup.find("h1")
        if h1:
            return _normalize(h1.get_text())
        return urlparse(url).path.split("/")[-1]


# ---------------------------------------------------------------------------
# NASA Mars Exploration Program crawler
# ---------------------------------------------------------------------------

NASA_MARS_URLS = [
    # Mars mission pages
    "https://science.nasa.gov/mission/mars-reconnaissance-orbiter/",
    "https://science.nasa.gov/mission/mars-2020-perseverance/",
    "https://science.nasa.gov/mission/msl-curiosity/",
    "https://science.nasa.gov/mission/insight/",
    "https://science.nasa.gov/mission/maven/",
    "https://science.nasa.gov/mission/mars-odyssey/",
    "https://science.nasa.gov/mission/mars-express/",
    "https://science.nasa.gov/mission/phoenix/",
    "https://science.nasa.gov/mission/mars-exploration-rovers-spirit-and-opportunity/",
    # Mars overview / science
    "https://science.nasa.gov/mars/",
    "https://science.nasa.gov/mars/facts/",
    # NSSDC fact sheets
    "https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html",
]


class NasaTechDocsCrawler:
    """Fetch NASA Mars mission and science pages."""

    def __init__(self, *, rate_delay: float = 2.0):
        self.rate_delay = rate_delay
        self._seen: set = set()

    def crawl(self, urls: Optional[List[str]] = None) -> List[MarsDocument]:
        urls = urls or NASA_MARS_URLS
        docs: List[MarsDocument] = []

        for url in urls:
            logger.info("[NASA Tech] Fetching: %s", url)
            time.sleep(self.rate_delay)

            resp = _safe_get(url)
            if not resp:
                continue

            text = extract_html_text(resp.text)
            if not text or len(text) < 100:
                logger.info("[NASA Tech]   Skipped (too short)")
                continue

            parsed = urlparse(url)
            dedupe = parsed.netloc + parsed.path.rstrip("/")
            if dedupe in self._seen:
                continue
            self._seen.add(dedupe)

            mission = self._infer_mission(url)
            title = self._extract_title(resp.text, url)

            docs.append(MarsDocument(
                title=title,
                text=text,
                source=url,
                dedupe_key=dedupe,
                metadata={
                    "type": "nasa_tech_doc",
                    "mission": mission,
                    "url": url,
                },
            ))
            logger.info("[NASA Tech]   OK: %s (%d chars)", title, len(text))

        logger.info("[NASA Tech] Crawl complete: %d documents", len(docs))
        return docs

    @staticmethod
    def _infer_mission(url: str) -> str:
        path = urlparse(url).path.lower()
        mapping = {
            "mars-reconnaissance-orbiter": "MRO",
            "mars-2020-perseverance": "Perseverance",
            "msl-curiosity": "Curiosity",
            "insight": "InSight",
            "maven": "MAVEN",
            "mars-odyssey": "Mars Odyssey",
            "mars-express": "Mars Express",
            "phoenix": "Phoenix",
            "exploration-rovers": "Spirit & Opportunity",
        }
        for key, name in mapping.items():
            if key in path:
                return name
        return "Mars"

    @staticmethod
    def _extract_title(html: str, url: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return _normalize(title_tag.string)
        h1 = soup.find("h1")
        if h1:
            return _normalize(h1.get_text())
        return urlparse(url).path.split("/")[-1]


# ---------------------------------------------------------------------------
# Ingestion pipeline adapter
# ---------------------------------------------------------------------------

class CrawlerIngestionPipeline:
    """Ingest MarsDocuments into the RAG vector store."""

    def __init__(self, *, collection: str = "mars_science",
                 chunk_size: int = 512, chunk_overlap: int = 64,
                 dry_run: bool = False):
        self.collection = collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dry_run = dry_run
        self.stats = {
            "ingested": 0,
            "skipped": 0,
            "errors": 0,
            "total_chunks": 0,
        }

    def ingest(self, docs: List[MarsDocument]) -> Dict:
        """Ingest all documents. Returns stats."""
        for doc in docs:
            if not doc.text or len(doc.text.strip()) < 50:
                self.stats["skipped"] += 1
                continue

            if self.dry_run:
                self.stats["ingested"] += 1
                self.stats["total_chunks"] += max(1, len(doc.text) // self.chunk_size)
                continue

            try:
                from .ingestion import ingest_text
                result = ingest_text(
                    text=doc.text,
                    source=doc.source,
                    title=doc.title,
                    collection=self.collection,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    metadata=doc.metadata,
                )
                chunks = result.get("chunks", 0)
                if result.get("status") == "ok" and chunks > 0:
                    self.stats["ingested"] += 1
                    self.stats["total_chunks"] += chunks
                else:
                    self.stats["skipped"] += 1
            except Exception as exc:
                logger.error("Ingestion error for %s: %s", doc.source, exc)
                self.stats["errors"] += 1

        return self.stats
