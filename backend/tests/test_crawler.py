"""Tests for rag.crawler module."""

import json
import types
from unittest.mock import MagicMock, patch

import pytest

from rag.crawler import (
    ArxivCrawler,
    CrawlerIngestionPipeline,
    MarsDocument,
    NasaPdsCrawler,
    NasaTechDocsCrawler,
    _normalize,
    extract_html_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v1</id>
    <title>Atmospheric Dynamics on Mars: A GCM Study</title>
    <summary>We present a comprehensive study of Mars atmospheric dynamics using a general circulation model. Our results show significant seasonal variations in dust lifting and transport mechanisms.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Johnson</name></author>
    <published>2023-01-15T00:00:00Z</published>
    <category term="astro-ph.EP"/>
    <category term="physics.ao-ph"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2302.67890v2</id>
    <title>InSight SEIS: Seismic Constraints on Mars Interior</title>
    <summary>Using data from the InSight SEIS instrument, we constrain the velocity structure of the Martian crust and upper mantle. P-wave velocities indicate a layered crustal structure.</summary>
    <author><name>Carol Lee</name></author>
    <published>2023-02-20T00:00:00Z</published>
    <category term="astro-ph.EP"/>
  </entry>
</feed>
"""

SAMPLE_NASA_HTML = """<!DOCTYPE html>
<html>
<head><title>CRISM - Compact Reconnaissance Imaging Spectrometer for Mars</title></head>
<body>
<header><nav>Navigation links here</nav></header>
<main>
<h1>CRISM Instrument</h1>
<p>The Compact Reconnaissance Imaging Spectrometer for Mars (CRISM) is a visible-infrared
spectrometer aboard Mars Reconnaissance Orbiter (MRO). CRISM searches for mineralogic
indicators of past and present water activity on Mars. It measures 544 wavelengths from
362 to 3920 nm, providing detailed mineral maps of the Martian surface.</p>
<p>CRISM has two modes: targeted observations at 18 m/pixel and survey mode at 200 m/pixel.
The instrument has discovered diverse hydrated minerals including phyllosilicates,
sulfates, and carbonates across the Martian surface.</p>
</main>
<footer>Footer content</footer>
<script>var x = 1;</script>
</body>
</html>
"""

SAMPLE_BOILERPLATE_HTML = """<html><body><nav>Menu</nav><footer>End</footer></body></html>"""


# ---------------------------------------------------------------------------
# Tests: extract_html_text
# ---------------------------------------------------------------------------

class TestExtractHtmlText:
    def test_extracts_main_content(self):
        text = extract_html_text(SAMPLE_NASA_HTML)
        assert "CRISM Instrument" in text
        assert "visible-infrared spectrometer" in text
        assert "544 wavelengths" in text

    def test_strips_nav_footer_script(self):
        text = extract_html_text(SAMPLE_NASA_HTML)
        assert "Navigation links" not in text
        assert "Footer content" not in text
        assert "var x = 1" not in text

    def test_skips_short_boilerplate(self):
        text = extract_html_text(SAMPLE_BOILERPLATE_HTML, min_length=100)
        assert text == ""

    def test_normalize_whitespace(self):
        html = "<html><body><main><p>Hello   \n\n   world</p></main></body></html>"
        text = extract_html_text(html, min_length=5)
        assert "Hello world" in text


# ---------------------------------------------------------------------------
# Tests: _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_collapses_whitespace(self):
        assert _normalize("  hello   world  \n\n  test  ") == "hello world test"

    def test_empty_string(self):
        assert _normalize("") == ""


# ---------------------------------------------------------------------------
# Tests: ArxivCrawler
# ---------------------------------------------------------------------------

class TestArxivCrawler:
    def _mock_response(self, text, status=200):
        resp = MagicMock()
        resp.status_code = status
        resp.text = text
        return resp

    @patch("rag.crawler._safe_get")
    def test_parses_arxiv_entries(self, mock_get):
        mock_get.return_value = self._mock_response(SAMPLE_ARXIV_ATOM)
        crawler = ArxivCrawler(max_per_query=10, rate_delay=0)
        docs = crawler.crawl(terms=["Mars atmosphere"])

        assert len(docs) == 2
        assert docs[0].title == "Atmospheric Dynamics on Mars: A GCM Study"
        assert "2301.12345v1" in docs[0].dedupe_key
        assert docs[0].metadata["type"] == "arxiv_paper"
        assert docs[0].metadata["year"] == "2023"
        assert "Alice Smith" in docs[0].metadata["authors"]

    @patch("rag.crawler._safe_get")
    def test_deduplicates_across_terms(self, mock_get):
        mock_get.return_value = self._mock_response(SAMPLE_ARXIV_ATOM)
        crawler = ArxivCrawler(max_per_query=10, rate_delay=0)
        # Same feed returned for two different terms
        docs = crawler.crawl(terms=["Mars atmosphere", "Mars GCM"])
        # Should only have 2 unique papers, not 4
        assert len(docs) == 2

    @patch("rag.crawler._safe_get")
    def test_handles_empty_response(self, mock_get):
        empty_feed = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        mock_get.return_value = self._mock_response(empty_feed)
        crawler = ArxivCrawler(max_per_query=10, rate_delay=0)
        docs = crawler.crawl(terms=["nonexistent"])
        assert len(docs) == 0

    @patch("rag.crawler._safe_get")
    def test_handles_network_failure(self, mock_get):
        mock_get.return_value = None
        crawler = ArxivCrawler(max_per_query=10, rate_delay=0)
        docs = crawler.crawl(terms=["Mars"])
        assert len(docs) == 0

    def test_parse_entry_extracts_fields(self):
        import feedparser
        feed = feedparser.parse(SAMPLE_ARXIV_ATOM)
        entry = feed.entries[1]
        doc = ArxivCrawler._parse_entry(entry, "test")
        assert doc is not None
        assert "InSight SEIS" in doc.title
        assert doc.metadata["arxiv_id"] == "2302.67890v2"
        assert doc.metadata["year"] == "2023"
        assert "Carol Lee" in doc.metadata["authors"]


# ---------------------------------------------------------------------------
# Tests: NasaPdsCrawler
# ---------------------------------------------------------------------------

class TestNasaPdsCrawler:
    @patch("rag.crawler._safe_get")
    def test_crawls_html_page(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = SAMPLE_NASA_HTML
        mock_get.return_value = resp

        crawler = NasaPdsCrawler(rate_delay=0)
        docs = crawler.crawl(urls=["https://pds-geosciences.wustl.edu/missions/mro/crism.htm"])
        assert len(docs) == 1
        assert "CRISM" in docs[0].title
        assert docs[0].metadata["type"] == "nasa_pds"
        assert docs[0].metadata["mission"] == "Mars Reconnaissance Orbiter"
        assert docs[0].metadata["instrument"] == "CRISM"

    def test_infer_mission_instrument(self):
        m, i = NasaPdsCrawler._infer_mission_instrument(
            "https://pds-geosciences.wustl.edu/missions/mars2020/sherloc.htm"
        )
        assert m == "Mars 2020 (Perseverance)"
        assert i == "SHERLOC"

    @patch("rag.crawler._safe_get")
    def test_skips_boilerplate(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = SAMPLE_BOILERPLATE_HTML
        mock_get.return_value = resp

        crawler = NasaPdsCrawler(rate_delay=0)
        docs = crawler.crawl(urls=["https://example.com/empty"])
        assert len(docs) == 0


# ---------------------------------------------------------------------------
# Tests: NasaTechDocsCrawler
# ---------------------------------------------------------------------------

class TestNasaTechDocsCrawler:
    def test_infer_mission(self):
        assert NasaTechDocsCrawler._infer_mission(
            "https://science.nasa.gov/mission/msl-curiosity/"
        ) == "Curiosity"
        assert NasaTechDocsCrawler._infer_mission(
            "https://science.nasa.gov/mission/insight/"
        ) == "InSight"
        assert NasaTechDocsCrawler._infer_mission(
            "https://science.nasa.gov/mars/facts/"
        ) == "Mars"


# ---------------------------------------------------------------------------
# Tests: CrawlerIngestionPipeline
# ---------------------------------------------------------------------------

class TestIngestionPipeline:
    def test_dry_run_skips_actual_ingest(self):
        pipeline = CrawlerIngestionPipeline(dry_run=True)
        docs = [
            MarsDocument(title="Test", text="A" * 200, source="test",
                         dedupe_key="k1", metadata={"type": "test"}),
        ]
        stats = pipeline.ingest(docs)
        assert stats["ingested"] == 1
        assert stats["errors"] == 0

    def test_skips_short_documents(self):
        pipeline = CrawlerIngestionPipeline(dry_run=True)
        docs = [
            MarsDocument(title="Short", text="Hi", source="test",
                         dedupe_key="k2", metadata={}),
        ]
        stats = pipeline.ingest(docs)
        assert stats["skipped"] == 1
        assert stats["ingested"] == 0

    @patch("rag.crawler.CrawlerIngestionPipeline.ingest")
    def test_pipeline_returns_stats(self, mock_ingest):
        mock_ingest.return_value = {"ingested": 5, "skipped": 1, "errors": 0, "total_chunks": 25}
        pipeline = CrawlerIngestionPipeline()
        result = pipeline.ingest([])
        assert "ingested" in result


# ---------------------------------------------------------------------------
# Tests: MarsDocument
# ---------------------------------------------------------------------------

class TestMarsDocument:
    def test_dataclass_fields(self):
        doc = MarsDocument(
            title="Test Paper",
            text="Some content about Mars",
            source="arXiv:1234.5678",
            dedupe_key="1234.5678",
            metadata={"type": "arxiv_paper"},
        )
        assert doc.title == "Test Paper"
        assert doc.metadata["type"] == "arxiv_paper"

    def test_default_metadata(self):
        doc = MarsDocument(title="T", text="X", source="S", dedupe_key="K")
        assert doc.metadata == {}
