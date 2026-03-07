"""
orchestrator.py
---------------
RAGOrchestrator — wires PortfolioScraper, DocumentChunker, FAISSDatabase.

Cache vs Index separation
--------------------------
The scraper cache (raw HTML + Gemini video summaries) is independent of the
FAISS index.  This means:

  - Change chunk_size, embedding model, dedup_threshold, etc.?
    → Call rebuild_index() — re-chunks and re-embeds from cache, zero HTTP
      requests and zero Gemini API calls.

  - Add new pages to the portfolio?
    → Call ingest_portfolio() again — only genuinely new URLs are fetched;
      everything already cached is served from disk instantly.

Typical flow
------------
1. rag.ingest_portfolio(root_url)  — first run, populates cache + FAISS
2. rag.save()                       — persist FAISS index to disk
3. Later: change chunk_size / model, call rag.rebuild_index(root_url)
4. rag.query(question)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from scraper import PortfolioScraper, ScraperCache, ScrapedDocument
from chunker import DocumentChunker, DocumentChunk
from database import FAISSDatabase

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Parameters
    ----------
    gemini_api_key  : Gemini API key (YouTube video summarisation + RAG queries).
    hf_model_name   : Gemini embedding model name (default: "gemini-embedding-001").
    chunk_size      : Characters per chunk.
    chunk_overlap   : Overlap between consecutive chunks.
    dedup_threshold : Cosine similarity threshold for chunk dedup (None = off).
    index_dir       : Directory to save/load the FAISS index.
    cache_dir       : Directory for the scraper disk cache (HTML + video summaries).
                      Defaults to <index_dir>/../scraper_cache.
    follow_external : Follow links to external domains (Medium, arXiv, etc.).
    device          : Torch device for embeddings ("cpu", "cuda", …).
    """

    def __init__(
        self,
        gemini_api_key: str,
        hf_model_name: str = "gemini-embedding-001",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        dedup_threshold: Optional[float] = None,
        min_tokens: int = 30,
        index_dir: str = "./rag_index",
        cache_dir: Optional[str] = None,
        follow_external: bool = True,
        device: str = "cpu",
    ) -> None:

        # Resolve cache directory (default: sibling of index_dir)
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(index_dir)),
                                     "scraper_cache")

        self.cache = ScraperCache(cache_dir)

        self.scraper = PortfolioScraper(
            gemini_api_key=gemini_api_key,
            cache=self.cache,
            follow_external=follow_external,
        )

        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            dedup_threshold=dedup_threshold,
            min_tokens=min_tokens,
        )

        self.db = FAISSDatabase(
            model_name=hf_model_name,
            gemini_api_key=gemini_api_key,
            index_path=index_dir if os.path.exists(index_dir) else None,
            device=device,
        )

        self.index_dir = index_dir
        self.min_tokens = min_tokens

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_portfolio(self, root_url: str) -> int:
        """
        Scrape an entire portfolio and store in the vector DB.
        Cached pages/videos are served from disk — no redundant requests.
        Returns total chunks stored.
        """
        docs = self.scraper.scrape_portfolio(root_url)
        logger.info("Scraping done: %d docs  (cache: %s)", len(docs), self.cache.stats())

        by_type: dict[str, int] = {}
        by_section: dict[str, int] = {}
        for d in docs:
            by_type[d.doc_type]       = by_type.get(d.doc_type, 0) + 1
            by_section[d.section]     = by_section.get(d.section, 0) + 1
        logger.info("  type breakdown:    %s", by_type)
        logger.info("  section breakdown: %s", by_section)

        return self._store_docs(docs)

    def ingest_section(self, section_url: str, section_name: str) -> int:
        """Add or refresh a single section (index + content pattern)."""
        docs = self.scraper.process_section(section_url, section_name)
        return self._store_docs(docs)

    def ingest_videos(self, youtube_urls: list[str], section: str = "video") -> int:
        """Summarise YouTube URLs via Gemini (cache-aware) and store."""
        docs = self.scraper.summarise_videos(youtube_urls, section=section)
        return self._store_docs(docs)

    def rebuild_index(self, root_url: str) -> int:
        """
        Re-chunk and re-embed the entire portfolio using the current
        chunker/model config WITHOUT making any new HTTP or Gemini calls.

        Use this when you change chunk_size, embedding model, or dedup_threshold
        and want to rebuild FAISS from scratch.

        Steps:
          1. Reset the scraper session (clears _visited).
          2. scrape_portfolio() — every URL is already cached, so only
             BeautifulSoup parsing and text extraction runs.
          3. Clear the old FAISS index.
          4. Re-chunk and re-embed everything.
        """
        logger.info("Rebuilding index from cache (no network/Gemini calls)...")
        self.scraper.reset()
        docs = self.scraper.scrape_portfolio(root_url)
        logger.info("  %d docs recovered from cache", len(docs))

        self.db.clear()
        return self._store_docs(docs)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        section_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search over the vector DB.

        Parameters
        ----------
        question        : natural language question
        top_k           : max results to return
        section_filter  : e.g. "education", "video", "research"
        doc_type_filter : "index" for directory chunks,
                          "text" for content, "video_summary" for videos
        """
        results = self.db.search(question, top_k=top_k * 4, section_filter=section_filter)
        if doc_type_filter:
            results = [r for r in results if r.get("doc_type") == doc_type_filter]
        return results[:top_k]

    def query_doc(self, doc_index: int) -> list[DocumentChunk]:
        return self.db.get_all_chunks_for_doc(doc_index)

    # ------------------------------------------------------------------
    # Persistence & diagnostics
    # ------------------------------------------------------------------

    def save(self) -> None:
        self.db.save(self.index_dir)
        logger.info("FAISS index saved → %s", self.index_dir)

    def stats(self) -> dict:
        s = self.db.stats()
        s["cache"] = self.cache.stats()
        return s

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _store_docs(self, docs: list[ScrapedDocument]) -> int:
        """
        Chunk, optionally deduplicate, embed and add to FAISS. After adding,
        run remove_short_chunks() using the configured thresholds and then
        remove any documents with fewer than `self.min_chunks_per_doc` chunks.
        Return the approximate net number of chunks added by this call.
        """
        if not docs:
            return 0

        chunks = self.chunker.chunk_documents(
            docs,
            embed_fn=self.db.embed_one if self.chunker.dedup_threshold else None,
        )
        # add to DB
        self.db.add(chunks)
        added_count = len(chunks)
        logger.info("Stored %d chunks (before cleanup).", added_count)

        # 1) Run short/junk chunk removal policy after ingestion/rebuild
        removed_short = self.db.remove_short_chunks(
            min_tokens=self.min_tokens,
        )
        if removed_short:
            logger.info("Cleanup removed %d short/junk chunks after ingestion.", removed_short)


        # Return the net number of chunks added by this call (approx).
        net_added = max(0, added_count - (removed_short))
        logger.info("Net chunks added by this operation (approx): %d", net_added)
        return net_added