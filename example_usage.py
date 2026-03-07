"""
example_usage.py
----------------
Full walkthrough of the portfolio RAG pipeline.
Replace PORTFOLIO_URL with the real URL before running.
"""

import logging
import os
from pprint import pprint

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

PORTFOLIO_URL = "https://yourportfolio.com"   # ← change this

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
from orchestrator import RAGOrchestrator
from rag_query import GeminiRAG

rag = RAGOrchestrator(
    gemini_api_key=os.environ["GEMINI_API_KEY"],
    hf_model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=50,
    dedup_threshold=0.98,

    index_dir="./rag_index",
    # cache_dir defaults to ./scraper_cache (sibling of index_dir)
    # Set explicitly if you want it elsewhere:
    # cache_dir="./scraper_cache",

    follow_external=True,    # follow Medium, arXiv, Huffington Post, FAMe, etc.
    device="cpu",
)

# ---------------------------------------------------------------------------
# First run — scrapes portfolio, populates cache and FAISS
# Sections auto-discovered: biography, video, opinion, employment, education,
# advisor, cases, research, associate_editor, contact, executive_teaching,
# research_in_progress, working_papers, fame, general
# ---------------------------------------------------------------------------
print("\n=== First run ===")
total = rag.ingest_portfolio(PORTFOLIO_URL)
print(f"Chunks stored: {total}")
rag.save()
pprint(rag.stats())
# Stats will now show:
#   cache: { cached_pages: N, cached_videos: M }
#   sections: { biography: x, video: y, research: z, ... }

# ---------------------------------------------------------------------------
# Second run (different machine / fresh process)
# ALL pages and video summaries served from cache — zero HTTP, zero Gemini
# ---------------------------------------------------------------------------
print("\n=== Second run (cache warm) ===")
rag2 = RAGOrchestrator(
    gemini_api_key=os.environ["GEMINI_API_KEY"],
    hf_model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=50,
    index_dir="./rag_index",
    # Same cache_dir → cache is reused
)
total2 = rag2.ingest_portfolio(PORTFOLIO_URL)
print(f"Chunks stored (from cache): {total2}")

# ---------------------------------------------------------------------------
# Experimenting with chunk size — rebuild FAISS, NO re-scraping at all
# ---------------------------------------------------------------------------
print("\n=== Rebuild index with different chunk size ===")
rag_exp = RAGOrchestrator(
    gemini_api_key=os.environ["GEMINI_API_KEY"],
    hf_model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=300,          # ← new chunk size
    chunk_overlap=30,
    index_dir="./rag_index_300",
    # Same cache_dir — reads from the same cache
)
# rebuild_index re-runs text extraction from cached HTML + Gemini responses
# No HTTP requests, no Gemini API calls
total_exp = rag_exp.rebuild_index(PORTFOLIO_URL)
print(f"Chunks with chunk_size=300: {total_exp}")
rag_exp.save()

# Experiment with a different embedding model
rag_exp2 = RAGOrchestrator(
    gemini_api_key=os.environ["GEMINI_API_KEY"],
    hf_model_name="sentence-transformers/all-mpnet-base-v2",  # larger model
    chunk_size=500,
    chunk_overlap=50,
    index_dir="./rag_index_mpnet",
)
total_exp2 = rag_exp2.rebuild_index(PORTFOLIO_URL)
print(f"Chunks with mpnet model: {total_exp2}")
rag_exp2.save()

# ---------------------------------------------------------------------------
# GeminiRAG — answer questions using retrieved chunks
# ---------------------------------------------------------------------------
gemini_rag = GeminiRAG(
    db=rag.db,
    gemini_api_key=os.environ["GEMINI_API_KEY"],
    gemini_model="gemini-2.0-flash",
    top_k=6,
)

# Single-shot answer
print("\n=== Q: What is her research about? ===")
result = gemini_rag.answer("What is her primary research focus?")
print(gemini_rag.format_answer(result))

# Section-filtered question
print("\n=== Q: Education (filtered) ===")
result = gemini_rag.answer(
    "What degrees does she hold and from which universities?",
    section_filter="education",
)
print(gemini_rag.format_answer(result))

# Ask about videos (index chunk gives the directory listing)
print("\n=== Q: What videos are available? ===")
result = gemini_rag.answer(
    "List all the videos or talks available on the portfolio.",
    doc_type_filter="index",
    section_filter="video",
)
print(gemini_rag.format_answer(result))

# Streaming answer
print("\n=== Streaming: What are her working papers about? ===")
gen = gemini_rag.stream_answer(
    "What are her current working papers about?",
    section_filter="working_papers",
)
try:
    while True:
        print(next(gen), end="", flush=True)
except StopIteration as e:
    final = e.value
print()
print(f"\nSources: {[s.doc_title for s in final.sources]}")
print(f"Tokens:  {final.total_tokens_used}")

# ---------------------------------------------------------------------------
# Direct class usage (fine-grained control)
# ---------------------------------------------------------------------------
from scraper import PortfolioScraper, ScraperCache
from chunker import DocumentChunker
from database import FAISSDatabase

cache   = ScraperCache("./scraper_cache")
scraper = PortfolioScraper(gemini_api_key=os.environ["GEMINI_API_KEY"], cache=cache)
chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
db      = FAISSDatabase(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Inspect cache stats before scraping
print("\nCache before:", cache.stats())

docs = scraper.scrape_portfolio(PORTFOLIO_URL)

print("\nCache after:", cache.stats())
print(f"Total docs: {len(docs)}")

# Breakdown
for dt in ("index", "text", "video_summary"):
    n = sum(1 for d in docs if d.doc_type == dt)
    print(f"  {dt}: {n}")

chunks = chunker.chunk_documents(docs)
db.add(chunks)
pprint(db.stats())

# Direct Gemini RAG from this db
g = GeminiRAG(db=db, gemini_api_key=os.environ["GEMINI_API_KEY"])
ans = g.answer("What Huffington Post articles has she written?", section_filter="opinion")
print(g.format_answer(ans))