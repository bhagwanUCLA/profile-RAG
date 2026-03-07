"""
server.py
---------
FastAPI backend that exposes the RAG pipeline over HTTP.
The React frontend connects to this.

Endpoints
---------
POST /ingest          — scrape & index a portfolio URL
POST /rebuild         — rebuild FAISS index from cache (no network calls)
POST /query           — semantic search, returns chunks + Gemini answer
GET  /query/stream    — same but streams Gemini tokens as SSE
GET  /stats           — cache + index statistics
GET  /cache/clear     — wipe scraper cache
POST /index/clear     — wipe FAISS index

Run
---
    pip install fastapi uvicorn[standard]
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from orchestrator import RAGOrchestrator
from rag_query import GeminiRAG
from dotenv import load_dotenv

# Optionally set the path to the .env file (defaults to current directory if omitted)
env_path = Path(__file__).parent / ".env"

# Load the .env file into environment variables
# load_dotenv returns True if the file was found and parsed
load_dotenv(dotenv_path=env_path)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state — single RAGOrchestrator instance, swappable via /config
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "gemini_api_key":  os.environ.get("GEMINI_API_KEY", ""),
    "hf_model_name":   "gemini-embedding-001",
    "chunk_size":      500,
    "chunk_overlap":   50,
    "dedup_threshold": None,
    "min_tokens":      20,
    "index_dir":       "./rag_index",
    "cache_dir":       "./scraper_cache",
    "follow_external": True,
    "device":          "cpu",
    "gemini_model":    "gemini-3-flash-preview",
    "top_k":           6,
}

_current_config: dict = dict(_DEFAULT_CONFIG)
_rag: Optional[RAGOrchestrator] = None
_gemini_rag: Optional[GeminiRAG] = None


def _get_rag() -> RAGOrchestrator:
    global _rag
    if _rag is None:
        _rag = RAGOrchestrator(
            gemini_api_key=_current_config["gemini_api_key"],
            hf_model_name=_current_config["hf_model_name"],
            chunk_size=_current_config["chunk_size"],
            chunk_overlap=_current_config["chunk_overlap"],
            dedup_threshold=_current_config["dedup_threshold"],
            min_tokens=_current_config["min_tokens"],
            index_dir=_current_config["index_dir"],
            cache_dir=_current_config["cache_dir"],
            follow_external=_current_config["follow_external"],
            device=_current_config["device"],
        )
    return _rag


def _get_gemini_rag(system_prompt: Optional[str] = None) -> GeminiRAG:
    return GeminiRAG(
        db=_get_rag().db,
        gemini_api_key=_current_config["gemini_api_key"],
        gemini_model=_current_config["gemini_model"],
        top_k=_current_config["top_k"],
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    gemini_api_key:  Optional[str]   = None
    hf_model_name:   Optional[str]   = None
    chunk_size:      Optional[int]   = None
    chunk_overlap:   Optional[int]   = None
    dedup_threshold: Optional[float] = None
    min_tokens:      Optional[int]   = None
    index_dir:       Optional[str]   = None
    cache_dir:       Optional[str]   = None
    follow_external: Optional[bool]  = None
    device:          Optional[str]   = None
    gemini_model:    Optional[str]   = None
    top_k:           Optional[int]   = None


class IngestRequest(BaseModel):
    url: str
    rebuild: bool = False   # if True, rebuild index from cache without scraping


class QueryRequest(BaseModel):
    question:        str
    top_k:           int            = Field(default=6, ge=1, le=20)
    section_filter:  Optional[str]  = None
    doc_type_filter: Optional[str]  = None
    system_prompt:   Optional[str]  = None
    gemini_model:    Optional[str]  = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def get_config():
    safe = dict(_current_config)
    if safe.get("gemini_api_key"):
        safe["gemini_api_key"] = "sk-…(set)"
    return safe


@app.post("/config")
def update_config(body: ConfigUpdate):
    """Update pipeline config. Reinitialises RAG objects on next request."""
    global _rag, _gemini_rag, _current_config
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    _current_config.update(updates)
    # Force re-init on next use
    _rag = None
    _gemini_rag = None
    return {"updated": list(updates.keys()), "config": get_config()}


@app.post("/ingest")
def ingest(body: IngestRequest):
    """Scrape a portfolio URL and populate the vector index."""
    rag = _get_rag()
    if body.rebuild:
        chunks = rag.rebuild_index(body.url)
        action = "rebuild"
    else:
        chunks = rag.ingest_portfolio(body.url)
        action = "ingest"
    rag.save()
    return {"action": action, "chunks_stored": chunks, "stats": rag.stats()}


@app.get("/stats")
def stats():
    return _get_rag().stats()


@app.post("/query")
def query(body: QueryRequest):
    """
    Retrieve relevant chunks AND generate a Gemini answer.
    Returns chunks + answer + sources in one response.
    """
    rag = _get_rag()

    # Override model per-request if provided
    if body.gemini_model:
        _current_config["gemini_model"] = body.gemini_model

    g = _get_gemini_rag(system_prompt=body.system_prompt)

    # Retrieve chunks for display
    chunks = rag.query(
        question=body.question,
        top_k=body.top_k,
        section_filter=body.section_filter,
        doc_type_filter=body.doc_type_filter,
    )

    # Generate answer
    result = g.answer(
        question=body.question,
        top_k=body.top_k,
        section_filter=body.section_filter,
        doc_type_filter=body.doc_type_filter,
    )

    return {
        "question": body.question,
        "answer":   result.answer,
        "tokens_used": result.total_tokens_used,
        "chunks": [
            {
                "rank":         i + 1,
                "score":        r["score"],
                "doc_index":    r["doc_index"],
                "doc_title":    r["doc_title"],
                "section":      r["section"],
                "doc_type":     r["doc_type"],
                "doc_url":      r["doc_url"],
                "chunk_index":  r["chunk_index"],
                "raw_content":  r["raw_content"],
                "full_text":    r["text"],
            }
            for i, r in enumerate(chunks)
        ],
        "sources": [
            {
                "doc_index": s.doc_index,
                "doc_title": s.doc_title,
                "section":   s.section,
                "doc_type":  s.doc_type,
                "doc_url":   s.doc_url,
                "score":     s.score,
            }
            for s in result.sources
        ],
    }


@app.get("/query/stream")
async def query_stream(
    question:        str,
    top_k:           int  = 6,
    section_filter:  Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    system_prompt:   Optional[str] = None,
    gemini_model:    Optional[str] = None,
):
    """
    Server-Sent Events streaming endpoint.
    Emits:
      event: chunk   → each retrieved chunk as JSON (sent first, before generation)
      event: token   → each Gemini text token
      event: done    → final metadata (sources, tokens_used)
      event: error   → on failure
    """
    rag = _get_rag()
    if gemini_model:
        _current_config["gemini_model"] = gemini_model

    g = _get_gemini_rag(system_prompt=system_prompt)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            # 1. Send retrieved chunks immediately
            chunks = rag.query(
                question=question,
                top_k=top_k,
                section_filter=section_filter,
                doc_type_filter=doc_type_filter,
            )
            for i, r in enumerate(chunks):
                payload = json.dumps({
                    "rank":        i + 1,
                    "score":       r["score"],
                    "doc_index":   r["doc_index"],
                    "doc_title":   r["doc_title"],
                    "section":     r["section"],
                    "doc_type":    r["doc_type"],
                    "doc_url":     r["doc_url"],
                    "chunk_index": r["chunk_index"],
                    "raw_content": r["raw_content"],
                })
                yield f"event: chunk\ndata: {payload}\n\n"
                await asyncio.sleep(0)

            # 2. Stream Gemini tokens
            gen = g.stream_answer(
                question=question,
                top_k=top_k,
                section_filter=section_filter,
                doc_type_filter=doc_type_filter,
            )
            final_answer = None
            try:
                while True:
                    token = next(gen)
                    yield f"event: token\ndata: {json.dumps(token)}\n\n"
                    await asyncio.sleep(0)
            except StopIteration as e:
                final_answer = e.value

            # 3. Done event
            if final_answer:
                done_payload = json.dumps({
                    "tokens_used": final_answer.total_tokens_used,
                    "sources": [
                        {
                            "doc_index": s.doc_index,
                            "doc_title": s.doc_title,
                            "section":   s.section,
                            "doc_type":  s.doc_type,
                            "doc_url":   s.doc_url,
                            "score":     s.score,
                        }
                        for s in final_answer.sources
                    ],
                })
                yield f"event: done\ndata: {done_payload}\n\n"

        except Exception as exc:
            logger.error("Stream error: %s", exc)
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/cache")
def clear_cache():
    """Delete all cached pages and videos."""
    import shutil
    cache_dir = _current_config.get("cache_dir", "./scraper_cache")
    shutil.rmtree(cache_dir, ignore_errors=True)
    return {"cleared": cache_dir}


@app.delete("/index")
def clear_index():
    """Wipe the FAISS vector index (keeps scraper cache)."""
    _get_rag().db.clear()
    return {"cleared": "faiss_index"}