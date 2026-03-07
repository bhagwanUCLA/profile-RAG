"""
server.py
---------
FastAPI backend for the Portfolio RAG pipeline.

New in this version
-------------------
- Conversational chat: every /query/stream call accepts a `session_id`.
  Passing the same session_id across requests makes the LLM remember
  previous turns (Gemini via chats.create, Claude via message history).
- Claude model support: any model name starting with "claude" is routed
  through the Anthropic SDK.
- GET /sessions              — list active session ids
- DELETE /sessions/{id}      — clear a specific session
- DELETE /sessions           — clear all sessions

Endpoints
---------
POST /ingest          — scrape & index a portfolio URL
POST /config          — update pipeline config
GET  /config          — read current config
GET  /query/stream    — SSE streaming query (pass session_id for chat)
POST /query           — single-shot query
GET  /stats           — cache + index statistics
DELETE /cache         — wipe scraper cache
DELETE /index         — wipe FAISS index
GET  /sessions        — list sessions
DELETE /sessions/{id} — clear one session
DELETE /sessions      — clear all sessions
GET  /health          — liveness probe
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

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "gemini_api_key":   os.environ.get("GEMINI_API_KEY", ""),
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    "hf_model_name":    "gemini-embedding-001",
    "chunk_size":       500,
    "chunk_overlap":    50,
    "dedup_threshold":  None,
    "min_tokens":       20,
    "index_dir":        "./rag_index",
    "cache_dir":        "./scraper_cache",
    "follow_external":  True,
    "device":           "cpu",
    "gemini_model":     "gemini-2.0-flash",
    "top_k":            6,
}

_current_config: dict = dict(_DEFAULT_CONFIG)
_rag:            Optional[RAGOrchestrator] = None


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


def _get_gemini_rag(
    system_prompt: Optional[str] = None,
    model_override: Optional[str] = None,
) -> GeminiRAG:
    return GeminiRAG(
        db=_get_rag().db,
        gemini_api_key=_current_config["gemini_api_key"],
        anthropic_api_key=_current_config["anthropic_api_key"],
        gemini_model=model_override or _current_config["gemini_model"],
        top_k=_current_config["top_k"],
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    gemini_api_key:   Optional[str]   = None
    anthropic_api_key: Optional[str]  = None
    hf_model_name:    Optional[str]   = None
    chunk_size:       Optional[int]   = None
    chunk_overlap:    Optional[int]   = None
    dedup_threshold:  Optional[float] = None
    min_tokens:       Optional[int]   = None
    index_dir:        Optional[str]   = None
    cache_dir:        Optional[str]   = None
    follow_external:  Optional[bool]  = None
    device:           Optional[str]   = None
    gemini_model:     Optional[str]   = None
    top_k:            Optional[int]   = None


class IngestRequest(BaseModel):
    url:     str
    rebuild: bool = False


class QueryRequest(BaseModel):
    question:        str
    top_k:           int            = Field(default=6, ge=1, le=20)
    section_filter:  Optional[str]  = None
    doc_type_filter: Optional[str]  = None
    system_prompt:   Optional[str]  = None
    gemini_model:    Optional[str]  = None
    session_id:      Optional[str]  = None


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
        safe["gemini_api_key"] = "***set***"
    if safe.get("anthropic_api_key"):
        safe["anthropic_api_key"] = "***set***"
    return safe


@app.post("/config")
def update_config(body: ConfigUpdate):
    global _rag, _current_config
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    _current_config.update(updates)
    _rag = None
    return {"updated": list(updates.keys()), "config": get_config()}


@app.post("/ingest")
def ingest(body: IngestRequest):
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


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@app.get("/sessions")
def list_sessions():
    g = _get_gemini_rag()
    return {"sessions": g.list_sessions()}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    g = _get_gemini_rag()
    g.clear_session(session_id)
    return {"cleared": session_id}


@app.delete("/sessions")
def clear_all_sessions():
    g = _get_gemini_rag()
    for sid in g.list_sessions():
        g.clear_session(sid)
    return {"cleared": "all"}


# ---------------------------------------------------------------------------
# Query — streaming SSE (main endpoint used by frontend)
# ---------------------------------------------------------------------------

@app.get("/query/stream")
async def query_stream(
    question:        str,
    top_k:           int  = 6,
    section_filter:  Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    system_prompt:   Optional[str] = None,
    gemini_model:    Optional[str] = None,
    session_id:      Optional[str] = None,
):
    """
    SSE streaming endpoint.
    Emits:
      event: chunk   — each retrieved chunk (before generation)
      event: token   — each LLM text token
      event: done    — final metadata (sources, tokens_used)
      event: error   — on failure
    """
    rag = _get_rag()
    g   = _get_gemini_rag(
        system_prompt=system_prompt,
        model_override=gemini_model,
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            # 1. Send retrieved chunks
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

            # 2. Stream tokens
            gen          = g.stream_answer(
                question=question,
                top_k=top_k,
                section_filter=section_filter,
                doc_type_filter=doc_type_filter,
                session_id=session_id,
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
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Query — single-shot (non-streaming)
# ---------------------------------------------------------------------------

@app.post("/query")
def query(body: QueryRequest):
    rag = _get_rag()
    g   = _get_gemini_rag(
        system_prompt=body.system_prompt,
        model_override=body.gemini_model,
    )

    chunks = rag.query(
        question=body.question,
        top_k=body.top_k,
        section_filter=body.section_filter,
        doc_type_filter=body.doc_type_filter,
    )

    result = g.answer(
        question=body.question,
        top_k=body.top_k,
        section_filter=body.section_filter,
        doc_type_filter=body.doc_type_filter,
        session_id=body.session_id,
    )

    return {
        "question":    body.question,
        "answer":      result.answer,
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


# ---------------------------------------------------------------------------
# Danger zone
# ---------------------------------------------------------------------------

@app.delete("/cache")
def clear_cache():
    import shutil
    cache_dir = _current_config.get("cache_dir", "./scraper_cache")
    shutil.rmtree(cache_dir, ignore_errors=True)
    return {"cleared": cache_dir}


@app.delete("/index")
def clear_index():
    _get_rag().db.clear()
    return {"cleared": "faiss_index"}