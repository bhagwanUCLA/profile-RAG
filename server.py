"""
server.py
---------
FastAPI backend for the Portfolio RAG pipeline.

Streaming architecture
----------------------
The Anthropic SDK is synchronous and blocking.  Running it directly inside
an `async def` would freeze the entire uvicorn event loop for the duration
of each Claude call, causing gunicorn WORKER TIMEOUT on longer queries.

Solution: _run_llm_in_thread() submits the sync stream_answer generator to a
ThreadPoolExecutor.  The generator puts ('token', text), ('done', answer),
or ('error', msg) items into a thread-safe queue.Queue.  The async
event_stream() coroutine polls that queue with short sleeps, keeping the
event loop free for other requests and heartbeat keepalives.

Gunicorn start command (Render):
  gunicorn -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:$PORT --timeout 120
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import queue as _sync_queue
from typing import AsyncGenerator, Optional
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

from orchestrator import RAGOrchestrator
from rag_query import RAG
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

# Thread pool for blocking Anthropic SDK calls
# (one thread per concurrent streaming request)
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "gemini_api_key":    os.environ.get("GEMINI_API_KEY", ""),
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    "hf_model_name":     "gemini-embedding-001",
    "chunk_size":        3500,
    "chunk_overlap":     50,
    "dedup_threshold":   None,
    "min_tokens":        20,
    "index_dir":         "./rag_index",
    "cache_dir":         "./scraper_cache",
    "follow_external":   True,
    "device":            "cpu",
    "model":      "claude-sonnet-4-6",
    "top_k":             10,
}

_current_config: dict          = dict(_DEFAULT_CONFIG)
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
) -> RAG:
    return RAG(
        db=_get_rag().db,
        gemini_api_key=_current_config["gemini_api_key"],
        anthropic_api_key=_current_config["anthropic_api_key"],
        model= _current_config["model"],
        top_k=_current_config["top_k"],
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    gemini_api_key:    Optional[str]   = None
    anthropic_api_key: Optional[str]   = None
    hf_model_name:     Optional[str]   = None
    chunk_size:        Optional[int]   = None
    chunk_overlap:     Optional[int]   = None
    dedup_threshold:   Optional[float] = None
    min_tokens:        Optional[int]   = None
    index_dir:         Optional[str]   = None
    cache_dir:         Optional[str]   = None
    follow_external:   Optional[bool]  = None
    device:            Optional[str]   = None
    model:      Optional[str]   = None
    top_k:             Optional[int]   = None


class IngestRequest(BaseModel):
    url:     str
    rebuild: bool = False


class QueryRequest(BaseModel):
    question:        str
    top_k:           int           = Field(default=6, ge=1, le=20)
    section_filter:  Optional[str] = None
    doc_type_filter: Optional[str] = None
    system_prompt:   Optional[str] = None
    model:    Optional[str] = None
    session_id:      Optional[str] = None


# ---------------------------------------------------------------------------
# Additional request models
# ---------------------------------------------------------------------------

class FolderIngestRequest(BaseModel):
    folder_path: str
    section:     str  = "general"
    recursive:   bool = True


class RawDocumentItem(BaseModel):
    title:    str           = "Untitled"
    content:  str
    section:  str           = "general"
    url:      str           = ""
    doc_type: str           = "text"


class RawDocumentsRequest(BaseModel):
    documents: list[RawDocumentItem]


class VideosIngestRequest(BaseModel):
    urls:    list[str]
    section: str = "video"


# ---------------------------------------------------------------------------
# Thread-pool helper
# ---------------------------------------------------------------------------

def _run_llm_in_thread(
    g: RAG,
    question: str,
    top_k: int,
    section_filter: Optional[str],
    doc_type_filter: Optional[str],
    session_id: Optional[str],
    token_queue: "_sync_queue.Queue[tuple[str, object]]",
) -> None:
    """
    Runs stream_answer() synchronously in a worker thread.
    Puts items into token_queue:
      ('chunk', dict)         — one retrieved chunk (from a tool call)
      ('token', str)          — one text token
      ('done',  GeminiAnswer) — generator exhausted normally
      ('error', str)          — exception message

    The on_chunks callback is invoked by RAG every time Claude fires
    the search_portfolio tool.  It serialises each result dict and puts a
    ('chunk', payload_dict) item into the shared queue so the SSE loop can
    forward them to the client in real time — interleaved with tokens.
    """
    chunk_rank = [0]   # mutable counter shared across all tool calls

    def on_chunks(results: list[dict]) -> None:
        for r in results:
            chunk_rank[0] += 1
            payload = {
                "rank":        chunk_rank[0],
                "score":       r["score"],
                "doc_index":   r["doc_index"],
                "doc_title":   r["doc_title"],
                "section":     r["section"],
                "doc_type":    r["doc_type"],
                "doc_url":     r["doc_url"],
                "chunk_index": r["chunk_index"],
                "raw_content": r["raw_content"],
            }
            token_queue.put(("chunk", payload))

    try:
        gen = g.stream_answer(
            question=question,
            top_k=top_k,
            section_filter=section_filter,
            doc_type_filter=doc_type_filter,
            session_id=session_id,
            on_chunks=on_chunks,
        )
        while True:
            try:
                token = next(gen)
                token_queue.put(("token", token))
            except StopIteration as e:
                token_queue.put(("done", e.value))
                return
    except Exception as exc:
        logger.error("LLM thread error: %s", exc)
        token_queue.put(("error", str(exc)))


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

@app.get("/", response_class=HTMLResponse)
def ingest_ui():
    """Serve the document ingestion UI."""
    html_path = Path(__file__).parent / "ingest_ui.html"
    if not html_path.exists():
        return HTMLResponse("<h1>ingest_ui.html not found next to server.py</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

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
    return {"sessions": _get_gemini_rag().list_sessions()}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    _get_gemini_rag().clear_session(session_id)
    return {"cleared": session_id}


@app.delete("/sessions")
def clear_all_sessions():
    g = _get_gemini_rag()
    for sid in g.list_sessions():
        g.clear_session(sid)
    return {"cleared": "all"}


# ---------------------------------------------------------------------------
# Additional ingest endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest/folder")
def ingest_folder(body: FolderIngestRequest):
    """
    Ingest all supported files from a server-side directory.

    Supported: .txt .md .html .pdf .docx .doc .odt .pptx .ppt
               .xlsx .xls .xlsm .xlsb .csv .ods

    Binary files are extracted via Gemini; text files are read directly.
    The folder_path must be accessible on the server's filesystem.
    """
    rag = _get_rag()
    try:
        chunks = rag.ingest_folder(
            folder_path=body.folder_path,
            section=body.section,
            recursive=body.recursive,
        )
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(exc))
    rag.save()
    return {
        "action":         "ingest_folder",
        "folder_path":    body.folder_path,
        "section":        body.section,
        "chunks_stored":  chunks,
        "stats":          rag.stats(),
    }


@app.post("/ingest/documents")
def ingest_documents(body: RawDocumentsRequest):
    """
    Inject pre-written text documents directly into the index.

    Each document goes through the normal chunker pipeline.
    Useful for adding custom bios, CVs, notes, or any text
    that doesn't have a URL to scrape.

    Body:
      documents: [{title, content, section, url, doc_type}, ...]
    """
    rag = _get_rag()
    docs = [d.model_dump() for d in body.documents]
    chunks = rag.ingest_raw_documents(docs)
    rag.save()
    return {
        "action":        "ingest_documents",
        "docs_received": len(docs),
        "chunks_stored": chunks,
        "stats":         rag.stats(),
    }


@app.post("/ingest/videos")
def ingest_videos(body: VideosIngestRequest):
    """
    Summarise a list of YouTube video URLs or playlist URLs via Gemini.

    Playlist URLs (youtube.com/playlist?list=…) are automatically expanded
    to individual videos.  Results are cached so replaying is free.
    """
    rag = _get_rag()
    chunks = rag.ingest_videos(body.urls, section=body.section)
    rag.save()
    return {
        "action":        "ingest_videos",
        "urls_received": len(body.urls),
        "chunks_stored": chunks,
        "stats":         rag.stats(),
    }


# ---------------------------------------------------------------------------
# Query — streaming SSE
# ---------------------------------------------------------------------------

@app.get("/query/stream")
async def query_stream(
    question:        str,
    top_k:           int  = 6,
    section_filter:  Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    system_prompt:   Optional[str] = None,
    session_id:      Optional[str] = None,
):
    """
    SSE streaming endpoint.
    Events: chunk | token | done | error | ping
    """
    rag = _get_rag()
    g   = _get_gemini_rag(system_prompt=system_prompt)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            # ── LLM streaming in a background thread ─────────────────
            # Chunks now arrive via the on_chunks callback inside the
            # thread, which puts ("chunk", payload) into the queue.
            # This means the UI shows exactly the chunks Claude retrieved,
            # not a separate pre-retrieval pass.
            token_queue: _sync_queue.Queue = _sync_queue.Queue()
            loop = asyncio.get_event_loop()

            future = loop.run_in_executor(
                _thread_pool,
                _run_llm_in_thread,
                g, question, top_k, section_filter, doc_type_filter,
                session_id, token_queue,
            )

            ping_counter = 0
            final_answer = None

            while True:
                try:
                    kind, value = token_queue.get_nowait()
                except _sync_queue.Empty:
                    # Keep connection alive with a ping every ~5 s
                    ping_counter += 1
                    if ping_counter % 100 == 0:
                        yield ": ping\n\n"
                    await asyncio.sleep(0.05)
                    continue

                if kind == "chunk":
                    yield f"event: chunk\ndata: {json.dumps(value)}\n\n"
                elif kind == "token":
                    yield f"event: token\ndata: {json.dumps(value)}\n\n"
                elif kind == "done":
                    final_answer = value
                    break
                elif kind == "error":
                    yield f"event: error\ndata: {json.dumps(value)}\n\n"
                    break

            # Wait for thread to finish cleanly
            await asyncio.wrap_future(future)

            # ── 3. Done event ─────────────────────────────────────────
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
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Query — single-shot POST
# ---------------------------------------------------------------------------

@app.post("/query")
def query(body: QueryRequest):
    rag = _get_rag()
    g   = _get_gemini_rag(
        system_prompt=body.system_prompt,
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
                "rank":        i + 1,
                "score":       r["score"],
                "doc_index":   r["doc_index"],
                "doc_title":   r["doc_title"],
                "section":     r["section"],
                "doc_type":    r["doc_type"],
                "doc_url":     r["doc_url"],
                "chunk_index": r["chunk_index"],
                "raw_content": r["raw_content"],
                "full_text":   r["text"],
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

@app.get("/cache/corrupt")
def audit_corrupt_cache():
    """
    List all cached pages whose stored HTML is binary/compressed junk
    (i.e. requests failed to decompress the response).

    Returns each entry with:
      url          - original requested URL
      final_url    - actual destination URL — open this in your browser
      content_type - Content-Type stored at scrape time
      cache_file   - path to the .json file on the server

    Workflow:
      1. GET  /cache/corrupt          → see which URLs failed
      2. Open each final_url manually, copy the content
      3. POST /ingest/documents        → inject that content as raw text
      4. DELETE /cache/corrupt         → remove the bad cache entries
      5. The next ingest will try those URLs fresh
    """
    corrupt = _get_rag().audit_corrupt_cache()
    return {
        "corrupt_count": len(corrupt),
        "entries": corrupt,
    }


@app.delete("/cache/corrupt")
def purge_corrupt_cache():
    """
    Delete all corrupt cache entries (binary/compressed pages stored by
    requests without decompressing).

    After this call, run POST /ingest with the portfolio URL to re-attempt
    those pages, or use POST /ingest/documents to inject content manually.
    """
    removed = _get_rag().purge_corrupt_cache()
    return {
        "deleted_count": len(removed),
        "deleted": removed,
    }


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