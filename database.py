"""
database.py
-----------
FAISSDatabase — vector store backed by FAISS with Google Gemini embeddings.

Embedding backend
-----------------
Uses the Gemini Embedding API (gemini-embedding-001) via google-genai.
No local model is loaded — all embedding is done via API call.

Key design points
-----------------
* Asymmetric task types — Gemini is instruction-tuned per task type:
    embed()       uses  RETRIEVAL_DOCUMENT  (for indexing chunks)
    embed_query() uses  RETRIEVAL_QUERY     (for search queries)
  Using the correct task type per side is how the model is designed to work
  and gives meaningfully better retrieval scores than using a single type.

* Output dimensionality — default 3072 (full quality, already L2-normalised
  by the API).  Can be reduced to 1536 or 768 via output_dimensionality param
  to save storage; the class re-normalises in those cases too.

* Batching — texts are sent in batches of `batch_size` (default 100) with
  detailed progress logs so you can see exactly what is happening during ingest.

Both embed() and embed_query() return L2-normalised float32 numpy arrays so
inner-product search == cosine similarity.
"""
from __future__ import annotations

import logging
import pickle
import re
import time
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from google import genai
from google.genai import types

from chunker import DocumentChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation.  Safe: zero rows stay zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


# ---------------------------------------------------------------------------
# FAISSDatabase
# ---------------------------------------------------------------------------

class FAISSDatabase:
    """
    Embedding + vector store for DocumentChunks.

    Parameters
    ----------
    model_name            Gemini embedding model. Default: "gemini-embedding-001".
    gemini_api_key        Google Gemini API key (required).
    index_path            Directory to load an existing index from on construction.
    batch_size            Texts per embed_content API call (default 100).
    output_dimensionality Vector size: 3072 (default), 1536, or 768.
    device                Ignored — kept for backward-compatible call sites.
    """

    _INDEX_FILE = "faiss.index"
    _META_FILE  = "metadata.pkl"

    def __init__(
        self,
        model_name:            str           = "gemini-embedding-001",
        gemini_api_key:        str           = "",
        index_path:            Optional[str] = None,
        batch_size:            int           = 100,
        output_dimensionality: int           = 3072,
        device:                str           = "cpu",   # unused, kept for compat
    ) -> None:
        self.model_name            = model_name
        self.batch_size            = batch_size
        self.output_dimensionality = output_dimensionality
        self._dim                  = output_dimensionality

        if not gemini_api_key:
            raise ValueError(
                "FAISSDatabase requires a Gemini API key.  "
                "Pass gemini_api_key= or set GEMINI_API_KEY in your environment."
            )

        self._client = genai.Client(api_key=gemini_api_key)

        logger.info(
            "FAISSDatabase ready: model=%s  dim=%d  batch_size=%d",
            model_name, self._dim, self.batch_size,
        )

        # FAISS: IndexFlatIP inside IndexIDMap (inner-product = cosine for L2-normed vecs)
        self._index: faiss.IndexIDMap = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))
        self._meta:  dict[int, DocumentChunk] = {}
        self._next_id: int = 0

        if index_path and Path(index_path).exists():
            self.load(index_path)

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    def _call_gemini_embed(
        self,
        texts:     list[str],
        task_type: str,
        label:     str = "embedding",
    ) -> np.ndarray:
        """
        Internal — call the Gemini Embedding API for a list of texts.

        Sends texts in batches of self.batch_size and logs each batch so
        you can track progress during a large ingest.

        Returns float32 ndarray (N, dim), L2-normalised.
        """
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")

        total     = len(texts)
        n_batches = (total + self.batch_size - 1) // self.batch_size
        all_vecs: list[np.ndarray] = []

        logger.info(
            "[embed:%s] starting — %d text(s)  %d batch(es)  task=%s",
            label, total, n_batches, task_type,
        )

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end   = min(start + self.batch_size, total)
            batch = texts[start:end]

            logger.info(
                "[embed:%s] batch %d/%d  items %d–%d ...",
                label, batch_idx + 1, n_batches, start + 1, end,
            )
            t0 = time.perf_counter()

            result = self._client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.output_dimensionality,
                ),
            )

            elapsed = time.perf_counter() - t0
            vecs    = np.array([e.values for e in result.embeddings], dtype="float32")
            all_vecs.append(vecs)

            logger.info(
                "[embed:%s] batch %d/%d done  shape=%s  %.2fs",
                label, batch_idx + 1, n_batches, vecs.shape, elapsed,
            )

        combined = np.vstack(all_vecs)          # (total, dim)
        combined = _l2_normalise(combined)      # safety-normalise (always correct)

        logger.info(
            "[embed:%s] complete — %d vector(s)  dim=%d",
            label, len(combined), self._dim,
        )
        return combined

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed document texts using task_type=RETRIEVAL_DOCUMENT.

        Call this when indexing chunks.  For search queries use embed_query().

        Returns float32 ndarray (N, dim), L2-normalised.
        """
        return self._call_gemini_embed(texts, task_type="RETRIEVAL_DOCUMENT", label="doc")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query using task_type=RETRIEVAL_QUERY.

        Gemini uses a different internal representation for queries vs. documents;
        always use this method at query time, not embed().

        Returns float32 ndarray (1, dim), L2-normalised.
        """
        logger.info("[embed:query] %r", query[:100])
        return self._call_gemini_embed([query], task_type="RETRIEVAL_QUERY", label="query")

    def embed_one(self, text: str) -> list[float]:
        """
        Embed a single document string → Python list.
        Used by DocumentChunker for chunk deduplication.
        """
        return self.embed([text])[0].tolist()

    # -------------------------------------------------------------------------
    # Write operations
    # -------------------------------------------------------------------------

    def add(self, chunks: list[DocumentChunk]) -> None:
        """Embed and add chunks; skip any already present (by chunk_id)."""
        existing = {c.chunk_id for c in self._meta.values()}
        new      = [c for c in chunks if c.chunk_id not in existing]
        if not new:
            logger.info("add() — nothing to do (%d chunk(s) already indexed).", len(chunks))
            return

        logger.info(
            "add() — embedding %d new chunk(s)  (skipping %d already present)",
            len(new), len(chunks) - len(new),
        )

        vecs    = self.embed([c.text for c in new])
        int_ids = np.arange(self._next_id, self._next_id + len(new), dtype=np.int64)
        self._index.add_with_ids(vecs, int_ids)

        for int_id, chunk in zip(int_ids, new):
            self._meta[int(int_id)] = chunk

        self._next_id += len(new)
        logger.info("add() done — index now contains %d chunk(s).", len(self._meta))

    def delete_by_chunk_id(self, chunk_ids: list[str]) -> int:
        to_del = {iid for iid, c in self._meta.items() if c.chunk_id in chunk_ids}
        return self._remove_int_ids(list(to_del))

    def delete_by_doc_index(self, doc_index: int) -> int:
        to_del = [iid for iid, c in self._meta.items() if c.doc_index == doc_index]
        return self._remove_int_ids(to_del)

    def delete_by_doc_title(self, title: str) -> int:
        to_del = [iid for iid, c in self._meta.items() if c.doc_title == title]
        return self._remove_int_ids(to_del)

    def delete_by_section(self, section: str) -> int:
        to_del = [iid for iid, c in self._meta.items() if c.section == section]
        return self._remove_int_ids(to_del)

    def clear(self) -> None:
        self._index.reset()
        self._meta.clear()
        self._next_id = 0
        logger.info("Index cleared.")

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search(
        self,
        query:            str,
        top_k:            int           = 5,
        section_filter:   Optional[str] = None,
        doc_index_filter: Optional[int] = None,
    ) -> list[dict]:
        if not self._meta:
            logger.info("[search] index is empty — returning []")
            return []

        logger.info(
            "[search] query=%r  top_k=%d  section_filter=%s",
            query[:80], top_k, section_filter,
        )
        q_vec   = self.embed_query(query)
        fetch_k = min(top_k * 10, len(self._meta))
        scores, int_ids = self._index.search(q_vec, fetch_k)

        results = []
        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id == -1:
                continue
            chunk = self._meta.get(int(int_id))
            if chunk is None:
                continue
            if section_filter and chunk.section != section_filter:
                continue
            if doc_index_filter is not None and chunk.doc_index != doc_index_filter:
                continue
            results.append({
                "score":       float(score),
                "chunk_id":    chunk.chunk_id,
                "doc_index":   chunk.doc_index,
                "doc_title":   chunk.doc_title,
                "section":     chunk.section,
                "doc_type":    chunk.doc_type,
                "doc_url":     chunk.doc_url,
                "chunk_index": chunk.chunk_index,
                "text":        chunk.text,
                "raw_content": chunk.raw_content,
            })
            if len(results) >= top_k:
                break

        logger.info("[search] returning %d result(s)  top score=%.4f",
                    len(results), results[0]["score"] if results else 0.0)
        return results

    def get_all_chunks_for_doc(self, doc_index: int) -> list[DocumentChunk]:
        return sorted(
            [c for c in self._meta.values() if c.doc_index == doc_index],
            key=lambda c: c.chunk_index,
        )

    def stats(self) -> dict:
        sections: dict[str, int] = {}
        titles:   set[str]       = set()
        for c in self._meta.values():
            sections[c.section] = sections.get(c.section, 0) + 1
            titles.add(c.doc_title)
        return {
            "total_chunks":    len(self._meta),
            "total_documents": len(titles),
            "sections":        sections,
            "embedding_dim":   self._dim,
            "model":           self.model_name,
        }

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / self._INDEX_FILE))
        with open(path / self._META_FILE, "wb") as f:
            pickle.dump({"meta": self._meta, "next_id": self._next_id}, f)
        logger.info("Index saved → %s  (%d chunks)", directory, len(self._meta))

    def load(self, directory: str) -> None:
        path       = Path(directory)
        index_file = path / self._INDEX_FILE
        meta_file  = path / self._META_FILE
        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"No saved index found in {directory!r}")
        self._index = faiss.read_index(str(index_file))
        with open(meta_file, "rb") as f:
            state = pickle.load(f)
        self._meta    = state["meta"]
        self._next_id = state["next_id"]
        logger.info("Index loaded ← %s  (%d chunks)", directory, len(self._meta))

    # -------------------------------------------------------------------------
    # Private / cleanup helpers
    # -------------------------------------------------------------------------

    def _remove_int_ids(self, int_ids: list[int]) -> int:
        if not int_ids:
            return 0
        arr = np.array(int_ids, dtype=np.int64)
        try:
            self._index.remove_ids(arr)
        except Exception as exc:
            logger.warning("FAISS remove_ids raised: %s — metadata still removed", exc)
        for iid in int_ids:
            self._meta.pop(iid, None)
        logger.info("Removed %d chunk(s) from index.", len(int_ids))
        return len(int_ids)

    def remove_docs_with_few_chunks(self, min_chunks: int = 20) -> int:
        """Remove all chunks belonging to documents with fewer than min_chunks chunks."""
        if not self._meta or min_chunks <= 0:
            return 0
        doc_buckets: dict[int, List[int]] = {}
        for int_id, chunk in self._meta.items():
            doc_buckets.setdefault(chunk.doc_index, []).append(int_id)
        to_remove = sorted({
            iid
            for ids in doc_buckets.values() if len(ids) < min_chunks
            for iid in ids
        })
        if not to_remove:
            logger.info("remove_docs_with_few_chunks: no documents with < %d chunks.", min_chunks)
            return 0
        removed = self._remove_int_ids(to_remove)
        logger.info("remove_docs_with_few_chunks: removed %d chunks.", removed)
        return removed

    def remove_short_chunks(
        self,
        min_tokens:     int            = 20,
        min_chars:      int            = 30,
        extra_patterns: Optional[List] = None,
    ) -> int:
        """Remove chunks that are too short or match junk regex patterns."""
        if not self._meta:
            return 0
        patterns: list = []
        if extra_patterns:
            for p in extra_patterns:
                try:
                    patterns.append(re.compile(p))
                except Exception:
                    pass

        to_remove: List[int] = []
        for int_id, chunk in list(self._meta.items()):
            parts: list[str] = []
            if getattr(chunk, "text", None):
                parts.append(chunk.text)
            if getattr(chunk, "raw_content", None):
                parts.append(chunk.raw_content)
            content = "\n".join(parts).strip()

            if not content:
                to_remove.append(int_id); continue
            if min_chars and len(content) < min_chars:
                to_remove.append(int_id); continue
            if min_tokens and len(content.split()) < min_tokens:
                to_remove.append(int_id); continue
            if any(p.search(content) for p in patterns):
                to_remove.append(int_id)

        to_remove = sorted(set(to_remove))
        if not to_remove:
            logger.info("remove_short_chunks: nothing to remove.")
            return 0
        removed = self._remove_int_ids(to_remove)
        logger.info(
            "remove_short_chunks: removed %d chunk(s)  (min_tokens=%d, min_chars=%d).",
            removed, min_tokens, min_chars,
        )
        return removed