"""
rag_query.py
------------
GeminiRAG — end-to-end RAG query class.

Pipeline
--------
1. Embed the user question with the same HuggingFace model used for indexing.
2. Retrieve top-k relevant chunks from FAISSDatabase.
3. Assemble a structured context prompt (metadata header + chunk content).
4. Call Gemini to generate a grounded answer with citations.
5. Return a structured GeminiAnswer with the answer text and source list.

Two query modes
---------------
answer(question)      → single-shot answer
stream_answer(question) → yields text tokens as they arrive (Gemini streaming)

Filtering
---------
Both methods accept optional section_filter and doc_type_filter so you can
restrict retrieval to e.g. only "video_summary" chunks or only "education".
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import Generator, Iterator, Optional

from google import genai
from google.genai import types

from database import FAISSDatabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer data model
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """One source document referenced in the answer."""
    doc_index: int
    doc_title: str
    section: str
    doc_type: str
    doc_url: str
    score: float
    chunk_index: int


@dataclass
class GeminiAnswer:
    """Structured result returned by GeminiRAG.answer()."""
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    total_tokens_used: int = 0


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a helpful assistant answering questions about a person's portfolio.
    You are given a set of retrieved context chunks from a vector database.
    Each chunk starts with a metadata header showing its source document index,
    title, section, type, and URL.

    Rules:
    - Answer ONLY from the provided context chunks.
    - If the answer is not present in the context, say so clearly.
    - When you reference information, cite the source using its [index] number,
      e.g. "According to [3] ..." or "As mentioned in [7] ...".
    - Be concise but complete.
    - Do not fabricate any information not present in the context.
""")


def _build_context_block(results: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    Each chunk already contains its metadata header (from the chunker),
    so we just wrap them cleanly.
    """
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(f"### Context chunk {i} (relevance score: {r['score']:.4f})")
        lines.append(r["text"])   # includes metadata header + raw content
        lines.append("")
    return "\n".join(lines)


def _build_user_message(question: str, context: str) -> str:
    return (
        f"## Retrieved Context\n\n"
        f"{context}\n"
        f"---\n\n"
        f"## Question\n\n"
        f"{question}"
    )


def _extract_sources(results: list[dict]) -> list[Source]:
    seen: set[str] = set()
    sources: list[Source] = []
    for r in results:
        key = f"{r['doc_index']}-{r['chunk_index']}"
        if key not in seen:
            seen.add(key)
            sources.append(Source(
                doc_index=r["doc_index"],
                doc_title=r["doc_title"],
                section=r["section"],
                doc_type=r["doc_type"],
                doc_url=r["doc_url"],
                score=r["score"],
                chunk_index=r["chunk_index"],
            ))
    return sources


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GeminiRAG:
    """
    Retrieval-augmented generation using FAISSDatabase + Google Gemini.

    Parameters
    ----------
    db              : FAISSDatabase — the populated vector store.
    gemini_api_key  : Google Gemini API key.
    gemini_model    : Gemini model name.  Default: "gemini-2.0-flash".
    top_k           : Number of chunks to retrieve per query.
    system_prompt   : Override the default system prompt if needed.
    """

    def __init__(
        self,
        db: FAISSDatabase,
        gemini_api_key: str,
        gemini_model: str = "gemini-2.0-flash",
        top_k: int = 6,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.db            = db
        self.gemini_model  = gemini_model
        self.top_k         = top_k
        self._system       = system_prompt or _SYSTEM_PROMPT
        self._client       = genai.Client(api_key=gemini_api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        section_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
    ) -> GeminiAnswer:
        """
        Retrieve relevant chunks and generate a grounded answer.

        Parameters
        ----------
        question        : user's natural language question
        top_k           : override default retrieval count
        section_filter  : restrict to a section, e.g. "education"
        doc_type_filter : restrict to "text", "index", or "video_summary"

        Returns
        -------
        GeminiAnswer with .answer (str), .sources (list[Source])
        """
        k = top_k or self.top_k
        results = self._retrieve(question, k, section_filter, doc_type_filter)

        if not results:
            return GeminiAnswer(
                question=question,
                answer="I could not find relevant information in the portfolio to answer this question.",
                sources=[],
            )

        context    = _build_context_block(results)
        user_msg   = _build_user_message(question, context)
        sources    = _extract_sources(results)

        logger.info("Calling Gemini [%s] with %d chunks", self.gemini_model, len(results))
        try:
            response = self._client.models.generate_content(
                model=self.gemini_model,
                contents=[
                    types.Content(role="user", parts=[types.Part(text=user_msg)])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=self._system,
                    temperature=0.2,
                    max_output_tokens=2048,
                ),
            )
            answer_text = response.text or ""
            tokens_used = (
                response.usage_metadata.total_token_count
                if response.usage_metadata else 0
            )
        except Exception as exc:
            logger.error("Gemini call failed: %s", exc)
            return GeminiAnswer(
                question=question,
                answer=f"Error calling Gemini: {exc}",
                sources=sources,
            )

        return GeminiAnswer(
            question=question,
            answer=answer_text,
            sources=sources,
            total_tokens_used=tokens_used,
        )

    def stream_answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        section_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
    ) -> Generator[str, None, GeminiAnswer]:
        """
        Streaming version of answer().

        Yields text tokens as they arrive from Gemini.
        Returns a GeminiAnswer (via StopIteration.value) on completion.

        Usage
        -----
            gen = rag.stream_answer("What is her PhD research about?")
            try:
                while True:
                    token = next(gen)
                    print(token, end="", flush=True)
            except StopIteration as e:
                final = e.value   # GeminiAnswer
            print()
            print("Sources:", [s.doc_title for s in final.sources])
        """
        k = top_k or self.top_k
        results = self._retrieve(question, k, section_filter, doc_type_filter)

        if not results:
            yield "I could not find relevant information in the portfolio to answer this question."
            return GeminiAnswer(question=question, answer="No relevant chunks found.", sources=[])

        context  = _build_context_block(results)
        user_msg = _build_user_message(question, context)
        sources  = _extract_sources(results)

        full_text: list[str] = []
        tokens_used = 0
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self.gemini_model,
                contents=[
                    types.Content(role="user", parts=[types.Part(text=user_msg)])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=self._system,
                    temperature=0.2,
                    max_output_tokens=2048,
                ),
            ):
                token = chunk.text or ""
                if token:
                    full_text.append(token)
                    yield token
                if chunk.usage_metadata:
                    tokens_used = chunk.usage_metadata.total_token_count or 0
        except Exception as exc:
            err = f"\n[Gemini streaming error: {exc}]"
            full_text.append(err)
            yield err

        return GeminiAnswer(
            question=question,
            answer="".join(full_text),
            sources=sources,
            total_tokens_used=tokens_used,
        )

    def format_answer(self, ga: GeminiAnswer, show_sources: bool = True) -> str:
        """
        Pretty-print a GeminiAnswer to a string.
        Useful for CLI / notebook display.
        """
        lines = [
            f"Q: {ga.question}",
            "",
            ga.answer,
        ]
        if show_sources and ga.sources:
            lines += ["", "─" * 60, "Sources:"]
            for s in ga.sources:
                lines.append(
                    f"  [{s.doc_index}] {s.doc_title}  |  {s.section}  |  {s.doc_type}"
                    f"  |  score {s.score:.4f}"
                )
                if s.doc_url:
                    lines.append(f"       {s.doc_url}")
        if ga.total_tokens_used:
            lines += ["", f"Tokens used: {ga.total_tokens_used}"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        question: str,
        k: int,
        section_filter: Optional[str],
        doc_type_filter: Optional[str],
    ) -> list[dict]:
        # Over-fetch to allow post-filtering
        raw = self.db.search(question, top_k=k * 4, section_filter=section_filter)
        if doc_type_filter:
            raw = [r for r in raw if r.get("doc_type") == doc_type_filter]
        results = raw[:k]
        logger.debug("Retrieved %d chunks for: %r", len(results), question[:60])
        return results