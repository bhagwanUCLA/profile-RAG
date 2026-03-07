"""
rag_query.py
------------
GeminiRAG — end-to-end RAG query class with conversational chat support.

Supports:
  - Google Gemini models  (gemini-2.0-flash, gemini-2.5-flash-*, etc.)
  - Anthropic Claude models (claude-sonnet-4-*, claude-opus-4-*, etc.)

Conversation history is maintained per session_id so the LLM
remembers previous turns within the same browser session.

Two query modes
---------------
answer(question)          → single-shot answer
stream_answer(question)   → yields text tokens as they arrive (streaming)

Both accept an optional session_id (str).  Pass the same id across calls
to maintain a conversation.  Omit / pass None for a stateless one-shot query.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import Generator, Optional

from google import genai
from google.genai import types as gtypes

from database import FAISSDatabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer data model
# ---------------------------------------------------------------------------

@dataclass
class Source:
    doc_index:   int
    doc_title:   str
    section:     str
    doc_type:    str
    doc_url:     str
    score:       float
    chunk_index: int


@dataclass
class GeminiAnswer:
    question:          str
    answer:            str
    sources:           list[Source] = field(default_factory=list)
    total_tokens_used: int = 0


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a helpful assistant answering questions about a person's portfolio.
    You are given a set of retrieved context chunks from a vector database.
    Each chunk starts with a metadata header showing its source document index,
    title, section, type, and URL.

    Rules:
    - Answer ONLY from the provided context chunks.
    - If the answer is not present in the context, say so clearly.
    - When you reference information, cite the source using its [index] number.
    - Be concise but complete.
    - Remember the full conversation history and refer back to it when relevant.
    - Do not fabricate any information not present in the context.
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context_block(results: list[dict]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(f"### Context chunk {i} (relevance score: {r['score']:.4f})")
        lines.append(r["text"])
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


def _is_claude_model(model: str) -> bool:
    return model.startswith("claude")


# ---------------------------------------------------------------------------
# Gemini chat session store  (session_id → google.genai Chat object)
# ---------------------------------------------------------------------------

_gemini_sessions: dict[str, object] = {}   # session_id → genai Chat

# ---------------------------------------------------------------------------
# Claude conversation history store  (session_id → list of message dicts)
# ---------------------------------------------------------------------------

_claude_histories: dict[str, list[dict]] = {}   # session_id → [{role, content}]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GeminiRAG:
    """
    Retrieval-augmented generation using FAISSDatabase + Google Gemini or Anthropic Claude.

    Parameters
    ----------
    db              : FAISSDatabase — populated vector store.
    gemini_api_key  : Google Gemini API key.
    anthropic_api_key : Anthropic API key (required for Claude models).
    gemini_model    : Model name.  Gemini or Claude.
    top_k           : Number of chunks to retrieve per query.
    system_prompt   : Override the default system prompt if needed.
    """

    def __init__(
        self,
        db: FAISSDatabase,
        gemini_api_key: str,
        anthropic_api_key: str = "",
        gemini_model: str = "gemini-2.0-flash",
        top_k: int = 6,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.db               = db
        self.gemini_model     = gemini_model
        self.top_k            = top_k
        self._system          = system_prompt or _SYSTEM_PROMPT
        self._gemini_client   = genai.Client(api_key=gemini_api_key)
        self._anthropic_key   = anthropic_api_key

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> None:
        """Wipe conversation history for a session."""
        _gemini_sessions.pop(session_id, None)
        _claude_histories.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(set(list(_gemini_sessions.keys()) + list(_claude_histories.keys())))

    # ------------------------------------------------------------------
    # Gemini chat helpers
    # ------------------------------------------------------------------

    def _get_gemini_chat(self, session_id: Optional[str]):
        """Return existing Gemini chat for the session, or create a new one."""
        if not session_id:
            # Stateless — new chat every time
            return self._gemini_client.chats.create(
                model=self.gemini_model,
                config=gtypes.GenerateContentConfig(
                    system_instruction=self._system,
                    temperature=0.2,
                    max_output_tokens=2048,
                ),
            )
        if session_id not in _gemini_sessions:
            _gemini_sessions[session_id] = self._gemini_client.chats.create(
                model=self.gemini_model,
                config=gtypes.GenerateContentConfig(
                    system_instruction=self._system,
                    temperature=0.2,
                    max_output_tokens=2048,
                ),
            )
        return _gemini_sessions[session_id]

    # ------------------------------------------------------------------
    # Claude history helpers
    # ------------------------------------------------------------------

    def _get_claude_history(self, session_id: Optional[str]) -> list[dict]:
        if not session_id:
            return []
        if session_id not in _claude_histories:
            _claude_histories[session_id] = []
        return _claude_histories[session_id]

    def _append_claude_turn(
        self,
        session_id: Optional[str],
        user_msg: str,
        assistant_msg: str,
    ) -> None:
        if not session_id:
            return
        history = self._get_claude_history(session_id)
        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})

    # ------------------------------------------------------------------
    # Public API — answer()
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        section_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> GeminiAnswer:
        k = top_k or self.top_k
        results = self._retrieve(question, k, section_filter, doc_type_filter)

        if not results:
            return GeminiAnswer(
                question=question,
                answer="I could not find relevant information in the portfolio to answer this question.",
                sources=[],
            )

        context  = _build_context_block(results)
        user_msg = _build_user_message(question, context)
        sources  = _extract_sources(results)

        if _is_claude_model(self.gemini_model):
            return self._answer_claude(question, user_msg, sources, session_id)
        else:
            return self._answer_gemini(question, user_msg, sources, session_id)

    def _answer_gemini(self, question, user_msg, sources, session_id):
        logger.info("Calling Gemini [%s] session=%s", self.gemini_model, session_id)
        try:
            chat     = self._get_gemini_chat(session_id)
            response = chat.send_message(user_msg)
            answer_text = response.text or ""
            tokens_used = (
                response.usage_metadata.total_token_count
                if response.usage_metadata else 0
            )
        except Exception as exc:
            logger.error("Gemini call failed: %s", exc)
            return GeminiAnswer(question=question, answer=f"Error calling Gemini: {exc}", sources=sources)

        return GeminiAnswer(
            question=question,
            answer=answer_text,
            sources=sources,
            total_tokens_used=tokens_used,
        )

    def _answer_claude(self, question, user_msg, sources, session_id):
        import anthropic
        logger.info("Calling Claude [%s] session=%s", self.gemini_model, session_id)
        client  = anthropic.Anthropic(api_key=self._anthropic_key)
        history = self._get_claude_history(session_id)
        messages = history + [{"role": "user", "content": user_msg}]

        try:
            response = client.messages.create(
                model=self.gemini_model,
                max_tokens=2048,
                system=self._system,
                messages=messages,
            )
            answer_text = response.content[0].text if response.content else ""
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
        except Exception as exc:
            logger.error("Claude call failed: %s", exc)
            return GeminiAnswer(question=question, answer=f"Error calling Claude: {exc}", sources=sources)

        self._append_claude_turn(session_id, user_msg, answer_text)
        return GeminiAnswer(
            question=question,
            answer=answer_text,
            sources=sources,
            total_tokens_used=tokens_used,
        )

    # ------------------------------------------------------------------
    # Public API — stream_answer()
    # ------------------------------------------------------------------

    def stream_answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        section_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Generator[str, None, GeminiAnswer]:
        k = top_k or self.top_k
        results = self._retrieve(question, k, section_filter, doc_type_filter)

        if not results:
            yield "I could not find relevant information in the portfolio to answer this question."
            return GeminiAnswer(question=question, answer="No relevant chunks found.", sources=[])

        context  = _build_context_block(results)
        user_msg = _build_user_message(question, context)
        sources  = _extract_sources(results)

        if _is_claude_model(self.gemini_model):
            return (yield from self._stream_claude(question, user_msg, sources, session_id))
        else:
            return (yield from self._stream_gemini(question, user_msg, sources, session_id))

    def _stream_gemini(self, question, user_msg, sources, session_id):
        chat = self._get_gemini_chat(session_id)
        full_text:   list[str] = []
        tokens_used: int       = 0
        try:
            for chunk in chat.send_message_stream(user_msg):
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

    def _stream_claude(self, question, user_msg, sources, session_id):
        import anthropic
        client  = anthropic.Anthropic(api_key=self._anthropic_key)
        history = self._get_claude_history(session_id)
        messages = history + [{"role": "user", "content": user_msg}]

        full_text:   list[str] = []
        tokens_used: int       = 0
        try:
            with client.messages.stream(
                model=self.gemini_model,
                max_tokens=2048,
                system=self._system,
                messages=messages,
            ) as stream:
                for token in stream.text_stream:
                    if token:
                        full_text.append(token)
                        yield token
                final = stream.get_final_message()
                tokens_used = final.usage.input_tokens + final.usage.output_tokens
        except Exception as exc:
            err = f"\n[Claude streaming error: {exc}]"
            full_text.append(err)
            yield err

        answer_text = "".join(full_text)
        self._append_claude_turn(session_id, user_msg, answer_text)

        return GeminiAnswer(
            question=question,
            answer=answer_text,
            sources=sources,
            total_tokens_used=tokens_used,
        )

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
        raw = self.db.search(question, top_k=k * 4, section_filter=section_filter)
        if doc_type_filter:
            raw = [r for r in raw if r.get("doc_type") == doc_type_filter]
        return raw[:k]