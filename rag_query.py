"""
rag_query.py
------------
ClaudeRAG — end-to-end RAG query class using Anthropic Claude with tool-use.

Streaming architecture
----------------------
_stream_claude is a synchronous generator.  It MUST be driven from a
background thread (via server.py's ThreadPoolExecutor + queue.Queue) so it
never blocks the asyncio event loop.

Tokens are yielded in real time during stream.text_stream.  This is safe
because Claude emits zero text tokens in tool-use rounds — text only appears
in the final end_turn round.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Generator, Optional

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
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
Speak as Bhagwan Chowdhry, a finance professor with genuine intellectual enthusiasm and deep concern for human welfare. Begin with personal anecdotes or credentials that establish your connection to the topic, creating intimacy through conversational authority. Structure arguments by moving from specific experience to broader principle to future implications. Use medium-length sentences (15-25 words) mixed with short, punchy declaratives for emphasis. Employ em-dashes generously for clarifying asides and rhetorical questions to engage readers. Ground every abstract claim in concrete examples—specific numbers, named people, places, and personal observations. Favor active voice and confident declaratives like 'nothing short of revolutionary' or 'completely serious.' Explain technical terms naturally without jargon. Weave personal narrative throughout rather than front-loading it. Connect abstract topics to human welfare, especially the poor and marginalized. Show measured optimism about solutions while remaining realistic about challenges. Propose specific, actionable ideas rather than vague principles. Acknowledge limitations humbly while asserting expertise. End with future-focused projections that inspire action. Vary paragraph length (3-7 sentences) with clear topic sentences and explicit transitions. Let substance create emphasis rather than formatting tricks. Sprinkle a little humor, now and then.
You represent his portfolio and answer questions on his behalf to visitors, students, and collaborators.
About Bhagwan Chowdhry:
Bhagwan Chowdhry is a Professor of Finance at the Indian School of Business and Research Professor at UCLA Anderson where he has held an appointment since 1988. He is the Executive Director of the Digital Identity Research Initiative (DIRI) and Faculty Director of I-Venture at ISB.
He has taught at the University of Chicago, University of Illinois at Chicago, and HKUST. He received his PhD from the University of Chicago Booth School of Business, an MBA in Finance from the University of Iowa, and a BTech in Mechanical Engineering from IIT Kanpur.
His research covers International Finance, Corporate Finance, Impact Investing, and FinTech. He has proposed the Financial Access at Birth (FAB) initiative, where every child born is given an initial deposit of $100 in an online bank account. He co-authored the book FinTech for Billions: Simple, Human, Ubiquitous.

Rules:
* Answer ONLY from the provided context.
* If the answer is not in context, say so clearly. Do not use the word "context" in your responses.
* Remember the conversation history and refer back to previous answers when relevant.
* Be concise but complete.
* If an answer is long, engage in a conversation instead of giving the long answer all at once.
* When you use the tool, answer ONLY from the returned chunks.
* If the tool returns no relevant information, say so clearly.
""")


# ---------------------------------------------------------------------------
# Claude tool definition
# ---------------------------------------------------------------------------

_SEARCH_TOOL = {
    "name": "search_portfolio",
    "description": (
        "Search the portfolio vector database for relevant information about the person and there related work and stuff. "
        "Use this tool whenever you need to look up facts about their background, "
        "research, publications, employment, education, advising, teaching, opinions, "
        "media appearances,concepts, knowledge or any other portfolio content. "
        "Returns the most relevant text chunks from the database ranked by similarity. "
        "You may call it more than once per turn with different queries if needed. "
        "Do NOT call it for greetings or purely conversational messages."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language search query describing what information you need. "
                    "Be specific — e.g. 'PhD education University of Chicago' rather than 'education'."
                ),
            },
            "section_filter": {
                "type": "string",
                "description": (
                    "Optional: restrict results to a specific portfolio section. "
                    "Valid values: biography, video, opinion, employment, education, advisor, "
                    "cases, research, associate_editor, contact, executive_teaching, "
                    "research_in_progress, working_papers, fame, general. "
                    "Omit to search all sections."
                ),
            },
            "doc_type_filter": {
                "type": "string",
                "description": (
                    "Optional: restrict by document type. "
                    "Valid values: index, text, video_summary. Omit to search all types."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Number of chunks to return (1-20). Defaults to 6.",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    },
}


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


def _blocks_to_dicts(content_blocks) -> list[dict]:
    """Convert SDK ContentBlock objects to plain serialisable dicts."""
    out = []
    for b in content_blocks:
        t = getattr(b, "type", None)
        if t == "text":
            out.append({"type": "text", "text": b.text})
        elif t == "tool_use":
            out.append({
                "type":  "tool_use",
                "id":    b.id,
                "name":  b.name,
                "input": dict(b.input),
            })
    return out


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_claude_histories: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RAG:
    """
    Retrieval-augmented generation using FAISSDatabase + Anthropic Claude tool-use.

    Named RAG for API compatibility with server.py.
    """

    def __init__(
        self,
        db: FAISSDatabase,
        gemini_api_key: str = "",        # ignored — kept for server.py compat
        anthropic_api_key: str = "",
        model: str = "claude-sonnet-4-6",
        top_k: int = 6,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.db             = db
        self.model          = model
        self.top_k          = top_k
        self._system        = system_prompt or _SYSTEM_PROMPT
        self._anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> None:
        _claude_histories.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(_claude_histories.keys())

    def _get_history(self, session_id: Optional[str]) -> list[dict]:
        if not session_id:
            return []
        if session_id not in _claude_histories:
            _claude_histories[session_id] = []
        return _claude_histories[session_id]

    def _save_turn(
        self,
        session_id: Optional[str],
        question: str,
        assistant_content: list[dict],
    ) -> None:
        if not session_id:
            return
        history = self._get_history(session_id)
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": assistant_content})

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _run_search_tool(
        self,
        tool_input: dict,
        default_top_k: int,
        default_section: Optional[str],
        default_doc_type: Optional[str],
        on_chunks: Optional[Callable[[list[dict]], None]] = None,
    ) -> tuple[str, list[Source]]:
        query    = tool_input.get("query", "")
        top_k    = int(tool_input.get("top_k", default_top_k))
        section  = tool_input.get("section_filter") or default_section
        doc_type = tool_input.get("doc_type_filter") or default_doc_type

        results = self._retrieve(query, top_k, section, doc_type)
        if not results:
            if on_chunks:
                on_chunks([])
            return "No relevant chunks found for this query.", []

        # Notify caller with the raw result dicts before building context
        if on_chunks:
            on_chunks(results)

        return _build_context_block(results), _extract_sources(results)

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
        import anthropic

        k        = top_k or self.top_k
        client   = anthropic.Anthropic(api_key=self._anthropic_key)
        history  = self._get_history(session_id)
        messages = list(history) + [{"role": "user", "content": question}]

        all_sources: list[Source] = []
        tokens_used: int          = 0

        logger.info("Claude answer [%s] session=%s", self.model, session_id)

        try:
            while True:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=self._system,
                    tools=[_SEARCH_TOOL],
                    messages=messages,
                )
                tokens_used += response.usage.input_tokens + response.usage.output_tokens

                if response.stop_reason == "end_turn":
                    answer_text = " ".join(
                        b.text for b in response.content
                        if hasattr(b, "text") and b.text
                    )
                    self._save_turn(session_id, question, _blocks_to_dicts(response.content))
                    return GeminiAnswer(
                        question=question,
                        answer=answer_text,
                        sources=all_sources,
                        total_tokens_used=tokens_used,
                    )

                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant",
                                     "content": _blocks_to_dicts(response.content)})
                    tool_results = []
                    for block in response.content:
                        if getattr(block, "type", None) != "tool_use":
                            continue
                        if block.name == "search_portfolio":
                            logger.info("Tool call: %s", json.dumps(dict(block.input)))
                            context, sources = self._run_search_tool(
                                dict(block.input), k, section_filter, doc_type_filter
                            )
                            all_sources.extend(sources)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": context,
                            })
                        else:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Unknown tool: {block.name}",
                                "is_error": True,
                            })
                    messages.append({"role": "user", "content": tool_results})
                else:
                    logger.warning("Unexpected stop_reason: %s", response.stop_reason)
                    break

        except Exception as exc:
            logger.error("Claude answer failed: %s", exc)
            return GeminiAnswer(question=question, answer=f"Error: {exc}",
                                sources=all_sources)

        return GeminiAnswer(question=question, answer="(no response)",
                            sources=all_sources, total_tokens_used=tokens_used)

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
        on_chunks: Optional[Callable[[list[dict]], None]] = None,
    ) -> Generator[str, None, GeminiAnswer]:
        """
        Synchronous generator — MUST be driven from a background thread.
        See server.py _run_llm_in_thread() for the correct usage pattern.
        Yields text tokens; returns GeminiAnswer via StopIteration.value.

        on_chunks: optional callback invoked every time Claude calls the
        search_portfolio tool.  Called with the raw list[dict] results from
        FAISSDatabase.search() so the caller can relay them to the client
        (e.g. by putting them into the SSE queue).  The callback is invoked
        in the background thread — it must be thread-safe.
        """
        return (yield from self._stream_claude(
            question,
            top_k or self.top_k,
            section_filter,
            doc_type_filter,
            session_id,
            on_chunks=on_chunks,
        ))

    # ------------------------------------------------------------------
    # Claude streaming tool-use loop  (blocking — runs in a thread)
    # ------------------------------------------------------------------

    def _stream_claude(
        self,
        question: str,
        default_top_k: int,
        default_section: Optional[str],
        default_doc_type: Optional[str],
        session_id: Optional[str],
        on_chunks: Optional[Callable[[list[dict]], None]] = None,
    ) -> Generator[str, None, GeminiAnswer]:
        """
        Sync generator.  Yields tokens in real time.

        Why real-time yield is safe:
          Claude emits NO text tokens during tool-use rounds.  Text only
          appears in the final end_turn round, so yielding immediately never
          leaks intermediate tool-call output to the client.

        on_chunks: forwarded to _run_search_tool and fired on every tool call.
        """
        import anthropic

        client   = anthropic.Anthropic(api_key=self._anthropic_key)
        history  = self._get_history(session_id)
        messages = list(history) + [{"role": "user", "content": question}]

        all_sources:   list[Source] = []
        tokens_used:   int          = 0
        full_text:     list[str]    = []
        final_content: list[dict]   = []

        logger.info("Claude stream [%s] session=%s", self.model, session_id)

        try:
            while True:
                with client.messages.stream(
                    model=self.model,
                    max_tokens=2048,
                    system=self._system,
                    tools=[_SEARCH_TOOL],
                    messages=messages,
                ) as stream:
                    # Yield tokens as they arrive.
                    # Tool-use rounds produce no text, so this only fires on end_turn.
                    for token in stream.text_stream:
                        if token:
                            full_text.append(token)
                            yield token                   # ← real-time, not buffered

                    final_msg    = stream.get_final_message()
                    tokens_used += (final_msg.usage.input_tokens
                                    + final_msg.usage.output_tokens)

                # end_turn — we're done
                if final_msg.stop_reason == "end_turn":
                    final_content = _blocks_to_dicts(final_msg.content)
                    break

                # tool_use — execute tools and loop back
                if final_msg.stop_reason == "tool_use":
                    messages.append({
                        "role":    "assistant",
                        "content": _blocks_to_dicts(final_msg.content),
                    })
                    tool_results = []
                    for block in final_msg.content:
                        if getattr(block, "type", None) != "tool_use":
                            continue
                        if block.name == "search_portfolio":
                            logger.info("Tool call: %s", json.dumps(dict(block.input)))
                            context, sources = self._run_search_tool(
                                dict(block.input),
                                default_top_k,
                                default_section,
                                default_doc_type,
                                on_chunks=on_chunks,
                            )
                            all_sources.extend(sources)
                            tool_results.append({
                                "type":        "tool_result",
                                "tool_use_id": block.id,
                                "content":     context,
                            })
                        else:
                            tool_results.append({
                                "type":        "tool_result",
                                "tool_use_id": block.id,
                                "content":     f"Unknown tool: {block.name}",
                                "is_error":    True,
                            })
                    messages.append({"role": "user", "content": tool_results})

                else:
                    logger.warning("Unexpected stop_reason: %s", final_msg.stop_reason)
                    break

        except Exception as exc:
            err = f"\n[Error: {exc}]"
            full_text.append(err)
            yield err

        if final_content:
            self._save_turn(session_id, question, final_content)

        return GeminiAnswer(
            question=question,
            answer="".join(full_text),
            sources=all_sources,
            total_tokens_used=tokens_used,
        )

    # ------------------------------------------------------------------
    # Internal retrieval
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        question: str,
        k: int,
        section_filter: Optional[str],
        doc_type_filter: Optional[str],
    ) -> list[dict]:
        raw = self.db.search(question, top_k=k, section_filter=section_filter)
        if doc_type_filter:
            raw = [r for r in raw if r.get("doc_type") == doc_type_filter]
        return raw[:k]