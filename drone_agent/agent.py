"""
agent.py — LangChain 1.x agent for natural-language querying.

LangChain 1.x (which ships with LangGraph) replaced AgentExecutor with
create_agent(), which returns a compiled LangGraph graph. The old
AgentExecutor and ConversationBufferWindowMemory no longer exist.

New invocation pattern:
    graph = build_agent(search, context)
    result = graph.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

Conversation memory is managed by passing the full messages list back on
each call — the caller accumulates it in a plain list (see app.py).

Six tools wrap the SearchEngine and ContextManager:
    query_by_time, query_by_object, semantic_search,
    get_entity_history, get_alerts, get_video_summary
"""

from __future__ import annotations

import json
import logging

from drone_agent.pipeline import ContextManager, summarize_video
from drone_agent.storage import SearchEngine

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a professional property security analyst reviewing drone patrol footage.
You have access to a searchable index of every frame recorded today.

Rules:
1. Answer ONLY from tool results. Do not guess or invent events.
2. Cite specific frame IDs and timestamps for every claim (e.g. "Frame 17 at 21:00").
3. When alerts exist in the results, give a concrete recommendation
   (e.g. "Dispatch a guard to Main Gate immediately.").
4. If no evidence is found, say "No evidence found for that query."

Session context (summaries, not raw frames):
{session_context}
"""


def build_tools(search: SearchEngine, context: ContextManager) -> list:
    """Build LangChain tools wired to live search and context instances."""
    from langchain_core.tools import tool

    def _norm(t: str) -> str | None:
        if not t:
            return None
        return f"2024-01-15T{t}:00" if (len(t) == 5 and ":" in t) else t

    @tool
    def query_by_time(start: str, end: str) -> str:
        """Query frames by time range. Use HH:MM (e.g. '08:00') or ISO 8601."""
        results = search.by_time(_norm(start), _norm(end))
        return json.dumps(results[:20], indent=2) if results else "No frames found in that time range."

    @tool
    def query_by_object(label: str, color: str = "", subtype: str = "") -> str:
        """Query frames by object type (person/vehicle/animal), optional color, optional subtype (truck/van/car)."""
        results = search.by_object(
            label=label or None,
            subtype=subtype or None,
            color=color or None,
        )
        return json.dumps(results[:20], indent=2) if results else f"No frames found with label={label!r}."

    @tool
    def semantic_search(query: str, k: int = 5) -> str:
        """Natural-language semantic search over frame descriptions. Use for fuzzy queries."""
        results = search.semantic(query, k=k)
        return json.dumps(results, indent=2) if results else "No semantic results (vector store may be unavailable)."

    @tool
    def get_entity_history(entity: str) -> str:
        """Get tracking history for an entity. Use partial keys like 'blue ford' or 'truck'."""
        result = context.get_entity_history(entity)
        return json.dumps(result, indent=2) if result.get("found") else f"No entity matching {entity!r} tracked."

    @tool
    def get_alerts(severity: str = "", since: str = "") -> str:
        """Get alerts filtered by severity (info/warning/critical) and optional since time (HH:MM)."""
        results = search.get_alerts(severity=severity or None, since=_norm(since))
        return json.dumps(results, indent=2) if results else "No alerts found."

    @tool
    def get_video_summary() -> str:
        """Get a one-sentence summary of the entire patrol session."""
        s = context.get_summaries()
        return summarize_video(
            chunk_summaries=s["chunk_summaries"],
            daily_summary=s["daily_summary"],
        )

    return [
        query_by_time,
        query_by_object,
        semantic_search,
        get_entity_history,
        get_alerts,
        get_video_summary,
    ]


def build_agent(search: SearchEngine, context: ContextManager):
    """
    Build and return a LangGraph compiled agent (langchain 1.x API).

    Usage:
        graph = build_agent(search, context)

        # First question
        messages = [{"role": "user", "content": "Show all truck events"}]
        result = graph.invoke({"messages": messages})
        answer = result["messages"][-1].content

        # Follow-up (pass full history back)
        messages = result["messages"] + [{"role": "user", "content": "Which came more than once?"}]
        result = graph.invoke({"messages": messages})
        answer = result["messages"][-1].content
    """
    from langchain.agents import create_agent
    from langchain_core.messages import SystemMessage
    from drone_agent.analyzer import _build_llm

    llm = _build_llm()
    tools = build_tools(search, context)
    system_msg = SystemMessage(
        content=_SYSTEM_PROMPT.format(session_context=context.get_context_for_agent())
    )
    return create_agent(llm, tools, system_prompt=system_msg)
