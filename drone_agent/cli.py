"""
cli.py — Command-line interface for the drone security agent.

Run from the drone-security-agent/ directory:

    python drone_agent/cli.py run -s data/scenario_day1.json
    python drone_agent/cli.py run -s data/scenario_day1.json --analyzer llm
    python drone_agent/cli.py ask "show all truck events"
    python drone_agent/cli.py summary
    python drone_agent/cli.py chat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `drone_agent` is importable
# whether the script is run as `python drone_agent/cli.py` or `python -m drone_agent`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Shared init helpers
# ---------------------------------------------------------------------------

def _init(db_path: str | None = None, chroma_path: str | None = None):
    """Return (sqlite, search, context) using configured or override paths."""
    from drone_agent.config import DB_PATH, CHROMA_PATH
    from drone_agent.storage import SQLiteStore, SearchEngine
    from drone_agent.pipeline import ContextManager

    sqlite = SQLiteStore(db_path or DB_PATH)

    vec = None
    try:
        from drone_agent.storage import VectorStore
        vec = VectorStore(chroma_path or CHROMA_PATH)
    except (RuntimeError, ImportError):
        print("[info] Vector store unavailable — semantic search disabled.")

    return sqlite, SearchEngine(sqlite, vec), ContextManager()


def _rebuild_context(search, ctx) -> None:
    """Reload context summaries from the DB (for ask/summary/chat after a separate run)."""
    rows = search.by_time("1970-01-01", "2099-12-31")
    for r in rows:
        if r.get("summary"):
            ctx._pending.append(r["summary"])
            ctx._total += 1
    ctx.flush_final()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args):
    from drone_agent.simulator import load_scenario
    from drone_agent.analyzer import RuleBasedAnalyzer, LLMAnalyzer
    from drone_agent.pipeline import AlertRulesEngine, run_pipeline, summarize_chunk

    sqlite, search, ctx = _init(args.db, args.chroma)
    sqlite.clear()  # wipe previous run so alerts don't accumulate across runs

    if args.analyzer == "llm":
        ana = LLMAnalyzer()
        ctx._summarize = summarize_chunk
    else:
        ana = RuleBasedAnalyzer()

    print(f"\n=== Drone Security Agent ===")
    print(f"Scenario : {args.scenario}")
    print(f"Analyzer : {args.analyzer}")
    print("=" * 30 + "\n")

    frames_iter = load_scenario(args.scenario, realtime=args.realtime, speed_factor=args.speed)
    analyzed, alerts = run_pipeline(frames_iter, ana, sqlite, search, ctx,
                                    AlertRulesEngine())
    print(f"\nDone. {len(analyzed)} frames processed, {len(alerts)} alerts fired.")


def cmd_ask(args):
    sqlite, search, ctx = _init(args.db, args.chroma)
    _rebuild_context(search, ctx)
    _run_agent(args.question, search, ctx)


def cmd_summary(args):
    sqlite, search, ctx = _init(args.db, args.chroma)
    _rebuild_context(search, ctx)

    from drone_agent.pipeline import summarize_video
    s = ctx.get_summaries()
    result = summarize_video(s["chunk_summaries"], s["daily_summary"])
    print("\n=== Video Summary ===")
    print(result)


def cmd_chat(args):
    sqlite, search, ctx = _init(args.db, args.chroma)
    _rebuild_context(search, ctx)

    from drone_agent.agent import build_agent
    try:
        agent = build_agent(search, ctx)
    except Exception as exc:
        print(f"Error building agent: {exc}")
        print("Make sure LLM_PROVIDER and API key are set in .env")
        sys.exit(1)

    print("\n=== Drone Security Agent Chat ===")
    print("Type your questions. Type 'exit' to quit.\n")

    messages = []  # accumulate full message history for LangGraph
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit", "q"):
            break
        try:
            messages.append({"role": "user", "content": question})
            result = agent.invoke({"messages": messages})
            messages = result["messages"]
            last = messages[-1]
            answer = last.content if hasattr(last, "content") else str(last)
            print(f"\nAgent: {answer}\n")
        except Exception as exc:
            print(f"Error: {exc}")


def _run_agent(question: str, search, ctx) -> None:
    from drone_agent.agent import build_agent
    try:
        graph = build_agent(search, ctx)
        result = graph.invoke({"messages": [{"role": "user", "content": question}]})
        last = result["messages"][-1]
        answer = last.content if hasattr(last, "content") else str(last)
        print(f"\nAgent: {answer}")
    except Exception as exc:
        print(f"Error: {exc}")
        print("Make sure the API key is configured.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="drone_agent",
        description="Drone Security Analyst Agent",
    )
    parser.add_argument("--db", default=None, help="SQLite DB path override")
    parser.add_argument("--chroma", default=None, help="ChromaDB path override")

    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Process a scenario file through the pipeline")
    p_run.add_argument("-s", "--scenario", required=True, help="Path to scenario JSON")
    p_run.add_argument("-a", "--analyzer", default="rule", choices=["rule", "llm"],
                       help="Analyzer to use (default: rule)")
    p_run.add_argument("--realtime", action="store_true", help="Simulate live stream with sleep")
    p_run.add_argument("--speed", type=float, default=60.0, help="Speed factor for realtime mode")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a one-shot question")
    p_ask.add_argument("question", help='e.g. "show all truck events"')

    # summary
    sub.add_parser("summary", help="Print the one-sentence video summary")

    # chat
    sub.add_parser("chat", help="Interactive Q&A with conversation memory")

    args = parser.parse_args()

    # Attach shared options to args for subcommands
    if not hasattr(args, "db"):
        args.db = None
    if not hasattr(args, "chroma"):
        args.chroma = None

    commands = {
        "run": cmd_run,
        "ask": cmd_ask,
        "summary": cmd_summary,
        "chat": cmd_chat,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
