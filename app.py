"""
app.py — Streamlit UI for the Drone Security Analyst Agent.

Run from the drone-security-agent/ directory:
    streamlit run app.py

Tabs:
  1. Pipeline   — select scenario + analyzer, run pipeline, see live frame output
  2. Alerts     — table of all alerts fired, filterable by severity
  3. Search     — query by time / object / location / semantic
  4. Entities   — entity tracker: who/what was seen, how many times
  5. Agent Chat — LangChain Q&A with conversation memory
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make drone_agent importable when running from the project root
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# ── page config must be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="Drone Security Agent",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

from drone_agent.config import DATA_DIR, DB_PATH, CHROMA_PATH, DATABASE_URL
from drone_agent.storage import make_store, SearchEngine
from drone_agent.pipeline import ContextManager, AlertRulesEngine, run_pipeline
from drone_agent.simulator import load_scenario
from drone_agent.analyzer import RuleBasedAnalyzer, LLMAnalyzer

# ── severity badge colours ─────────────────────────────────────────────────
_SEV_COLOR = {"critical": "#d32f2f", "warning": "#f57c00", "info": "#1976d2"}
_SEV_EMOJI = {"critical": "🔴", "warning": "🟡", "info": "🔵"}


# ============================================================================
# Session-state helpers
# ============================================================================

def _init_state():
    """Initialise all session-state keys on first load."""
    if "sqlite" not in st.session_state:
        # On page refresh, restore the latest run_id from the DB so the session
        # reconnects to the previous run's data automatically.
        restore_run_id = None
        if DATABASE_URL:
            try:
                _probe = make_store(DATABASE_URL)
                restore_run_id = _probe.get_latest_run_id()
                _probe.close()
            except Exception:
                pass

        store = make_store(DATABASE_URL, run_id=restore_run_id)
        vec = None
        try:
            from drone_agent.storage import VectorStore
            vec = VectorStore(CHROMA_PATH)
        except (RuntimeError, ImportError):
            pass
        st.session_state.sqlite = store
        st.session_state.search = SearchEngine(store, vec)
        st.session_state.has_vec = vec is not None

    if "context" not in st.session_state:
        st.session_state.context = ContextManager()

    if "analyzed" not in st.session_state:
        st.session_state.analyzed = []   # list[AnalyzedFrame]

    if "alerts" not in st.session_state:
        st.session_state.alerts = []     # list[Alert]

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {"role": "user"|"assistant", "content": str}
        st.session_state.chat_messages = []  # full LangGraph messages list (includes tool calls)

    if "pipeline_ran" not in st.session_state:
        # If the DB already has data from a previous run, treat it as ran —
        # this is what makes Search, Alerts, Entities, Agent Chat work after a page refresh.
        try:
            st.session_state.pipeline_ran = st.session_state.sqlite.frame_count() > 0
        except Exception:
            st.session_state.pipeline_ran = False


def _reset_pipeline():
    """Clear this run's data and prepare a fresh run_id for the next run."""
    try:
        st.session_state.sqlite.clear()  # deletes only this run_id's rows
    except Exception:
        pass
    # Assign a new run_id so the next pipeline run is fully isolated
    import uuid as _uuid
    new_run_id = str(_uuid.uuid4())
    st.session_state.sqlite.run_id = new_run_id
    st.session_state.context = ContextManager()
    st.session_state.analyzed = []
    st.session_state.alerts = []
    st.session_state.agent = None
    st.session_state.chat_history = []
    st.session_state.chat_messages = []
    st.session_state.pipeline_ran = False


# ============================================================================
# Sidebar
# ============================================================================

def sidebar():
    st.sidebar.title("🚁 Drone Security Agent")
    st.sidebar.markdown("---")

    # Scenario file picker
    scenario_files = sorted(DATA_DIR.glob("*.json")) if DATA_DIR.exists() else []
    scenario_names = [f.name for f in scenario_files]
    if not scenario_names:
        st.sidebar.warning("No scenario files found in data/")
        return None, "rule"

    selected_name = st.sidebar.selectbox("Scenario file", scenario_names, index=0)
    scenario_path = DATA_DIR / selected_name

    # Analyzer choice
    analyzer_choice = st.sidebar.radio("Analyzer", ["rule", "llm"], index=0,
                                        help="rule = offline regex (fast, no API key)\nllm = LLM-backed (requires API key)")

    st.sidebar.markdown("---")

    # Pipeline controls
    col1, col2 = st.sidebar.columns(2)
    run_clicked = col1.button("▶  Run", use_container_width=True, type="primary")
    col2.button("↺  Reset", use_container_width=True, on_click=_reset_pipeline)

    # Status
    st.sidebar.markdown("---")
    if st.session_state.pipeline_ran:
        # Always read counts from DB — in-memory lists are empty after a page refresh
        try:
            n_frames = st.session_state.search.frame_count_global()
            n_alerts = len(st.session_state.search.get_alerts_global())
        except Exception:
            n_frames = len(st.session_state.analyzed)
            n_alerts = len(st.session_state.alerts)
        st.sidebar.success(f"✅ {n_frames} frames  ·  {n_alerts} alerts")
        vec_status = "✅ enabled" if st.session_state.has_vec else "⚠️ unavailable"
        st.sidebar.caption(f"Semantic search: {vec_status}")
    else:
        st.sidebar.info("No pipeline run yet.")

    return (scenario_path, analyzer_choice) if run_clicked else (None, analyzer_choice)


# ============================================================================
# Tab 1 — Pipeline
# ============================================================================

def tab_pipeline(scenario_path, analyzer_choice):
    st.header("🎬 Pipeline Run")

    if not st.session_state.pipeline_ran and scenario_path is None:
        st.info("Select a scenario and click **▶ Run** in the sidebar to start.")
        return

    if scenario_path is not None:
        # Build analyzer
        if analyzer_choice == "llm":
            from drone_agent.pipeline import summarize_chunk
            ana = LLMAnalyzer()
            st.session_state.context._summarize = summarize_chunk
        else:
            ana = RuleBasedAnalyzer()

        rules = AlertRulesEngine()
        sqlite = st.session_state.sqlite
        # Generate a fresh run_id for this pipeline run so it doesn't
        # collide with previous runs (all scenarios use frame_id 1-20).
        import uuid as _uuid
        st.session_state.sqlite.run_id = str(_uuid.uuid4())
        sqlite = st.session_state.sqlite
        sqlite.register_run(
            scenario=scenario_path.name,
            analyzer=analyzer_choice,
        )
        # Reset context so previous run's rolling window and entity tracker
        # don't bleed into this run's alert evaluation.
        st.session_state.context = ContextManager()
        search = st.session_state.search
        ctx = st.session_state.context

        st.subheader(f"Running: `{scenario_path.name}` · analyzer=`{analyzer_choice}`")
        progress_bar = st.progress(0)
        log_area = st.empty()
        log_lines: list[str] = []

        from drone_agent.config import ALERTS_LOG_PATH
        from drone_agent.models import Alert

        all_analyzed = []
        all_alerts = []

        pairs = list(load_scenario(scenario_path))
        total = len(pairs)

        with open(ALERTS_LOG_PATH, "a", encoding="utf-8") as log_file:
            for i, (frame, telemetry) in enumerate(pairs):
                af = ana.analyze(frame, telemetry)
                all_analyzed.append(af)

                sqlite.insert_analyzed_frame(af)

                vec = search._vec
                if vec is not None:
                    vec.upsert_frame(
                        frame_id=af.frame.frame_id,
                        description=af.frame.description,
                        summary=af.summary,
                        timestamp=af.frame.timestamp.isoformat(),
                        location=af.telemetry.location,
                        labels=[d.label for d in af.detections],
                    )

                ctx.update(af)
                frame_alerts = rules.evaluate(af, ctx)

                for alert in frame_alerts:
                    sqlite.insert_alert(alert)
                    all_alerts.append(alert)
                    log_file.write(
                        f"[{alert.severity.upper()}] {alert.message} "
                        f"(rule={alert.rule})\n"
                    )
                    log_file.flush()
                    emoji = _SEV_EMOJI.get(alert.severity, "⚪")
                    log_lines.append(f"  {emoji} **ALERT [{alert.severity.upper()}]** — {alert.message}")

                ts = af.frame.timestamp.strftime("%H:%M")
                log_lines.append(f"`[{ts}]` **{af.telemetry.location}** — {af.summary}")

                # Update live display (last 20 lines)
                log_area.markdown("\n\n".join(log_lines[-20:]))
                progress_bar.progress((i + 1) / total)

        ctx.flush_final()

        st.session_state.analyzed = all_analyzed
        st.session_state.alerts = all_alerts
        st.session_state.pipeline_ran = True

        st.success(f"✅ Done — {len(all_analyzed)} frames, {len(all_alerts)} alerts.")

    elif st.session_state.pipeline_ran:
        # Show summary of previous run
        st.subheader("Last run summary")

        # After a page refresh, in-memory lists are empty — read from DB instead
        db_frames = st.session_state.search.by_time("1970-01-01T00:00:00", "2099-12-31T23:59:59")
        db_alerts = st.session_state.search.get_alerts_global()
        analyzed = st.session_state.analyzed  # may be empty after refresh

        c1, c2, c3 = st.columns(3)
        c1.metric("Frames processed", st.session_state.search.frame_count_global())
        c2.metric("Alerts fired", len(db_alerts))
        c3.metric("Entities tracked", len(st.session_state.context._entities))

        st.markdown("### Frame log")
        # Use DB rows when in-memory list is empty (after refresh)
        if analyzed:
            rows = [{
                "Frame": af.frame.frame_id,
                "Time": af.frame.timestamp.strftime("%H:%M"),
                "Location": af.telemetry.location,
                "Altitude (m)": af.telemetry.altitude_m,
                "Summary": af.summary,
            } for af in analyzed]
        else:
            rows = [{
                "Frame": r["frame_id"],
                "Time": r["timestamp"][11:16],
                "Location": r["location"],
                "Altitude (m)": r["altitude_m"],
                "Summary": r["summary"],
            } for r in db_frames]
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ============================================================================
# Tab 2 — Alerts
# ============================================================================

def tab_alerts():
    st.header("🚨 Alerts")

    if not st.session_state.pipeline_ran:
        st.info("Run the pipeline first.")
        return

    all_alerts = st.session_state.search.get_alerts_global()
    if not all_alerts:
        st.success("No alerts were fired.")
        return

    # Severity filter
    severities = ["all", "critical", "warning", "info"]
    sev_filter = st.selectbox("Filter by severity", severities, index=0)

    filtered = all_alerts if sev_filter == "all" else [
        a for a in all_alerts if a["severity"] == sev_filter
    ]

    st.caption(f"Showing {len(filtered)} of {len(all_alerts)} alerts")

    for a in filtered:
        color = _SEV_COLOR.get(a["severity"], "#666")
        emoji = _SEV_EMOJI.get(a["severity"], "⚪")
        ts = a["timestamp"][:16].replace("T", " ")
        with st.container(border=True):
            st.markdown(
                f"{emoji} &nbsp; **{a['severity'].upper()}** &nbsp;·&nbsp; "
                f"<span style='color:{color};font-weight:bold'>{a['message']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Rule: `{a['rule']}` · Time: `{ts}` · "
                f"Evidence frames: {a['evidence_frame_ids']}"
            )


# ============================================================================
# Tab 3 — Search
# ============================================================================

def tab_search():
    st.header("🔍 Search Frames")

    if not st.session_state.pipeline_ran:
        st.info("Run the pipeline first.")
        return

    search = st.session_state.search
    mode = st.radio("Query type", ["By Object", "By Time", "By Location", "Semantic"],
                    horizontal=True)

    results = []

    if mode == "By Object":
        c1, c2, c3 = st.columns(3)
        label = c1.selectbox("Label", ["", "vehicle", "person", "animal", "unknown"])
        subtype = c2.selectbox("Subtype", ["", "truck", "van", "car", "suv", "motorcycle"])
        color = c3.text_input("Color", placeholder="blue, red, silver…")
        if st.button("Search", key="obj_search"):
            results = search.by_object(
                label=label or None,
                subtype=subtype or None,
                color=color.strip().lower() or None,
            )

    elif mode == "By Time":
        c1, c2 = st.columns(2)
        start = c1.text_input("Start (HH:MM or ISO)", value="08:00")
        end = c2.text_input("End   (HH:MM or ISO)", value="12:00")

        def _norm(t):
            return f"2024-01-15T{t}:00" if len(t) == 5 and ":" in t else t

        if st.button("Search", key="time_search"):
            results = search.by_time(_norm(start), _norm(end))

    elif mode == "By Location":
        loc = st.text_input("Location (partial match)", placeholder="Main Gate")
        if st.button("Search", key="loc_search"):
            results = search.by_location(loc)

    elif mode == "Semantic":
        if not st.session_state.has_vec:
            st.warning("Vector store not available. Install `chromadb` and `sentence-transformers`.")
        else:
            query = st.text_input("Natural language query", placeholder="vehicle at night near fence")
            k = st.slider("Results", 1, 20, 5)
            if st.button("Search", key="sem_search"):
                results = search.semantic(query, k=k)

    if results:
        st.caption(f"{len(results)} result(s)")
        st.dataframe(results, use_container_width=True, hide_index=True)
    elif st.session_state.get(f"{mode}_searched"):
        st.info("No results found.")


# ============================================================================
# Tab 4 — Entities
# ============================================================================

def tab_entities():
    st.header("🏷️ Tracked Entities")

    if not st.session_state.pipeline_ran:
        st.info("Run the pipeline first.")
        return

    entities = st.session_state.context._entities
    if not entities:
        st.info("No entities tracked yet.")
        return

    # Summary metrics
    persons = sum(1 for e in entities.values() if e.label == "person")
    vehicles = sum(1 for e in entities.values() if e.label == "vehicle")
    animals = sum(1 for e in entities.values() if e.label == "animal")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total entities", len(entities))
    c2.metric("Persons", persons)
    c3.metric("Vehicles", vehicles)
    c4.metric("Animals", animals)

    st.markdown("---")

    # Entity table
    rows = []
    for key, info in entities.items():
        color = info.attributes.get("color", "")
        make = info.attributes.get("make", "").title()
        model = info.attributes.get("model", "").upper()
        name = " ".join(p for p in [color, make, model] if p) or info.label.title()
        rows.append({
            "Name": name,
            "Label": info.label,
            "Entity Key": key,
            "First Seen": info.first_seen.strftime("%H:%M"),
            "Last Seen": info.last_seen.strftime("%H:%M"),
            "Locations": ", ".join(info.locations),
            "Entries": info.entry_count,
            "Exits": info.exit_count,
            "Frames": len(info.frame_ids),
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Drill-down
    st.markdown("---")
    st.subheader("Entity detail")
    query = st.text_input("Search entity (e.g. 'blue ford', 'truck', 'person')")
    if query:
        result = st.session_state.context.get_entity_history(query)
        if result["found"]:
            for e in result["entities"]:
                with st.expander(f"🔑 `{e['entity_key']}`", expanded=True):
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Entry count", e["entry_count"])
                    cc2.metric("Exit count", e["exit_count"])
                    cc3.metric("Total frames", e["total_frames"])
                    st.write(f"**First seen:** {e['first_seen'][:16]}  |  **Last seen:** {e['last_seen'][:16]}")
                    st.write(f"**Locations:** {', '.join(e['locations'])}")
                    st.write(f"**Attributes:** {e['attributes']}")
                    st.write(f"**Frame IDs:** {e['frame_ids']}")
        else:
            st.info(f"No entity found matching `{query}`.")


# ============================================================================
# Tab 5 — Agent Chat
# ============================================================================

def tab_agent():
    st.header("💬 Agent Q&A")

    if not st.session_state.pipeline_ran:
        st.info("Run the pipeline first.")
        return

    # Build agent once per session
    if st.session_state.agent is None:
        with st.spinner("Building agent…"):
            try:
                from drone_agent.agent import build_agent
                st.session_state.agent = build_agent(
                    st.session_state.search,
                    st.session_state.context,
                )
                st.success("Agent ready.")
            except Exception as exc:
                st.error(f"Could not build agent: {exc}")
                st.caption("Make sure the API key / LLM provider is set correctly.")
                return

    # Display conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Quick example prompts
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        examples = [
            "Show all truck events",
            "What happened at the main gate last night?",
            "Were there any critical alerts?",
            "Give me the video summary",
            "How many times did the blue Ford F150 enter today?",
        ]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex, use_container_width=True):
                _ask(ex)
                st.rerun()

    # Chat input
    question = st.chat_input("Ask a question about the patrol…")
    if question:
        _ask(question)
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.session_state.chat_messages = []
            # Rebuild agent to reset its internal state
            try:
                from drone_agent.agent import build_agent
                st.session_state.agent = build_agent(
                    st.session_state.search,
                    st.session_state.context,
                )
            except Exception:
                pass
            st.rerun()


def _ask(question: str):
    """Send a question to the agent using the LangGraph messages API."""
    # Append the new user message to the persistent messages list
    st.session_state.chat_messages.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "user", "content": question})
    try:
        result = st.session_state.agent.invoke(
            {"messages": st.session_state.chat_messages}
        )
        # The last message in the result is the assistant reply
        last_msg = result["messages"][-1]
        answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        # Persist the full updated messages list for next turn
        st.session_state.chat_messages = result["messages"]
    except Exception as exc:
        answer = f"Agent error: {exc}"
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ============================================================================
# Main layout
# ============================================================================

def main():
    _init_state()

    scenario_path, analyzer_choice = sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎬 Pipeline",
        "🚨 Alerts",
        "🔍 Search",
        "🏷️ Entities",
        "💬 Agent Chat",
    ])

    with tab1:
        tab_pipeline(scenario_path, analyzer_choice)
    with tab2:
        tab_alerts()
    with tab3:
        tab_search()
    with tab4:
        tab_entities()
    with tab5:
        tab_agent()


if __name__ == "__main__":
    main()
