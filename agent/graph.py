"""LangGraph assembly: planner -> researcher -> evaluator -(loop | finish)-> writer."""

import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, Iterator, Literal, Optional

from langgraph.graph import END, START, StateGraph

from agent.nodes import evaluator_node, planner_node, researcher_node, writer_node
from agent.state import ResearchState, create_initial_state

logger = logging.getLogger(__name__)


def route_after_evaluation(state: ResearchState) -> Literal["researcher", "writer"]:
    """Conditional edge: re-search with refined queries, or proceed to the writer.

    `max_loops` is the maximum number of evaluator-triggered re-search loops, so
    the researcher runs at most `1 + max_loops` times. The evaluator already
    enforces the limit; it is re-checked here as a hard safety net.
    """
    evaluation = state.get("evaluation", {})
    refinements_used = state.get("loop_count", 1) - 1
    within_limit = refinements_used < state.get("max_loops", 3)
    if evaluation.get("needs_more") and state.get("search_plan") and within_limit:
        return "researcher"
    return "writer"


def build_graph():
    """Compile the research StateGraph."""
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {"researcher": "researcher", "writer": "writer"},
    )
    graph.add_edge("writer", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def _default_max_loops() -> int:
    try:
        return int(os.getenv("MAX_RESEARCH_LOOPS", "3"))
    except ValueError:
        return 3


def _clamp_loops(max_loops: Optional[int]) -> int:
    value = _default_max_loops() if max_loops is None else max_loops
    return max(0, min(int(value), 5))


def build_result(state: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    """Convert a final graph state into the JSON payload returned to clients."""
    return {
        "session_id": state.get("session_id", ""),
        "query": state.get("query", ""),
        "status": state.get("status", "failed"),
        "sub_questions": state.get("sub_questions", []),
        "report": state.get("final_report", ""),
        "citations": state.get("citations", []),
        "logs": state.get("logs", []),
        "errors": state.get("errors", []),
        "loops_used": max(0, state.get("loop_count", 0) - 1),
        "elapsed_seconds": round(elapsed, 2),
    }


def run_research(
    query: str, max_loops: Optional[int] = None, session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Run the full graph synchronously and return the result payload."""
    started = time.time()
    state = create_initial_state(query, _clamp_loops(max_loops), session_id)
    final_state = get_graph().invoke(state, config={"recursion_limit": 60})
    return build_result(final_state, time.time() - started)


def stream_research(
    query: str, max_loops: Optional[int] = None, session_id: Optional[str] = None
) -> Iterator[Dict[str, Any]]:
    """Run the graph and yield events as they happen.

    Event types:
      {"type": "start", "session_id", "query"}
      {"type": "log", "log": {...}}          (live step logs)
      {"type": "error", "message": "..."}    (unexpected failure)
      {"type": "final", "result": {...}}     (always the last event)
    """
    started = time.time()
    state = create_initial_state(query, _clamp_loops(max_loops), session_id)
    yield {"type": "start", "session_id": state["session_id"], "query": state["query"]}

    last_state: Dict[str, Any] = dict(state)
    try:
        for mode, chunk in get_graph().stream(
            state, config={"recursion_limit": 60}, stream_mode=["custom", "values"]
        ):
            if mode == "custom":
                yield chunk
            elif mode == "values":
                last_state = chunk
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research run failed")
        message = f"{type(exc).__name__}: {exc}"
        yield {"type": "error", "message": message}
        last_state = dict(last_state)
        last_state["status"] = "failed"
        last_state["errors"] = list(last_state.get("errors", [])) + [message]

    yield {"type": "final", "result": build_result(last_state, time.time() - started)}
