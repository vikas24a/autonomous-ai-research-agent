"""State schema and structured-output models for the research agent graph."""

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------
def merge_sources(
    left: Optional[List[Dict[str, Any]]], right: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Merge two source lists, de-duplicating by URL (first occurrence wins)."""
    merged: Dict[str, Dict[str, Any]] = {}
    for item in (left or []) + (right or []):
        url = item.get("url")
        if url and url not in merged:
            merged[url] = item
    return list(merged.values())


# ---------------------------------------------------------------------------
# Structured LLM outputs (used with `llm.with_structured_output(...)`)
# ---------------------------------------------------------------------------
class SubQuestionPlan(BaseModel):
    """One sub-question and the web search query intended to answer it."""

    question: str = Field(description="A self-contained sub-question to research.")
    search_query: str = Field(
        description="A concise, keyword-rich web search query for this sub-question."
    )


class PlannerOutput(BaseModel):
    """Decomposition of the user's research query (3-5 sub-questions)."""

    sub_questions: List[SubQuestionPlan] = Field(
        description="Between 3 and 5 distinct sub-questions covering the topic."
    )


class SubQuestionAssessment(BaseModel):
    """Critic's verdict for a single sub-question."""

    question: str = Field(description="The sub-question being assessed (verbatim).")
    status: Literal["answered", "partial", "missing"] = Field(
        description="Whether the evidence answers the question."
    )
    reasoning: str = Field(description="One or two sentences justifying the status.")
    refined_queries: List[str] = Field(
        default_factory=list,
        description="New, different search queries to fill gaps (empty if answered).",
    )


class EvaluationResult(BaseModel):
    """Critic's overall verdict on the evidence gathered so far."""

    is_sufficient: bool = Field(
        description="True only if all sub-questions are adequately answered."
    )
    assessments: List[SubQuestionAssessment] = Field(default_factory=list)
    feedback: str = Field(default="", description="Short overall feedback.")


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class ResearchState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes.

    Fields annotated with a reducer are *accumulated* across node updates;
    all other fields are overwritten by the latest node output.
    """

    # Inputs / control
    session_id: str                 # Keys the per-run FAISS index
    query: str                      # Original user query
    max_loops: int                  # Max evaluator -> researcher cycles
    loop_count: int                 # Completed research passes so far

    # Planning
    sub_questions: List[str]
    search_plan: List[Dict[str, str]]   # [{"sub_question": ..., "query": ...}]

    # Evidence
    sources: Annotated[List[Dict[str, Any]], merge_sources]
    search_summaries: Annotated[List[Dict[str, Any]], operator.add]

    # Evaluation
    evaluation: Dict[str, Any]

    # Observability
    logs: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]

    # Output
    final_report: str
    status: str                     # running | complete | failed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_log(
    step: str,
    message: str,
    level: str = "info",
    loop: int = 0,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a structured, JSON-serialisable step-log entry."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "level": level,
        "loop": loop,
        "message": message,
        "data": data or {},
    }


def create_initial_state(
    query: str, max_loops: int = 3, session_id: Optional[str] = None
) -> ResearchState:
    """Create a fresh state for a new research run."""
    return ResearchState(
        session_id=session_id or uuid.uuid4().hex[:16],
        query=query.strip(),
        max_loops=max_loops,
        loop_count=0,
        sub_questions=[],
        search_plan=[],
        sources=[],
        search_summaries=[],
        evaluation={},
        logs=[],
        errors=[],
        final_report="",
        status="running",
    )
