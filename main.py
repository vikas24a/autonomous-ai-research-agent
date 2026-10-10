"""FastAPI backend for the Autonomous AI Research Agent.

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
  or: python main.py
"""

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv

load_dotenv()

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

from agent.graph import run_research, stream_research  # noqa: E402
from agent.nodes import validate_llm_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("research-api")

app = FastAPI(
    title="Autonomous AI Research Agent API",
    version="1.0.0",
    description="LangGraph multi-agent research pipeline (planner → researcher → evaluator → writer).",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Autonomous AI Research Agent API is running"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000, description="The research question.")
    max_loops: Optional[int] = Field(
        None, ge=0, le=5, description="Max evaluator-triggered re-search loops (default from env, 3)."
    )
    stream: bool = Field(False, description="If true, stream NDJSON events instead of one JSON body.")

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("Query must be at least 5 characters long.")
        return value


class ResearchResponse(BaseModel):
    session_id: str
    query: str
    status: str
    sub_questions: List[str]
    report: str
    citations: List[Dict[str, Any]]
    logs: List[Dict[str, Any]]
    errors: List[str]
    loops_used: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_llm() -> None:
    problem = validate_llm_config()
    if problem:
        raise HTTPException(status_code=503, detail=f"LLM not configured: {problem}")


def _ndjson(events: Iterator[Dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield json.dumps(event, default=str, ensure_ascii=False) + "\n"


def _stream_response(body: ResearchRequest) -> StreamingResponse:
    return StreamingResponse(
        _ndjson(stream_research(body.query, body.max_loops)),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    problem = validate_llm_config()
    return {
        "status": "ok" if problem is None else "degraded",
        "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
        "llm_configured": problem is None,
        "llm_problem": problem,
        "search": "tavily+duckduckgo" if os.getenv("TAVILY_API_KEY") else "duckduckgo",
        "default_max_loops": int(os.getenv("MAX_RESEARCH_LOOPS", "3")),
    }


@app.post("/api/research", response_model=None)
def research(body: ResearchRequest):
    """Run the research graph.

    - `stream=false` (default): returns one JSON `ResearchResponse` with the
      step-by-step logs and the final report.
    - `stream=true`: streams newline-delimited JSON events (start/log/error/final).
    """
    _require_llm()
    if body.stream:
        return _stream_response(body)
    try:
        return ResearchResponse(**run_research(body.query, body.max_loops))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research run failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/research/stream")
def research_stream(body: ResearchRequest) -> StreamingResponse:
    """Always-streaming variant (NDJSON), used by the Streamlit frontend."""
    _require_llm()
    return _stream_response(body)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
