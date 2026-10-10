"""LangGraph node implementations: planner, researcher, evaluator, writer.

Every node:
  * never raises on expected failures (LLM/search/scrape errors are logged and
    handled with fallbacks),
  * emits live step logs through LangGraph's custom stream writer (when running
    inside a graph) and also returns them in state["logs"].
"""

import difflib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel

from agent.state import (
    EvaluationResult,
    PlannerOutput,
    ResearchState,
    SubQuestionAssessment,
    make_log,
)
from agent.tools import get_memory, search_and_index

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
}


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def validate_llm_config() -> Optional[str]:
    """Return a human-readable configuration problem, or None if the LLM is usable."""
    provider = _provider()
    key_var = _KEY_VARS.get(provider)
    if key_var is None:
        return f"Unsupported LLM_PROVIDER '{provider}'. Use one of: openai, anthropic, groq."
    if not os.getenv(key_var):
        return f"{key_var} is not set (required for LLM_PROVIDER={provider})."
    return None


def get_llm(temperature: Optional[float] = None, max_tokens: Optional[int] = None):
    """Build a chat model for the configured provider (openai | anthropic | groq)."""
    provider = _provider()
    if temperature is None:
        try:
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        except ValueError:
            temperature = 0.2
    try:
        timeout = int(os.getenv("LLM_TIMEOUT", "120"))
    except ValueError:
        timeout = 120
    max_tokens = max_tokens or 4096

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            timeout=timeout,
            max_retries=2,
            max_tokens=max_tokens,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            temperature=temperature,
            timeout=timeout,
            max_retries=2,
            max_tokens=max_tokens,
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            temperature=temperature,
            timeout=timeout,
            max_retries=2,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Use openai, anthropic or groq.")


def _message_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def _extract_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def invoke_structured(
    schema: Type[T],
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    attempts: int = 2,
) -> T:
    """Call the LLM and return a validated `schema` instance.

    Tries native structured output first; falls back to plain-JSON parsing for
    models/providers where tool-calling structured output is unreliable.
    """
    llm = get_llm(temperature=temperature)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    last_exc: Optional[Exception] = None

    for _ in range(attempts):
        try:
            result = llm.with_structured_output(schema).invoke(messages)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    try:
        schema_json = json.dumps(schema.model_json_schema())
        raw = llm.invoke(
            [
                SystemMessage(
                    content=system_prompt
                    + "\n\nRespond ONLY with a valid JSON object (no prose, no code "
                    + "fences) matching this JSON schema:\n"
                    + schema_json
                ),
                HumanMessage(content=user_prompt),
            ]
        )
        return schema.model_validate(_extract_json(_message_text(raw)))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Structured LLM output failed: {last_exc or exc}") from exc


# ---------------------------------------------------------------------------
# Live logging
# ---------------------------------------------------------------------------
def _emit(entry: Dict[str, Any]) -> None:
    """Push a log entry to the live stream (no-op outside a graph run)."""
    try:
        get_stream_writer()({"type": "log", "log": entry})
    except Exception:  # noqa: BLE001
        pass


class StepLogger:
    """Collects log entries for a node and streams them live."""

    def __init__(self, step: str, loop: int = 0) -> None:
        self.step = step
        self.loop = loop
        self.entries: List[Dict[str, Any]] = []

    def log(self, message: str, level: str = "info", **data: Any) -> None:
        entry = make_log(self.step, message, level=level, loop=self.loop, data=data)
        self.entries.append(entry)
        _emit(entry)
        getattr(logger, "warning" if level != "info" else "info")("[%s] %s", self.step, message)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fallback_plan(query: str) -> List[Dict[str, str]]:
    year = datetime.now().year
    return [
        {"question": f"What is the background and key concepts behind: {query}?",
         "search_query": f"{query} overview background"},
        {"question": f"What are the latest developments, data and evidence regarding: {query}?",
         "search_query": f"{query} latest developments statistics {year}"},
        {"question": f"What are the main alternatives, comparisons, challenges and limitations of: {query}?",
         "search_query": f"{query} comparison challenges limitations"},
    ]


def _default_refinements(sub_question: str, attempt: int) -> List[str]:
    year = datetime.now().year
    variants = [
        f"{sub_question} detailed explanation data",
        f"{sub_question} research report {year}",
        f"{sub_question} case study examples evidence",
        f"{sub_question} expert analysis review",
    ]
    return [variants[attempt % len(variants)]]


def _match_question(question: str, sub_questions: List[str]) -> Optional[str]:
    if question in sub_questions:
        return question
    match = difflib.get_close_matches(question, sub_questions, n=1, cutoff=0.4)
    return match[0] if match else None


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------------------------------------------------------------------------
# 1. Planner
# ---------------------------------------------------------------------------
def planner_node(state: ResearchState) -> Dict[str, Any]:
    query = state["query"]
    sl = StepLogger("planner", 0)
    sl.log("Decomposing the query into sub-questions…", phase="start")
    errors: List[str] = []

    system = (
        "You are the Planner in an autonomous multi-agent research system.\n"
        "Break the user's research query into 3 to 5 distinct, non-overlapping "
        "sub-questions that together cover the topic: background/definitions, current "
        "state and recent developments, concrete data and evidence, comparisons or "
        "alternatives, and risks/limitations/open issues (use the ones that fit).\n"
        "For each sub-question write a concise, keyword-rich web search query "
        "(4-12 words, no quotes, no search operators).\n"
        f"Today's date is {datetime.now().strftime('%B %d, %Y')}; prefer recent information "
        "where relevant."
    )

    plan: List[Dict[str, str]] = []
    try:
        out = invoke_structured(PlannerOutput, system, f"Research query: {query}")
        seen = set()
        for item in out.sub_questions:
            q, sq = item.question.strip(), (item.search_query or "").strip()
            if not q or q.lower() in seen:
                continue
            seen.add(q.lower())
            plan.append({"question": q, "search_query": sq or q})
        plan = plan[:5]
    except Exception as exc:  # noqa: BLE001
        msg = f"Planner LLM failed ({exc}); using a generic plan."
        sl.log(msg, level="warning")
        errors.append(msg)

    if len(plan) < 3:
        existing = {p["question"].lower() for p in plan}
        for fb in _fallback_plan(query):
            if len(plan) >= 3:
                break
            if fb["question"].lower() not in existing:
                plan.append(fb)

    sub_questions = [p["question"] for p in plan]
    search_plan = [{"sub_question": p["question"], "query": p["search_query"]} for p in plan]
    for i, p in enumerate(plan, 1):
        sl.log(f"Sub-question {i}: {p['question']}", query=p["search_query"])
    sl.log(f"Plan ready with {len(plan)} sub-questions.", phase="end")

    return {
        "sub_questions": sub_questions,
        "search_plan": search_plan,
        "loop_count": 0,
        "status": "running",
        "logs": sl.entries,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# 2. Researcher
# ---------------------------------------------------------------------------
def researcher_node(state: ResearchState) -> Dict[str, Any]:
    loop = state.get("loop_count", 0) + 1
    session_id = state["session_id"]
    sl = StepLogger("researcher", loop)

    plan = state.get("search_plan") or [
        {"sub_question": q, "query": q} for q in state.get("sub_questions", [])
    ]
    sl.log(f"Research pass {loop}: running {len(plan)} targeted searches…", phase="start")

    summaries: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    errors: List[str] = []

    if plan:
        workers = max(1, min(len(plan), 3))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(search_and_index, session_id, t["query"], t["sub_question"]): t
                for t in plan
            }
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    summary = fut.result()
                except Exception as exc:  # noqa: BLE001
                    summary = {
                        "query": task["query"], "sub_question": task["sub_question"],
                        "provider": None, "results_found": 0, "pages_scraped": 0,
                        "snippet_fallbacks": 0, "chunks_added": 0, "sources": [],
                        "errors": [f"Unexpected failure: {exc}"],
                    }
                sources.extend(summary.get("sources", []))
                for err in summary.get("errors", []):
                    errors.append(f"[{task['query']}] {err}")
                level = "info" if summary["results_found"] else "warning"
                sl.log(
                    f"“{_clip(task['query'], 80)}” → {summary['results_found']} results via "
                    f"{summary['provider'] or 'none'}, {summary['pages_scraped']} pages scraped, "
                    f"{summary['chunks_added']} chunks indexed",
                    level=level,
                    query=task["query"],
                )
                compact = {k: v for k, v in summary.items() if k != "sources"}
                compact["loop"] = loop
                summaries.append(compact)

    stats = get_memory(session_id).stats()
    sl.log(
        f"Evidence store now holds {stats['chunks']} chunks from {stats['urls']} sources.",
        phase="end",
        **stats,
    )
    return {
        "loop_count": loop,
        "sources": sources,
        "search_summaries": summaries,
        "logs": sl.entries,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# 3. Evaluator / Critic
# ---------------------------------------------------------------------------
def evaluator_node(state: ResearchState) -> Dict[str, Any]:
    loop = state.get("loop_count", 1)
    max_loops = state.get("max_loops", 3)
    sub_questions = state.get("sub_questions", [])
    session_id = state["session_id"]
    sl = StepLogger("evaluator", loop)
    sl.log("Checking the gathered evidence against each sub-question…", phase="start")
    errors: List[str] = []

    memory = get_memory(session_id)
    stats = memory.stats()
    blocks: List[str] = []
    for i, sq in enumerate(sub_questions, 1):
        hits = memory.retrieve(sq, k=4)
        if hits:
            excerpts = "\n".join(f"  - ({h['url']}) {_clip(h['content'], 450)}" for h in hits)
        else:
            excerpts = "  (no evidence retrieved)"
        blocks.append(f"Sub-question {i}: {sq}\n{excerpts}")

    used_queries = sorted({s.get("query", "") for s in state.get("search_summaries", [])})
    used_lower = {q.lower() for q in used_queries}

    system = (
        "You are the Critic in an autonomous research system. For EACH sub-question, "
        "judge whether the evidence excerpts contain concrete, relevant information "
        "that answers it.\n"
        "- status 'answered': specific facts/data are present.\n"
        "- status 'partial': related but vague, one-sided, outdated or ambiguous.\n"
        "- status 'missing': nothing relevant.\n"
        "For partial/missing items, propose 1-2 NEW search queries (4-12 words) that "
        "differ from the queries already used. Set is_sufficient=true only if every "
        "sub-question is answered well enough to write a solid report. Copy each "
        "sub-question text verbatim into the 'question' field."
    )
    user = (
        f"Original research query: {state['query']}\n\n"
        f"Queries already used:\n" + "\n".join(f"- {q}" for q in used_queries) + "\n\n"
        f"Evidence store: {stats['chunks']} chunks from {stats['urls']} sources.\n\n"
        + "\n\n".join(blocks)
    )

    result: Optional[EvaluationResult] = None
    try:
        result = invoke_structured(EvaluationResult, system, user)
    except Exception as exc:  # noqa: BLE001
        msg = f"Evaluator LLM failed ({exc}); using a heuristic assessment."
        sl.log(msg, level="warning")
        errors.append(msg)

    # Map assessments onto the real sub-questions (fill any the critic skipped).
    assessed: Dict[str, SubQuestionAssessment] = {}
    if result is not None:
        for a in result.assessments:
            sq = _match_question(a.question, sub_questions)
            if sq and sq not in assessed:
                assessed[sq] = a
    for sq in sub_questions:
        if sq not in assessed:
            if result is None:
                status = "answered" if stats["chunks"] > 0 else "missing"
                reason = "Heuristic: evidence exists." if stats["chunks"] else "Heuristic: no evidence."
            else:
                status, reason = "partial", "Not assessed by the critic."
            assessed[sq] = SubQuestionAssessment(
                question=sq, status=status, reasoning=reason, refined_queries=[]
            )

    gaps = [a for a in assessed.values() if a.status != "answered"]
    llm_sufficient = bool(result.is_sufficient) if result is not None else not gaps
    sufficient = (not gaps) or (llm_sufficient and not any(a.status == "missing" for a in gaps))

    counts = {s: sum(1 for a in assessed.values() if a.status == s) for s in ("answered", "partial", "missing")}
    for sq, a in assessed.items():
        sl.log(f"[{a.status.upper()}] {_clip(sq, 90)} — {_clip(a.reasoning, 140)}",
               level="info" if a.status == "answered" else "warning")

    refinements_used = loop - 1
    can_loop = refinements_used < max_loops
    new_plan: List[Dict[str, str]] = []
    if not sufficient and can_loop:
        for a in gaps:
            sq = _match_question(a.question, sub_questions) or a.question
            fresh = [q.strip() for q in a.refined_queries if q.strip() and q.strip().lower() not in used_lower]
            if not fresh:
                fresh = [q for q in _default_refinements(sq, refinements_used) if q.lower() not in used_lower]
            for q in fresh[:2]:
                new_plan.append({"sub_question": sq, "query": q})
        new_plan = new_plan[:6]
    needs_more = bool(new_plan)

    if needs_more:
        sl.log(
            f"{counts['answered']} answered, {counts['partial']} partial, {counts['missing']} missing → "
            f"re-searching with {len(new_plan)} refined queries "
            f"(refinement {refinements_used + 1}/{max_loops}).",
            phase="end",
        )
    elif sufficient:
        sl.log("Evidence is sufficient → handing over to the writer.", phase="end")
    else:
        sl.log(
            f"Loop limit reached ({max_loops}) with {counts['partial']} partial / "
            f"{counts['missing']} missing → writing with the evidence available.",
            level="warning",
            phase="end",
        )

    evaluation = {
        "loop": loop,
        "is_sufficient": sufficient,
        "needs_more": needs_more,
        "feedback": result.feedback if result is not None else "",
        "counts": counts,
        "assessments": [a.model_dump() for a in assessed.values()],
    }
    return {
        "evaluation": evaluation,
        "search_plan": new_plan,
        "logs": sl.entries,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# 4. Writer / Synthesizer
# ---------------------------------------------------------------------------
def _gather_evidence(
    state: ResearchState, per_query_k: int = 6, max_chunks: int = 36, max_per_source: int = 4
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Collect de-duplicated evidence chunks and assign citation numbers per URL."""
    memory = get_memory(state["session_id"])
    queries = list(state.get("sub_questions", [])) + [state["query"]]
    chunks: List[Dict[str, Any]] = []
    seen = set()
    per_source: Dict[str, int] = {}

    for q in queries:
        for hit in memory.retrieve(q, k=per_query_k):
            key = (hit["url"], hit["content"][:120])
            if key in seen or per_source.get(hit["url"], 0) >= max_per_source:
                continue
            seen.add(key)
            per_source[hit["url"]] = per_source.get(hit["url"], 0) + 1
            chunks.append(hit)
            if len(chunks) >= max_chunks:
                break
        if len(chunks) >= max_chunks:
            break

    order: Dict[str, int] = {}
    for h in chunks:
        if h["url"] not in order:
            order[h["url"]] = len(order) + 1
    return chunks, order


def _link_citations(text: str, id_to_url: Dict[int, str]) -> str:
    """Turn [1] / [1, 3] markers into clickable markdown links; drop invalid numbers."""
    pattern = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\](?!\()")

    def repl(match: "re.Match[str]") -> str:
        nums = [int(n) for n in re.findall(r"\d+", match.group(1))]
        links = [f"[[{n}]]({id_to_url[n]})" for n in nums if n in id_to_url]
        return " ".join(links)

    return pattern.sub(repl, text)


def _safe_title(title: str) -> str:
    return re.sub(r"[\[\]]", "", " ".join((title or "").split())) or "Untitled source"


def _fallback_report(state: ResearchState, chunks: List[Dict[str, Any]], order: Dict[str, int]) -> str:
    """Deterministic report used when the writer LLM is unavailable."""
    lines = [
        f"# Research Report: {state['query']}",
        "",
        "## Executive Summary",
        "",
        "The automated writer was unavailable, so this report lists the most relevant "
        "evidence retrieved for each sub-question. Review the excerpts and sources below.",
        "",
        "## Detailed Analysis",
        "",
    ]
    memory = get_memory(state["session_id"])
    for sq in state.get("sub_questions", []):
        lines += [f"### {sq}", ""]
        for hit in memory.retrieve(sq, k=3):
            n = order.get(hit["url"])
            cite = f" [{n}]" if n else ""
            lines.append(f"- {_clip(hit['content'], 400)}{cite}")
        lines.append("")
    lines += ["## Key Findings", "", "- See the evidence excerpts above.", ""]
    return "\n".join(lines)


def writer_node(state: ResearchState) -> Dict[str, Any]:
    loop = state.get("loop_count", 0)
    sl = StepLogger("writer", loop)
    sl.log("Synthesizing the final research report…", phase="start")
    errors: List[str] = []

    chunks, order = _gather_evidence(state, per_query_k=int(os.getenv("RETRIEVAL_K", "6")))
    if not chunks:
        sl.log("No evidence could be gathered; cannot write a grounded report.", level="error", phase="end")
        report = (
            f"# Research Report: {state['query']}\n\n"
            "> **No usable evidence was gathered.** All searches failed or returned no "
            "readable content. Check your network, `TAVILY_API_KEY`, and the execution log.\n"
        )
        return {
            "final_report": report,
            "citations": [],
            "status": "failed",
            "logs": sl.entries,
            "errors": ["Writer: no evidence available."],
        }

    # Build numbered source blocks.
    source_meta = {s["url"]: s for s in state.get("sources", [])}
    grouped: Dict[str, List[str]] = {}
    titles: Dict[str, str] = {}
    for h in chunks:
        grouped.setdefault(h["url"], []).append(h["content"])
        titles.setdefault(h["url"], h["title"] or source_meta.get(h["url"], {}).get("title", ""))

    evidence_text = "\n\n".join(
        f"[{n}] {_safe_title(titles[url])} — {url}\n" + "\n".join(f"  • {_clip(c, 900)}" for c in grouped[url])
        for url, n in order.items()
    )
    id_to_url = {n: url for url, n in order.items()}
    citations = [
        {
            "id": n,
            "title": _safe_title(titles[url]),
            "url": url,
            "snippet": _clip(source_meta.get(url, {}).get("snippet", ""), 300),
        }
        for url, n in order.items()
    ]

    gaps = [
        a for a in state.get("evaluation", {}).get("assessments", []) if a.get("status") != "answered"
    ]
    gaps_text = (
        "\n".join(f"- {a['question']} ({a['status']}): {a['reasoning']}" for a in gaps)
        if gaps else "None — all sub-questions were adequately answered."
    )

    system = (
        "You are a senior research analyst writing a rigorous, well-structured report in "
        "Markdown. Use ONLY the numbered evidence provided; never invent facts, numbers, "
        "names or URLs. If sources conflict, say so explicitly.\n\n"
        "Required structure (use these exact H2 headings, in this order):\n"
        "# <Descriptive report title>\n"
        "## Executive Summary  (one short paragraph, 4-6 sentences)\n"
        "## Detailed Analysis  (one H3 subsection per sub-question, with substantive paragraphs)\n"
        "## Key Findings  (5-8 bullet points, each a concrete finding)\n"
        "## Comparative Analysis  (at least one Markdown table comparing relevant options, "
        "approaches, entities or viewpoints; include a 'Source' or evidence column where sensible)\n"
        "## Limitations & Gaps  (what the evidence did not cover)\n\n"
        "Citation rules: cite every factual claim with its source number in square brackets, "
        "e.g. [2] or [1, 4]. Only use numbers that appear in the evidence list. Do NOT write a "
        "references/sources section — it is appended automatically. Do not wrap the output in "
        "code fences."
    )
    user = (
        f"Research query: {state['query']}\n"
        f"Today's date: {datetime.now().strftime('%B %d, %Y')}\n\n"
        "Sub-questions:\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(state.get("sub_questions", []), 1))
        + f"\n\nKnown evidence gaps:\n{gaps_text}\n\n"
        f"=== NUMBERED EVIDENCE ===\n{evidence_text}"
    )

    try:
        llm = get_llm(temperature=0.3, max_tokens=8000)
        raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        report = _message_text(raw).strip()
        report = re.sub(r"^```(?:markdown|md)?\s*\n|\n```\s*$", "", report).strip()
        if len(report) < 200:
            raise ValueError("Writer returned an unusably short report")
    except Exception as exc:  # noqa: BLE001
        msg = f"Writer LLM failed ({exc}); generated a fallback evidence digest."
        sl.log(msg, level="warning")
        errors.append(msg)
        report = _fallback_report(state, chunks, order)

    report = _link_citations(report, id_to_url)
    refs = "\n".join(f"{c['id']}. [{c['title']}]({c['url']})" for c in citations)
    report = f"{report.rstrip()}\n\n## References\n\n{refs}\n"

    sl.log(f"Report complete: {len(report.split())} words, {len(citations)} cited sources.", phase="end")
    return {
        "final_report": report,
        "citations": citations,
        "status": "complete",
        "logs": sl.entries,
        "errors": errors,
    }
