"""Streamlit frontend for the Autonomous AI Research Agent.

Run:  streamlit run app.py     (backend must be running: uvicorn main:app --port 8000)
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Set

import requests
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
STEPS = [
    ("planner", "🧭 Planner"),
    ("researcher", "🔎 Researcher"),
    ("evaluator", "🧪 Evaluator"),
    ("writer", "✍️ Writer"),
]
EXAMPLES = [
    "Compare solid-state batteries and lithium-ion batteries for electric vehicles: maturity, cost, safety and timelines.",
    "What are the most effective techniques for reducing hallucinations in large language models?",
    "Impact of quantum computing on modern cryptography and the state of post-quantum standards.",
]
LEVEL_ICONS = {"info": "•", "warning": "⚠", "error": "✖"}


# ---------------------------------------------------------------------------
# PDF export (fpdf2, core fonts only -> text is sanitised to latin-1)
# ---------------------------------------------------------------------------
_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-",
    "\u2014": "-", "\u2026": "...", "\u2022": "*", "\u00a0": " ", "\u2192": "->",
    "\u2248": "~", "\u2265": ">=", "\u2264": "<=", "\u00d7": "x",
}


def _prep(text: str) -> str:
    """Strip Markdown inline syntax and make text safe for core PDF fonts."""
    text = re.sub(r"\[\[(\d+)\]\]\([^)]*\)", r"[\1]", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1 (\2)", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(?<!\w)[*_]([^*_]+?)[*_](?!\w)", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"(\S{50})(?=\S)", r"\1 ", text)  # let very long URLs wrap
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def markdown_to_pdf(markdown_text: str) -> bytes:
    """Render the report Markdown (headings, paragraphs, bullets, tables) as a PDF."""
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    def write(text: str, size: float = 10.5, style: str = "", height: float = 5.6, indent: float = 0) -> None:
        pdf.set_font("Helvetica", style, size)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(0, height, _prep(text), new_x="LMARGIN", new_y="NEXT")

    heading_sizes = {1: 18, 2: 14, 3: 12, 4: 11}
    lines = markdown_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped or stripped.startswith("```") or re.fullmatch(r"[-*_]{3,}", stripped):
            if not stripped:
                pdf.ln(2)
            i += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            pdf.ln(3 if level > 1 else 1)
            write(heading.group(2), size=heading_sizes[level], style="B", height=heading_sizes[level] * 0.55)
            pdf.ln(1)
            i += 1
            continue

        if stripped.startswith("|"):
            rows: List[List[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            width = max((len(r) for r in rows), default=0)
            rows = [r + [""] * (width - len(r)) for r in rows]
            try:
                pdf.set_font("Helvetica", "", 8.5)
                with pdf.table(text_align="LEFT") as table:
                    for r in rows:
                        row = table.row()
                        for cell in r:
                            row.cell(_prep(cell))
                pdf.set_x(pdf.l_margin)
                pdf.ln(2)
            except Exception:  # noqa: BLE001
                for r in rows:
                    write(" | ".join(r), size=8.5)
            continue

        bullet = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if bullet:
            indent = 4 + 4 * (len(bullet.group(1)) // 2)
            write("- " + bullet.group(2), indent=indent)
            i += 1
            continue

        numbered = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", line)
        if numbered:
            write(f"{numbered.group(2)}. {numbered.group(3)}", indent=4)
            i += 1
            continue

        write(stripped.lstrip("> ").strip() if stripped.startswith(">") else stripped)
        i += 1

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_steps(placeholder, active: Optional[str], done: Set[str], loop: int) -> None:
    with placeholder.container():
        cols = st.columns(len(STEPS))
        for col, (key, label) in zip(cols, STEPS):
            icon = "✅" if key in done else ("⏳" if key == active else "⚪")
            col.markdown(f"**{icon} {label}**")
        st.caption(f"Research pass: {max(loop, 0)}")


def format_log_line(log: Dict[str, Any]) -> str:
    ts = str(log.get("timestamp", ""))[11:19]
    icon = LEVEL_ICONS.get(log.get("level", "info"), "•")
    loop = log.get("loop", 0)
    loop_tag = f" #{loop}" if loop else ""
    return f"{ts}  {icon} [{log.get('step', '')}{loop_tag}] {log.get('message', '')}"


def check_backend(api_url: str) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreachable", "llm_problem": str(exc)}


def run_research_stream(api_url: str, query: str, max_loops: int) -> None:
    """Call the streaming endpoint and render live progress. Stores result in session_state."""
    st.session_state.pop("result", None)
    st.session_state.pop("pdf_bytes", None)

    logs: List[Dict[str, Any]] = []
    done: Set[str] = set()
    active: Optional[str] = None
    loop = 0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    with st.status("Research agent running…", expanded=True) as status:
        steps_ph = st.empty()
        log_ph = st.empty()
        render_steps(steps_ph, active, done, loop)

        try:
            with requests.post(
                f"{api_url.rstrip('/')}/api/research/stream",
                json={"query": query, "max_loops": max_loops},
                stream=True,
                timeout=(10, 1200),
            ) as resp:
                if resp.status_code != 200:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:  # noqa: BLE001
                        detail = resp.text
                    raise RuntimeError(f"Backend returned {resp.status_code}: {detail}")
                resp.encoding = "utf-8"

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    event = json.loads(raw_line)
                    etype = event.get("type")

                    if etype == "log":
                        log = event["log"]
                        logs.append(log)
                        step = log.get("step")
                        phase = (log.get("data") or {}).get("phase")
                        loop = max(loop, int(log.get("loop", 0) or 0))
                        if phase == "start":
                            active = step
                            done.discard(step)
                            if step == "researcher":
                                done.discard("evaluator")
                        elif phase == "end":
                            done.add(step)
                            if active == step:
                                active = None
                        render_steps(steps_ph, active, done, loop)
                        log_ph.code("\n".join(format_log_line(l) for l in logs[-80:]), language="text")
                    elif etype == "error":
                        error_message = event.get("message")
                    elif etype == "final":
                        result = event["result"]
        except requests.exceptions.ConnectionError:
            error_message = f"Cannot reach the backend at {api_url}. Start it with: uvicorn main:app --port 8000"
        except Exception as exc:  # noqa: BLE001
            error_message = f"{type(exc).__name__}: {exc}"

        if result and result.get("report") and result.get("status") == "complete":
            render_steps(steps_ph, None, {k for k, _ in STEPS}, loop)
            status.update(label="Research complete", state="complete", expanded=False)
        else:
            status.update(label="Research failed", state="error", expanded=True)

    if result:
        if not result.get("logs"):
            result["logs"] = logs
        st.session_state["result"] = result
        if result.get("report"):
            try:
                st.session_state["pdf_bytes"] = markdown_to_pdf(result["report"])
            except Exception as exc:  # noqa: BLE001
                st.session_state["pdf_error"] = str(exc)
    if error_message:
        st.error(error_message)


def render_result(result: Dict[str, Any]) -> None:
    status = result.get("status", "failed")
    citations = result.get("citations", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status.capitalize())
    c2.metric("Sources cited", len(citations))
    c3.metric("Re-search loops", result.get("loops_used", 0))
    c4.metric("Time (s)", result.get("elapsed_seconds", 0))

    slug = re.sub(r"[^a-z0-9]+", "_", result.get("query", "report").lower()).strip("_")[:40] or "report"
    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button(
        "⬇️ Markdown",
        data=result.get("report", ""),
        file_name=f"{slug}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    if st.session_state.get("pdf_bytes"):
        d2.download_button(
            "⬇️ PDF",
            data=st.session_state["pdf_bytes"],
            file_name=f"{slug}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    elif st.session_state.get("pdf_error"):
        d2.caption(f"PDF unavailable: {st.session_state['pdf_error']}")

    tab_report, tab_cites, tab_plan, tab_log = st.tabs(
        ["📄 Report", f"🔗 Citations ({len(citations)})", "🗺️ Plan", "🪵 Execution log"]
    )
    with tab_report:
        st.markdown(result.get("report", "_No report generated._"))
    with tab_cites:
        if not citations:
            st.info("No citations were produced.")
        for c in citations:
            st.markdown(f"**{c['id']}.** [{c['title']}]({c['url']})")
            if c.get("snippet"):
                st.caption(c["snippet"])
    with tab_plan:
        for i, q in enumerate(result.get("sub_questions", []), 1):
            st.markdown(f"{i}. {q}")
    with tab_log:
        st.code("\n".join(format_log_line(l) for l in result.get("logs", [])) or "No logs.", language="text")
        errors = result.get("errors", [])
        if errors:
            with st.expander(f"⚠️ {len(errors)} non-fatal warnings/errors"):
                for e in errors:
                    st.text(e)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Autonomous AI Research Agent", page_icon="🔬", layout="wide")
st.title("🔬 Autonomous AI Research Agent")
st.caption("Planner → Researcher → Critic (re-search loop) → Writer, powered by LangGraph + FAISS.")

with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("Backend URL", value=DEFAULT_API_URL)
    max_loops = st.slider("Max re-search loops", 0, 5, 3, help="How many times the critic may send the researcher back for more evidence.")
    if st.button("Check backend"):
        health = check_backend(api_url)
        if health.get("status") == "ok":
            st.success(f"Backend OK · LLM: {health.get('llm_provider')} · Search: {health.get('search')}")
        elif health.get("status") == "degraded":
            st.warning(f"Backend up, but: {health.get('llm_problem')}")
        else:
            st.error(f"Backend unreachable: {health.get('llm_problem')}")
    st.divider()
    st.subheader("Examples")


def _set_query(text: str) -> None:
    st.session_state["query_input"] = text


for example in EXAMPLES:
    st.sidebar.button(example[:60] + "…", on_click=_set_query, args=(example,), key=f"ex_{hash(example)}")

query = st.text_area(
    "Research question",
    key="query_input",
    height=110,
    placeholder="e.g. Compare the leading approaches to grid-scale energy storage and their economics…",
)

if st.button("🚀 Run research", type="primary"):
    if len((query or "").strip()) < 5:
        st.warning("Please enter a research question (at least 5 characters).")
    else:
        run_research_stream(api_url, query.strip(), max_loops)

if "result" in st.session_state:
    st.divider()
    render_result(st.session_state["result"])
