"""Streamlit frontend for the Autonomous AI Research Agent (glassmorphism landing-page theme).

Run:  streamlit run app.py     (backend must be running: uvicorn main:app --port 8000)
"""

import json
import os
import re
from html import escape
from typing import Any, Dict, List, Optional, Set

import requests
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

# ---------------------------------------------------------------------------
# Branding / copy (edit these to change the landing page text)
# ---------------------------------------------------------------------------
BRAND_NAME = "ResearchAgent"
BADGE_TEXT = "Multi-agent · Self-critiquing · Cited"
HERO_LINES = ["The Smarter,", "{AI-Powered}", "Research Agent"]  # {..} gets the gradient accent
HERO_SUBTITLE = (
    "Harness the power of an autonomous multi-agent loop: plan, search, verify and write "
    "fully cited research reports in minutes."
)
NAV_LINKS = [
    ("Research", "#research"),
    ("How it works", "#how-it-works"),
    ("Features", "#features"),
    ("About", "#about"),
]

DEFAULT_API_URL = os.getenv("API_URL", "https://autonomous-ai-research-agent-s6j0.onrender.com")
STEPS = [
    ("planner", "Planner"),
    ("researcher", "Researcher"),
    ("evaluator", "Evaluator"),
    ("writer", "Writer"),
]
EXAMPLES = [
    ("🔋 Solid-state vs Li-ion",
     "Compare solid-state batteries and lithium-ion batteries for electric vehicles: maturity, cost, safety and timelines."),
    ("🧠 Reducing LLM hallucinations",
     "What are the most effective techniques for reducing hallucinations in large language models?"),
    ("🔐 Quantum vs cryptography",
     "Impact of quantum computing on modern cryptography and the state of post-quantum standards."),
]
LEVEL_ICONS = {"info": "•", "warning": "⚠", "error": "✖"}


# ---------------------------------------------------------------------------
# Theme: CSS
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#F8FAFC; --ink:#0F172A; --muted:#64748B;
  --violet:#7C3AED; --violet-2:#C084FC; --cyan:#67E8F9; --pink:#F472B6;
  --glass:rgba(255,255,255,.65); --glass-border:rgba(255,255,255,.8);
  --shadow:0 10px 30px -10px rgba(0,0,0,.05);
  --font:'Plus Jakarta Sans','Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
}

/* ---------- base + ambient background ---------- */
html,[data-testid="stMain"]{scroll-behavior:smooth}
.stApp{background:var(--bg);color:var(--ink)}
.stApp,.stApp :is(p,h1,h2,h3,h4,label,li,a,button,textarea,input,td,th){font-family:var(--font)!important}
.stApp::before,.stApp::after,[data-testid="stAppViewContainer"]::before{
  content:"";position:fixed;z-index:0;pointer-events:none;border-radius:50%;filter:blur(120px)}
.stApp::before{width:660px;height:660px;left:-180px;top:-140px;opacity:.55;
  background:radial-gradient(circle at 35% 35%,var(--violet-2),transparent 66%)}
.stApp::after{width:720px;height:720px;right:-220px;top:22vh;opacity:.5;
  background:radial-gradient(circle at 60% 40%,var(--cyan),transparent 64%)}
[data-testid="stAppViewContainer"]{position:relative;z-index:1}
[data-testid="stAppViewContainer"]::before{width:640px;height:640px;left:18vw;bottom:-260px;z-index:-1;opacity:.42;
  background:radial-gradient(circle at 50% 50%,var(--pink),transparent 66%)}
[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"]{background:transparent!important}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"],#MainMenu,footer{display:none!important}
[data-testid="stElementContainer"]:has(style){display:none}
[data-testid="stMainBlockContainer"]{max-width:1200px;padding:0 24px 72px!important}

/* ---------- navbar ---------- */
.nav-wrap{position:fixed;top:14px;left:0;right:0;z-index:999;display:flex;justify-content:center;padding:0 16px;pointer-events:none}
.nav{pointer-events:auto;width:min(1180px,100%);display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px;
  padding:8px 8px 8px 20px;border-radius:999px;background:rgba(255,255,255,.6);
  -webkit-backdrop-filter:blur(16px) saturate(1.4);backdrop-filter:blur(16px) saturate(1.4);
  border:1px solid var(--glass-border);box-shadow:var(--shadow),inset 0 1px 0 rgba(255,255,255,.9)}
.brand{display:inline-flex;align-items:center;gap:10px;font-weight:800;font-size:1.08rem;letter-spacing:-.02em;
  color:var(--ink)!important;text-decoration:none!important;transition:all .3s}
.brand:hover{transform:translateY(-2px)}
.gem{width:22px;height:22px;border-radius:7px;transform:rotate(45deg) scale(.82);
  background:conic-gradient(from 210deg,#7C3AED,#C084FC,#F472B6,#67E8F9,#7C3AED);box-shadow:0 4px 14px rgba(124,58,237,.45)}
.pill{display:flex;gap:2px;padding:4px;border-radius:999px;background:rgba(255,255,255,.7);border:1px solid rgba(15,23,42,.06)}
.pill a{padding:8px 16px;border-radius:999px;font-size:.88rem;font-weight:600;white-space:nowrap;
  color:var(--muted)!important;text-decoration:none!important;transition:all .3s}
.pill a:hover{background:#fff;color:var(--ink)!important;transform:translateY(-2px);box-shadow:0 6px 16px -8px rgba(15,23,42,.25)}
.nav-cta{justify-self:end}

/* ---------- html buttons ---------- */
.btn{display:inline-flex;align-items:center;gap:8px;padding:13px 26px;border-radius:999px;font-weight:700;font-size:.95rem;
  text-decoration:none!important;white-space:nowrap;cursor:pointer;transition:all .3s cubic-bezier(.2,.8,.2,1)}
.btn:hover{transform:translateY(-2px)}
.btn-sm{padding:10px 20px;font-size:.86rem}
.btn-dark{background:linear-gradient(135deg,#0F172A,#4C1D95);color:#fff!important;box-shadow:0 12px 26px -12px rgba(76,29,149,.8)}
.btn-dark:hover{box-shadow:0 18px 32px -12px rgba(124,58,237,.85)}
.btn-glass{background:rgba(255,255,255,.55);color:var(--ink)!important;border:1px solid rgba(255,255,255,.95);
  -webkit-backdrop-filter:blur(12px);backdrop-filter:blur(12px);box-shadow:var(--shadow)}
.btn-glass:hover{background:#fff;box-shadow:0 14px 28px -14px rgba(76,29,149,.3)}

/* ---------- hero ---------- */
.hero{display:grid;grid-template-columns:1.05fr .95fr;gap:32px;align-items:center;padding:132px 0 36px;min-height:640px;overflow-x:clip}
.badge{display:inline-flex;align-items:center;gap:8px;padding:7px 14px 7px 8px;border-radius:999px;font-size:.8rem;font-weight:700;
  color:#5B21B6;border:1px solid rgba(124,58,237,.18);box-shadow:var(--shadow);
  background:linear-gradient(110deg,rgba(255,255,255,.6) 30%,rgba(255,255,255,.98) 50%,rgba(255,255,255,.6) 70%) 0 0/250% 100%;
  animation:shimmer 3.4s linear infinite}
.badge .spark{width:20px;height:20px;border-radius:50%;display:grid;place-items:center;font-size:.7rem;color:#fff;
  background:linear-gradient(135deg,#7C3AED,#F472B6)}
.hero-title{margin:22px 0 18px;font-size:clamp(2.5rem,5.4vw,4.5rem);line-height:1.04;letter-spacing:-.035em;font-weight:800;color:var(--ink)}
.hero-title .ln{display:block}
.grad{background:linear-gradient(90deg,#7C3AED,#C084FC 55%,#F472B6);-webkit-background-clip:text;background-clip:text;
  color:transparent;-webkit-text-fill-color:transparent}
.hero-sub{max-width:520px;font-size:1.1rem;line-height:1.65;color:var(--muted)}
.hero-cta{display:flex;flex-wrap:wrap;gap:12px;margin-top:30px}
.trust{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:34px;color:var(--muted);font-size:.82rem;font-weight:600}
.trust span::before{content:"✦";color:var(--violet-2);margin-right:8px}

/* 3D glass scene (sizes in container-query units so it scales on any screen) */
.scene{position:relative;width:100%;max-width:560px;margin:0 auto;aspect-ratio:1/.95;container-type:inline-size}
.s-orb{position:absolute;left:18cqw;top:12cqw;width:64cqw;height:64cqw;border-radius:50%;filter:blur(24px);
  background:radial-gradient(circle,rgba(192,132,252,.55),rgba(244,114,182,.28) 45%,transparent 70%);animation:pulse 7s ease-in-out infinite}
.s-shadow{position:absolute;left:10cqw;top:77cqw;width:70cqw;height:11cqw;border-radius:50%;filter:blur(10px);
  background:radial-gradient(ellipse,rgba(76,29,149,.3),transparent 70%);animation:shadow 9s ease-in-out infinite}
.s-slab{position:absolute;left:14cqw;top:26cqw;width:50cqw;height:50cqw;border-radius:5cqw;
  background:linear-gradient(135deg,rgba(255,255,255,.98),rgba(221,227,255,.85) 45%,rgba(244,114,182,.55) 78%,rgba(103,232,249,.55));
  border:1px solid rgba(255,255,255,.95);
  box-shadow:1px 1px 0 #EDEBFB,2px 2px 0 #E6E3F8,3px 3px 0 #DFDBF5,4px 4px 0 #D8D3F2,5px 5px 0 #D0CBEE,6px 6px 0 #C9C2EA,
    7px 7px 0 #C2BAE6,8px 8px 0 #BAB1E2,30px 36px 60px -10px rgba(124,58,237,.4);
  transform:perspective(900px) rotateX(58deg) rotateZ(-34deg);animation:bob 9s ease-in-out infinite;transition:transform .6s ease}
.scene:hover .s-slab{transform:perspective(900px) rotateX(52deg) rotateZ(-28deg)}
.s-ring{position:absolute;right:3cqw;top:8cqw;width:48cqw;height:48cqw;border-radius:50%;border:7cqw solid transparent;
  background:conic-gradient(from 40deg,rgba(255,255,255,.98),rgba(192,132,252,.62),rgba(103,232,249,.55),rgba(255,255,255,.95),
    rgba(244,114,182,.56),rgba(255,255,255,.98)) border-box;
  -webkit-mask:linear-gradient(#000 0 0) padding-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;
  mask:linear-gradient(#000 0 0) padding-box exclude,linear-gradient(#000 0 0);
  transform:perspective(900px) rotateX(64deg) rotateY(-8deg) rotateZ(-14deg);filter:drop-shadow(0 18px 24px rgba(124,58,237,.3));
  animation:bob 11s ease-in-out -3s infinite;transition:transform .6s ease}
.scene:hover .s-ring{transform:perspective(900px) rotateX(58deg) rotateY(-4deg) rotateZ(-6deg)}
.s-bead{position:absolute;left:47cqw;top:36cqw;width:3.2cqw;height:7cqw;border-radius:2cqw;
  background:linear-gradient(180deg,#FDE68A,#D97706);box-shadow:0 0 18px rgba(251,191,36,.7);animation:bob 6s ease-in-out -1s infinite}
.chip{position:absolute;display:flex;align-items:center;gap:.6em;padding:.7em 1.1em;border-radius:999px;white-space:nowrap;
  background:rgba(255,255,255,.72);-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.95);
  box-shadow:0 14px 30px -14px rgba(76,29,149,.35);font-size:max(10.5px,2.2cqw);font-weight:700;color:var(--ink);
  animation:bob 7s ease-in-out infinite;transition:all .3s}
.chip small{color:var(--muted);font-weight:600;font-size:1em}
.chip.c1{left:1cqw;top:6cqw}.chip.c2{right:0;top:58cqw}.chip.c3{left:10cqw;bottom:1cqw}
.dot-live{width:.7em;height:.7em;border-radius:50%;background:#22C55E;box-shadow:0 0 0 .35em rgba(34,197,94,.18)}

@keyframes bob{0%,100%{translate:0 0}50%{translate:0 -2.2cqw}}
@keyframes pulse{0%,100%{opacity:.85;scale:1}50%{opacity:1;scale:1.06}}
@keyframes shadow{0%,100%{scale:1;opacity:.9}50%{scale:.92;opacity:.65}}
@keyframes shimmer{from{background-position:0 0}to{background-position:100% 0}}
@keyframes spin{to{rotate:360deg}}

/* ---------- sections + glass cards ---------- */
.section{padding:56px 0 6px;scroll-margin-top:110px}
.kicker{font-size:.78rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--violet)}
.sec-title{margin:10px 0 8px;font-size:clamp(1.7rem,3.2vw,2.5rem);font-weight:800;letter-spacing:-.03em;line-height:1.1;color:var(--ink)}
.sec-sub{max-width:640px;line-height:1.65;color:var(--muted)}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:26px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:26px}
.gcard{padding:22px;border-radius:24px;background:var(--glass);-webkit-backdrop-filter:blur(16px);backdrop-filter:blur(16px);
  border:1px solid var(--glass-border);box-shadow:var(--shadow);transition:all .3s}
.gcard:hover{transform:translateY(-2px);box-shadow:0 20px 40px -20px rgba(76,29,149,.28)}
.ico{width:42px;height:42px;border-radius:14px;display:grid;place-items:center;font-size:1.2rem;margin-bottom:14px;
  background:linear-gradient(135deg,rgba(192,132,252,.35),rgba(103,232,249,.35));border:1px solid rgba(255,255,255,.95)}
.step-no{font-size:.72rem;font-weight:800;letter-spacing:.1em;color:var(--violet);margin-bottom:6px}
.g-title{font-weight:700;font-size:1.02rem;color:var(--ink);margin-bottom:6px}
.g-text{font-size:.9rem;line-height:1.6;color:var(--muted)}
.footer{margin-top:64px;padding:28px 0 0;border-top:1px solid rgba(15,23,42,.08);text-align:center;color:var(--muted);font-size:.85rem;scroll-margin-top:110px}
.footer b{color:var(--ink)}

/* ---------- streamlit widgets ---------- */
.st-key-research_card,.st-key-result_card{padding:28px;border-radius:28px;background:var(--glass);
  -webkit-backdrop-filter:blur(16px);backdrop-filter:blur(16px);border:1px solid var(--glass-border);
  box-shadow:var(--shadow),0 30px 60px -40px rgba(76,29,149,.35)}
.card-title{margin:6px 0 4px;font-size:1.5rem;font-weight:800;letter-spacing:-.02em;color:var(--ink)}
.card-sub{margin-bottom:6px;color:var(--muted);font-size:.95rem}
.hint{color:var(--muted);font-size:.84rem;padding-top:.7rem}

[data-baseweb="textarea"]{background:rgba(255,255,255,.75)!important;border:1px solid rgba(15,23,42,.08)!important;
  border-radius:18px!important;transition:all .3s}
[data-baseweb="textarea"]:focus-within{border-color:#A78BFA!important;box-shadow:0 0 0 4px rgba(167,139,250,.22)!important}
[data-baseweb="textarea"] textarea{background:transparent!important;color:var(--ink)!important;font-size:1rem;padding:14px 18px}
[data-baseweb="input"],[data-baseweb="base-input"]{border-radius:14px!important;background:rgba(255,255,255,.75)!important}

.stApp button p{color:inherit!important;font-weight:inherit}
.stApp [data-testid="stBaseButton-secondary"],.stApp [data-testid="stBaseButton-primary"]{
  border-radius:999px!important;font-weight:700;transition:all .3s cubic-bezier(.2,.8,.2,1)!important}
.stApp [data-testid="stBaseButton-secondary"]{background:rgba(255,255,255,.6)!important;border:1px solid rgba(255,255,255,.95)!important;
  color:var(--ink)!important;box-shadow:var(--shadow);-webkit-backdrop-filter:blur(12px);backdrop-filter:blur(12px)}
.stApp [data-testid="stBaseButton-secondary"]:hover{transform:translateY(-2px);background:#fff!important;
  border-color:rgba(124,58,237,.35)!important;color:var(--violet)!important;box-shadow:0 14px 28px -14px rgba(76,29,149,.3)}
.stApp [data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#0F172A,#4C1D95)!important;border:none!important;
  color:#fff!important;padding:.65rem 1.7rem;box-shadow:0 12px 26px -12px rgba(76,29,149,.8)}
.stApp [data-testid="stBaseButton-primary"]:hover{transform:translateY(-2px);box-shadow:0 18px 32px -12px rgba(124,58,237,.85)}
.st-key-examples [data-testid="stBaseButton-secondary"]{font-size:.8rem!important;padding:.3rem .9rem!important;min-height:0!important;font-weight:600}

[data-testid="stExpander"]{border:1px solid var(--glass-border)!important;border-radius:18px!important;
  background:rgba(255,255,255,.55)!important;-webkit-backdrop-filter:blur(16px);backdrop-filter:blur(16px);box-shadow:var(--shadow)}
[data-testid="stExpander"] summary{font-weight:700;color:var(--ink)}

.stTabs [data-baseweb="tab-list"]{gap:4px;padding:5px;border-radius:999px;width:fit-content;max-width:100%;overflow-x:auto;
  background:rgba(255,255,255,.6);border:1px solid var(--glass-border)}
.stTabs button[data-baseweb="tab"]{height:auto;padding:8px 18px;border-radius:999px;color:var(--muted);font-weight:700;transition:all .3s}
.stTabs button[data-baseweb="tab"]:hover{color:var(--ink);background:rgba(255,255,255,.8)}
.stTabs button[data-baseweb="tab"][aria-selected="true"]{background:linear-gradient(135deg,#0F172A,#4C1D95);color:#fff}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}

[data-testid="stMetric"]{padding:14px 18px;border-radius:20px;background:var(--glass);-webkit-backdrop-filter:blur(16px);
  backdrop-filter:blur(16px);border:1px solid var(--glass-border);box-shadow:var(--shadow);transition:all .3s}
[data-testid="stMetric"]:hover{transform:translateY(-2px)}
[data-testid="stMetricLabel"] p{color:var(--muted);font-weight:600}
[data-testid="stMetricValue"]{color:var(--ink);font-weight:800}
[data-testid="stCode"] pre{background:rgba(15,23,42,.92)!important;border-radius:16px;color:#E2E8F0}
[data-testid="stAlert"]{border-radius:16px;-webkit-backdrop-filter:blur(12px);backdrop-filter:blur(12px)}
[data-testid="stMarkdownContainer"] table{border-collapse:separate;border-spacing:0;border:1px solid rgba(15,23,42,.08);
  border-radius:14px;overflow:hidden;background:rgba(255,255,255,.6)}
[data-testid="stMarkdownContainer"] th{background:rgba(124,58,237,.08)!important}
[data-testid="stMarkdownContainer"] a{color:var(--violet)}
.st-key-result_card [data-testid="stMarkdownContainer"] :is(h1,h2,h3){letter-spacing:-.02em;color:var(--ink)}

/* ---------- live progress stepper ---------- */
.stepper{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:6px 0 2px}
.step{display:flex;align-items:center;gap:10px;padding:7px 16px 7px 7px;border-radius:999px;font-weight:700;font-size:.88rem;
  color:var(--muted);background:rgba(255,255,255,.75);border:1px solid rgba(15,23,42,.06);transition:all .3s}
.step .dot{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;font-size:.75rem;background:#EEF2F7;color:var(--muted)}
.step.active{color:var(--ink);border-color:rgba(124,58,237,.35);box-shadow:0 0 0 4px rgba(124,58,237,.1)}
.step.active .dot{background:conic-gradient(#7C3AED,#67E8F9,#F472B6,#7C3AED);animation:spin 1.1s linear infinite;color:transparent}
.step.active .dot::after{content:"";width:12px;height:12px;border-radius:50%;background:#fff}
.step.done{color:var(--ink)}
.step.done .dot{background:linear-gradient(135deg,#34D399,#10B981);color:#fff}
.step-bar{width:20px;height:2px;border-radius:2px;background:rgba(15,23,42,.12)}
.pass{margin-top:8px;font-size:.78rem;font-weight:600;color:var(--muted)}

/* ---------- responsive ---------- */
@media (max-width:960px){
  .hero{grid-template-columns:1fr;gap:12px;padding-top:150px;min-height:0}
  .grid-4{grid-template-columns:repeat(2,1fr)}.grid-3{grid-template-columns:1fr}
}
@media (max-width:820px){
  .nav{grid-template-columns:1fr auto;border-radius:28px;padding:10px 10px 10px 18px}
  .pill{grid-column:1/-1;order:3;overflow-x:auto;justify-content:flex-start}
  .hero{padding-top:176px}
}
@media (max-width:560px){
  .grid-4{grid-template-columns:1fr}
  .pill{justify-content:space-between}
  .pill a{padding:8px 10px;font-size:.82rem}
  [data-testid="stMainBlockContainer"]{padding:0 16px 56px!important}
  .st-key-research_card,.st-key-result_card{padding:18px}
  .btn{padding:12px 20px}
}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
"""


# ---------------------------------------------------------------------------
# Theme: HTML blocks
# ---------------------------------------------------------------------------
def html(markup: str, target: Any = None) -> None:
    """Render raw HTML. Lines are stripped/blank lines dropped so Markdown never
    mistakes indented HTML for a code block."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    (target or st).markdown(cleaned, unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def render_nav() -> None:
    links = "".join(f'<a href="{href}" target="_self">{escape(label)}</a>' for label, href in NAV_LINKS)
    html(f"""
    <div class="nav-wrap"><div class="nav">
      <a class="brand" href="#top" target="_self"><span class="gem"></span>{escape(BRAND_NAME)}</a>
      <nav class="pill">{links}</nav>
      <div class="nav-cta"><a class="btn btn-dark btn-sm" href="#research" target="_self">Get started</a></div>
    </div></div>
    """)


def render_hero() -> None:
    title = "".join(
        '<span class="ln"><span class="grad">%s</span></span>' % escape(line[1:-1])
        if line.startswith("{") and line.endswith("}")
        else '<span class="ln">%s</span>' % escape(line)
        for line in HERO_LINES
    )
    html(f"""
    <section class="hero" id="top">
      <div>
        <span class="badge"><span class="spark">✦</span>{escape(BADGE_TEXT)}</span>
        <div class="hero-title" role="heading" aria-level="1">{title}</div>
        <p class="hero-sub">{escape(HERO_SUBTITLE)}</p>
        <div class="hero-cta">
          <a class="btn btn-dark" href="#research" target="_self">Start researching →</a>
          <a class="btn btn-glass" href="#how-it-works" target="_self">See how it works</a>
        </div>
        <div class="trust"><span>LangGraph</span><span>FAISS memory</span><span>Inline citations</span></div>
      </div>
      <div class="scene" aria-hidden="true">
        <div class="s-orb"></div><div class="s-shadow"></div>
        <div class="s-slab"></div><div class="s-ring"></div><div class="s-bead"></div>
        <div class="chip c1" style="animation-delay:-1s"><span class="dot-live"></span>Planner <small>3–5 sub-questions</small></div>
        <div class="chip c2" style="animation-delay:-3s">🔎 FAISS <small>evidence indexed</small></div>
        <div class="chip c3" style="animation-delay:-5s">✓ Critic <small>loops until sufficient</small></div>
      </div>
    </section>
    """)


def render_pipeline() -> None:
    html("""
    <section class="section" id="how-it-works">
      <div class="kicker">How it works</div>
      <div class="sec-title" role="heading" aria-level="2">Four agents. One autonomous loop.</div>
      <p class="sec-sub">Each question flows through a LangGraph state machine that plans, gathers, critiques and writes.</p>
      <div class="grid-4">
        <div class="gcard"><div class="ico">🧭</div><div class="step-no">STEP 01</div><div class="g-title">Planner</div>
          <div class="g-text">Splits your question into 3–5 focused sub-questions and search intents.</div></div>
        <div class="gcard"><div class="ico">🔎</div><div class="step-no">STEP 02</div><div class="g-title">Researcher</div>
          <div class="g-text">Searches the web, scrapes and cleans pages, and embeds excerpts into FAISS.</div></div>
        <div class="gcard"><div class="ico">🧪</div><div class="step-no">STEP 03</div><div class="g-title">Evaluator</div>
          <div class="g-text">Scores evidence per sub-question and loops back with sharper queries when gaps remain.</div></div>
        <div class="gcard"><div class="ico">✍️</div><div class="step-no">STEP 04</div><div class="g-title">Writer</div>
          <div class="g-text">Synthesizes summary, analysis, findings and comparison tables with citations.</div></div>
      </div>
    </section>
    <section class="section" id="features">
      <div class="kicker">Features</div>
      <div class="sec-title" role="heading" aria-level="2">Built to be trusted, not just fast.</div>
      <div class="grid-3">
        <div class="gcard"><div class="ico">🔗</div><div class="g-title">Grounded citations</div>
          <div class="g-text">Every claim links to a scraped source with inline [n] citations and a reference list.</div></div>
        <div class="gcard"><div class="ico">♻️</div><div class="g-title">Self-critiquing loop</div>
          <div class="g-text">A critic reviews the evidence and sends the researcher back, up to your loop limit.</div></div>
        <div class="gcard"><div class="ico">🧠</div><div class="g-title">Persistent FAISS memory</div>
          <div class="g-text">Chunks are embedded per session so the writer retrieves only the most relevant evidence.</div></div>
      </div>
    </section>
    """)


def render_footer() -> None:
    html(f"""
    <div class="footer" id="about"><b>{escape(BRAND_NAME)}</b> · Autonomous AI Research Agent<br>
    Built with LangGraph · LangChain · FAISS · FastAPI · Streamlit</div>
    """)


# ---------------------------------------------------------------------------
# PDF export (fpdf2, core fonts only -> text is sanitised to latin-1)
# ---------------------------------------------------------------------------
_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
    "—": "-", "…": "...", "•": "*", " ": " ", "→": "->",
    "≈": "~", "≥": ">=", "≤": "<=", "×": "x",
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
    """Live pipeline stepper: done (green check), active (spinning ring), idle (number)."""
    parts: List[str] = []
    for idx, (key, label) in enumerate(STEPS, 1):
        if key in done:
            state, dot = "done", "✓"
        elif key == active:
            state, dot = "active", ""
        else:
            state, dot = "", str(idx)
        if idx > 1:
            parts.append('<span class="step-bar"></span>')
        parts.append(f'<div class="step {state}"><span class="dot">{dot}</span>{escape(label)}</div>')
    html(
        f'<div class="stepper">{"".join(parts)}</div>'
        f'<div class="pass">Research pass {max(loop, 0)}</div>',
        target=placeholder,
    )


def format_log_line(log: Dict[str, Any]) -> str:
    ts = str(log.get("timestamp", ""))[11:19]
    icon = LEVEL_ICONS.get(log.get("level", "info"), "•")
    loop = log.get("loop", 0)
    loop_tag = f" #{loop}" if loop else ""
    return f"{ts}  {icon} [{log.get('step', '')}{loop_tag}] {log.get('message', '')}"


def check_backend(api_url: str) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreachable", "llm_problem": str(exc)}


def run_research_stream(api_url: str, query: str, max_loops: int) -> None:
    """Call the streaming endpoint and render live progress. Stores result in session_state."""
    st.session_state.pop("result", None)
    st.session_state.pop("pdf_bytes", None)
    st.session_state.pop("pdf_error", None)

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


def _set_query(text: str) -> None:
    st.session_state["query_input"] = text


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{BRAND_NAME} · Autonomous AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()
render_nav()
render_hero()

with st.container(key="research_card"):
    html("""
    <div class="kicker" id="research" style="scroll-margin-top:110px">Start a research run</div>
    <div class="card-title">What should the agent investigate?</div>
    <div class="card-sub">Ask a complex question. The agents plan, search, critique and write a cited report.</div>
    """)
    query = st.text_area(
        "Research question",
        key="query_input",
        height=120,
        label_visibility="collapsed",
        placeholder="e.g. Compare the leading approaches to grid-scale energy storage and their economics…",
    )

    with st.container(key="examples"):
        example_cols = st.columns(len(EXAMPLES))
        for i, (label, text) in enumerate(EXAMPLES):
            example_cols[i].button(label, key=f"ex_{i}", on_click=_set_query, args=(text,), use_container_width=True)

    with st.expander("⚙️ Advanced settings"):
        s1, s2, s3 = st.columns([3, 2, 1.3])
        api_url = s1.text_input("Backend URL", value=DEFAULT_API_URL)
        max_loops = s2.slider(
            "Max re-search loops", 0, 5, 3,
            help="How many times the critic may send the researcher back for more evidence.",
        )
        s3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if s3.button("Check backend", use_container_width=True):
            health = check_backend(api_url)
            if health.get("status") == "ok":
                st.success(f"Backend OK · LLM: {health.get('llm_provider')} · Search: {health.get('search')}")
            elif health.get("status") == "degraded":
                st.warning(f"Backend up, but: {health.get('llm_problem')}")
            else:
                st.error(f"Backend unreachable: {health.get('llm_problem')}")

    hint_col, run_col = st.columns([3, 1.2])
    hint_col.markdown('<div class="hint">Typical run: 1–3 minutes depending on loops and sources.</div>', unsafe_allow_html=True)
    run_clicked = run_col.button("Run research →", type="primary", key="run", use_container_width=True)

if run_clicked:
    if len((query or "").strip()) < 5:
        st.warning("Please enter a research question (at least 5 characters).")
    else:
        run_research_stream(api_url, query.strip(), max_loops)

if "result" in st.session_state:
    with st.container(key="result_card"):
        render_result(st.session_state["result"])

render_pipeline()
render_footer()
