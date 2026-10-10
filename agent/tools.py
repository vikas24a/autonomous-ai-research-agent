"""Tools: web search (Tavily -> DuckDuckGo fallback), URL scraping, FAISS memory.

Nothing in this module raises on network/search failures; errors are collected
and returned so the graph nodes can log them and keep going.
"""

import hashlib
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 ResearchAgent/1.0"
)
SKIP_EXTENSIONS = (
    ".pdf", ".zip", ".gz", ".tar", ".rar", ".7z", ".exe", ".dmg", ".mp3",
    ".mp4", ".avi", ".mov", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".pptx", ".docx", ".xlsx",
)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class HashingEmbeddings(Embeddings):
    """Dependency-free bag-of-words/bigram hashing embeddings (last-resort fallback)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"\w+", text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feat in features:
            h = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def _build_embeddings(provider: str) -> Embeddings:
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    if provider == "fastembed":
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        return FastEmbedEmbeddings(
            model_name=os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
        )
    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=os.getenv(
                "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
    if provider == "hash":
        return HashingEmbeddings()
    raise ValueError(f"Unknown embedding provider: {provider}")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a working embeddings backend, trying providers in priority order."""
    configured = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
    if configured == "auto":
        candidates = []
        if os.getenv("OPENAI_API_KEY"):
            candidates.append("openai")
        candidates += ["fastembed", "hash"]
    else:
        candidates = [configured, "fastembed", "hash"]

    seen = set()
    for provider in candidates:
        if provider in seen:
            continue
        seen.add(provider)
        try:
            emb = _build_embeddings(provider)
            emb.embed_query("healthcheck")  # fail fast if the backend is unusable
            logger.info("Using embedding provider: %s", provider)
            return emb
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding provider '%s' unavailable: %s", provider, exc)
    return HashingEmbeddings()


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
def _tavily_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    resp = client.search(
        query=query,
        max_results=max_results,
        search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "advanced"),
    )
    return [
        {
            "title": r.get("title") or r.get("url", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content") or "",
            "provider": "tavily",
        }
        for r in resp.get("results", [])
        if r.get("url")
    ]


def _ddg_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    try:
        from ddgs import DDGS  # newer package name
    except ImportError:
        from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        raw = ddgs.text(query, max_results=max_results) or []
    return [
        {
            "title": r.get("title") or r.get("href", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body") or "",
            "provider": "duckduckgo",
        }
        for r in raw
        if r.get("href")
    ]


def web_search(query: str, max_results: Optional[int] = None) -> Dict[str, Any]:
    """Search the web. Tavily first (with retries), then DuckDuckGo as fallback.

    Returns {"query", "provider", "results": [...], "errors": [...]}. Never raises.
    """
    max_results = max_results or _env_int("SEARCH_MAX_RESULTS", 5)
    retries = max(1, _env_int("SEARCH_RETRIES", 2))
    errors: List[str] = []
    query = (query or "").strip()
    if not query:
        return {"query": query, "provider": None, "results": [], "errors": ["Empty query"]}

    providers = []
    if os.getenv("TAVILY_API_KEY"):
        providers.append(("tavily", _tavily_search))
    providers.append(("duckduckgo", _ddg_search))

    for name, fn in providers:
        for attempt in range(1, retries + 1):
            try:
                results = fn(query, max_results)
                if results:
                    return {"query": query, "provider": name, "results": results, "errors": errors}
                errors.append(f"{name}: no results for '{query}'")
                break  # empty result is not retried; move to next provider
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name} attempt {attempt}/{retries} failed: {exc}")
                time.sleep(min(2 ** attempt, 6))
    return {"query": query, "provider": None, "results": [], "errors": errors}


# ---------------------------------------------------------------------------
# URL fetching / HTML cleaning
# ---------------------------------------------------------------------------
def clean_html(html: Any) -> Dict[str, str]:
    """Strip boilerplate from HTML and return {"title", "text"}."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(
        ["script", "style", "noscript", "nav", "footer", "header", "aside",
         "form", "iframe", "svg", "canvas", "button", "template"]
    ):
        tag.decompose()

    def _extract(root: Any) -> str:
        raw = root.get_text(separator="\n")
        lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in raw.splitlines()]
        lines = [ln for ln in lines if len(ln.split()) >= 3]  # drop menu crumbs
        return "\n".join(lines)

    text = ""
    for root in (soup.find("article"), soup.find("main"), soup.body, soup):
        if root is None:
            continue
        text = _extract(root)
        if len(text) >= 300:
            break
    return {"title": title, "text": text}


def fetch_url(
    url: str, timeout: Optional[int] = None, max_chars: Optional[int] = None
) -> Dict[str, Any]:
    """Download a page and return cleaned text. Never raises.

    Returns {"url", "title", "text", "ok", "error"}.
    """
    timeout = timeout or _env_int("FETCH_TIMEOUT", 12)
    max_chars = max_chars or _env_int("MAX_PAGE_CHARS", 20000)
    max_bytes = _env_int("MAX_PAGE_BYTES", 2_000_000)
    result: Dict[str, Any] = {"url": url, "title": "", "text": "", "ok": False, "error": None}

    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        result["error"] = "Unsupported or invalid URL"
        return result
    if parsed.path.lower().endswith(SKIP_EXTENSIONS):
        result["error"] = "Unsupported file type"
        return result

    try:
        with requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"},
            timeout=timeout,
            stream=True,
            allow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "").lower()
            if ctype and not any(t in ctype for t in ("html", "text/plain", "xml")):
                result["error"] = f"Unsupported content type: {ctype}"
                return result
            data = b""
            for chunk in resp.iter_content(chunk_size=65536):
                data += chunk
                if len(data) >= max_bytes:
                    break

        if "text/plain" in ctype:
            text = data.decode("utf-8", errors="ignore")
            title = parsed.netloc
        else:
            cleaned = clean_html(data)
            text, title = cleaned["text"], cleaned["title"]

        text = text.strip()
        if not text:
            result["error"] = "No readable text extracted"
            return result
        result.update(title=title or parsed.netloc, text=text[:max_chars], ok=True)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# FAISS research memory
# ---------------------------------------------------------------------------
def _safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", session_id or "")
    return cleaned[:64] or "default"


class ResearchMemory:
    """Per-session FAISS index of scraped chunks, persisted to disk after each write."""

    def __init__(self, session_id: str) -> None:
        self.session_id = _safe_session_id(session_id)
        self._lock = threading.RLock()
        self._store: Optional[FAISS] = None
        self._seen_chunks: set = set()
        self._urls: set = set()
        self._dir = Path(os.getenv("FAISS_INDEX_DIR", "./faiss_indexes")) / self.session_id
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=_env_int("CHUNK_SIZE", 900),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._try_load()

    # -- persistence -------------------------------------------------------
    def _try_load(self) -> None:
        if not (self._dir / "index.faiss").exists():
            return
        try:
            self._store = FAISS.load_local(
                str(self._dir), get_embeddings(), allow_dangerous_deserialization=True
            )
            for doc in self._store.docstore._dict.values():  # noqa: SLF001
                self._seen_chunks.add(doc.metadata.get("chunk_hash"))
                if doc.metadata.get("url"):
                    self._urls.add(doc.metadata["url"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load FAISS index for %s: %s", self.session_id, exc)
            self._store = None

    def _persist(self) -> None:
        if self._store is None:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self._dir))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist FAISS index: %s", exc)

    # -- writes ------------------------------------------------------------
    def has_url(self, url: str) -> bool:
        with self._lock:
            return url in self._urls

    def add_page(
        self, url: str, title: str, text: str, sub_question: str = "", query: str = ""
    ) -> int:
        """Chunk and embed a page. Returns the number of NEW chunks added."""
        max_chunks = _env_int("MAX_CHUNKS_PER_PAGE", 25)
        chunks = self._splitter.split_text(text)[:max_chunks]
        docs: List[Document] = []
        hashes: List[str] = []
        with self._lock:
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if len(chunk) < 40:
                    continue
                digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
                if digest in self._seen_chunks or digest in hashes:
                    continue
                hashes.append(digest)
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "url": url, "title": title, "sub_question": sub_question,
                            "query": query, "chunk_index": i, "chunk_hash": digest,
                        },
                    )
                )
            if not docs:
                self._urls.add(url)
                return 0
            if self._store is None:
                self._store = FAISS.from_documents(docs, get_embeddings())
            else:
                self._store.add_documents(docs)
            self._seen_chunks.update(hashes)
            self._urls.add(url)
            self._persist()
        return len(docs)

    # -- reads -------------------------------------------------------------
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the top-k most relevant chunks for a query (lower score = closer)."""
        k = k or _env_int("RETRIEVAL_K", 6)
        with self._lock:
            if self._store is None:
                return []
            try:
                hits = self._store.similarity_search_with_score(query, k=k)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FAISS retrieval failed: %s", exc)
                return []
        return [
            {
                "content": doc.page_content,
                "url": doc.metadata.get("url", ""),
                "title": doc.metadata.get("title", ""),
                "sub_question": doc.metadata.get("sub_question", ""),
                "score": float(score),
            }
            for doc, score in hits
        ]

    def stats(self) -> Dict[str, int]:
        with self._lock:
            chunks = int(self._store.index.ntotal) if self._store is not None else 0
            return {"chunks": chunks, "urls": len(self._urls)}


_MEMORIES: Dict[str, ResearchMemory] = {}
_MEMORIES_LOCK = threading.Lock()


def get_memory(session_id: str) -> ResearchMemory:
    """Get (or create) the FAISS memory for a session."""
    key = _safe_session_id(session_id)
    with _MEMORIES_LOCK:
        if key not in _MEMORIES:
            _MEMORIES[key] = ResearchMemory(key)
        return _MEMORIES[key]


def drop_memory(session_id: str) -> None:
    """Release a session's in-process index (the on-disk copy is kept)."""
    with _MEMORIES_LOCK:
        _MEMORIES.pop(_safe_session_id(session_id), None)


# ---------------------------------------------------------------------------
# Composite operation used by the researcher node
# ---------------------------------------------------------------------------
def search_and_index(
    session_id: str,
    query: str,
    sub_question: str = "",
    max_results: Optional[int] = None,
    max_fetch: Optional[int] = None,
) -> Dict[str, Any]:
    """Search, scrape the top pages in parallel, and index chunks into FAISS.

    Pages that cannot be scraped fall back to the search snippet. Never raises.
    """
    max_fetch = max_fetch or _env_int("FETCH_MAX_PAGES_PER_QUERY", 4)
    min_chars = _env_int("MIN_PAGE_CHARS", 200)
    memory = get_memory(session_id)

    summary: Dict[str, Any] = {
        "query": query, "sub_question": sub_question, "provider": None,
        "results_found": 0, "pages_scraped": 0, "snippet_fallbacks": 0,
        "chunks_added": 0, "sources": [], "errors": [],
    }

    search = web_search(query, max_results)
    summary["provider"] = search["provider"]
    summary["errors"].extend(search["errors"])
    results = search["results"]
    summary["results_found"] = len(results)
    if not results:
        return summary

    # Scrape the top pages concurrently (skip URLs already indexed).
    to_fetch = [r for r in results[:max_fetch] if not memory.has_url(r["url"])]
    pages: Dict[str, Dict[str, Any]] = {}
    if to_fetch:
        workers = max(1, min(len(to_fetch), _env_int("FETCH_WORKERS", 4)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_url, r["url"]): r["url"] for r in to_fetch}
            for fut in as_completed(futures):
                url = futures[fut]
                try:
                    pages[url] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    pages[url] = {"ok": False, "error": str(exc), "text": "", "title": ""}

    # Index sequentially (embedding calls + FAISS writes).
    for r in results:
        url, snippet = r["url"], (r.get("snippet") or "").strip()
        title = r.get("title") or url
        if memory.has_url(url):
            summary["sources"].append(
                {"url": url, "title": title, "snippet": snippet[:300],
                 "sub_question": sub_question, "query": query}
            )
            continue

        page = pages.get(url)
        text, used_snippet = "", False
        if page and page.get("ok") and len(page.get("text", "")) >= min_chars:
            text = page["text"]
            title = page.get("title") or title
            summary["pages_scraped"] += 1
        elif len(snippet) >= 40:
            text, used_snippet = snippet, True
            if page and page.get("error"):
                summary["errors"].append(f"Scrape failed for {url}: {page['error']}")
        else:
            if page and page.get("error"):
                summary["errors"].append(f"Scrape failed for {url}: {page['error']}")
            continue

        try:
            added = memory.add_page(url, title, text, sub_question, query)
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"Indexing failed for {url}: {exc}")
            continue

        if used_snippet:
            summary["snippet_fallbacks"] += 1
        summary["chunks_added"] += added
        summary["sources"].append(
            {"url": url, "title": title, "snippet": (snippet or text)[:300],
             "sub_question": sub_question, "query": query}
        )
    return summary


# ---------------------------------------------------------------------------
# LangChain tool definitions (usable by any tool-calling agent)
# ---------------------------------------------------------------------------
@tool
def search_web_tool(query: str, max_results: int = 5) -> str:
    """Search the web (Tavily, falling back to DuckDuckGo) and return titles, URLs and snippets."""
    out = web_search(query, max_results)
    if not out["results"]:
        return "No results. " + "; ".join(out["errors"])
    return "\n\n".join(
        f"[{i}] {r['title']}\n{r['url']}\n{r['snippet'][:400]}"
        for i, r in enumerate(out["results"], 1)
    )


@tool
def fetch_url_tool(url: str) -> str:
    """Download a web page and return its cleaned main text (truncated)."""
    page = fetch_url(url)
    if not page["ok"]:
        return f"Failed to fetch {url}: {page['error']}"
    return f"{page['title']}\n\n{page['text'][:4000]}"


@tool
def retrieve_evidence_tool(session_id: str, query: str, k: int = 6) -> str:
    """Retrieve the most relevant previously indexed chunks (with source URLs) from FAISS."""
    hits = get_memory(session_id).retrieve(query, k)
    if not hits:
        return "No indexed evidence yet."
    return "\n\n".join(f"({h['url']})\n{h['content']}" for h in hits)


ALL_TOOLS = [search_web_tool, fetch_url_tool, retrieve_evidence_tool]
