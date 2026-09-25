"""Fetch papers from arXiv, extract page text, chunk, and index.

Two sources:
- Category feed (rss.arxiv.org): today's announcements for a category such as
  cs.CL, optionally filtered by keywords in the title/abstract. Reliable.
- Search API (export.arxiv.org/api): full arXiv search syntax. Some networks
  get 406 responses from its CDN; if that happens, use the category feed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import feedparser
import httpx
from pypdf import PdfReader

from .config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, PDF_DIR
from .store import Chunk, PaperStore

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_FEED = "https://rss.arxiv.org/atom/{category}"
HEADERS = {"User-Agent": "paper-digest/0.1 (https://github.com/tdiprima/paper-digest)"}


@dataclass
class ArxivEntry:
    arxiv_id: str
    title: str
    published: str
    pdf_url: str
    abstract: str = ""


def _clean_id(raw: str) -> str:
    raw = raw.rsplit(":", 1)[-1].rsplit("/abs/", 1)[-1]
    return re.sub(r"v\d+$", "", raw)


def fetch_category(category: str, match: list[str] | None = None) -> list[ArxivEntry]:
    """Today's announcements for an arXiv category, filtered by keywords (any match, case-insensitive)."""
    r = httpx.get(ARXIV_FEED.format(category=category), headers=HEADERS, timeout=30)
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    pats = [re.compile(re.escape(m), re.I) for m in (match or [])]
    out: list[ArxivEntry] = []
    for e in feed.entries:
        if e.get("arxiv_announce_type", "new") not in ("new", "cross"):
            continue  # skip replaced versions
        title = " ".join(e.title.split())
        abstract = e.get("summary", "").split("Abstract:", 1)[-1].strip()
        if pats and not any(p.search(title) or p.search(abstract) for p in pats):
            continue
        aid = _clean_id(e.id)
        out.append(
            ArxivEntry(
                arxiv_id=aid,
                title=title,
                published=e.published[:10],
                pdf_url=f"https://arxiv.org/pdf/{aid}",
                abstract=abstract,
            )
        )
    return out


def search_arxiv(query: str, max_results: int = 20) -> list[ArxivEntry]:
    """Query the arXiv search API. `query` uses arXiv syntax, e.g. 'cat:cs.CL AND ti:retrieval'."""
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = httpx.get(ARXIV_API, params=params, headers=HEADERS, timeout=30, follow_redirects=True)
    if r.status_code == 406:
        raise RuntimeError(
            "arXiv search API refused the request (406). Use the category feed instead: "
            "paper-digest add cs.CL --match retrieval"
        )
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    entries: list[ArxivEntry] = []
    for e in feed.entries:
        aid = _clean_id(e.id)
        pdf_url = next(
            (l.href for l in e.links if l.get("type") == "application/pdf"),
            f"https://arxiv.org/pdf/{aid}",
        )
        entries.append(
            ArxivEntry(
                arxiv_id=aid,
                title=" ".join(e.title.split()),
                published=e.published[:10],
                pdf_url=pdf_url,
                abstract=e.get("summary", ""),
            )
        )
    return entries


def download_pdf(entry: ArxivEntry) -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    path = PDF_DIR / f"{entry.arxiv_id.replace('/', '_')}.pdf"
    if not path.exists():
        with httpx.stream("GET", entry.pdf_url, headers=HEADERS, timeout=60, follow_redirects=True) as r:
            r.raise_for_status()
            with path.open("wb") as f:
                for part in r.iter_bytes():
                    f.write(part)
    return path


def chunk_words(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    return [" ".join(words[i : i + CHUNK_WORDS]) for i in range(0, len(words), step)]


def index_pdf(entry: ArxivEntry, path: Path, store: PaperStore) -> int:
    reader = PdfReader(str(path))
    chunks: list[Chunk] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for piece in chunk_words(text):
            chunks.append(
                Chunk(
                    arxiv_id=entry.arxiv_id,
                    title=entry.title,
                    published=entry.published,
                    page=page_no,
                    text=piece,
                )
            )
    store.add_chunks(chunks, pages=len(reader.pages))
    return len(chunks)


def ingest(entries: list[ArxivEntry], store: PaperStore, log=print) -> int:
    """Download and index entries not already in the store. Returns count newly indexed."""
    new = 0
    for entry in entries:
        if store.has_paper(entry.arxiv_id):
            log(f"skip  {entry.arxiv_id}  {entry.title[:70]}")
            continue
        try:
            path = download_pdf(entry)
            n = index_pdf(entry, path, store)
        except Exception as exc:  # network or PDF parse failure; keep going
            log(f"FAIL  {entry.arxiv_id}  {exc}")
            continue
        log(f"added {entry.arxiv_id}  {entry.title[:70]}  ({n} chunks)")
        new += 1
    return new
