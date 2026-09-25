"""Pydantic AI agent: retrieval tools over the paper index, structured output with citations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from .config import MODEL
from .store import PaperStore


@dataclass
class Deps:
    store: PaperStore
    since: date | None = None


class Citation(BaseModel):
    arxiv_id: str
    title: str
    pages: list[int] = Field(description="1-indexed PDF page numbers that support the claim")


class Finding(BaseModel):
    claim: str = Field(description="One concrete finding, method, or result")
    citations: list[Citation] = Field(min_length=1)


class Answer(BaseModel):
    summary: str = Field(description="Direct answer in a few sentences")
    findings: list[Finding]
    gaps: list[str] = Field(
        default_factory=list,
        description="Things the indexed papers do not cover, if the question asked for them",
    )


class DigestItem(BaseModel):
    arxiv_id: str
    title: str
    published: str
    one_liner: str = Field(description="What the paper does, in one sentence")
    why_it_matters: str = Field(description="Why a reader tracking this area should care")
    key_pages: list[int] = Field(description="Pages with the main method or result")


class Digest(BaseModel):
    period: str
    themes: list[str] = Field(description="2-5 cross-paper themes seen this period")
    items: list[DigestItem]


INSTRUCTIONS = """
You answer questions about a local library of research papers. You can only use
what the tools return; never rely on prior knowledge about a paper.

Workflow:
- Call search_chunks with several queries using different technical terms.
  Retrieval is keyword-based (BM25), so use the words a paper would use:
  method names, dataset names, metrics, not paraphrases.
- When a chunk looks relevant, call read_pages on that paper for the pages
  around it to get full context before citing it.
- Every claim needs at least one citation with the arXiv id and the page
  number(s) the supporting text came from. Cite pages you actually read.
- If the library does not answer the question, say so in `gaps` rather than
  guessing.
"""


def _fmt(chunks) -> str:
    if not chunks:
        return "No results."
    return "\n\n".join(
        f"[{c.arxiv_id} | {c.title} | {c.published} | page {c.page}]\n{c.text}"
        for c in chunks
    )


def build_agent(output_type):
    agent = Agent(
        MODEL,
        deps_type=Deps,
        output_type=output_type,
        instructions=INSTRUCTIONS,
        retries=2,
    )

    @agent.tool
    def search_chunks(ctx: RunContext[Deps], query: str, k: int = 8) -> str:
        """Keyword (BM25) search over all indexed paper pages. Returns chunks tagged with arXiv id and page."""
        return _fmt(ctx.deps.store.search(query, k=k, since=ctx.deps.since))

    @agent.tool
    def list_papers(ctx: RunContext[Deps]) -> str:
        """List every indexed paper (id, title, published date, page count), newest first."""
        papers = ctx.deps.store.list_papers(since=ctx.deps.since)
        if not papers:
            return "No papers indexed for this period."
        return "\n".join(
            f"{p.arxiv_id} | {p.title} | {p.published} | {p.pages} pages" for p in papers
        )

    @agent.tool
    def read_pages(ctx: RunContext[Deps], arxiv_id: str, pages: list[int]) -> str:
        """Read the full text of specific pages (1-indexed) of one paper."""
        return _fmt(ctx.deps.store.get_pages(arxiv_id, pages))

    return agent


ask_agent = build_agent(Answer)
digest_agent = build_agent(Digest)


def ask(question: str, store: PaperStore, since: date | None = None) -> Answer:
    result = ask_agent.run_sync(question, deps=Deps(store=store, since=since))
    return result.output


def digest(store: PaperStore, since: date) -> Digest:
    prompt = (
        f"Produce a digest of every paper published on or after {since.isoformat()}. "
        "Call list_papers first, then read_pages on each paper's opening pages "
        "(abstract, intro, and the main results) before writing its entry. "
        f"Set period to 'since {since.isoformat()}'."
    )
    result = digest_agent.run_sync(prompt, deps=Deps(store=store, since=since))
    return result.output
