"""Pydantic AI agent: retrieval tools over the paper index, structured output with citations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import cache

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from .config import MODEL
from .store import PaperStore


@dataclass
class Deps:
    store: PaperStore
    since: date | None = None
    # arxiv_id -> pages whose text was actually returned to the model by read_pages.
    read: dict[str, set[int]] = field(default_factory=dict)


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


class Themes(BaseModel):
    themes: list[str] = Field(description="2-5 cross-paper themes seen this period")


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


def _unread(read: dict[str, set[int]], refs: list[tuple[str, list[int]]]) -> list[str]:
    """Return 'id p.N' for every cited page that was never returned by read_pages."""
    bad = []
    for arxiv_id, pages in refs:
        seen = read.get(arxiv_id, set())
        bad += [f"{arxiv_id} p.{p}" for p in pages if p not in seen]
    return bad


def _retry_unread(bad: list[str]) -> None:
    if bad:
        raise ModelRetry(
            "These citations point to pages you did not read: "
            + ", ".join(bad)
            + ". Call read_pages on them first, or drop the claim."
        )


def build_agent(output_type, instructions: str = INSTRUCTIONS):
    agent = Agent(
        MODEL,
        deps_type=Deps,
        output_type=output_type,
        instructions=instructions,
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
        chunks = ctx.deps.store.get_pages(arxiv_id, pages)
        for c in chunks:
            ctx.deps.read.setdefault(c.arxiv_id, set()).add(c.page)
        return _fmt(chunks)

    if output_type is Answer:

        @agent.output_validator
        def validate_citations(ctx: RunContext[Deps], out: Answer) -> Answer:
            refs = [(c.arxiv_id, c.pages) for f in out.findings for c in f.citations]
            _retry_unread(_unread(ctx.deps.read, refs))
            return out

    if output_type is DigestItem:

        @agent.output_validator
        def validate_key_pages(ctx: RunContext[Deps], out: DigestItem) -> DigestItem:
            if out.arxiv_id not in ctx.deps.read:
                raise ModelRetry(f"You did not read any pages of {out.arxiv_id}. Call read_pages first.")
            _retry_unread(_unread(ctx.deps.read, [(out.arxiv_id, out.key_pages)]))
            return out

    return agent


@cache
def get_agent(output_type):
    """Construct (once, on first use) the agent for a given output type."""
    return build_agent(output_type)


def _run(output_type, prompt: str, deps: Deps, model=None):
    agent = get_agent(output_type)
    if model is None:
        return agent.run_sync(prompt, deps=deps).output
    with agent.override(model=model):
        return agent.run_sync(prompt, deps=deps).output


def ask(question: str, store: PaperStore, since: date | None = None, *, model=None) -> Answer:
    return _run(Answer, question, Deps(store=store, since=since), model)


def digest(store: PaperStore, since: date, *, model=None) -> Digest:
    """One bounded model run per paper, then one run to synthesize themes."""
    period = f"since {since.isoformat()}"
    items: list[DigestItem] = []
    for p in store.list_papers(since=since):
        prompt = (
            f"Write a digest entry for the paper {p.arxiv_id} ({p.title!r}, published {p.published}, "
            f"{p.pages} pages). Call read_pages on its opening pages (abstract, intro) and the main "
            f"results before writing. Set arxiv_id to {p.arxiv_id!r}, title to {p.title!r}, "
            f"and published to {p.published!r}."
        )
        items.append(_run(DigestItem, prompt, Deps(store=store, since=since), model))
    if not items:
        return Digest(period=period, themes=[], items=[])
    listing = "\n".join(f"- {it.arxiv_id} | {it.title}: {it.one_liner}" for it in items)
    prompt = (
        f"Here are one-line summaries of every paper published {period}:\n{listing}\n\n"
        "Name 2-5 cross-paper themes. Use the tools only if a summary is unclear."
    )
    themes = _run(Themes, prompt, Deps(store=store, since=since), model)
    return Digest(period=period, themes=themes.themes, items=items)
