from __future__ import annotations

import re
from datetime import date, timedelta

import typer

from . import agent as agent_mod
from .ingest import fetch_category, ingest, search_arxiv
from .store import PaperStore

app = typer.Typer(add_completion=False, help="Index arXiv PDFs and ask questions with citations.")


def _since(spec: str | None) -> date | None:
    """Accept '7d', '3w', '2m', or an ISO date."""
    if not spec:
        return None
    m = re.fullmatch(r"(\d+)([dwm])", spec)
    if m:
        n, unit = int(m[1]), m[2]
        days = n * {"d": 1, "w": 7, "m": 30}[unit]
        return date.today() - timedelta(days=days)
    return date.fromisoformat(spec)


@app.command()
def add(
    category: str = typer.Argument(None, help="arXiv category feed, e.g. cs.CL, cs.IR, stat.ML"),
    match: list[str] = typer.Option(None, "--match", "-m", help="Keep papers whose title/abstract contains this (repeatable)"),
    query: str = typer.Option(None, "--query", "-q", help="arXiv search-API query instead of a category feed"),
    max_results: int = typer.Option(20, "--max", help="Max papers to fetch"),
):
    """Fetch new papers from arXiv, download PDFs, and index them."""
    if query:
        entries = search_arxiv(query, max_results)
    elif category:
        entries = fetch_category(category, match)[:max_results]
    else:
        raise typer.BadParameter("Give a category (e.g. cs.CL) or --query.")
    typer.echo(f"{len(entries)} candidate paper(s)")
    n = ingest(entries, PaperStore(), log=typer.echo)
    typer.echo(f"\n{n} new paper(s) indexed.")


@app.command()
def papers(since: str = typer.Option(None, help="e.g. 30d, 2w, 2026-01-01")):
    """List indexed papers."""
    for p in PaperStore().list_papers(since=_since(since)):
        typer.echo(f"{p.published}  {p.arxiv_id:<12}  {p.title}")


@app.command()
def ask(
    question: str,
    since: str = typer.Option(None, help="Restrict to papers published since, e.g. 30d"),
):
    """Ask a question; the answer cites arXiv id and page numbers."""
    a = agent_mod.ask(question, PaperStore(), since=_since(since))
    typer.echo(a.summary + "\n")
    for f in a.findings:
        cites = "; ".join(
            f"{c.arxiv_id} p.{','.join(map(str, c.pages))}" for c in f.citations
        )
        typer.echo(f"- {f.claim}  [{cites}]")
    if a.gaps:
        typer.echo("\nNot covered by the library:")
        for g in a.gaps:
            typer.echo(f"- {g}")


@app.command()
def digest(since: str = typer.Option("7d", help="Period to cover, e.g. 7d, 1m")):
    """Write a digest of papers from the period, as Markdown."""
    d = agent_mod.digest(PaperStore(), since=_since(since))
    typer.echo(f"# Paper digest ({d.period})\n")
    typer.echo("**Themes:** " + "; ".join(d.themes) + "\n")
    for it in d.items:
        pages = ", ".join(map(str, it.key_pages))
        typer.echo(f"## {it.title}")
        typer.echo(f"arXiv:{it.arxiv_id} · {it.published} · key pages {pages}\n")
        typer.echo(f"{it.one_liner}\n")
        typer.echo(f"*Why it matters:* {it.why_it_matters}\n")


if __name__ == "__main__":
    app()
