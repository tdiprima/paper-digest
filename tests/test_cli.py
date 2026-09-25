"""TEST-4 CLI behavior: fixed dates, deterministic agent output, rendering."""

from __future__ import annotations

from datetime import date

import pytest
from typer.testing import CliRunner

from paper_digest import agent as agent_mod
from paper_digest import cli
from paper_digest.agent import Answer, Citation, Digest, DigestItem, Finding
from paper_digest.store import Chunk, PaperStore

runner = CliRunner()
TODAY = date(2026, 9, 25)


class FixedDate(date):
    @classmethod
    def today(cls):
        return cls(TODAY.year, TODAY.month, TODAY.day)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(cli, "date", FixedDate)


@pytest.mark.parametrize(
    "spec, expected",
    [
        (None, None),
        ("", None),
        ("7d", date(2026, 9, 18)),
        ("2w", date(2026, 9, 11)),
        ("1m", date(2026, 8, 26)),
        ("2026-01-15", date(2026, 1, 15)),
    ],
)
def test_since_parsing(spec, expected):
    assert cli._since(spec) == expected


def test_since_rejects_garbage():
    with pytest.raises(ValueError):
        cli._since("yesterday")


def test_papers_lists_and_filters_by_since(data_dir):
    s = PaperStore()
    s.add_chunks([Chunk("old.1", "Old Paper", "2026-08-01", 1, "x")], pages=1)
    s.add_chunks([Chunk("new.1", "New Paper", "2026-09-20", 1, "x")], pages=1)

    r = runner.invoke(cli.app, ["papers"])
    assert r.exit_code == 0, r.output
    assert r.output.splitlines() == [
        "2026-09-20  new.1         New Paper",
        "2026-08-01  old.1         Old Paper",
    ]

    r = runner.invoke(cli.app, ["papers", "--since", "7d"])  # cutoff 2026-09-18
    assert r.output.splitlines() == ["2026-09-20  new.1         New Paper"]

    r = runner.invoke(cli.app, ["papers", "--since", "2026-09-21"])
    assert r.output == ""


def test_papers_on_empty_library(data_dir):
    r = runner.invoke(cli.app, ["papers"])
    assert r.exit_code == 0
    assert r.output == ""


def test_ask_renders_citations_and_gaps(data_dir, monkeypatch):
    calls = {}

    def fake_ask(question, store, since=None):
        calls["question"], calls["since"] = question, since
        assert isinstance(store, PaperStore)
        return Answer(
            summary="Two papers address it.",
            findings=[
                Finding(claim="Claim one.", citations=[Citation(arxiv_id="a.1", title="A", pages=[2, 5])]),
                Finding(
                    claim="Claim two.",
                    citations=[Citation(arxiv_id="a.1", title="A", pages=[1]), Citation(arxiv_id="b.2", title="B", pages=[7])],
                ),
            ],
            gaps=["No paper covers X."],
        )

    monkeypatch.setattr(agent_mod, "ask", fake_ask)
    r = runner.invoke(cli.app, ["ask", "what?", "--since", "30d"])

    assert r.exit_code == 0, r.output
    assert calls == {"question": "what?", "since": date(2026, 8, 26)}
    assert r.output == (
        "Two papers address it.\n\n"
        "- Claim one.  [a.1 p.2,5]\n"
        "- Claim two.  [a.1 p.1; b.2 p.7]\n"
        "\nNot covered by the library:\n"
        "- No paper covers X.\n"
    )


def test_ask_without_since_passes_none_and_omits_gaps_section(data_dir, monkeypatch):
    calls = {}

    def fake_ask(question, store, since=None):
        calls["since"] = since
        return Answer(summary="Nothing indexed.", findings=[], gaps=[])

    monkeypatch.setattr(agent_mod, "ask", fake_ask)
    r = runner.invoke(cli.app, ["ask", "anything?"])
    assert r.exit_code == 0
    assert calls == {"since": None}
    assert r.output == "Nothing indexed.\n\n"
    assert "Not covered" not in r.output


def test_digest_renders_markdown(data_dir, monkeypatch):
    calls = {}

    def fake_digest(store, since):
        calls["since"] = since
        return Digest(
            period="since 2026-09-18",
            themes=["retrieval", "evaluation"],
            items=[
                DigestItem(
                    arxiv_id="a.1", title="Paper A", published="2026-09-20",
                    one_liner="Does A.", why_it_matters="Because A.", key_pages=[1, 4],
                ),
                DigestItem(
                    arxiv_id="b.2", title="Paper B", published="2026-09-22",
                    one_liner="Does B.", why_it_matters="Because B.", key_pages=[2],
                ),
            ],
        )

    monkeypatch.setattr(agent_mod, "digest", fake_digest)
    r = runner.invoke(cli.app, ["digest"])  # default 7d

    assert r.exit_code == 0, r.output
    assert calls == {"since": date(2026, 9, 18)}
    assert r.output == (
        "# Paper digest (since 2026-09-18)\n\n"
        "**Themes:** retrieval; evaluation\n\n"
        "## Paper A\n"
        "arXiv:a.1 · 2026-09-20 · key pages 1, 4\n\n"
        "Does A.\n\n"
        "*Why it matters:* Because A.\n\n"
        "## Paper B\n"
        "arXiv:b.2 · 2026-09-22 · key pages 2\n\n"
        "Does B.\n\n"
        "*Why it matters:* Because B.\n\n"
    )


def test_digest_custom_since_and_empty(data_dir, monkeypatch):
    calls = {}

    def fake_digest(store, since):
        calls["since"] = since
        return Digest(period="since 2026-08-26", themes=[], items=[])

    monkeypatch.setattr(agent_mod, "digest", fake_digest)
    r = runner.invoke(cli.app, ["digest", "--since", "1m"])
    assert r.exit_code == 0
    assert calls == {"since": date(2026, 8, 26)}
    assert r.output == "# Paper digest (since 2026-08-26)\n\n**Themes:** \n\n"


def test_add_requires_category_or_query(data_dir):
    r = runner.invoke(cli.app, ["add"])
    assert r.exit_code != 0
    assert "Give a category" in r.output


def test_add_ingests_from_category(data_dir, monkeypatch):
    from paper_digest.ingest import ArxivEntry

    entries = [ArxivEntry("x.1", "X", "2026-09-20", "u1"), ArxivEntry("x.2", "Y", "2026-09-20", "u2")]
    seen = {}
    monkeypatch.setattr(cli, "fetch_category", lambda cat, match: seen.update(cat=cat, match=match) or entries)

    def fake_ingest(es, store, log):
        seen["entries"] = es
        for e in es:
            log(f"added {e.arxiv_id}")
        return len(es)

    monkeypatch.setattr(cli, "ingest", fake_ingest)
    r = runner.invoke(cli.app, ["add", "cs.CL", "-m", "retrieval", "--max", "1"])
    assert r.exit_code == 0, r.output
    assert seen["cat"] == "cs.CL" and seen["match"] == ["retrieval"]
    assert seen["entries"] == entries[:1]
    assert r.output == "1 candidate paper(s)\nadded x.1\n\n1 new paper(s) indexed.\n"
