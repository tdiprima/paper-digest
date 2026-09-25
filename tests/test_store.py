"""TEST-3 retrieval correctness on a real temporary SQLite database."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from paper_digest import store as store_mod
from paper_digest.store import Chunk, PaperStore, _fts_query


def chunk(aid: str, page: int, text: str, published: str = "2026-09-01", title: str | None = None) -> Chunk:
    return Chunk(arxiv_id=aid, title=title or f"Title {aid}", published=published, page=page, text=text)


def test_persists_across_connections(data_dir: Path):
    s1 = PaperStore()
    s1.add_chunks([chunk("a", 1, "transformer attention mechanism")], pages=1)
    assert store_mod.DB_PATH.exists()

    s2 = PaperStore()
    assert s2.has_paper("a")
    assert [p.arxiv_id for p in s2.list_papers()] == ["a"]
    assert len(s2.search("attention")) == 1


def test_add_chunks_empty_is_noop(store: PaperStore):
    store.add_chunks([], pages=3)
    assert store.list_papers() == []


def test_search_returns_matching_chunks_with_scores(store: PaperStore):
    store.add_chunks([chunk("a", 1, "retrieval augmented generation with BM25"), chunk("a", 2, "unrelated cooking recipe")], pages=2)
    store.add_chunks([chunk("b", 1, "graph neural networks for molecules")], pages=1)

    hits = store.search("retrieval BM25")
    assert [(h.arxiv_id, h.page) for h in hits] == [("a", 1)]
    assert hits[0].score is not None and hits[0].score > 0
    assert hits[0].title == "Title a"
    assert hits[0].published == "2026-09-01"


def test_search_no_match_returns_empty(store: PaperStore):
    store.add_chunks([chunk("a", 1, "some text here")], pages=1)
    assert store.search("zzzzqqq") == []
    assert store.search("") == []


def test_search_ranks_more_relevant_chunk_first(store: PaperStore):
    store.add_chunks([
        chunk("a", 1, "attention attention attention transformer"),
        chunk("a", 2, "attention mentioned once among many other words about training data"),
    ], pages=2)
    hits = store.search("attention")
    assert [h.page for h in hits] == [1, 2]
    assert hits[0].score >= hits[1].score


def test_search_respects_k(store: PaperStore):
    store.add_chunks([chunk("a", i, f"word{i} shared") for i in range(1, 6)], pages=5)
    assert len(store.search("shared", k=2)) == 2


def test_search_stems_terms(store: PaperStore):
    store.add_chunks([chunk("a", 1, "we evaluate several retrievers")], pages=1)
    assert len(store.search("retriever")) == 1


def test_fts_query_quotes_and_drops_short_terms():
    assert _fts_query("a BM25 re-rank!") == '"BM25" OR "re-rank"'
    assert _fts_query("") == '""'


def test_search_and_list_since_is_inclusive(store: PaperStore):
    store.add_chunks([chunk("old", 1, "shared keyword", published="2026-08-31")], pages=1)
    store.add_chunks([chunk("edge", 1, "shared keyword", published="2026-09-01")], pages=1)
    store.add_chunks([chunk("new", 1, "shared keyword", published="2026-09-02")], pages=1)

    since = date(2026, 9, 1)
    assert {h.arxiv_id for h in store.search("shared", since=since)} == {"edge", "new"}
    assert [p.arxiv_id for p in store.list_papers(since=since)] == ["new", "edge"]

    assert {h.arxiv_id for h in store.search("shared")} == {"old", "edge", "new"}
    assert [p.arxiv_id for p in store.list_papers()] == ["new", "edge", "old"]
    assert store.list_papers(since=date(2026, 9, 3)) == []


def test_paper_and_page_attribution(store: PaperStore):
    store.add_chunks([
        chunk("a", 1, "alpha intro", published="2026-01-01", title="Alpha"),
        chunk("a", 3, "alpha needle result", published="2026-01-01", title="Alpha"),
    ], pages=3)
    store.add_chunks([
        chunk("b", 2, "beta needle discussion", published="2026-02-02", title="Beta"),
    ], pages=2)

    hits = store.search("needle")
    assert {(h.arxiv_id, h.title, h.published, h.page) for h in hits} == {
        ("a", "Alpha", "2026-01-01", 3),
        ("b", "Beta", "2026-02-02", 2),
    }
    for h in hits:
        assert "needle" in h.text


def test_get_pages_filters_by_paper_and_page_in_order(store: PaperStore):
    store.add_chunks([chunk("a", 1, "a1"), chunk("a", 2, "a2 first"), chunk("a", 2, "a2 second"), chunk("a", 3, "a3")], pages=3)
    store.add_chunks([chunk("b", 2, "b2")], pages=2)

    got = store.get_pages("a", [3, 2])
    assert [(c.arxiv_id, c.page, c.text) for c in got] == [("a", 2, "a2 first"), ("a", 2, "a2 second"), ("a", 3, "a3")]
    assert got[0].title == "Title a"
    assert store.get_pages("a", []) == []
    assert store.get_pages("a", [99]) == []
    assert store.get_pages("zzz", [1]) == []


def test_list_papers_records_page_count(store: PaperStore):
    store.add_chunks([chunk("a", 1, "x")], pages=12)
    p = store.list_papers()[0]
    assert (p.arxiv_id, p.title, p.published, p.pages) == ("a", "Title a", "2026-09-01", 12)
