"""TEST-1 download recovery, TEST-2 ingestion correctness."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from paper_digest import ingest as ingest_mod
from paper_digest.ingest import ArxivEntry, download_pdf, ingest
from paper_digest.store import PaperStore

from conftest import count_chunks, make_pdf


def entry(aid: str = "2401.00001", title: str = "A Paper") -> ArxivEntry:
    return ArxivEntry(arxiv_id=aid, title=title, published="2026-09-20", pdf_url=f"https://x/{aid}.pdf")


class FakeStream:
    """Stand-in for httpx.stream: serves `payloads[url]`, optionally failing mid-stream N times."""

    def __init__(self, payloads: dict[str, bytes], fail_first: int = 0, status: int = 200):
        self.payloads = payloads
        self.fail_first = fail_first
        self.status = status
        self.calls = 0

    @contextmanager
    def __call__(self, method, url, **kw):
        self.calls += 1
        data = self.payloads[url]
        should_fail = self.calls <= self.fail_first
        req = httpx.Request(method, url)
        resp = httpx.Response(self.status, request=req)

        def iter_bytes():
            half = max(1, len(data) // 2)
            yield data[:half]
            if should_fail:
                raise httpx.ReadError("connection dropped", request=req)
            yield data[half:]

        resp.iter_bytes = iter_bytes  # type: ignore[method-assign]
        yield resp


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf(["hello world page one", "second page text"])


# ---- TEST-1: download recovery -----------------------------------------------


def test_interrupted_download_leaves_no_files(data_dir: Path, monkeypatch, pdf_bytes):
    e = entry()
    fake = FakeStream({e.pdf_url: pdf_bytes}, fail_first=1)
    monkeypatch.setattr(ingest_mod.httpx, "stream", fake)

    with pytest.raises(httpx.ReadError):
        download_pdf(e)

    pdf_dir = ingest_mod.PDF_DIR
    assert not (pdf_dir / "2401.00001.pdf").exists()
    assert not (pdf_dir / "2401.00001.pdf.part").exists()
    assert list(pdf_dir.iterdir()) == []


def test_retry_after_interrupt_downloads_complete_file(data_dir: Path, monkeypatch, pdf_bytes):
    e = entry()
    fake = FakeStream({e.pdf_url: pdf_bytes}, fail_first=1)
    monkeypatch.setattr(ingest_mod.httpx, "stream", fake)

    with pytest.raises(httpx.ReadError):
        download_pdf(e)
    path = download_pdf(e)

    assert fake.calls == 2
    assert path.read_bytes() == pdf_bytes
    assert not path.with_suffix(".pdf.part").exists()


def test_http_error_leaves_no_files(data_dir: Path, monkeypatch, pdf_bytes):
    e = entry()
    fake = FakeStream({e.pdf_url: pdf_bytes}, status=404)
    monkeypatch.setattr(ingest_mod.httpx, "stream", fake)

    with pytest.raises(httpx.HTTPStatusError):
        download_pdf(e)
    assert list(ingest_mod.PDF_DIR.iterdir()) == []


def test_existing_pdf_is_not_redownloaded(data_dir: Path, monkeypatch, pdf_bytes):
    e = entry()
    fake = FakeStream({e.pdf_url: pdf_bytes})
    monkeypatch.setattr(ingest_mod.httpx, "stream", fake)

    download_pdf(e)
    download_pdf(e)
    assert fake.calls == 1


# ---- TEST-2: ingestion correctness ------------------------------------------


def test_textless_pdf_fails_and_indexes_nothing(data_dir: Path, monkeypatch, store: PaperStore):
    e = entry()
    monkeypatch.setattr(ingest_mod.httpx, "stream", FakeStream({e.pdf_url: make_pdf(["", ""])}))
    logs: list[str] = []

    n = ingest([e], store, log=logs.append)

    assert n == 0
    assert any(l.startswith("FAIL  2401.00001") and "no extractable text" in l for l in logs)
    assert not store.has_paper(e.arxiv_id)
    assert store.list_papers() == []
    assert count_chunks(ingest_mod.PDF_DIR.parent / "index" / "papers.sqlite") == 0


def test_failed_paper_does_not_block_following_paper(data_dir: Path, monkeypatch, store: PaperStore, pdf_bytes):
    bad = entry("2401.00001", "Bad")
    good = entry("2401.00002", "Good")
    monkeypatch.setattr(
        ingest_mod.httpx, "stream",
        FakeStream({bad.pdf_url: make_pdf([""]), good.pdf_url: pdf_bytes}),
    )
    logs: list[str] = []

    n = ingest([bad, good], store, log=logs.append)

    assert n == 1
    assert not store.has_paper(bad.arxiv_id)
    assert store.has_paper(good.arxiv_id)
    assert [p.arxiv_id for p in store.list_papers()] == [good.arxiv_id]
    assert any(l.startswith("FAIL  2401.00001") for l in logs)
    assert any(l.startswith("added 2401.00002") and "(2 chunks)" in l for l in logs)


def test_network_failure_does_not_block_following_paper(data_dir: Path, monkeypatch, store: PaperStore, pdf_bytes):
    bad = entry("2401.00001", "Bad")
    good = entry("2401.00002", "Good")
    fake = FakeStream({bad.pdf_url: pdf_bytes, good.pdf_url: pdf_bytes}, fail_first=1)
    monkeypatch.setattr(ingest_mod.httpx, "stream", fake)

    n = ingest([bad, good], store, log=lambda _: None)

    assert n == 1
    assert not store.has_paper(bad.arxiv_id)
    assert store.has_paper(good.arxiv_id)


def test_repeated_ingestion_creates_no_duplicate_chunks(data_dir: Path, monkeypatch, store: PaperStore, pdf_bytes):
    e = entry()
    fake = FakeStream({e.pdf_url: pdf_bytes})
    monkeypatch.setattr(ingest_mod.httpx, "stream", fake)
    db = ingest_mod.PDF_DIR.parent / "index" / "papers.sqlite"
    logs: list[str] = []

    assert ingest([e], store, log=logs.append) == 1
    first = count_chunks(db, e.arxiv_id)
    assert first == 2

    assert ingest([e], store, log=logs.append) == 0
    assert count_chunks(db, e.arxiv_id) == first
    assert len(store.list_papers()) == 1
    assert fake.calls == 1
    assert any(l.startswith("skip  2401.00001") for l in logs)

    # A fresh store on the same DB also skips it.
    assert ingest([e], PaperStore(), log=logs.append) == 0
    assert count_chunks(db, e.arxiv_id) == first


def test_indexed_chunks_carry_page_numbers(data_dir: Path, monkeypatch, store: PaperStore, pdf_bytes):
    e = entry()
    monkeypatch.setattr(ingest_mod.httpx, "stream", FakeStream({e.pdf_url: pdf_bytes}))
    ingest([e], store, log=lambda _: None)

    hits = store.search("second")
    assert len(hits) == 1
    assert hits[0].page == 2
    assert store.list_papers()[0].pages == 2
