from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from paper_digest import ingest as ingest_mod
from paper_digest import store as store_mod


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal PDF with one Helvetica text line per page (empty string = blank page)."""
    objs: list[bytes] = []
    n_pages = len(pages)
    # 1 catalog, 2 pages, 3 font, then (page, content) pairs
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n_pages))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        page_no = 4 + 2 * i
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_no + 1} 0 R >>".encode()
        )
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode() if text else b""
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point PDF and index storage at a temp dir."""
    pdf_dir = tmp_path / "pdfs"
    index_dir = tmp_path / "index"
    monkeypatch.setattr(ingest_mod, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(store_mod, "DB_PATH", index_dir / "papers.sqlite")
    return tmp_path


@pytest.fixture
def store(data_dir: Path) -> store_mod.PaperStore:
    return store_mod.PaperStore()


def count_chunks(db_path: Path, arxiv_id: str | None = None) -> int:
    con = sqlite3.connect(db_path)
    try:
        if arxiv_id:
            return con.execute("SELECT COUNT(*) FROM chunks WHERE arxiv_id=?", (arxiv_id,)).fetchone()[0]
        return con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        con.close()
