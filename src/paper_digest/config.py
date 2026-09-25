import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

DATA_DIR = Path(os.getenv("PAPER_DIGEST_DATA", "./data")).resolve()
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"
COLLECTION = "papers"

MODEL = "anthropic:claude-opus-5"

CHUNK_WORDS = 350
CHUNK_OVERLAP_WORDS = 60
