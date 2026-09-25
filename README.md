# paper-digest 📄

Keep up with an arXiv topic without reading every PDF.

`paper-digest` watches an arXiv category, pulls the papers that match your
keywords, indexes the full text of every PDF page by page, and gives you two
things:

- **Ask.** Questions across the whole library, answered with citations that
  point at an arXiv id *and a page number* you can go check.
- **Digest.** A Markdown roundup of what landed in the last week or month:
  cross-paper themes, one line per paper on what it does and why it matters.

Built with [Pydantic AI](https://ai.pydantic.dev) on Claude. Retrieval is
SQLite FTS5 (BM25), so there is no embedding model to download and it runs
offline apart from the model call.

## Why this shape

Most "RAG agent" demos retrieve three sentences about the weather. That is not
the problem RAG solves. RAG is for a corpus that will never fit in a context
window: dozens of 20-page papers is a few million tokens. The agent here has to
search, decide which pages are worth reading in full, read them, and cite them.
The structured output forces every claim to carry a citation, and the CLI
prints those citations so you can verify the answer instead of trusting it.

## Install

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env_sample .env   # add your ANTHROPIC_API_KEY
```

## Use

```bash
# 1. Pull today's cs.CL announcements whose title/abstract mention retrieval or RAG
uv run paper-digest add cs.CL -m retrieval -m RAG --max 25

# Or use full arXiv search syntax (some networks get a 406 from this endpoint)
uv run paper-digest add --query "cat:cs.CL AND ti:retrieval" --max 25

# 2. See what's indexed
uv run paper-digest papers --since 30d

# 3. Ask across the library
uv run paper-digest ask "What retrieval methods reduce hallucination, and how are they evaluated?"
uv run paper-digest ask "Which papers use the BEIR benchmark?" --since 2w

# 4. Weekly digest, as Markdown
uv run paper-digest digest --since 7d > digest.md
```

`--since` accepts `7d`, `2w`, `1m`, or an ISO date.

Example `ask` output:

```
Only two of the four indexed papers involve retrieval, and in very different
senses. PTC-Bias uses retrieval literally as a component ...

- PTC-Bias evaluates on LibriSpeech test-clean/test-other under the Rare5k
  protocol with N in {100, 500, 1000, 2000} distractors ...  [2609.28727 p.3]
- BM25 and cross-encoder retrieval are used as data-construction machinery in
  CTC-BENCH: distractors are BM25-mined from each dataset's corpus ...  [2609.29245 p.17,18]

Not covered by the library:
- No indexed paper implements classical retrieve-then-generate RAG
```

## Run it weekly

Add a cron entry (or a launchd job on macOS):

```
# Ingest every weekday morning (the feed only lists that day's announcements)
0 8 * * 1-5  cd /path/to/paper-digest && uv run paper-digest add cs.CL -m retrieval -m RAG --max 30
# Digest on Friday afternoon
0 16 * * 5   cd /path/to/paper-digest && uv run paper-digest digest --since 7d > digests/$(date +\%F).md
```

## How it works

```
src/paper_digest/
├── ingest.py   arXiv feed/API → PDF download → pypdf page text → word chunks
├── store.py    SQLite: papers table + chunks table + FTS5 index (BM25)
├── agent.py    Pydantic AI agent, 3 tools, two output schemas
├── cli.py      typer commands: add, papers, ask, digest
└── config.py   paths, model id, chunk sizes
```

The agent gets three tools:

| Tool | What it does |
|---|---|
| `search_chunks(query, k)` | BM25 search over every chunk, tagged with arXiv id and page |
| `list_papers()` | Every indexed paper, newest first |
| `read_pages(arxiv_id, pages)` | Full text of specific pages of one paper |

Its output is validated against a Pydantic model. For `ask`, every `Finding`
requires at least one `Citation` with page numbers. For `digest`, every paper
gets a `DigestItem` with the key pages. If the model returns something that
does not validate, Pydantic AI sends the error back and retries.

Data lives in `./data` (PDFs and the SQLite index). Override with
`PAPER_DIGEST_DATA`.

## Configuration

| Setting | Where | Default |
|---|---|---|
| Model | `config.py` `MODEL` | `anthropic:claude-opus-5` |
| Chunk size | `config.py` `CHUNK_WORDS` | 350 words, 60 overlap |
| Data directory | `PAPER_DIGEST_DATA` env | `./data` |

## License

[MIT](LICENSE)

<BR>
