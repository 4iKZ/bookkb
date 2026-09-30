# bookkb — Personal Book Q&A CLI

English | [简体中文](README.zh-CN.md)

![bookkb](docs/assets/hero.webp)

Turn your PDF ebooks and Markdown notes into a **local, citable** knowledge base:
`ingest` to load → `ask` to query, getting back source passages with **precise
citations** (book title + page / file + section).
The CLI only retrieves — it never writes answers for you. Feed the evidence to
any LLM (or agent) to compose the final answer.

## Features

- **Fully local**: embeddings run on-device via ONNX; no external API calls except downloading the model
- **Bilingual**: `intfloat/multilingual-e5-small`, handles mixed Chinese/English queries
- **Traceable citations**: PDFs cited down to the page, Markdown down to file + section, every hit carries a similarity score
- **Idempotent ingest**: files deduplicated by sha256 — re-runs only process new/changed files
- **Zero services**: one package + one SQLite file, no vector DB, no web server

## Install

```bash
pip install --user onnxruntime tokenizers numpy pypdf
pip install --user -e .          # installs the bookkb package + console script
```

Then prepare the embedding model (ONNX build of `intfloat/multilingual-e5-small`):

```
<model-dir>/
  tokenizer.json
  onnx/model.onnx
```

Two ways to point bookkb at the model (pick one):

1. HuggingFace cache layout (default): under `~/.cache/huggingface` or `HF_HOME`,
   `models--intfloat--multilingual-e5-small/snapshots/*/onnx/model.onnx`
2. Environment variable: `export BOOKKB_MODEL_DIR=/path/to/<model-dir>`

## Quick start

```bash
# 1. Ingest: a PDF file/dir, or a markdown repo dir containing book/ + docs/
bookkb ingest ~/books/naval.pdf
bookkb ingest ~/books/pdfs/
bookkb ingest ~/notes-repo/      # needs book/*.md (body) and optionally docs/*.md

# 2. Ask: returns top-k evidence passages (default 8)
bookkb ask "What does Naval say about leverage?" -k 5
bookkb ask "失业了能领什么补助" -k 3
```

Equivalent: `python3 -m bookkb ingest ...` / `python3 -m bookkb ask ...`.
The old `python3 bookkb.py ...` no longer works (single file removed in favor of the package layout).

Example output:

```
[1] 《纳瓦尔宝典-英文原版》第34页 (0.83)
"Fortunes require leverage. Business leverage comes from capital, people,
and products with no marginal cost of replication (code and media)."

[2] HowToLiveBetter/book/07-没钱的时候怎么活.md · 7. 没钱的时候怎么活 (0.91)
<chunk text>
```

Database location: `data/bookkb.sqlite` (under the project root), overridable via the `BOOKKB_DB` env var.

## How it works

![How it works](docs/assets/how-it-works.webp)

```
ingest:  PDFs split by page / Markdown split by h2 → chunked → e5-small embeddings
         → stored in SQLite (documents, chunks), embeddings as float32 BLOBs
ask:     question encoded the same way → brute-force cosine similarity via NumPy
         → top-k passages + citations
```

- e5 convention: `passage: ` prefix for ingested text, `query: ` prefix for questions, L2-normalized vectors
- 512-token chunk cap (e5-small context); overlong PDF pages split further by paragraph
- Brute-force search is plenty fast up to tens of thousands of chunks — no vector DB needed

## Citation format

| Source   | Format | Example |
|----------|--------|---------|
| PDF      | 《title》p.N | 《纳瓦尔宝典-英文原版》第34页 |
| Markdown | path · section | HowToLiveBetter/book/05-不要浪费钱.md · 5. 不要浪费钱 |

## Known limitations

- Scanned PDFs (no text layer) are skipped with a warning; no OCR in v1
- Retrieval is pure vector similarity, no rerank; ordering among near-ties is imperfect
- Re-ingesting an edited file has replace semantics that aren't fully tested; when in doubt, delete the DB and rebuild

## Project layout

```
src/bookkb/     package source
  __main__.py     python -m bookkb entry point
  cli.py          subcommand wiring (ingest / ask)
  ingest.py       document discovery + chunking (PDF / markdown)
  embed.py        ONNX embeddings (injectable test seam)
  store.py        SQLite read/write
  retrieve.py     retrieval + citation formatting
tests/          unit tests (chunking, store round-trip, ranking) — no model needed
docs/SPEC.md    v1 spec (Chinese)
AGENTS.md       agent-facing usage notes (kept at root)
data/           local data (gitignored): the SQLite database
```

### Migrating from the old version

- Old: `python3 bookkb.py ingest ...` → new: `bookkb ingest ...` (or `python3 -m bookkb ingest ...`)
- Old: `python3 bookkb.py ask ...` → new: `bookkb ask ...` (or `python3 -m bookkb ask ...`)
- Default DB location moved from `data/bookkb.sqlite` next to the script to
  `data/bookkb.sqlite` under the current working directory;
  still overridable via `BOOKKB_DB`.

## License

MIT — see [LICENSE](LICENSE).
