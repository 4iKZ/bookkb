"""CLI subcommand assembly and dispatch."""

from __future__ import annotations

import argparse
from pathlib import Path

from bookkb import ingest, retrieve, store
from bookkb.embed import encode


def cmd_ingest(args: argparse.Namespace) -> int:
    conn = store.connect()
    total_files = total_chunks = total_skipped = 0
    ok_paths = 0
    for raw in args.paths:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            print(f"[error] path not found: {p}")
            continue
        ok_paths += 1
        pdfs = ingest.iter_pdfs(p)
        if pdfs:
            print(f"\n== PDF: {p} ==")
            for pdf in pdfs:
                f, c, s = ingest.ingest_pdf(pdf, conn)
                total_files += f
                total_chunks += c
                total_skipped += s
        if p.is_dir() and ((p / "book").is_dir() or (p / "docs").is_dir()):
            mds = ingest.iter_markdown(p)
            if mds:
                label = p.name
                print(f"\n== Markdown: {p} ==")
                for md in mds:
                    f, c, s = ingest.ingest_markdown(md, conn, label)
                    total_files += f
                    total_chunks += c
                    total_skipped += s
    if not ok_paths:
        print("[error] no valid paths given; nothing ingested")
        conn.close()
        return 1
    conn.close()
    print("\n=== Ingest complete ===")
    print(f"  files ingested : {total_files}")
    print(f"  chunks created : {total_chunks}")
    print(f"  skipped (dup)  : {total_skipped}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    conn = store.connect()
    chunks = store.fetch_all_chunks(conn)
    if not chunks:
        print("[error] no chunks in database. Run `bookkb ingest <path>` first.")
        conn.close()
        return 1
    try:
        mat, _ = retrieve.build_matrix(chunks)
    except ValueError as e:
        print(f"[error] {e}")
        conn.close()
        return 1
    qvec = encode([args.question], "query: ")[0]
    results = retrieve.search(qvec, chunks, mat, args.k)
    for row in results:
        cite = retrieve.format_citation(row)
        print(f"[{row['rank']}] {cite} ({row['score']:.2f})")
        print(retrieve.format_text(row["text"]))
        print()
    conn.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bookkb",
        description="Personal book knowledge-base: ingest PDFs & markdown, ask for evidence.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest PDF files/dirs or a cloned repo dir.")
    p_ingest.add_argument("paths", nargs="+", help="PDF file/dir or repo dir containing book/ and docs/")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question; returns top-k evidence chunks.")
    p_ask.add_argument("question", help="Question text")
    p_ask.add_argument("-k", type=int, default=8, help="Number of results (default 8)")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    return args.func(args)
