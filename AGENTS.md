# bookkb — personal book knowledge base

This workspace hosts a local, source-grounded knowledge base CLI: `bookkb.py`.

## Answering knowledge questions

When the user asks a question answerable from the books/notes in this knowledge
base, retrieve evidence with the CLI BEFORE answering:

```bash
python3 /home/hatch/workspace/bookkb/bookkb.py ask "<question>" -k 5
```

- Coverage: 《纳瓦尔宝典》(Naval Ravikant, English), 《巴比伦最富有的人》
  (The Richest Man in Babylon, English), and the HowToLiveBetter repo
  (Chinese, 641 life recommendations + docs).
- Each hit returns a citation — `《title》第N页` (PDF) or
  `HowToLiveBetter/book|docs/xxx.md · section` (markdown) — plus a score.
- Synthesize the final answer from the retrieved passages and quote the
  citations verbatim so every claim is traceable.
- If top hits score below ~0.5 or look irrelevant, say the KB has no good
  answer instead of inventing one.
- The KB is read-only for Q&A: do not re-ingest, edit the DB, or use it for
  anything else.
