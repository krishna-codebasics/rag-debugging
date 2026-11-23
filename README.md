# Why does my RAG app give wrong answers?

A small HR assistant for AtliQ's leave and travel policy, with four common
RAG failures built in, and a fix for each one.

```
Indexing:   PDF ──> text ──> chunks ──> embeddings ──> Qdrant
Answering:  question ──> closest chunks ──> Groq model ──> answer
```

## Setup

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync                     # creates .venv and installs everything
cp .env.example .env        # Windows: copy .env.example .env, then paste in your Groq key
uv run rag.py
```

Get a free Groq key at [console.groq.com/keys](https://console.groq.com/keys).
The first run downloads two small models (about 180 MB), so give it a minute.

## Using it

```text
uv run rag.py                  chat with the assistant; it prints the chunks it found
uv run rag.py --show-prompt    also print exactly what the model received
```

## The four failures

All four fixes are already in `rag.py`, switched off. Turn on one setting at a
time, restart `rag.py`, and ask the question again to see the answer change.

| # | Step | What goes wrong | Try asking | Setting |
|---|---|---|---|---|
| 1 | Reading the PDF | The table comes out jumbled, so the wrong number gets read | What is the hotel limit for L4 in a metro city? | `READ_TABLES` |
| 2 | Chunking | A rule and its exception end up in different chunks | Can I carry forward my earned leave during my notice period? | `SPLIT_BY_SECTION` |
| 3 | The question | A follow-up is searched on its own | How many sick leaves do I get? then: Can I carry them forward? | `REWRITE_FOLLOW_UPS` |
| 4 | Search | The right chunk is found but ranked too low to reach the model | What do I do with leftover travel advance money? | `KEYWORD_SEARCH` |

## Checking your own app

When an answer is wrong, ask one question first: **was the answer in the chunks
that were retrieved?**

- **Yes:** the problem is the answer step: the model ignored the chunks or added
  things of its own.
- **No:** check whether the answer is in the index at all. If it isn't, look at
  reading and chunking (failures 1 and 2). If it is, look at the question and
  the search (failures 3 and 4).

## In a real project

Here, Qdrant runs inside the script and the policy is re-indexed on every run.
In production you'd run Qdrant as a server and index once. The ideas stay the
same.
