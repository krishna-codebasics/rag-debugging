"""An HR assistant for AtliQ's leave and travel policy, built with RAG.

    Indexing:  PDF -> text -> chunks -> embeddings -> Qdrant
    Answering: question -> find the closest chunks -> give them to the model -> answer

Each of the four settings below switches on the fix for one common failure.
Turn them on one at a time, restart, and ask the same question again to see the answer change.

Run it:   uv run rag.py                 (chat)
          uv run rag.py --show-prompt   (also print exactly what the model receives)
"""

import os
import re
import sys

import pdfplumber
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---- Settings: each fix in the session changes one of these ----
READ_TABLES = False         # Failure 1: the table in the PDF comes out jumbled
SPLIT_BY_SECTION = False    # Failure 2: a rule and its exception end up in different chunks
REWRITE_FOLLOW_UPS = False  # Failure 3: "Can I carry them forward?" means nothing on its own
KEYWORD_SEARCH = False      # Failure 4: the right chunk is found, but ranked too low to reach the model

CHUNK_SIZE = 300            # characters per chunk, when not splitting by section
TOP_K = 3                   # how many chunks the model gets

PDF_FILE = os.path.join(os.path.dirname(__file__), "atliq_leave_travel_policy.pdf")
ANSWER_MODEL = "openai/gpt-oss-20b"

load_dotenv()
groq = Groq(max_retries=5)  # reads GROQ_API_KEY; waits and retries when rate-limited
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
db = QdrantClient(":memory:")  # Qdrant running inside this script, nothing to install


def ask_model(messages, model=ANSWER_MODEL):
    reply = groq.chat.completions.create(model=model, messages=messages, temperature=0)
    return reply.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Step 1 · Read the PDF
# What can go wrong: tables come out as a jumble of numbers with no column names.
# ---------------------------------------------------------------------------

def read_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(read_page(page) for page in pdf.pages)


def read_page(page):
    if not READ_TABLES:
        return page.extract_text()

    # Fix: read the text above each table, then the table row by row, then the rest.
    parts, top = [], 0
    for table in page.find_tables():
        parts.append(page.crop((0, top, page.width, table.bbox[1])).extract_text())
        parts.append(table_to_sentences(table.extract()))
        top = table.bbox[3]
    parts.append(page.crop((0, top, page.width, page.height)).extract_text())
    return "\n".join(parts)


def table_to_sentences(rows):
    """Turn every row into one line that carries its own column names:
    'Grade: L4, Daily allowance: Rs. 1,500, Hotel limit per night (metro cities): Rs. 7,500, ...'"""
    header = [name.replace("\n", " ") for name in rows[0]]
    return "\n".join(", ".join(f"{name}: {value}" for name, value in zip(header, row))
                     for row in rows[1:])


# ---------------------------------------------------------------------------
# Step 2 · Split into chunks
# What can go wrong: a cut in the wrong place separates a rule from its exception.
# ---------------------------------------------------------------------------

def split_into_chunks(text):
    if not SPLIT_BY_SECTION:
        return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    # Fix: cut wherever a new numbered section starts ("4. Carrying forward..."),
    # so every chunk is one complete section.
    sections = re.split(r"\n(?=\d+\. [A-Z])", text)
    return [section.strip() for section in sections if section.strip()]


# ---------------------------------------------------------------------------
# Step 3 · Embed the chunks and store them in Qdrant
# ---------------------------------------------------------------------------

def build_index():
    chunks = split_into_chunks(read_pdf(PDF_FILE))
    vectors = embedder.encode(chunks)

    if db.collection_exists("policy"):
        db.delete_collection("policy")
    db.create_collection("policy", vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
    db.upsert("policy", points=[PointStruct(id=i, vector=vector.tolist(), payload={"text": chunk})
                                for i, (chunk, vector) in enumerate(zip(chunks, vectors))])
    return chunks


# ---------------------------------------------------------------------------
# Step 4 · Understand the question
# What can go wrong: a follow-up like "And for L4?" is searched on its own.
# ---------------------------------------------------------------------------

def rewrite_question(question, history):
    if not REWRITE_FOLLOW_UPS or not history:
        return question

    # Fix: use the chat so far to turn the follow-up into a full question.
    chat = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    return ask_model([{"role": "user", "content":
        f"Chat so far:\n{chat}\n\nLatest question: {question}\n\n"
        "Rewrite the latest question so it makes sense on its own. Reply with only the question."}])


# ---------------------------------------------------------------------------
# Step 5 · Search for the closest chunks
# What can go wrong: the right chunk comes back at #5, but only the top 3 reach the model.
# Embeddings match meaning loosely, so "leftover money" may not rank close to "unused advance".
# ---------------------------------------------------------------------------

def search(question, chunks):
    vector = embedder.encode(question).tolist()

    if not KEYWORD_SEARCH:
        hits = db.query_points("policy", query=vector, limit=TOP_K).points
        return [(hit.payload["text"], hit.score) for hit in hits]

    # Fix, part 1: collect candidates from both meaning search and keyword search.
    by_meaning = [hit.payload["text"] for hit in db.query_points("policy", query=vector, limit=10).points]
    candidates = list(dict.fromkeys(by_meaning + keyword_search(question, chunks)))

    # Fix, part 2: a reranker reads the question and each chunk together and scores them.
    scores = reranker.predict([(question, chunk) for chunk in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [(chunk, float(score)) for chunk, score in ranked[:TOP_K]]


def keyword_search(question, chunks, limit=10):
    """Score chunks by the question's words they contain. Rare words like 'TR-9'
    count for a lot; common words like 'the' count for almost nothing."""
    def words(text):
        return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()))

    chunk_words = [words(chunk) for chunk in chunks]
    scores = []
    for chunk, found in zip(chunks, chunk_words):
        score = sum(1 / sum(word in other for other in chunk_words)
                    for word in words(question) if word in found)
        scores.append((score, chunk))
    return [chunk for score, chunk in sorted(scores, reverse=True)[:limit] if score > 0]


# ---------------------------------------------------------------------------
# Step 6 · Answer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are AtliQ's friendly HR assistant. Answer the employee's question."


def answer(question, found_chunks):
    extracts = "\n\n---\n\n".join(chunk for chunk, _ in found_chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Policy extracts:\n{extracts}\n\nQuestion: {question}"}]
    return ask_model(messages), messages


# ---------------------------------------------------------------------------
# All the steps together
# ---------------------------------------------------------------------------

def ask(question, chunks, history):
    """The model only ever sees this one question and the chunks, never the chat
    so far. That's why a follow-up needs rewriting first (Step 4)."""
    question = rewrite_question(question, history)
    found = search(question, chunks)
    reply, messages = answer(question, found)
    return {"question": question, "found": found, "answer": reply, "messages": messages}


def show(result, show_prompt=False):
    print(f"\n  Question used: {result['question']}")
    for rank, (chunk, score) in enumerate(result["found"], 1):
        print(f"\n  [{rank}] score {score:.2f}")
        print("      " + chunk.replace("\n", "\n      "))
    if show_prompt:
        print("\n  ---- What the model received ----")
        for m in result["messages"]:
            print(f"  [{m['role']}]\n      " + m["content"].replace("\n", "\n      "))
    print(f"\nAnswer: {result['answer']}")


def main():
    show_prompt = "--show-prompt" in sys.argv
    chunks = build_index()
    print(f"AtliQ HR assistant. {len(chunks)} chunks indexed. Type 'exit' to quit.")

    history = []
    while True:
        question = input("\n> ").strip()
        if question in ("exit", "quit"):
            break
        if question:
            result = ask(question, chunks, history)
            show(result, show_prompt)
            history += [{"role": "user", "content": question},
                        {"role": "assistant", "content": result["answer"]}]


if __name__ == "__main__":
    main()
