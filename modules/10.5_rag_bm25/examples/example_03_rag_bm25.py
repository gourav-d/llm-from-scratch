"""
Example 03: Full BM25-RAG Pipeline
Module 10.5: RAG Without Vectors

Builds a complete RAG pipeline:
  1. Chunk documents
  2. Index with BM25
  3. Retrieve top-K chunks for a query
  4. Build a prompt
  5. Simulate LLM answer (no API key needed)

Run:  python example_03_rag_bm25.py
Deps: none (pure Python)
"""

import math
import re
from collections import Counter


# ─────────────────────────────────────────────────────────
# BM25 (reused from example_02)
# ─────────────────────────────────────────────────────────

STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","to","for","of",
    "and","or","but","in","on","at","it","its","this","that","with",
    "has","have","had","not","by","from","as","so","do","does","will",
    "can","all","also","we","our","their","they","you","your"
}

def preprocess(text):
    tokens = text.lower().split()
    return [t.strip(".,!?;:'\"()[]") for t in tokens
            if t.strip(".,!?;:'\"()[]") not in STOP_WORDS
            and len(t.strip(".,!?;:'\"()[]")) > 1]

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.corpus = corpus; self.N = len(corpus)
        self.doc_lengths = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N else 1
        self.df = {}
        for doc in corpus:
            for w in set(doc): self.df[w] = self.df.get(w, 0) + 1
        self.idf = {w: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                    for w, df in self.df.items()}

    def score(self, query_tokens, doc_idx):
        doc = self.corpus[doc_idx]
        tf_counts = Counter(doc)
        dl = self.doc_lengths[doc_idx]
        total = 0.0
        for w in query_tokens:
            if w not in self.idf: continue
            tf = tf_counts.get(w, 0)
            num = self.idf[w] * tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += num / den
        return total

    def search(self, query_tokens, top_k=5):
        scores = [self.score(query_tokens, i) for i in range(self.N)]
        return sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]


# ─────────────────────────────────────────────────────────
# FAKE KNOWLEDGE BASE (simulates internal company docs)
# ─────────────────────────────────────────────────────────

# Imagine these are pages from your company's internal wiki
DOCUMENTS = {
    "payments.md": """
# Payment Service

The payment service handles all financial transactions in our platform.
We use Stripe as our primary payment processor for credit and debit cards.
All payments go through 3D Secure authentication for cards over $100.

Payment flow:
1. Customer submits payment form with card details
2. Frontend sends card token to our API (never raw card numbers)
3. Payment service calls Stripe API with the token and amount
4. Stripe returns a payment intent object with status
5. We publish a payment.completed event to Kafka on success
6. Order service listens to the Kafka topic and fulfills the order

We support Visa, Mastercard, American Express, and PayPal.
Refunds are processed within 5-7 business days.
The payment service uses PostgreSQL to store transaction records.
""",

    "authentication.md": """
# Authentication Service

Users authenticate via JWT tokens. Tokens expire after 24 hours.
Refresh tokens are valid for 30 days and stored in Redis.

Login flow:
1. User submits email and password
2. We hash the password with bcrypt (12 rounds)
3. Compare with stored hash in PostgreSQL users table
4. On success, generate JWT with user ID and roles
5. Return access token (24h) and refresh token (30 days)

We use OAuth 2.0 for Google and GitHub social login.
Multi-factor authentication (MFA) is available via TOTP (Google Authenticator).
Failed login attempts are rate-limited: 5 attempts per 10 minutes per IP.
Passwords must be at least 12 characters with mixed case and numbers.
""",

    "deployment.md": """
# Deployment Guide

All services are deployed on AWS using Kubernetes (EKS).
We use Docker containers and Helm charts for deployment configuration.

Deployment steps:
1. Push code to GitHub main branch
2. GitHub Actions CI runs tests and builds Docker image
3. Image pushed to Amazon ECR (container registry)
4. ArgoCD detects new image and deploys to staging
5. After QA approval, ArgoCD promotes to production

Production runs in us-east-1 and eu-west-1 for redundancy.
Horizontal pod autoscaling (HPA) scales pods based on CPU usage.
We use Istio service mesh for traffic management and observability.
Logs are shipped to Datadog. Alerts trigger PagerDuty on-call.
""",

    "database.md": """
# Database Architecture

We use PostgreSQL 15 as our primary relational database.
Redis is used for caching and session storage.
MongoDB stores unstructured data like user activity logs.

PostgreSQL setup:
- Primary in us-east-1a
- Read replicas in us-east-1b and us-east-1c
- Connection pooling via PgBouncer (max 1000 connections)
- Backups run daily to S3 with 30-day retention

Redis configuration:
- Redis Cluster with 6 nodes (3 primary, 3 replica)
- Used for: session cache, rate limiting, real-time counters
- TTL: user sessions 24 hours, API cache 5 minutes

All databases encrypted at rest with AWS KMS.
Database migrations use Flyway for version control.
"""
}


# ─────────────────────────────────────────────────────────
# STEP 1: CHUNK DOCUMENTS
# ─────────────────────────────────────────────────────────

def chunk_by_paragraph(text: str, source: str, max_words: int = 100) -> list[dict]:
    """
    Split a document into chunks at paragraph boundaries.
    Keeps chunks under max_words. Adds overlap via context prefix.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for i, para in enumerate(paragraphs):
        words = para.split()
        if not words:
            continue

        # If paragraph is short enough, keep as one chunk
        if len(words) <= max_words:
            chunks.append({
                "id": f"{source}:chunk{len(chunks)}",
                "source": source,
                "text": para,
                "preview": para[:80] + ("..." if len(para) > 80 else ""),
            })
        else:
            # Split long paragraphs with overlap
            stride = max_words // 2
            for start in range(0, len(words), stride):
                chunk_words = words[start:start + max_words]
                if len(chunk_words) < 10:   # skip tiny trailing fragments
                    break
                text_chunk = " ".join(chunk_words)
                chunks.append({
                    "id": f"{source}:chunk{len(chunks)}",
                    "source": source,
                    "text": text_chunk,
                    "preview": text_chunk[:80] + "...",
                })

    return chunks


# ─────────────────────────────────────────────────────────
# STEP 2: BUILD INDEX
# ─────────────────────────────────────────────────────────

def build_index(documents: dict) -> tuple[list[dict], BM25]:
    """Chunk all docs, tokenize, build BM25 index."""
    all_chunks = []
    for source, text in documents.items():
        all_chunks.extend(chunk_by_paragraph(text, source))

    tokenized = [preprocess(chunk["text"]) for chunk in all_chunks]
    index = BM25(tokenized)
    return all_chunks, index


# ─────────────────────────────────────────────────────────
# STEP 3: RETRIEVE
# ─────────────────────────────────────────────────────────

def retrieve(query: str, chunks: list[dict], index: BM25,
             top_k: int = 3) -> list[tuple[dict, float]]:
    """BM25-search the index, return top-K (chunk, score) pairs."""
    q_tokens = preprocess(query)
    results = index.search(q_tokens, top_k=top_k)
    return [(chunks[idx], score) for idx, score in results if score > 0]


# ─────────────────────────────────────────────────────────
# STEP 4: BUILD RAG PROMPT
# ─────────────────────────────────────────────────────────

def build_rag_prompt(question: str, retrieved: list[tuple[dict, float]]) -> str:
    """Assemble the context + question into an LLM prompt."""
    if not retrieved:
        context = "No relevant documents found."
    else:
        parts = []
        for i, (chunk, score) in enumerate(retrieved, 1):
            parts.append(
                f"[Source {i}: {chunk['source']} | relevance={score:.2f}]\n"
                f"{chunk['text']}"
            )
        context = "\n\n".join(parts)

    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# ─────────────────────────────────────────────────────────
# STEP 5: SIMULATED LLM (no API key needed)
# ─────────────────────────────────────────────────────────

def simulated_llm(prompt: str, question: str, retrieved: list) -> str:
    """
    Fake LLM that extracts key sentences from retrieved chunks.
    In production, replace this with a real LLM call:
        response = client.messages.create(model="claude-sonnet-4-6", ...)
    """
    if not retrieved:
        return "I don't know based on the provided documents."

    # Find sentences in retrieved chunks containing question keywords
    q_words = set(preprocess(question))
    best_sentences = []

    for chunk, score in retrieved[:2]:
        sentences = re.split(r'[.!?]', chunk["text"])
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_words = set(preprocess(sent))
            overlap = len(q_words & sent_words)
            if overlap > 0:
                best_sentences.append((overlap, sent))

    best_sentences.sort(key=lambda x: -x[0])
    if best_sentences:
        answer_parts = [s for _, s in best_sentences[:3]]
        return " ".join(answer_parts).strip() + "."
    else:
        return retrieved[0][0]["text"][:200] + "..."


# ─────────────────────────────────────────────────────────
# DEMO: Full RAG Pipeline
# ─────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


print_section("STEP 1: Building Index from Documents")

chunks, index = build_index(DOCUMENTS)
print(f"\n  Documents:  {len(DOCUMENTS)}")
print(f"  Chunks:     {len(chunks)}")
print(f"  Vocabulary: {len(index.df)} unique terms")
print(f"  Avg chunk length: {index.avgdl:.0f} tokens")
print(f"\n  Sample chunks:")
for chunk in chunks[:3]:
    print(f"    [{chunk['id']}] {chunk['preview']}")


print_section("STEP 2: Retrieval Examples")

test_queries = [
    "How does payment processing work?",
    "How do users log in and what token is used?",
    "Where are services deployed and how does CI/CD work?",
    "What database does the system use for sessions?",
]

for question in test_queries:
    retrieved = retrieve(question, chunks, index, top_k=3)
    print(f"\n  Q: '{question}'")
    if not retrieved:
        print(f"    No results found (query terms not in any document)")
    for rank, (chunk, score) in enumerate(retrieved, 1):
        print(f"    {rank}. [{chunk['source']}] score={score:.2f}  {chunk['preview']}")


print_section("STEP 3: Full RAG Answer Generation")

demo_question = "How does payment processing work?"
retrieved = retrieve(demo_question, chunks, index, top_k=3)
prompt = build_rag_prompt(demo_question, retrieved)
answer = simulated_llm(prompt, demo_question, retrieved)

print(f"\n  Question: {demo_question}")
print(f"\n  Retrieved chunks:")
for chunk, score in retrieved:
    print(f"    [{chunk['source']}] score={score:.2f}")

print(f"\n  Prompt sent to LLM:")
print(f"  {'-'*50}")
for line in prompt.split('\n')[:15]:
    print(f"  {line}")
print(f"  ... [truncated]")

print(f"\n  Simulated LLM answer:")
print(f"  {'-'*50}")
print(f"  {answer}")


print_section("STEP 4: Missing Answer Demo (Grounding)")

unknown_q = "What is the CEO's name?"
retrieved_unknown = retrieve(unknown_q, chunks, index, top_k=3)
answer_unknown = simulated_llm(
    build_rag_prompt(unknown_q, retrieved_unknown),
    unknown_q, retrieved_unknown
)

print(f"\n  Q: '{unknown_q}'")
if retrieved_unknown:
    print(f"  Retrieved chunks (low relevance): {[(c['source'], round(s,2)) for c,s in retrieved_unknown]}")
else:
    print(f"  No chunks retrieved (no matching keywords)")
print(f"  Answer: {answer_unknown}")
print(f"""
  LESSON: RAG with grounded prompting says "I don't know" instead of hallucinating.
  Without RAG, LLM would make up a CEO name confidently.
  With RAG: "I don't know based on the provided documents."
""")


print_section("STEP 5: Chunking Strategy Comparison")

short_doc = "Python is great. Machine learning uses Python. Neural networks need data. Data science is popular."
print(f"\n  Document: '{short_doc}'")

# Fixed-size chunks (word count)
def fixed_chunks(text, size=5):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# Overlapping chunks
def overlap_chunks(text, size=7, stride=4):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words)-size+1, stride)]

# Sentence chunks
def sentence_chunks(text):
    return [s.strip() for s in text.split('.') if s.strip()]

print(f"\n  Fixed-size chunks (5 words each):")
for c in fixed_chunks(short_doc): print(f"    '{c}'")

print(f"\n  Overlapping chunks (7 words, stride 4):")
for c in overlap_chunks(short_doc): print(f"    '{c}'")

print(f"\n  Sentence chunks:")
for c in sentence_chunks(short_doc): print(f"    '{c}'")

print(f"""
  LESSON: Overlapping chunks prevent context loss at boundaries.
  Sentence chunks are cleanest but require sentence-ending punctuation.
  Rule of thumb: 200-500 words per chunk, 10-20% overlap.
""")
