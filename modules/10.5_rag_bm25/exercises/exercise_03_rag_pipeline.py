"""
Exercise 03: Build a RAG Pipeline with BM25
Module 10.5: RAG Without Vectors

TASKS:
  1. Implement chunk_text() — split a document into overlapping chunks
  2. Implement build_rag_index() — tokenize + build BM25 index from chunks
  3. Implement retrieve() — find top-K relevant chunks for a query
  4. Implement build_prompt() — assemble context + question for LLM
  5. Run a full RAG query and verify grounding

Run:  python exercise_03_rag_pipeline.py
Deps: none (pure Python)
"""

import math
from collections import Counter


# ─────────────────────────────────────────────────────────
# PROVIDED: BM25 and preprocessing (do not modify)
# ─────────────────────────────────────────────────────────

STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","to","for","of",
    "and","or","but","in","on","at","it","its","this","that","with",
    "has","have","had","not","by","from","as","so","do","does","will"
}

def preprocess(text):
    tokens = text.lower().split()
    return [t.strip(".,!?;:'\"()") for t in tokens
            if t.strip(".,!?;:'\"()") not in STOP_WORDS
            and len(t.strip(".,!?;:'\"()")) > 1]

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus; self.N = len(corpus)
        self.k1 = k1; self.b = b
        self.dl = [len(d) for d in corpus]
        self.avgdl = sum(self.dl) / self.N if self.N else 1
        self.df = {}
        for doc in corpus:
            for w in set(doc): self.df[w] = self.df.get(w, 0) + 1
        self.idf = {w: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                    for w, df in self.df.items()}

    def get_scores(self, q):
        scores = []
        for i, doc in enumerate(self.corpus):
            tf = Counter(doc)
            s = sum(self.idf.get(w, 0) * tf.get(w, 0) * (self.k1 + 1) /
                    (tf.get(w, 0) + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
                    for w in q if w in self.idf)
            scores.append(s)
        return scores


# ─────────────────────────────────────────────────────────
# TASK 1: Chunk a Document
# ─────────────────────────────────────────────────────────

def chunk_text(text: str, source: str,
               chunk_size: int = 60, overlap: int = 15) -> list[dict]:
    """
    Split text into overlapping word-based chunks.

    Algorithm:
      1. Split text into words
      2. Create chunks of chunk_size words
      3. Each chunk starts overlap words before the previous chunk ended
         (stride = chunk_size - overlap)
      4. Skip chunks with fewer than 10 words (trailing fragments)

    Args:
        text:       document text to chunk
        source:     filename/ID for the source document
        chunk_size: words per chunk
        overlap:    words of overlap between consecutive chunks

    Returns:
        list of dicts: {"id", "source", "text"}

    Example:
        words = ["a", "b", "c", "d", "e", "f", "g", "h"]
        chunk_size=4, overlap=2 → stride=2
        Chunk 0: words[0:4] = "a b c d"
        Chunk 1: words[2:6] = "c d e f"
        Chunk 2: words[4:8] = "e f g h"

    HINT:
        words = text.split()
        stride = chunk_size - overlap
        chunks = []
        for start in range(0, len(words), stride):
            chunk_words = words[start : start + chunk_size]
            if len(chunk_words) < 10:
                break
            chunks.append({
                "id": f"{source}:chunk{len(chunks)}",
                "source": source,
                "text": " ".join(chunk_words),
            })
        return chunks
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Build RAG Index
# ─────────────────────────────────────────────────────────

def build_rag_index(documents: dict[str, str]) -> tuple[list[dict], BM25]:
    """
    Chunk all documents and build a BM25 index over the chunks.

    Args:
        documents: dict of {filename: text}

    Returns:
        (chunks, bm25_index)
        chunks: list of chunk dicts with "id", "source", "text"
        bm25_index: BM25 instance built on tokenized chunks

    HINT:
        all_chunks = []
        for source, text in documents.items():
            all_chunks.extend(chunk_text(text, source))

        tokenized = [preprocess(chunk["text"]) for chunk in all_chunks]
        index = BM25(tokenized)
        return all_chunks, index
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Retrieve
# ─────────────────────────────────────────────────────────

def retrieve(query: str, chunks: list[dict], index: BM25,
             top_k: int = 3) -> list[tuple[dict, float]]:
    """
    Find the top-K most relevant chunks for a query.

    Steps:
      1. Preprocess the query
      2. Get BM25 scores for all chunks
      3. Sort by score (highest first)
      4. Return top-K (chunk, score) pairs where score > 0

    Args:
        query:   user's question
        chunks:  list of chunk dicts
        index:   BM25 instance
        top_k:   number of chunks to return

    Returns:
        list of (chunk_dict, score) tuples, best first

    HINT:
        q_tokens = preprocess(query)
        scores = index.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(chunks[i], s) for i, s in ranked[:top_k] if s > 0]
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Build RAG Prompt
# ─────────────────────────────────────────────────────────

def build_prompt(question: str, retrieved: list[tuple[dict, float]]) -> str:
    """
    Assemble a RAG prompt from the question and retrieved chunks.

    Format:
        System instruction: answer only from context, say "I don't know" if not found
        CONTEXT: numbered list of retrieved chunks with source labels
        QUESTION: the user's question
        ANSWER: (empty — LLM fills this in)

    Args:
        question:  user's question
        retrieved: list of (chunk, score) pairs from retrieve()

    Returns:
        formatted prompt string

    HINT:
        if not retrieved:
            context = "No relevant documents found."
        else:
            parts = []
            for i, (chunk, score) in enumerate(retrieved, 1):
                parts.append(f"[{i}. {chunk['source']} | score={score:.2f}]\n{chunk['text']}")
            context = "\n\n".join(parts)

        return f\"\"\"Answer using ONLY the context. Say "I don't know" if not in context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:\"\"\"
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────

DOCS = {
    "payments.md": """
The payment service uses Stripe for credit card processing. Payments go through
3D Secure authentication for transactions over 100 dollars. The payment flow
starts when the customer submits their card details on the checkout form.
The frontend sends a tokenized card reference to our API endpoint.
The payment service then calls the Stripe API with the token and amount.
Stripe returns a payment intent object with a confirmed or failed status.
On success we publish a payment completed event to the Kafka message queue.
The order service subscribes to this Kafka topic and fulfills the order.
We support Visa Mastercard American Express and PayPal payment methods.
Refunds take 5 to 7 business days to appear on the customer statement.
""",
    "auth.md": """
Users authenticate using JWT tokens that expire after 24 hours.
Refresh tokens are stored in Redis and remain valid for 30 days.
The login flow begins when users submit their email and password.
We hash passwords with bcrypt using 12 rounds before storing them.
The hashed password is compared with the stored hash in PostgreSQL.
On successful authentication we generate a JWT containing the user ID and roles.
We return both an access token valid 24 hours and a refresh token valid 30 days.
Google and GitHub social login use OAuth 2.0 for third party authentication.
Multi factor authentication MFA is available via TOTP with Google Authenticator.
Failed login attempts are rate limited to 5 per 10 minutes per IP address.
""",
}


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 60)
    print("  Exercise 03: RAG Pipeline")
    print("=" * 60)

    # Test 1: chunk_text
    print("\n--- Test 1: chunk_text ---")
    result = chunk_text("word " * 80, source="test.md", chunk_size=20, overlap=5)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        stride = 20 - 5   # chunk_size - overlap
        expected_chunks = len(range(0, 80 - 20 + 1, stride))
        print(f"  Chunked 80-word doc: {len(result)} chunks (expected ~{expected_chunks})")
        if len(result) >= 3:
            print(f"  PASS  got {len(result)} chunks")
            print(f"  First chunk word count: {len(result[0]['text'].split())}")
            print(f"  First chunk id: {result[0]['id']}")
            print(f"  Second chunk starts with: {result[1]['text'][:30]}")
        else:
            print(f"  FAIL  expected at least 3 chunks, got {len(result)}")

    # Test 2: build_rag_index
    print("\n--- Test 2: build_rag_index ---")
    result2 = build_rag_index(DOCS)
    if result2 is None:
        print("  NOT IMPLEMENTED YET")
    else:
        chunks, index = result2
        print(f"  Chunks created:  {len(chunks)}")
        print(f"  Vocabulary size: {len(index.df)} terms")
        if len(chunks) >= 4 and len(index.df) > 10:
            print(f"  PASS  index built with {len(chunks)} chunks")
        else:
            print(f"  FAIL  expected more chunks/vocabulary")

    # Test 3: retrieve
    print("\n--- Test 3: retrieve ---")
    if result2 is not None:
        chunks, index = result2
        result3 = retrieve("how do users login and what token is used", chunks, index, top_k=3)
        if result3 is None:
            print("  NOT IMPLEMENTED YET")
        else:
            print(f"  Query: 'how do users login and what token is used'")
            print(f"  Retrieved {len(result3)} chunks:")
            found_auth = False
            for chunk, score in result3:
                print(f"    [{chunk['source']}] score={score:.2f}")
                print(f"    {chunk['text'][:70]}...")
                if chunk['source'] == 'auth.md':
                    found_auth = True
            if found_auth:
                print(f"  PASS  retrieved auth.md chunk for login query")
            else:
                print(f"  FAIL  expected auth.md in results")

    # Test 4: build_prompt
    print("\n--- Test 4: build_prompt ---")
    if result2 is not None and result3 is not None:
        question = "How does user authentication work?"
        result4 = build_prompt(question, result3)
        if result4 is None:
            print("  NOT IMPLEMENTED YET")
        else:
            has_question = question in result4
            has_context  = "CONTEXT" in result4.upper() or "context" in result4
            has_answer   = "ANSWER" in result4.upper() or "answer" in result4
            has_source   = "auth.md" in result4

            checks = [
                (has_question, "Contains the question"),
                (has_context,  "Has CONTEXT section"),
                (has_answer,   "Has ANSWER placeholder"),
                (has_source,   "Includes source label (auth.md)"),
            ]
            for passed, label in checks:
                status = "PASS" if passed else "FAIL"
                print(f"  {status}  {label}")

            print(f"\n  Prompt preview (first 15 lines):")
            print("  " + "-" * 50)
            for line in result4.split('\n')[:15]:
                print(f"  {line}")

    # Bonus: Unknown question grounding
    print("\n--- BONUS: Grounding Test (unknown question) ---")
    if result2 is not None:
        chunks, index = result2
        unknown_q = "What is the CEO's email address?"
        unknown_retrieved = retrieve(unknown_q, chunks, index, top_k=3)
        if build_prompt is not None:
            prompt = build_prompt(unknown_q, unknown_retrieved)
            if prompt:
                print(f"  Query: '{unknown_q}'")
                print(f"  Retrieved chunks: {len(unknown_retrieved)}")
                if not unknown_retrieved:
                    print(f"  PASS  no chunks retrieved — LLM should say 'I don't know'")
                else:
                    print(f"  Chunks found but scores may be low:")
                    for c, s in unknown_retrieved:
                        print(f"    [{c['source']}] score={s:.3f}")
                    print(f"  Grounded prompt means LLM will correctly say 'I don't know'")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def chunk_text(text, source, chunk_size=60, overlap=15):
#     words = text.split()
#     stride = chunk_size - overlap
#     chunks = []
#     for start in range(0, len(words), stride):
#         chunk_words = words[start : start + chunk_size]
#         if len(chunk_words) < 10:
#             break
#         chunks.append({
#             "id":     f"{source}:chunk{len(chunks)}",
#             "source": source,
#             "text":   " ".join(chunk_words),
#         })
#     return chunks
#
# def build_rag_index(documents):
#     all_chunks = []
#     for source, text in documents.items():
#         all_chunks.extend(chunk_text(text, source))
#     tokenized = [preprocess(chunk["text"]) for chunk in all_chunks]
#     index = BM25(tokenized)
#     return all_chunks, index
#
# def retrieve(query, chunks, index, top_k=3):
#     q_tokens = preprocess(query)
#     scores = index.get_scores(q_tokens)
#     ranked = sorted(enumerate(scores), key=lambda x: -x[1])
#     return [(chunks[i], s) for i, s in ranked[:top_k] if s > 0]
#
# def build_prompt(question, retrieved):
#     if not retrieved:
#         context = "No relevant documents found."
#     else:
#         parts = [f"[{i}. {c['source']} | score={s:.2f}]\n{c['text']}"
#                  for i, (c, s) in enumerate(retrieved, 1)]
#         context = "\n\n".join(parts)
#     return f"""Answer using ONLY the context. Say "I don't know" if not in context.
#
# CONTEXT:
# {context}
#
# QUESTION: {question}
#
# ANSWER:"""
