# Lesson 2 -- Bi-Encoders

---

## GLOSSARY

| Term                 | Plain English Definition                                                        |
|----------------------|---------------------------------------------------------------------------------|
| bi-encoder           | A model that encodes the QUERY and each DOCUMENT separately into vectors        |
| sentence embedding   | A single vector (list of numbers) representing the meaning of an entire sentence|
| cosine similarity    | Measures angle between two vectors. 1.0 = identical direction, 0.0 = unrelated  |
| dot product          | Multiply matching positions of two vectors and sum. Another similarity measure  |
| sentence-transformers| A Python library that provides pre-trained bi-encoder models                    |
| offline encoding     | Pre-computing document vectors BEFORE any user queries arrive                   |
| latency              | How long the user has to wait for a result                                      |
| word2vec             | An older model that creates word-level embeddings (one vector per word)         |
| BERT                 | A transformer model -- base of most modern sentence encoders                    |
| all-MiniLM-L6-v2    | A popular small, fast bi-encoder model from HuggingFace                         |

---

## 1. What Is a Bi-Encoder?

The word "bi" means TWO. A bi-encoder has two separate encoding paths:

- **Path 1**: Encode the DOCUMENT --> get Document Vector
- **Path 2**: Encode the QUERY --> get Query Vector
- **Final step**: Compare vectors with cosine similarity --> get similarity score

The KEY insight: both paths use the EXACT SAME MODEL WEIGHTS.
It is not two different models -- it is one model used twice.

---

## 2. C#/.NET Analogy

Think of it like this C# code:

```csharp
// One method that converts ANY text to a "fingerprint" (vector)
double[] GetFingerprint(string text)
{
    // ... model magic inside ...
    return new double[] { 0.3, 0.8, 0.2, 0.7, ... };
}

// Bi-encoder works exactly like calling this method twice:
double[] documentFingerprint = GetFingerprint("We sell quick automobiles");
double[] queryFingerprint    = GetFingerprint("fast car");

// Then compare the fingerprints
double similarity = CosineSimilarity(documentFingerprint, queryFingerprint);
// similarity = 0.91 -- they are very similar in meaning!
```

The magic is that GetFingerprint() was trained to make similar meanings
produce similar fingerprints.

Compare to keyword search in C#:
```csharp
// Keyword search -- simple but misses synonyms
bool Matches(string doc, string query)
{
    return doc.Contains(query);  // "quick automobile" does NOT contain "fast car"
}
```

---

## 3. The Bi-Encoder Architecture

```
              QUERY                          DOCUMENT
         "fast car"                   "quick automobiles"
               |                               |
               v                               v
    +--------------------+        +--------------------+
    |   Tokenizer        |        |   Tokenizer        |
    | "fast" "car"       |        | "quick" "auto..."  |
    +--------------------+        +--------------------+
               |                               |
               v                               v
    +--------------------+        +--------------------+
    |  Transformer       |        |  Transformer       |
    |  (BERT-style)      |        |  (SAME weights!)   |
    +--------------------+        +--------------------+
               |                               |
               v                               v
    +--------------------+        +--------------------+
    |  Pooling Layer     |        |  Pooling Layer     |
    | (avg of all tokens)|        | (avg of all tokens)|
    +--------------------+        +--------------------+
               |                               |
               v                               v
    [0.4, 0.7, 0.3, 0.6]        [0.3, 0.8, 0.2, 0.7]
         Query Vector                Document Vector
               |                               |
               +---------------+---------------+
                               |
                               v
                    Cosine Similarity
                         = 0.91
                    (Very similar!)
```

Both paths share the SAME transformer. Training adjusts these weights so that
semantically similar texts produce similar vectors.

---

## 4. Why Bi-Encoders Are Fast

Here is the crucial business case for bi-encoders:

```
SCENARIO: You have 1 million documents and 10,000 queries per second.

OPTION A (slow): Re-encode every document for every query
  - 1 query x 1,000,000 docs = 1,000,000 encode calls
  - At 100ms each = 100,000 seconds per query. IMPOSSIBLE.

OPTION B (bi-encoder): Pre-encode documents ONCE, offline
  - OFFLINE (before any users arrive):
    Encode 1,000,000 docs once = 1,000,000 encode calls
    (This takes hours, but you do it once and save the results)

  - ONLINE (when user searches):
    Encode 1 query = 1 encode call (~10ms)
    Compare query vector to 1,000,000 stored vectors (~20ms)
    Total: ~30ms per query!
```

The timeline looks like this:

```
OFFLINE (done once):
  t=0:   Encode doc_1  --> save vector
  t=10ms: Encode doc_2 --> save vector
  ...
  t=hours later: All 1M docs encoded and saved

ONLINE (per query):
  User types "fast car"
  t=0ms:  Encode query --> query_vector
  t=10ms: Compare to all stored vectors --> find top 10
  t=12ms: Return results to user

User experience: ~12ms response time!
```

---

## 5. Cosine Similarity Deep Dive

Cosine similarity measures the ANGLE between two vectors.
It does not care about length -- only direction.

```
Two vectors pointing in SAME direction:
  query_vec    = [1.0, 1.0]
  doc_vec_good = [1.0, 0.9]  <- almost same direction
  cosine = 0.998  <-- near 1.0, very similar

Two vectors pointing in DIFFERENT directions:
  query_vec    = [1.0, 1.0]
  doc_vec_bad  = [-0.8, 0.2]  <- opposite direction
  cosine = -0.42  <-- near -1.0, opposite meaning

                 ^
          [1, 1] |  [1, 0.9]
               \ | /
                \|/
  ---------------+---------------->
               / |
              /  |
        [-0.8, 0.2]
```

The formula:

```
cosine_similarity(A, B) = (A dot B) / (|A| * |B|)

where:
  A dot B = A[0]*B[0] + A[1]*B[1] + A[2]*B[2] + ...
  |A|     = sqrt(A[0]^2 + A[1]^2 + ...)
  |B|     = sqrt(B[0]^2 + B[1]^2 + ...)
```

In NumPy:
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## 6. Bi-Encoder vs Word-Level Embeddings

Before bi-encoders, people used WORD-LEVEL models like word2vec.

### word2vec (older approach):
- Each WORD gets one vector
- To represent a sentence: average all word vectors
- Problem: "I love dogs, not cats" averaged with "I love cats, not dogs"
  produces the SAME vector! Meaning is lost.

### Bi-encoder (modern approach):
- The ENTIRE SENTENCE goes through a transformer
- Tokens interact via attention -- "not" modifies "love"
- The final vector captures the FULL sentence meaning
- "I love dogs, not cats" gets a DIFFERENT vector than "I love cats, not dogs"

```
Word2vec (OLD):
  "I love dogs not cats" --> average([I, love, dogs, not, cats])
  "I love cats not dogs" --> average([I, love, cats, not, dogs])
  Both averages are IDENTICAL! The model is confused.

Bi-encoder (NEW):
  "I love dogs not cats" --> transformer with attention --> unique vector A
  "I love cats not dogs" --> transformer with attention --> unique vector B
  A != B because "not" changes the meaning, and attention captures this.
```

---

## 7. Popular Bi-Encoder Models

| Model Name              | Size    | Speed   | Quality | Best For              |
|-------------------------|---------|---------|---------|----------------------|
| all-MiniLM-L6-v2        | 22MB    | Very fast| Good  | Most use cases        |
| all-mpnet-base-v2       | 420MB   | Moderate | Great | Higher quality needed |
| paraphrase-MiniLM-L6-v2 | 22MB    | Very fast| Good  | Paraphrase detection  |
| multi-qa-mpnet-base-v2  | 420MB   | Moderate | Great | Q&A search            |

All are available via the `sentence-transformers` library.

---

## 8. Quick Example: Bi-Encoder in Action

Using real code (Part B in the example files):

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model (downloads ~22MB first time)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode documents (do this ONCE, save results)
documents = [
    "We sell quick automobiles at competitive prices",
    "Scientists discovered new Amazon species",
    "Car sales are up this quarter",
]
doc_vectors = model.encode(documents)  # shape: (3, 384)

# Encode query (at query time)
query = "fast car"
query_vector = model.encode([query])[0]  # shape: (384,)

# Compare (cosine similarity)
from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity([query_vector], doc_vectors)[0]

# Rank
for i, score in sorted(enumerate(scores), key=lambda x: -x[1]):
    print(f"Score: {score:.3f} | {documents[i]}")

# Expected output:
# Score: 0.712 | We sell quick automobiles at competitive prices
# Score: 0.608 | Car sales are up this quarter
# Score: 0.089 | Scientists discovered new Amazon species
```

"Quick automobiles" scored highest for "fast car" -- semantic match works!

---

## Quiz -- Lesson 2

**Question 1:**
In a bi-encoder, when are document vectors computed?

A) At query time, for each user search
B) OFFLINE, once before any queries arrive, then saved
C) Inside the transformer during training only
D) They are never stored -- always recomputed

**Question 2:**
What does cosine similarity measure?

A) The length difference between two vectors
B) The angle between two vectors (direction similarity)
C) The exact element-by-element difference
D) How many elements two vectors have in common

**Question 3:**
Why is the bi-encoder faster than re-encoding documents for every query?

A) Because it uses a smaller model
B) Because documents are pre-encoded offline -- only 1 encode call needed at query time
C) Because it uses GPU acceleration
D) Because it only looks at the first 10 words of each document

**Question 4:**
What is the main weakness of word2vec compared to bi-encoders?

A) word2vec is slower
B) word2vec uses more memory
C) word2vec averages word vectors and loses sentence structure
D) word2vec only works for English

**Question 5:**
What is the output of a bi-encoder when you feed it a sentence?

A) A single number (like a score)
B) A list of words (like a summary)
C) A vector (list of ~384 numbers) representing the sentence meaning
D) A probability distribution over all words in the vocabulary

### Answers:
1. B -- Offline pre-encoding is the key efficiency of bi-encoders
2. B -- Cosine = angle between vectors (1.0 = same direction = similar meaning)
3. B -- Documents encoded once and saved; only 1 encode call per query
4. C -- Averaging loses the effect of words like "not" on sentence meaning
5. C -- A fixed-size vector (e.g., 384 numbers) encoding the sentence meaning

---

## Summary -- Lesson 2

**Key Takeaways:**

1. **Bi-encoder encodes query and document SEPARATELY** -- two encoding calls
2. **Both use the SAME model weights** -- one model, used twice
3. **Documents are pre-encoded OFFLINE** -- this is what makes it fast
4. **Cosine similarity measures angle** -- closer angle = more similar meaning
5. **Better than word2vec** -- sentence-level context via attention

**What is coming next:**

The bi-encoder is fast but not perfectly accurate. For a TOP-10 list, the top
result might actually be rank 3 when judged by a human expert.

Lesson 3 introduces the CROSS-ENCODER -- a slower but much more accurate model
that re-ranks the bi-encoder's results. Together they form the most powerful
search pipeline available today.
