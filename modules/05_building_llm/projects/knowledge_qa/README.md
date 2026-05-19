# Project: Personal Knowledge Base Q&A

Ask questions about your own Module 05 learning notes in plain English.
Runs 100% locally. No API. No internet. No cost.

---

## What This Project Is

You built embeddings in `example_02`. You built GPT in `example_04`.
This project **combines both** into one real, daily-use application.

```
You type:  "what is nn.Embedding?"
App does:  search your notes --> find relevant paragraphs --> generate answer
```

Every skill used here was already taught in Module 05.

---

## Quick Start (run these 4 commands in order)

```bash
# Make sure you are in this folder first
cd modules/05_building_llm/projects/knowledge_qa

# Step 1: Train the embedding model (~3 minutes)
python embedder.py

# Step 2: Index your notes (~1 minute)
python indexer.py

# Step 3: Train the GPT generator (~5 minutes)
python generator.py

# Step 4: Start asking questions (every day!)
python app.py
```

Steps 1, 2, 3 run **once**. Step 4 runs every time you want to use the app.

---

## File-by-File Explanation

### `embedder.py` -- Phase 1: Train the Embedding Model

**What it does:**
- Reads all `.md` files from `modules/05_building_llm/`
- Builds a character vocabulary (same as every module 05 example)
- Trains a character embedding model using `nn.Embedding` + next-char prediction
- Saves the trained model to `saved/embedding_model.pt`
- Saves the vocabulary to `saved/vocab.json`

**Module 05 connection:**
- `nn.Embedding` -- same as `example_02_embeddings_pytorch.py`
- Training loop (zero_grad / backward / step) -- same as `example_03_bigram_pytorch.py`

**Run:** Once. Re-run only if you add new notes and want to retrain.

**Output files:**
```
saved/embedding_model.pt   <-- trained model weights
saved/vocab.json           <-- character vocabulary
```

---

### `indexer.py` -- Phase 2: Chunk Notes and Store Their Vectors

**What it does:**
- Loads `embedding_model.pt` from Phase 1
- Reads every `.md` file and splits it into paragraphs (~300 chars each)
- Embeds each paragraph using `encode_chunk()` (mean pooling of char vectors)
- Saves all chunks + their vectors to `saved/index.json`

**Module 05 connection:**
- Tokenization concept -- same as `example_01_tokenization_pytorch.py`
- `encode_chunk()` uses `nn.Embedding` mean pooling -- from `example_02`

**Run:** Once. Re-run whenever you add or change `.md` notes.

**Output files:**
```
saved/index.json   <-- list of {text, source_file, vector} for every paragraph
```

**What index.json looks like:**
```json
[
  {
    "text": "nn.Embedding is a lookup table. Maps token IDs to dense vectors...",
    "source": "03_bigram_pytorch.md",
    "vector": [0.12, -0.45, 0.78, ...]
  },
  ...
]
```

---

### `retriever.py` -- Phase 3: Find Relevant Chunks by Similarity

**What it does:**
- Loads the embedding model and `index.json`
- Given a question, embeds it using the same `encode_chunk()` method
- Computes cosine similarity between the question vector and every chunk vector
- Returns the top 3 most similar chunks

**Module 05 connection:**
- `F.cosine_similarity()` -- exact same call as `example_02_embeddings_pytorch.py` Part 3

**Run standalone to test search:**
```bash
python retriever.py
```
This runs 5 test questions and shows which chunks it finds.

**Key function used by app.py:**
```python
results = retrieve("what is cross entropy?", model, char_to_idx, texts, sources, vectors)
# returns: [{"text": "...", "source": "file.md", "score": 0.84}, ...]
```

**How cosine similarity works here:**
```
Question: "what is nn.Embedding?"
       |
       v  encode_chunk()
Question vector: [0.2, -0.1, 0.8, ...]
       |
       v  F.cosine_similarity vs every chunk
Scores: [0.84, 0.71, 0.23, 0.65, ...]
       |
       v  torch.topk(3)
Top 3 chunks returned
```

---

### `generator.py` -- Phase 4: Train GPT and Generate Answers

**What it does:**
- Reads all `.md` notes (same as embedder.py)
- Trains a TinyGPT model on that text so it learns the content
- Saves the trained GPT to `saved/gpt_model.pt`
- Provides `generate_answer()` function used by `app.py`

**Module 05 connection:**
- `TinyGPT` class -- **direct copy** of `example_04_gpt_pytorch.py`
- Same `GPTConfig`, `CausalSelfAttention`, `FeedForward`, `TransformerBlock`
- Same `generate()` method with temperature + torch.multinomial

**Run:** Once to train. Re-run to retrain with more steps for better quality.

**Output files:**
```
saved/gpt_model.pt    <-- trained GPT weights
saved/gpt_vocab.json  <-- GPT character vocabulary + config
```

**How generation works:**
```python
prompt = "Context: nn.Embedding is a lookup table...\nQuestion: what is it?\nAnswer:"
# GPT continues from "Answer:" --> generates answer text
```

**Honest note:** This is a tiny GPT (~42K parameters). Answers will read like notes,
not like ChatGPT. That is expected. The architecture is identical to GPT-2 -- just smaller.

---

### `app.py` -- Phase 5: The Daily-Use Application

**What it does:**
- Loads everything: embedding model, index, GPT model
- Runs an interactive loop in the terminal
- For each question: retrieve top 3 chunks --> build prompt --> generate answer --> print

**Commands inside the app:**

| Command | What it does |
|---|---|
| Type any question | Full Q&A: retrieve + generate |
| `search <query>` | Show retrieved chunks only (no generation) |
| `quit` or `exit` | Stop the app |

**Example session:**
```
Personal Knowledge Base Q&A
============================================================
Index: 127 chunks loaded
GPT model loaded (42,304 parameters)

Ready! Type your question below.
------------------------------------------------------------

Your question: what is cross entropy?

  Searching notes for: 'what is cross entropy?'...
  Retrieved 3 relevant chunk(s)
  Generating answer...

  ANSWER:
  --------------------------------------------------------
  Cross entropy measures how wrong the model predictions
  are. Lower loss means higher confidence on correct
  tokens. F.cross_entropy(logits, targets) applies softmax
  then negative log likelihood in one call.
  --------------------------------------------------------

  Sources:
    - 03_bigram_pytorch.md  (similarity: 0.861)
    - 04_building_gpt_pytorch.md  (similarity: 0.802)

Your question: search optimizer

  Found 3 relevant chunk(s):

  [1] Score: 0.879  |  Source: 03_bigram_pytorch.md
  optimizer.zero_grad() clears accumulated gradients...

Your question: quit
Goodbye!
```

---

## How All 5 Files Connect

```
embedder.py  ------>  saved/embedding_model.pt
                      saved/vocab.json
                             |
indexer.py   ------>  saved/index.json
(uses embedding model)
                             |
retriever.py  <---- used by app.py at query time
(loads embedding model + index, does cosine search)
                             |
generator.py  ------>  saved/gpt_model.pt
                       saved/gpt_vocab.json
                             |
app.py       <---- ties retriever + generator together
                   this is the file you run every day
```

---

## Saved Files Reference

All generated files go into the `saved/` folder (auto-created):

| File | Created by | Used by | What it contains |
|---|---|---|---|
| `embedding_model.pt` | embedder.py | indexer.py, retriever.py | Trained embedding weights |
| `vocab.json` | embedder.py | indexer.py, retriever.py | char_to_idx mapping |
| `index.json` | indexer.py | retriever.py, app.py | All chunks + vectors |
| `gpt_model.pt` | generator.py | app.py | Trained GPT weights |
| `gpt_vocab.json` | generator.py | app.py | GPT vocabulary + config |

---

## Tuning for Better Results

| Setting | File | Default | Try |
|---|---|---|---|
| `TRAIN_STEPS` | embedder.py | 3000 | 5000 for better embeddings |
| `TRAIN_STEPS` | generator.py | 1000 | 3000 for better generation |
| `EMBED_DIM` | embedder.py | 32 | 64 for richer vectors (slower) |
| `CHUNK_SIZE` | indexer.py | 300 | 500 for more context per chunk |
| `TOP_K` | retriever.py | 3 | 5 for more sources |
| `temperature` | app.py generate_answer call | 0.7 | 0.4 for more focused answers |

---

## Honest Limitations

This is an **educational build** (Path A). Limitations are expected:

- GPT is tiny (~42K params vs GPT-4's ~1.8 trillion). Answers read like notes, not clean Q&A.
- Embeddings are character-level. Works well for finding similar content, not perfect semantics.
- Training data = only your notes. Model knows nothing outside them.
- Total training time: ~10 minutes on CPU. Fast enough to run today.

**Path B upgrade (when ready):**
Replace the 10 lines inside `generate_answer()` in `generator.py` with one API call.
The entire retrieval layer (phases 1-3) stays 100% unchanged.
This is how production RAG systems work.
