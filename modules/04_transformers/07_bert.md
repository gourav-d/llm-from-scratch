# Lesson 07 — BERT: Bidirectional Encoder Representations from Transformers

**Module:** 04 — Transformers  
**Prerequisite:** Lessons 01–06 (attention, self-attention, multi-head, positional encoding, transformer block, GPT overview)

---

## What is BERT?

BERT is a **pre-trained language model** built by Google in 2018.

Think of it as a very smart reader that has read billions of web pages and learned to understand language deeply. Then you can hire that reader for your specific task (sentiment analysis, question answering, etc.) — this is called **fine-tuning**.

> **C# analogy:** BERT is like a pre-built NuGet package trained on massive data. You don't build it from scratch. You download it and plug it into your app.

---

## Encoder-Only Architecture

Transformers have three possible shapes:

```
┌─────────────────────────────────────────────────────────────┐
│  Encoder Only     │  Decoder Only     │  Encoder + Decoder  │
│  (BERT)           │  (GPT)            │  (T5, original)     │
│                   │                   │                     │
│  Reads input      │  Generates text   │  Reads then writes  │
│  Understands it   │  one token at     │  (translation,      │
│  Classifies it    │  a time           │  summarization)     │
└─────────────────────────────────────────────────────────────┘
```

BERT uses **only the encoder stack** from the original transformer paper.

---

## The Key Innovation: Bidirectionality

Before BERT, language models read text **left to right only** (like GPT).

BERT reads **both directions simultaneously**.

```
GPT reads:   "The cat sat on the ___"   → left to right → predicts "mat"

BERT reads:  "The cat [MASK] on the mat" → left AND right → fills "sat"
                 ↑                 ↑
             reads left         reads right
             "The cat"         "on the mat"
```

This gives BERT much deeper understanding of context. It knows what came before AND after a word.

> **C# analogy:** GPT is like reading a string with `string[0]` to `string[n]`. BERT is like having the full string in memory and checking characters on both sides simultaneously.

---

## How BERT Was Trained

Google trained BERT with two tasks on massive text (BooksCorpus + English Wikipedia, 3.3 billion words):

### Task 1: Masked Language Modeling (MLM)

- 15% of words in each sentence are replaced with `[MASK]`
- BERT must predict the original word
- This forces BERT to understand context from both sides

```
Input:  "The cat [MASK] on the mat"
Target: "sat"

Input:  "Paris is the [MASK] of France"
Target: "capital"
```

### Task 2: Next Sentence Prediction (NSP)

- Two sentences are given
- BERT must predict: are these sentences consecutive in the original text?

```
Input A: "She bought a ticket."  →  [SEP]  →  "She boarded the plane."   → IsNext? YES
Input A: "She bought a ticket."  →  [SEP]  →  "The sky is blue."         → IsNext? NO
```

This helps BERT understand relationships between sentences (useful for Q&A).

---

## BERT's Special Tokens

BERT uses special tokens that C# developers should recognize as sentinel values:

| Token | Meaning | C# analogy |
|-------|---------|-----------|
| `[CLS]` | Start of every input — holds the "sentence meaning" vector | Like a header byte in a protocol |
| `[SEP]` | Separator between two sentences | Like a delimiter character |
| `[MASK]` | Hidden word during training | Like a null placeholder |
| `[PAD]` | Padding to make all sequences same length | Like `new T[maxLength]` |

---

## BERT Input Format

Every input to BERT has three parts added together:

```
Token:     [CLS]  I    love  Paris  [SEP]  It   is   beautiful  [SEP]
            ↓     ↓     ↓     ↓      ↓     ↓    ↓       ↓        ↓
Token ID:  101   146   1567  3000   102   156  119     3567      102

Position:   0     1     2     3      4     5    6       7         8
            ↓     ↓     ↓     ↓      ↓     ↓    ↓       ↓        ↓
Pos Emb:  [PE0] [PE1] [PE2] [PE3] [PE4] [PE5] [PE6]  [PE7]   [PE8]

Segment:    A     A     A     A      A     B    B       B         B
            ↓     ↓     ↓     ↓      ↓     ↓    ↓       ↓        ↓
Seg Emb:  [SA] [SA]  [SA]  [SA]  [SA]  [SB] [SB]    [SB]     [SB]

Final Input = Token Embedding + Position Embedding + Segment Embedding
```

> **C# analogy:** Segment embeddings are like tagging each item in a list with which "group" it belongs to — Group A = first sentence, Group B = second sentence.

---

## BERT Variants

| Model | Layers | Hidden Size | Attention Heads | Parameters |
|-------|--------|-------------|-----------------|------------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |
| DistilBERT | 6 | 768 | 12 | 66M (smaller, faster) |
| RoBERTa | 12–24 | 768–1024 | 12–16 | 125M–355M |

> **RoBERTa** = "Robustly Optimized BERT" — Facebook AI trained BERT longer, with more data, and dropped the NSP task. Generally outperforms BERT.

---

## What BERT is Good At (Downstream Tasks)

After pre-training, BERT is fine-tuned on specific tasks:

```
┌──────────────────────────────────────────────────────────┐
│                    Pre-trained BERT                      │
└─────────────────────┬────────────────────────────────────┘
                      │  Fine-tune on labeled data
          ┌───────────┼───────────┬──────────────┐
          ▼           ▼           ▼              ▼
   Sentiment      Question    Named Entity    Text
   Analysis       Answering   Recognition   Similarity
   (classify      (find       (find names,  (are these
   [CLS] output)  answer      dates, etc.)  sentences
                  in text)                  similar?)
```

| Task | What BERT does | Real-world use |
|------|---------------|----------------|
| Sentiment analysis | Classifies `[CLS]` token output | Product review scoring |
| Q&A | Finds answer start/end positions in text | Search engines |
| NER | Labels each token (Person, Location, Date) | Parsing documents |
| Text similarity | Compares `[CLS]` embeddings | Duplicate detection |

---

## What BERT is NOT Good At

BERT **cannot generate text**. It only understands text.

```
Ask BERT: "Write me a poem about Paris"
BERT:      ??? (not designed for this)

Ask GPT:  "Write me a poem about Paris"
GPT:       "In Paris lights that softly glow..." ✓
```

> **Key rule:** BERT = understanding tasks. GPT = generation tasks.

---

## BERT vs GPT — Side by Side

| Feature | BERT | GPT |
|---------|------|-----|
| Architecture | Encoder only | Decoder only |
| Direction | Bidirectional | Left to right only |
| Training task | Masked word prediction | Next word prediction |
| Good for | Classification, Q&A, NER | Text generation, completion |
| Output | Vector for each input token | Next token probability |
| Can generate text? | No | Yes |
| Creator | Google (2018) | OpenAI (2018) |

---

## Visual: BERT Architecture

```
Input: [CLS] The cat sat on the mat [SEP]
         │    │   │   │   │   │   │   │
         ▼    ▼   ▼   ▼   ▼   ▼   ▼   ▼
    ┌─────────────────────────────────────┐
    │        Token + Position +           │
    │        Segment Embeddings           │
    └─────────────────────────────────────┘
         │    │   │   │   │   │   │   │
         ▼    ▼   ▼   ▼   ▼   ▼   ▼   ▼
    ┌─────────────────────────────────────┐
    │   Transformer Encoder Block 1       │  ← Multi-head attention
    │   (sees ALL tokens at once)         │    (bidirectional)
    └─────────────────────────────────────┘
         │    │   │   │   │   │   │   │
         ▼    ▼   ▼   ▼   ▼   ▼   ▼   ▼
    ┌─────────────────────────────────────┐
    │   Transformer Encoder Block 2       │
    └─────────────────────────────────────┘
         ... (12 blocks total for BERT-base)
         │    │   │   │   │   │   │   │
         ▼    ▼   ▼   ▼   ▼   ▼   ▼   ▼
    [CLS]  The  cat  sat  on  the  mat [SEP]
    output  (rich context vectors for each token)
       │
       ▼  (use [CLS] for sentence-level tasks)
  Fine-tune head → Sentiment / Classification / etc.
```

---

## Quiz Questions

**Q1:** What does "bidirectional" mean in BERT?  
→ It reads context from both left and right of each word simultaneously.

**Q2:** BERT was trained with two tasks. Name them.  
→ Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).

**Q3:** What is the `[CLS]` token used for?  
→ It represents the entire sentence meaning. Used as input to classification heads.

**Q4:** Can BERT generate new text?  
→ No. BERT is for understanding (classification, Q&A). Use GPT for generation.

**Q5:** What is the difference between BERT and RoBERTa?  
→ RoBERTa is BERT trained longer, on more data, without NSP. Generally better performance.

---

## Key Takeaways

- BERT = encoder-only transformer. Reads bidirectionally. Understands language deeply.
- Pre-trained on masked word prediction + next sentence prediction.
- Cannot generate text. Best for classification, Q&A, similarity tasks.
- Special tokens: `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`.
- Fine-tune BERT on your labeled data for specific downstream tasks.
- Think of BERT as a smart reader for hire; GPT is a writer for hire.
