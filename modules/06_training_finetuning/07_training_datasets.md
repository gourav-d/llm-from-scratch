# Lesson 07 — Training Datasets: C4, Common Crawl, and How LLMs Get Their Data

**Module:** 06 — Training & Fine-tuning  
**Prerequisite:** Lesson 03 (training GPT), Lesson 04 (fine-tuning)

---

## Why Training Data Matters

A model is only as good as its training data. Before learning about C4, understand this:

> **An LLM learns everything from text. If the text is bad, the model is bad.**

> **C# analogy:** Garbage in, garbage out. Training data is the schema + seed data for your model's "database of knowledge." If your seed data has errors, your queries will return wrong results forever.

---

## The Internet as Training Data: Common Crawl

Most large LLMs start with **Common Crawl** — a free, public archive of the internet.

```
What is Common Crawl?
─────────────────────
• Non-profit organization
• Has been crawling the internet since 2008
• Every month: downloads ~3-4 billion web pages
• Stores raw HTML, extracted text, metadata
• Available for free download (petabytes of data)
• URL: commoncrawl.org

Scale (as of 2024):
• ~100 billion web pages in archive
• ~400TB new data per crawl
• Used to train: GPT-3, T5, LLaMA, BERT, and nearly every major LLM
```

---

## The Problem: Raw Web Text is Dirty

You cannot train an LLM directly on Common Crawl. The raw data contains:

```
Problems in raw Common Crawl:
──────────────────────────────────────────────────────────────
Problem                      Example
──────────────────────────────────────────────────────────────
HTML boilerplate             <nav><ul><li><a href="...">Home
Footer/menu text             "About Us | Contact | Privacy Policy"
Duplicate content            Same article scraped 10,000 times
Non-English text             Chinese, Arabic, German pages (if English only)
Spam and gibberish           "buy cheap pills!!! click here!!!"
Adult/offensive content      Harmful text that would corrupt model behavior
Short/meaningless text       "OK" "Yes" "Thanks" (3-word pages)
Code/markup artifacts        {json:"value", key:"data"}
SEO spam                     Keyword-stuffed doorway pages
──────────────────────────────────────────────────────────────
```

Every major dataset applies heavy filtering to fix these problems.

---

## C4: Colossal Clean Crawled Corpus

**C4** is the cleaned version of Common Crawl used to train **T5**.  
Created by Google in 2019. Paper: "Exploring the Limits of Transfer Learning with T5."

### Size

| Property | Value |
|----------|-------|
| Raw Common Crawl input | ~20TB per crawl snapshot |
| After filtering | **750GB** (compressed) |
| Tokens | ~156 billion tokens |
| Documents | ~365 million documents |
| Language | English only |

---

## How C4 Was Created: The Filtering Pipeline

Google applied a series of filters to Common Crawl to produce C4. Each step removes more junk:

```
Step 1: Extract text from HTML
        ───────────────────────
        Raw HTML → strip all tags → keep visible text only
        Remove JavaScript, CSS, navigation menus


Step 2: Language detection
        ──────────────────
        Keep only pages where langdetect says English > 99% confidence
        (removes ~70% of raw data)


Step 3: Quality heuristics (remove pages if ANY are true):
        ─────────────────────────────────────────────────
        • Contains fewer than 3 sentences
        • Any line ends with a bad word (profanity list)
        • Contains "lorem ipsum" placeholder text
        • Contains "{" (likely code or templates, not prose)
        • Contains words "javascript must be enabled"
        • Less than 200 characters
        • More than 10% punctuation marks
        • More than 30% numeric characters


Step 4: Deduplicate
        ───────────
        Remove exact duplicate 3-sentence spans
        If a sentence appears in 3+ documents, remove from all
        (Wikipedia gets duplicated many times on the web)


Step 5: Final output = C4
        ────────────────────
        750GB of clean, English, deduplicated prose text
```

> **C# analogy:** This is like a LINQ pipeline — `.Where()`, `.Select()`, `.Distinct()` — but applied to billions of web pages.

---

## Visualizing the C4 Pipeline

```
Common Crawl (20TB raw HTML)
         │
         ▼  Step 1: Extract text
Text corpus (~6TB extracted text)
         │
         ▼  Step 2: Keep English only
English text (~1.8TB)
         │
         ▼  Step 3: Quality filters
Quality text (~900GB)
         │
         ▼  Step 4: Deduplicate
C4 (~750GB, ~156B tokens)
         │
         ▼  Used to train
       T5 model
```

750GB sounds large but it's only ~4% of the original Common Crawl data.

---

## What's in C4?

C4 is dominated by news articles, blog posts, and general web prose:

```
Content distribution (approximate):
┌───────────────────────────────────────┐
│ News / current events      ~35%       │
│ General web prose          ~30%       │
│ Wikipedia (deduplicated)    ~5%       │
│ Books / long-form           ~5%       │
│ Forums / discussions       ~15%       │
│ Other                      ~10%       │
└───────────────────────────────────────┘
```

**C4 does NOT contain:**
- Source code (filtered out by the `{` rule)
- Academic papers (mostly)
- Private communications
- Data after the crawl date

---

## Training Data for Other Major Models

| Model | Primary Dataset | Size | Notes |
|-------|----------------|------|-------|
| T5 | C4 | 750GB / 156B tokens | Clean English web prose |
| BERT | BooksCorpus + Wikipedia | ~16GB / 3.3B tokens | Books + Wikipedia only |
| GPT-2 | WebText | 45GB / 8B tokens | Reddit-linked pages (karma filtered) |
| GPT-3 | Common Crawl (filtered) + others | 570GB / 300B tokens | Multiple sources mixed |
| LLaMA 2 | Mixed (CommonCrawl, C4, GitHub, Wikipedia, books) | ~2TB / 2T tokens | Diverse sources |
| LLaMA 3 | Mixed, heavily filtered | ~15T tokens | Much more data |

---

## Data Quality vs Data Quantity

A key research finding from recent years:

> **Fewer tokens of higher quality > More tokens of lower quality**

```
Example:
────────────────────────────────────────────────────────
LLaMA 1 (2023): 1.4T tokens from mixed sources
LLaMA 2 (2023): 2T tokens with better filtering
Phi-1 (2023):   7B tokens — only textbook-quality content
                             (MIT research papers, synthetically generated exercises)
                → Phi-1 outperformed models trained on 100x more raw tokens
                   on coding benchmarks
────────────────────────────────────────────────────────
```

> **C# analogy:** 100 rows of clean, validated data beats 10,000 rows with NULLs, duplicates, and wrong types.

---

## The Deduplication Problem

Duplicate text causes models to memorize rather than generalize:

```
Problem scenario:
  If "The quick brown fox jumps over the lazy dog" appears 50,000 times
  in training data, the model will memorize and repeat it verbatim.
  
  This is bad because:
  - Model wastes capacity memorizing one sentence
  - Model might reproduce copyrighted text
  - Model doesn't learn general patterns, just specific text

Solution in C4:
  Remove any 3-sentence span that appears in 3+ documents.
  Remove exact duplicates entirely.
```

---

## Tokenization and Token Count

When researchers say "trained on 156 billion tokens" they mean:

```
Text:    "The cat sat on the mat"
Tokens:  ["The", "cat", "sat", "on", "the", "mat"]  → 6 tokens

But BPE tokenization might give:
Tokens:  ["The", " cat", " sat", " on", " the", " mat"]  → still 6 tokens

Longer word: "transformers"
BPE tokens: ["transform", "ers"]  → 2 tokens

156 billion tokens ≈ ~120 billion words ≈ ~750GB of text
(average ~5 chars per token)
```

> **C# analogy:** Tokens are like `string.Split()` but smarter — not splitting on spaces, splitting on learned subword boundaries.

---

## Why This Matters for You (as a Developer)

When you fine-tune a model (Module 12) or evaluate it (Module 13), you need to understand:

1. **Base model biases** come from training data. C4 is mostly US/UK English web content from 2019. The model reflects that.

2. **Knowledge cutoff** = the date of the training crawl. C4's crawl was April 2019. T5 doesn't know about events after that.

3. **Data contamination** = if your benchmark test questions were in the training data, performance scores are inflated. This is a real problem in LLM evaluation.

4. **Your fine-tuning data quality matters as much as quantity.** 100 clean examples often beat 10,000 noisy examples.

---

## Quiz Questions

**Q1:** What is Common Crawl?  
→ A free public archive of billions of web pages, crawled monthly since 2008. Used as source data for most major LLMs.

**Q2:** Why can't you train on raw Common Crawl data directly?  
→ Contains HTML boilerplate, spam, duplicates, non-English content, offensive text, and low-quality pages.

**Q3:** What is C4, and what model was it used to train?  
→ Colossal Clean Crawled Corpus. 750GB of filtered English web prose. Used to pre-train T5 (Google, 2019).

**Q4:** Name 3 filters used to create C4 from Common Crawl.  
→ Language detection (English only), quality heuristics (min sentence count, remove `{`, remove profanity), deduplication (remove repeated 3-sentence spans).

**Q5:** Why does deduplication matter for LLM training?  
→ Duplicate text causes memorization instead of generalization. Model wastes capacity and may reproduce copyrighted text verbatim.

**Q6:** What does "knowledge cutoff" mean?  
→ The date of the training data crawl. The model has no knowledge of events after that date.

---

## Key Takeaways

- Common Crawl = free internet archive. Source for most LLM training data.
- Raw web data is too dirty to use directly. Filtering reduces 20TB → 750GB for C4.
- C4 pipeline: extract text → language filter → quality heuristics → deduplicate.
- Different models use different datasets: BERT (books + Wikipedia), GPT-2 (Reddit-linked), T5 (C4), GPT-3 (mixed).
- Quality beats quantity: fewer clean tokens often outperforms more noisy tokens.
- Knowledge cutoff = when the crawl happened. Model knows nothing after that date.
- These principles apply when you build your own fine-tuning datasets in Module 12.
