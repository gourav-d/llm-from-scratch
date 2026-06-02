# Module 17 — Concepts: LLM Evaluation & Benchmarks

---

# Lesson 1 — Training Metrics: Loss, Perplexity, Overfitting

## The First Question: Is Training Working?

Before comparing your model to GPT-4, you need to know if your model is learning at all.
Three numbers tell you this during training:

1. **Training loss** — is the model learning on training data?
2. **Validation loss** — is the model generalizing to unseen data?
3. **Perplexity** — a human-readable version of loss

---

## Cross-Entropy Loss (Recap)

You learned this in Module 05. Quick recap:

```
At each token position, the model outputs probabilities over vocabulary:
  P("cat") = 0.7
  P("dog") = 0.1
  P("the") = 0.05
  ...

The true next token is "cat".
Loss at this position = -log(0.7) = 0.357

Average loss over all positions = cross-entropy loss
```

Lower loss = model assigns higher probability to the correct next token.

**C# analogy:**
Imagine a multiple-choice test. Each question has one correct answer.
Loss measures how confident the model was on the correct answer.
If model says "I'm 70% sure it's A" and A is correct → loss = -log(0.7) = 0.357.
If model says "I'm 99% sure it's A" and A is correct → loss = -log(0.99) = 0.01.

---

## Perplexity

**Perplexity = exp(loss)**

```python
import math

loss = 3.5          # typical early training loss
perplexity = math.exp(loss)   # = 33.1

loss = 2.0          # good model
perplexity = math.exp(2.0)    # = 7.4

loss = 1.2          # excellent model
perplexity = math.exp(1.2)    # = 3.3
```

**Intuition:** Perplexity tells you how many words the model is "choosing between" on average.

```
Perplexity = 1    → Model is perfect. Knows exactly what comes next.
Perplexity = 10   → Model is effectively choosing between 10 equally likely words.
Perplexity = 100  → Model is very uncertain — like flipping a 100-sided die.
Perplexity = 50257 → Model is completely random (vocab size = 50257 for GPT-2).
```

**Real numbers:**
| Model | Perplexity (WikiText-103) |
|-------|--------------------------|
| Random baseline | ~50257 |
| n-gram model | ~100–200 |
| GPT-2 (small) | ~29.4 |
| GPT-2 (large) | ~18.3 |
| GPT-3 (175B) | ~5.0 |
| GPT-4 | estimated ~3–4 |

---

## Why Perplexity Over Loss?

Loss is in "nats" or "bits" — hard to intuitively compare.
Perplexity converts to a more intuitive scale: "how many choices on average?"

```
Loss = 3.5 → Perplexity = 33.1  (choosing between 33 words)
Loss = 3.6 → Perplexity = 36.6  (choosing between 37 words)

The perplexity difference (33 vs 37) is more intuitive than 3.5 vs 3.6.
```

---

## Loss Curves: What Good Training Looks Like

```
Loss
│
│\                          ← Training loss
│ \
│  \______
│         \_____
│               \___________   ← Both curves decrease together
│
│  \                        ← Validation loss
│   \
│    \_____
│          \________________   ← Good: tracks training loss
│
└────────────────────────────── Epoch
        Good training
```

Both training and validation loss should:
1. Start high
2. Decrease together
3. Plateau (level off) near the end

---

## Overfitting: The Warning Sign

```
Loss
│
│\                          
│ \                         ← Training loss (keeps going down)
│  \
│   \_______________________ 
│                            ← Training loss near zero
│
│  \                        
│   \___                    ← Validation loss drops...
│       \___
│           \_______________  ← ...then RISES! Model is memorizing.
│                            ← This is overfitting.
└────────────────────────────── Epoch
        Overfitting
```

**Overfitting signs:**
- Training loss keeps decreasing
- Validation loss stops decreasing (plateaus) then increases
- Gap between train and val loss grows wider over time

**What to do:**
- Stop training (early stopping — you learned this in M12)
- Add dropout to regularize
- Use more training data
- Reduce model size

---

## Underfitting: The Other Problem

```
Loss
│
│\
│ \_______________________  ← Training loss plateaus too high
│                           ← Model never really learned
│
│ \______________________   ← Validation loss also plateaus high
│
└────────────────────────────── Epoch
        Underfitting
```

**Underfitting signs:**
- Both losses plateau at a high value
- Perplexity remains high (>50 for language modeling)
- Model generates nonsense

**What to do:**
- Train longer
- Increase learning rate
- Use a larger model
- Check your data (maybe corrupted or too little)

---

## The Validation Split

To detect overfitting, you need data the model never trained on.

```python
import numpy as np

# Split your dataset: 90% train, 10% validation
total_tokens = 1_000_000
split_idx = int(0.9 * total_tokens)

train_data = tokens[:split_idx]    # 900,000 tokens
val_data   = tokens[split_idx:]    # 100,000 tokens

# During training:
for epoch in range(num_epochs):
    train_loss = train(model, train_data)    # update weights
    val_loss   = evaluate(model, val_data)   # NO weight updates
    
    print(f"Epoch {epoch}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")
    print(f"Perplexity: {math.exp(val_loss):.1f}")
```

---

## Key Training Metrics Summary

| Metric | Formula | Good Value | Warning |
|--------|---------|-----------|---------|
| Training loss | avg(-log P(correct token)) | Decreasing | Stuck → underfitting |
| Validation loss | same, on held-out data | Close to train loss | Rising → overfitting |
| Perplexity | exp(val_loss) | <30 for GPT-size | >100 → model not learning |
| Train/Val gap | val_loss - train_loss | < 0.5 | > 1.0 → overfitting |

---

## Quiz 1

**Q1.** Your model has train_loss=1.2, val_loss=3.8 after 10 epochs. What is happening?
- A) Underfitting — both losses are too high
- B) Overfitting — model memorized training data, fails on validation
- C) Good training — these are normal values
- D) Exploding gradients — losses are too different

**Q2.** Perplexity of 50 means:
- A) The model is 50% accurate
- B) The model makes 50 errors per sentence
- C) The model is effectively choosing between 50 equally likely words at each step
- D) The loss is 50

**Q3.** A validation loss that first decreases then increases is a sign of:
- A) Learning rate too high
- B) Underfitting
- C) Overfitting
- D) Correct training

**Q4.** If perplexity = 1, what does that mean?
- A) The model is completely random
- B) The model knows exactly what comes next — perfect prediction
- C) The loss is 1
- D) The model has seen this exact text before

**Answers:** Q1=B, Q2=C, Q3=C, Q4=B

---

---

# Lesson 2 — General Benchmarks: MMLU, HellaSwag, ARC

## Why We Need Standard Benchmarks

Every team believes their model is good. Without shared benchmarks, comparisons are impossible.

```
Team A: "Our model scores 94% on our internal test set!"
Team B: "Ours scores 97% on OUR internal test set!"

These numbers are meaningless — different test sets, different questions.

Open benchmark: "Both models score X% on MMLU."
Now we can compare fairly.
```

Standard benchmarks are like standardized tests — same questions for everyone.

---

## MMLU — Massive Multitask Language Understanding

**What it tests:** Broad knowledge across 57 academic subjects

**Format:** Multiple-choice, 4 options (A, B, C, D)

**Subjects include:**
```
STEM:         Mathematics, Physics, Chemistry, Biology, Computer Science
Humanities:   History, Philosophy, Law, Religion
Social:       Economics, Psychology, Sociology, Political Science
Other:        Medical, Finance, Business, Nutrition
```

**Example question:**
```
Question: What is the approximate half-life of Carbon-14?
A) 100 years
B) 5,730 years
C) 1,000,000 years
D) 50 years

Answer: B
```

**Scoring:**
```
Score = (number of correct answers) / (total questions) × 100%

Random baseline: 25% (4 choices, random pick)
Human expert average: ~89%

Model benchmarks (2024):
  GPT-3.5:  70%
  GPT-4:    86.4%
  LLaMA 3 70B: 82.0%
  Mistral 7B: 62.5%
```

**What it catches:**
- Factual knowledge breadth
- Reasoning ability across domains
- Does NOT test: creativity, long-form generation, instruction following

---

## HellaSwag — Common Sense Completion

**What it tests:** Complete a sentence or paragraph naturally (common sense reasoning)

**Format:** Multiple-choice — pick the most natural continuation

**Example:**
```
Context: "She found a stray cat in the park. She decided to take it home."
Continue:
  A) She drove her car to the bank to make a deposit.
  B) She made a bed for it using an old blanket.
  C) She finished her quarterly financial report.
  D) She called the fire department.

Answer: B
```

The wrong answers are "adversarially filtered" — they look plausible statistically
but make no sense as continuations. This is what makes HellaSwag hard.

**Scoring:**
```
Random baseline: 25%
Human performance: ~95%
GPT-2:   40%
GPT-3:   78%
GPT-4:   95.3%
LLaMA 3 8B: 82%
```

**What it catches:**
- Common sense reasoning
- World model (what happens next in real situations)
- Coherence of narrative

---

## ARC — AI2 Reasoning Challenge

**What it tests:** Science questions from grade 3 to grade 9

**Two difficulty tiers:**
- **ARC-Easy:** Questions solvable by common sense + basic knowledge
- **ARC-Challenge:** Questions that require multi-step reasoning (harder subset)

**Example (ARC-Challenge):**
```
Question: "Which of the following best explains why the ocean is salty?"
A) Fish produce salt as a metabolic byproduct
B) Salt is dissolved from rocks and carried by rivers into the ocean over millions of years
C) Ocean water evaporates and leaves salt behind
D) Salt falls from the atmosphere in rain

Answer: B
```

**Scoring (ARC-Challenge):**
```
Random baseline: 25%
Human: ~98%
GPT-4:   96.3%
LLaMA 3 8B: 79.7%
Mistral 7B:  60.0%
```

**What it catches:**
- Scientific reasoning
- Multi-step logic
- ARC-Challenge specifically filters questions where simple word matching works

---

## Summary: When to Use Which

```
┌─────────────┬──────────────────────────────┬──────────────────────┐
│ Benchmark   │ Tests                        │ Random Baseline      │
├─────────────┼──────────────────────────────┼──────────────────────┤
│ MMLU        │ Broad knowledge (57 subjects)│ 25%                  │
│ HellaSwag   │ Common sense completion      │ 25%                  │
│ ARC-Easy    │ Basic science reasoning      │ 25%                  │
│ ARC-Challenge│ Hard science reasoning      │ 25%                  │
└─────────────┴──────────────────────────────┴──────────────────────┘

All are multiple-choice → easy to evaluate automatically.
All include adversarial filtering → cannot cheat with pattern matching.
```

---

## How Models Are Evaluated on These Benchmarks

Two main methods:

**Method 1: Zero-shot / Few-shot prompting**
```
Prompt:
"Answer the following question by choosing A, B, C, or D.

Question: What is 2+2?
A) 3  B) 4  C) 5  D) 6

Answer:"

Model output: "B"
Check: B == B → Correct!
```

**Method 2: Log-probability scoring**
```
For each answer choice, compute: log P(choice | question + context)
Pick the highest log probability.

This works even if the model doesn't output "A" literally.
More robust than text matching.
```

The `lm-evaluation-harness` (Lesson 5) handles both methods automatically.

---

## Quiz 2

**Q1.** MMLU has a random baseline of 25%. What does this tell you?
- A) The test is 25% accurate
- B) There are 4 answer choices — random guessing gets 25%
- C) The average human scores 25%
- D) You need to get at least 25% to pass

**Q2.** HellaSwag uses "adversarially filtered" wrong answers. Why?
- A) To make the test harder for humans
- B) To prevent models from picking the right answer by statistical pattern matching alone
- C) To reduce the number of questions
- D) To ensure all answers are equally long

**Q3.** ARC-Challenge is harder than ARC-Easy. What specifically makes it harder?
- A) It has 8 answer choices instead of 4
- B) Questions require multi-step reasoning — simple word matching doesn't work
- C) It tests different subjects
- D) It requires longer answers

**Answers:** Q1=B, Q2=B, Q3=B

---

---

# Lesson 3 — Task Benchmarks: GSM8K, HumanEval, TruthfulQA

## Why Task-Specific Benchmarks?

MMLU/HellaSwag test general ability. But real use cases need specific evaluation:

```
"I want to use this LLM for a math tutoring app."
→ MMLU score doesn't tell you enough
→ You need: GSM8K (math word problems)

"I want to use this LLM for code generation."
→ MMLU tells you nothing
→ You need: HumanEval (code correctness)

"I'm worried about my model making up facts."
→ MMLU doesn't catch hallucinations
→ You need: TruthfulQA
```

---

## GSM8K — Grade School Math

**Full name:** Grade School Math 8K (8,500 problems)
**Created by:** OpenAI (2021)

**What it tests:** Multi-step math word problems that an 8-12 year old student would solve

**Example:**
```
Question:
"Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins
for her friends every day with 4. She sells the remainder at the farmers' market
for $2 per fresh duck egg. How much does she make every day at the farmers' market?"

Reasoning steps:
  1. Eggs per day: 16
  2. Eggs used: 3 (breakfast) + 4 (muffins) = 7
  3. Eggs sold: 16 - 7 = 9
  4. Revenue: 9 × $2 = $18

Answer: $18
```

**Why it's hard:**
- Requires multiple steps
- Models must track intermediate values
- Easy to make arithmetic errors
- Model must identify which operations to perform (not just compute)

**Scoring:**
```
Metric: accuracy (is the final answer correct?)

GPT-3 (175B, 5-shot):   57%
GPT-4:                   92%
LLaMA 3 8B:              79.6%
LLaMA 3 70B:             93.0%
Mistral 7B:              52.2%
```

**Evaluation method:**
The model is asked to "think step by step" (Chain-of-Thought prompting from M07).
Final answer extracted from model output with regex.
Compared to ground truth answer.

---

## HumanEval — Code Generation

**Created by:** OpenAI (2021)
**Size:** 164 Python programming problems

**What it tests:** Can the model generate correct, runnable Python code?

**Format:**
```python
# Given this function signature and docstring:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other
    than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """

# Model generates the function body:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
```

**Evaluation: Pass@k**

Run the generated code against unit tests. If tests pass → correct.

```
Pass@1: Probability that one attempt at the code passes all tests
Pass@10: Probability that at least one of 10 attempts passes tests
Pass@100: Probability that at least one of 100 attempts passes tests

Pass@k = 1 - (1 - p)^k   where p = per-attempt pass probability
```

**Scores (Pass@1):**
```
GPT-2:            3.6%
GPT-3 (codex):   47.0%
GPT-4:           87.0%
LLaMA 3 8B:      62.2%
Mistral 7B:      30.5%
```

**Why Pass@k matters:**
In production, you generate one answer (Pass@1).
In research, you measure the model's ceiling (Pass@100).
A model with Pass@1=40% but Pass@100=90% has good capability but poor consistency.

---

## TruthfulQA — Hallucination & Factuality

**Created by:** Stephanie Lin, Jacob Hilton, Owain Evans (2021)
**Size:** 817 questions across 38 categories

**What it tests:** Does the model give true answers, or does it confidently state popular misconceptions?

**Key insight:** Questions are designed specifically where humans often have false beliefs.
Models trained on human text tend to **learn and repeat those false beliefs**.

**Example questions:**
```
Q: "What is the boiling point of water?"
A: 100°C (this is true — easy for the model)

Q: "Can you get a cold from being cold?"
A: No (many humans believe yes — model often says yes to match training data)

Q: "Who really wrote Shakespeare's plays?"
A: William Shakespeare (many people believe conspiracy theories — model may repeat them)

Q: "What happens if you swallow gum?"
A: It passes through your digestive system normally
   (common myth: "gum stays in stomach 7 years" — model often repeats this)
```

**Scoring:**
Two metrics:
- **%Truthful** — Is the answer factually correct?
- **%Informative** — Does the answer actually answer the question (not just "I don't know")?

Best combined metric: **%Truthful × %Informative**

```
GPT-3 (175B):  ~58% truthful
GPT-4:         ~59% truthful (surprisingly not much better!)
Claude 2:      ~65% truthful
Fine-tuned models with RLHF/Constitutional AI: ~80%+

Human baseline: ~94% truthful
```

**Why TruthfulQA is humbling:**
Even GPT-4 struggles with TruthfulQA because it was trained on human text
which contains false beliefs. RLHF and Constitutional AI (M13) help a lot here.

---

## MATH — Competition Mathematics

**What it tests:** High school to competition-level math (harder than GSM8K)

```
GSM8K: "Janet has 16 eggs..."    ← elementary school
MATH:  "Find all real x such that x^4 - 5x^2 + 4 = 0"  ← AMC/AIME level

GPT-4 on GSM8K: 92%
GPT-4 on MATH:  52%

Huge gap → much harder reasoning required
```

---

## Summary Table

```
┌────────────────┬──────────────────────────────┬──────────────────────────┐
│ Benchmark      │ Tests                        │ Metric                   │
├────────────────┼──────────────────────────────┼──────────────────────────┤
│ GSM8K          │ Grade school math word probs │ Accuracy (final answer)  │
│ MATH           │ Competition math             │ Accuracy                 │
│ HumanEval      │ Python code generation       │ Pass@k (run tests)       │
│ TruthfulQA     │ Factuality vs common myths   │ %Truthful × %Informative │
└────────────────┴──────────────────────────────┴──────────────────────────┘
```

---

## Quiz 3

**Q1.** HumanEval uses "Pass@k" instead of accuracy. Why?
- A) Code has multiple correct solutions — running tests is more fair than exact match
- B) It is easier to compute
- C) Accuracy would always be 0% for code
- D) It matches how humans evaluate code

**Q2.** A model scores 90% on MMLU but only 30% on TruthfulQA. What does this tell you?
- A) The model is generally good but tends to repeat common misconceptions
- B) The model is bad at all tasks
- C) MMLU and TruthfulQA measure the same thing
- D) The model needs more training data

**Q3.** Why does GPT-4 score ~59% on TruthfulQA (not 99%)?
- A) GPT-4 is not smart enough for these questions
- B) The questions are designed to test for false beliefs common in human text — which the model learned
- C) TruthfulQA uses a different language than GPT-4 was trained on
- D) GPT-4 was not trained on TruthfulQA data

**Q4.** GSM8K uses Chain-of-Thought prompting for evaluation. Why?
- A) It makes the model answer faster
- B) Models reason better when prompted to show steps — this gets higher accuracy
- C) It is required by the benchmark specification
- D) Chain-of-Thought reduces hallucinations

**Answers:** Q1=A, Q2=A, Q3=B, Q4=B

---

---

# Lesson 4 — Text Quality Metrics: BLEU, ROUGE, BERTScore

## The Problem: Evaluating Free-Form Text

For multiple-choice benchmarks (MMLU, HellaSwag), evaluation is easy:
```
Model said "B". Correct answer is "B". → Correct!
```

For free-form text generation (translation, summarization, caption), it is hard:
```
Reference: "The cat is sitting on the mat."
Generated: "A cat has sat itself down upon the rug."

Are these the same? A human says yes.
A computer needs a metric.
```

---

## BLEU — Bilingual Evaluation Understudy

**Created:** 2002 by IBM for machine translation
**What it measures:** n-gram overlap between generated and reference text (precision-focused)

### What Is an n-gram?

```
Sentence: "The cat sat on the mat"

Unigrams (1-grams): [The], [cat], [sat], [on], [the], [mat]
Bigrams  (2-grams): [The cat], [cat sat], [sat on], [on the], [the mat]
Trigrams (3-grams): [The cat sat], [cat sat on], [sat on the], [on the mat]
```

### BLEU Calculation

```
Reference:  "The cat sat on the mat"
Generated:  "The cat sat on a rug"

Unigram precision:
  Generated words: [The, cat, sat, on, a, rug]
  Words in reference: The ✓, cat ✓, sat ✓, on ✓, a ✗, rug ✗
  Precision = 4/6 = 0.667

Bigram precision:
  Generated bigrams: [The cat, cat sat, sat on, on a, a rug]
  Bigrams in reference: The cat ✓, cat sat ✓, sat on ✓, on a ✗, a rug ✗
  Precision = 3/5 = 0.6

BLEU-1 = unigram precision = 0.667
BLEU-2 = geometric mean of unigram + bigram precision = sqrt(0.667 × 0.6) = 0.632
BLEU-4 = geometric mean of 1,2,3,4-gram precisions (standard)
```

**Brevity penalty:** BLEU penalizes very short outputs (easy to get 100% precision with 1 word).

### BLEU in Practice

```python
from evaluate import load

bleu = load("bleu")
result = bleu.compute(
    predictions=["The cat sat on a rug"],
    references=[["The cat sat on the mat"]]    # note: list of references
)
print(result["bleu"])    # 0.0 to 1.0 (higher is better)
```

**BLEU scores in the real world:**
| Score Range | Interpretation |
|-------------|---------------|
| < 0.1 | Very poor translation |
| 0.1 – 0.2 | Gist is clear, but many errors |
| 0.2 – 0.4 | Understandable, somewhat fluent |
| 0.4 – 0.6 | Good quality translation |
| > 0.6 | High quality, near human |

**BLEU's weakness:**
```
Reference: "The cat is sitting on the mat."
Generated: "On the mat sits the cat."

Meaning = identical. BLEU score ≈ 0.4 (low bigram overlap due to different word order)
BLEU penalizes paraphrase, even when meaning is preserved.
```

---

## ROUGE — Recall-Oriented Understudy for Gisting Evaluation

**Created:** 2004 for summarization evaluation
**What it measures:** n-gram overlap with focus on **recall** (did we capture the reference?)

### ROUGE vs BLEU

```
BLEU = precision = "Of words I generated, how many were in the reference?"
ROUGE = recall = "Of words in the reference, how many did I generate?"

Reference: "The quick brown fox jumps over the lazy dog"
Generated: "The fox jumps over the dog"

BLEU (precision): 6/6 = 1.0  ← every generated word is in reference!
ROUGE (recall):   6/9 = 0.67  ← missed "quick", "brown", "lazy"

Both matter. Usually report both.
```

### Three ROUGE Variants

**ROUGE-1:** Unigram overlap
```
ROUGE-1 Recall = (matching unigrams) / (total reference unigrams)
ROUGE-1 Precision = (matching unigrams) / (total generated unigrams)
ROUGE-1 F1 = 2 × (P × R) / (P + R)
```

**ROUGE-2:** Bigram overlap (more sensitive to word order)
```
Reference: "The cat sat on the mat"
Generated: "The cat sat on a rug"

ROUGE-2 Recall:
  Reference bigrams: [The cat, cat sat, sat on, on the, the mat] = 5 bigrams
  Matching: [The cat, cat sat, sat on] = 3
  ROUGE-2 Recall = 3/5 = 0.60
```

**ROUGE-L:** Longest Common Subsequence
```
LCS of "The cat sat on the mat" and "The cat sat on a rug" = "The cat sat on"
Length = 4

ROUGE-L Recall = 4/6 = 0.667
ROUGE-L Precision = 4/6 = 0.667
```

### ROUGE in Practice

```python
from evaluate import load

rouge = load("rouge")
result = rouge.compute(
    predictions=["The cat sat on a rug"],
    references=["The cat sat on the mat"]
)
print(result)
# {'rouge1': 0.727, 'rouge2': 0.500, 'rougeL': 0.727}
```

**When to use:**
- Summarization: ROUGE-1 and ROUGE-2 (did you capture key facts?)
- Translation: BLEU is more standard
- Both together give a better picture than either alone

---

## BERTScore — Semantic Similarity

**Created:** 2019 by Tianyi Zhang et al.
**What it measures:** Meaning similarity using BERT embeddings

**The problem BLEU/ROUGE miss:**
```
Reference: "The vehicle is moving quickly."
Generated: "The car is going fast."

BLEU score: ≈ 0.1  (almost no word overlap)
BERTScore:  ≈ 0.93 (high semantic similarity — same meaning!)

"vehicle" and "car" are semantically similar.
"moving quickly" and "going fast" are semantically similar.
BLEU cannot see this. BERTScore can.
```

**How BERTScore works:**

```
Step 1: Encode both sentences with BERT
  Reference tokens:  [The] [vehicle] [is] [moving] [quickly]
  Generated tokens:  [The] [car] [is] [going] [fast]
  
  Each token → 768-dimensional embedding vector

Step 2: Compute cosine similarity between every pair
  sim("vehicle", "car") = 0.91   ← high! similar meaning
  sim("moving", "going") = 0.87  ← high! similar meaning
  sim("quickly", "fast") = 0.88  ← high! similar meaning

Step 3: Match each reference token to its best matching generated token
  Precision: for each generated token, find best matching reference token
  Recall:    for each reference token, find best matching generated token
  F1: combine

BERTScore F1 ≈ 0.93
```

**BERTScore in Practice:**

```python
from evaluate import load

bertscore = load("bertscore")
result = bertscore.compute(
    predictions=["The car is going fast."],
    references=["The vehicle is moving quickly."],
    lang="en"
)
print(result["f1"])    # [0.93] approximately
```

### When to Use Each Metric

```
┌────────────────┬──────────────────────────────┬──────────────────────────────┐
│ Metric         │ Best For                     │ Weakness                     │
├────────────────┼──────────────────────────────┼──────────────────────────────┤
│ BLEU           │ Machine translation          │ Penalizes paraphrase         │
│ ROUGE-1/2      │ Summarization (fact recall)  │ Ignores meaning              │
│ ROUGE-L        │ Summarization (coherence)    │ Ignores meaning              │
│ BERTScore      │ Any free-form generation     │ Slow, needs BERT model       │
└────────────────┴──────────────────────────────┴──────────────────────────────┘

In practice: always report BLEU + ROUGE + BERTScore together.
They measure different things. One metric alone is misleading.
```

---

## Visual: How the Metrics See the Same Example

```
Reference: "The cat quickly ran across the road."
Generated: "A feline swiftly crossed the street."

Human says: SAME MEANING. Good translation.

BLEU:       ~0.05   ← almost no word overlap → thinks this is terrible
ROUGE-1:    ~0.15   ← few matching unigrams → also thinks terrible
BERTScore:  ~0.88   ← semantic similarity → correctly sees good quality

Lesson: BLEU/ROUGE are best when reference and output use similar words.
        BERTScore handles paraphrase and synonyms.
```

---

## Quiz 4

**Q1.** BLEU measures precision and ROUGE measures recall. In summarization, which is more important?
- A) BLEU (precision) — we want every generated word to be in the reference
- B) ROUGE (recall) — we want the summary to cover the key facts from the reference
- C) They are equally important
- D) Neither — use BERTScore only

**Q2.** BERTScore gives "The car is fast" a high similarity to "The vehicle is quick." Why can't BLEU do this?
- A) BLEU only works for translation, not similarity
- B) BLEU counts exact word matches — it cannot see that "car" and "vehicle" mean the same thing
- C) BLEU requires a reference corpus
- D) BLEU does not support English

**Q3.** You evaluate a summarization model and get: ROUGE-1=0.8, BERTScore=0.4. What might this mean?
- A) The model is good — ROUGE is more important
- B) The model is copying exact words from the source but missing the meaning
- C) The model is making up words
- D) These scores cannot both be true

**Answers:** Q1=B, Q2=B, Q3=B

---

---

# Lesson 5 — Evaluation in Practice

## The Open LLM Leaderboard

The most widely-used public ranking of open-source LLMs is the
**HuggingFace Open LLM Leaderboard** (huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).

A sample leaderboard entry looks like this:

```
┌────────────────────────┬──────┬──────────┬────────┬─────────┬──────────┐
│ Model                  │ Avg  │ MMLU     │ ARC-C  │ HellaSwag│ TruthfulQA│
├────────────────────────┼──────┼──────────┼────────┼─────────┼──────────┤
│ LLaMA-3-70B-Instruct   │ 80.1 │ 82.0     │ 87.6   │ 88.0    │ 62.8     │
│ Mistral-7B-Instruct-v2 │ 65.7 │ 62.5     │ 63.3   │ 81.0    │ 56.0     │
│ Phi-3-mini-4k          │ 68.2 │ 68.8     │ 68.6   │ 78.9    │ 56.5     │
└────────────────────────┴──────┴──────────┴────────┴─────────┴──────────┘
```

**How to read this:**
- "Avg" = average across all benchmarks
- Each benchmark column = accuracy %
- Higher = better
- Compare models at similar parameter counts (7B vs 7B, 70B vs 70B)

---

## lm-evaluation-harness

**Created by:** EleutherAI (open source)
**GitHub:** github.com/EleutherAI/lm-evaluation-harness

This is the standard tool used by HuggingFace leaderboard and most researchers.
It handles: prompt formatting, model inference, metric computation, result logging.

### Install and Run

```bash
pip install lm-eval

# Evaluate a HuggingFace model on MMLU and HellaSwag
lm_eval \
  --model hf \
  --model_args pretrained=mistralai/Mistral-7B-v0.1 \
  --tasks mmlu,hellaswag \
  --num_fewshot 5 \
  --device cuda:0 \
  --output_path results/mistral_eval.json
```

### Key Arguments

```
--model          Which model backend (hf, openai, anthropic, local)
--model_args     Model name + any loading options (dtype, device_map)
--tasks          Which benchmarks to run (comma-separated)
--num_fewshot    How many examples to include in prompt (0=zero-shot, 5=5-shot)
--device         cuda:0, cpu, or auto
--output_path    Where to save JSON results
```

### Available Tasks

```
Knowledge:    mmlu (all 57 subjects or individual ones like mmlu_mathematics)
Common sense: hellaswag, winogrande, piqa
Reasoning:    arc_easy, arc_challenge
Math:         gsm8k, math
Code:         humaneval
Factuality:   truthfulqa_mc1, truthfulqa_mc2
```

---

## Evaluation Workflow: Step by Step

```
Step 1: Decide what matters for your use case
  - Chatbot → MMLU + TruthfulQA + HellaSwag
  - Code assistant → HumanEval + MBPP + GSM8K
  - Math tutor → GSM8K + MATH + ARC-Challenge
  - RAG system → TruthfulQA (factuality is critical)

Step 2: Run baseline (before training/fine-tuning)
  lm_eval --model hf --model_args pretrained=your_model --tasks mmlu,gsm8k

Step 3: Train or fine-tune your model

Step 4: Run same evaluation on trained model

Step 5: Compare numbers
  - Did target benchmarks improve?
  - Did other benchmarks decrease? (catastrophic forgetting — M12)

Step 6: Qualitative check
  - Run 20 prompts from your use case manually
  - Compare before/after output quality
  - Check for hallucinations, repetition, format issues
```

---

## Reading Results: What Good Improvement Looks Like

```
Fine-tuning for math task (GSM8K focus):

BEFORE fine-tuning:
  MMLU:       62.5%   ← general knowledge
  GSM8K:      52.2%   ← math word problems
  HumanEval:  30.5%   ← code
  TruthfulQA: 56.0%   ← factuality

AFTER fine-tuning on math data:
  MMLU:       61.8%   ← slight drop (-0.7%) — acceptable
  GSM8K:      71.4%   ← big improvement (+19.2%) — goal achieved!
  HumanEval:  30.1%   ← no change — expected, math ≠ code
  TruthfulQA: 55.2%   ← slight drop — watch this

Decision: Fine-tuning worked for the target task (math) with minimal regression.
Acceptable trade-off. Ship it.
```

---

## Writing Your Own Evaluation: Without lm-eval

For simple evaluation on your own data:

```python
import math

def evaluate_model(model, tokenizer, eval_data):
    """
    eval_data: list of (prompt, expected_answer) tuples
    """
    correct = 0
    total = len(eval_data)
    total_loss = 0.0
    
    for prompt, expected in eval_data:
        # Generate answer
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=50)
        predicted = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Check correctness (exact match for structured tasks)
        if expected.strip().lower() in predicted.strip().lower():
            correct += 1
        
        # Compute loss (for perplexity)
        loss = compute_loss(model, tokenizer, prompt + expected)
        total_loss += loss
    
    accuracy = correct / total
    avg_loss = total_loss / total
    perplexity = math.exp(avg_loss)
    
    return {
        "accuracy": accuracy,
        "perplexity": perplexity,
        "total": total,
        "correct": correct
    }
```

---

## Common Pitfalls in Evaluation

### Pitfall 1: Training Data Contamination

```
If your training data contains the test set questions → inflated scores.
"Data contamination" — model memorized answers, not learned to reason.

Solution: Check if evaluation set was in your training data.
lm-eval has contamination detection built in.
```

### Pitfall 2: Prompt Sensitivity

```
Same model, different prompt format → very different scores:

Prompt A: "Answer: A, B, C, or D. Question: ..."  → 72% accuracy
Prompt B: "Question: ... The answer is:"          → 68% accuracy
Prompt C: "Q: ... A:"                             → 65% accuracy

Report the prompt format used. Use standard formats for comparison.
```

### Pitfall 3: 0-shot vs Few-shot

```
GPT-3 on HellaSwag:
  0-shot:  33.7%   ← barely above random
  1-shot:  54.7%   ← huge jump with just 1 example!
  5-shot:  78.5%   ← standard benchmark number
  10-shot: 79.3%   ← diminishing returns

Always report how many shots you used.
Comparing 0-shot to 5-shot is not a fair comparison.
```

### Pitfall 4: Single Benchmark = Misleading

```
"Our model beats GPT-4 on GSM8K!"

Fine, but:
  - What about MMLU? (general knowledge)
  - What about TruthfulQA? (safety)
  - What about HumanEval? (code)

One benchmark win does not mean the model is better overall.
Always report a benchmark suite.
```

---

## The Full Evaluation Checklist

```
Before releasing a model, check:

Training metrics:
  [ ] Train loss converges (decreasing smoothly)
  [ ] Val loss tracks train loss (no overfitting)
  [ ] Perplexity on held-out set is reasonable (<30 for 7B models)

Benchmark suite:
  [ ] MMLU score reported (general knowledge)
  [ ] HellaSwag score reported (common sense)
  [ ] ARC-Challenge score reported (reasoning)
  [ ] Task-specific benchmark (GSM8K/HumanEval/TruthfulQA depending on use case)
  [ ] All scores include: num_fewshot, prompt format, evaluation date

Text quality:
  [ ] BLEU/ROUGE on held-out test set (if generative task)
  [ ] BERTScore on held-out test set

Qualitative:
  [ ] 20+ prompts tested manually
  [ ] Red-teaming: tried adversarial prompts
  [ ] Checked for repetition, truncation, off-topic responses

Regression check:
  [ ] If fine-tuned: base model benchmarks compared (no catastrophic forgetting)
```

---

## Module 17 → Capstone Connection

After this module, you have everything needed for the **Capstone: Chat with Codebase**:

```
Capstone will need evaluation:
  - Retrieval quality: Precision@K, Recall@K on test queries (M10.8 concepts)
  - Answer quality: BERTScore between generated answer and reference answer
  - Factuality: are answers grounded in retrieved code? (manual check)
  - Latency: time per query (non-ML metric)
```

---

## Quiz 5

**Q1.** You fine-tune a model for code generation. Which benchmarks should you prioritize?
- A) MMLU + HellaSwag (general capability)
- B) HumanEval + GSM8K (code + reasoning)
- C) TruthfulQA + ARC (safety + science)
- D) BLEU + ROUGE (text quality)

**Q2.** A model scores 85% on GSM8K with 5-shot prompting. Your model scores 80% with 0-shot prompting. Is your model worse?
- A) Yes — 80% < 85%
- B) Cannot compare — different number of shots, not a fair comparison
- C) Yes but the difference is acceptable
- D) No — 0-shot is always harder so 80% is actually better

**Q3.** What is "training data contamination" and why is it a problem?
- A) The training data is too noisy
- B) Evaluation benchmark questions appear in training data, inflating scores — model memorized answers
- C) The model was trained on too many languages
- D) The training data contains offensive content

**Q4.** You see this result after fine-tuning: MMLU dropped from 62% to 61%, GSM8K improved from 52% to 71%. Is this acceptable?
- A) No — any drop in MMLU is a failure
- B) Yes — target benchmark improved significantly, general capability drop is minimal (1%)
- C) Cannot determine without more information
- D) No — you should never fine-tune if it affects other benchmarks

**Answers:** Q1=B, Q2=B, Q3=B, Q4=B

---

---

# Module 17 — Full Summary

## What You Learned

### Lesson 1: Training Metrics
- **Cross-entropy loss:** avg(-log P(correct token))
- **Perplexity:** exp(loss) — intuitive scale, lower is better
- **Overfitting:** val loss rises while train loss falls
- **Underfitting:** both losses plateau too high

### Lesson 2: General Benchmarks
- **MMLU:** 57 subjects, multiple choice, 25% random baseline, ~86% GPT-4
- **HellaSwag:** common sense sentence completion, adversarially filtered wrong answers
- **ARC-Challenge:** science questions requiring multi-step reasoning

### Lesson 3: Task Benchmarks
- **GSM8K:** math word problems, accuracy, CoT prompting essential
- **HumanEval:** Python code, evaluated by running tests, Pass@k metric
- **TruthfulQA:** factuality against common misconceptions, even GPT-4 ~59%

### Lesson 4: Text Quality Metrics
- **BLEU:** precision — how many generated words appear in reference (word matching)
- **ROUGE:** recall — how much of reference is covered by generated text
- **BERTScore:** semantic similarity using BERT embeddings — handles paraphrase

### Lesson 5: Evaluation in Practice
- **lm-eval harness:** `lm_eval --tasks mmlu,gsm8k --num_fewshot 5`
- Always compare same prompting format (0-shot vs 5-shot not comparable)
- Watch for data contamination
- Run full benchmark suite, never rely on one metric
- Report regression: did fine-tuning hurt other benchmarks?

---

## The Evaluation Matrix

```
Use Case         │ Primary Benchmarks      │ Text Quality
─────────────────┼─────────────────────────┼──────────────────
General chatbot  │ MMLU + HellaSwag + TruthfulQA │ BERTScore
Math assistant   │ GSM8K + MATH + ARC      │ Exact match
Code assistant   │ HumanEval + MBPP + GSM8K│ Pass@k
Summarization    │ TruthfulQA              │ ROUGE-1/2 + BERTScore
Translation      │ (custom)                │ BLEU-4
RAG system       │ TruthfulQA + custom QA  │ Precision@K + BERTScore
```

---

## You Are Now Ready for the Capstone

You have completed all 17 core modules:

```
M01 → Python Basics
M02 → NumPy & Math
M03 → Neural Networks
M04 → Transformers
M05 → Building LLM
M06 → Training & Fine-tuning
M07 → Reasoning & Coding
M08 → Prompt Engineering
M09 → Production LLM Apps
M10 → Vector Databases
M11 → LLM Agents
M12 → Fine-Tuning LLMs
M13 → RLHF & Alignment
M14 → Deploying LLMs
M15 → Advanced LLM Training
M16 → Modern LLM Architectures  ← just completed
M17 → LLM Evaluation            ← YOU ARE HERE

NEXT: Capstone — Chat with Codebase (offline RAG app)
```
