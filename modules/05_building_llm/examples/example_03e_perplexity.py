"""
=============================================================================
EXAMPLE 03E: Sentence Scoring and Perplexity
=============================================================================

GLOSSARY
---------
Probability of a sentence : How likely is this exact sentence according to
                             the model? Computed by multiplying together
                             the probability of each word given the previous.

Log-probability  : log(probability). We use log because multiplying many
                   small probabilities causes "underflow" (number too small
                   for float to represent). Adding logs avoids this.
                   C# analogy: Math.Log(probability)

Perplexity       : exp(average_negative_log_prob_per_token)
                   Lower perplexity = model finds sentence more natural.
                   Higher perplexity = model finds sentence surprising/weird.

                   Intuition: "How many words would I need to guess at random
                   to generate this sentence?" Lower is better.

Cross-entropy    : The average negative log-probability per token.
                   cross_entropy = -sum(log P(each word)) / num_words
                   Perplexity = exp(cross_entropy)

                   NOTE: When you call F.cross_entropy() in PyTorch during
                   training, it IS computing this. Loss = cross-entropy.
                   Perplexity = exp(loss). They are the same thing!

Language model   : Any model that assigns probabilities to text sequences.
                   Bigram, trigram, GPT -- all are language models.
                   The BETTER the LM, the LOWER the perplexity on real text.

Sentence ranking : Using perplexity to compare sentences.
                   "Which phrasing sounds more natural?"
                   The sentence with lower perplexity is more natural
                   according to the model.

=============================================================================
WHY PERPLEXITY MATTERS
=============================================================================

After you train an LLM, how do you measure if it is GOOD?

  Option 1: Read the output and judge (slow, expensive, subjective)
  Option 2: Use perplexity (fast, automatic, objective)

Perplexity is the CORE metric for evaluating language models.
It is computed on a held-out test set (sentences the model never saw).

  Low perplexity  = model predicts test sentences well  = good model
  High perplexity = model is surprised by test sentences = bad model

Real example:
  GPT-2 small:   perplexity ~29 on WikiText-103
  GPT-2 large:   perplexity ~19 on WikiText-103
  GPT-3:         perplexity ~20 on WebText (different dataset)
  Human text:    perplexity ~12-15 (ceiling -- humans are very predictable)

Our toy bigram will have MUCH higher perplexity because it is tiny.
But the concept and calculation are identical.

=============================================================================
THE MATH (explained simply)
=============================================================================

Sentence: "the meeting is scheduled"
Bigram model assigns:

  P(meeting | the)     = 0.33  (model says: "meeting" follows "the" 33% of time)
  P(is | meeting)      = 0.90  (model says: "is" follows "meeting" 90% of time)
  P(scheduled | is)    = 0.25  (model says: "scheduled" follows "is" 25% of time)

  P(sentence) = 0.33 x 0.90 x 0.25 = 0.074

  BUT: with many tokens, multiplying small numbers -> 0.00000001
  Python float can't represent this accurately.

SOLUTION: use log-probabilities

  log P(sentence) = log(0.33) + log(0.90) + log(0.25)
                  = -1.11     + -0.105    + -1.386
                  = -2.60

  Average log-prob per token = -2.60 / 3 = -0.867
  Cross-entropy              = 0.867
  Perplexity                 = exp(0.867) = 2.38

Interpretation: "On average, the model had ~2.4 equally-likely choices
                 at each token position when predicting this sentence."

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("EXAMPLE 03E: Sentence Scoring and Perplexity")
print("=" * 60)

# =============================================================================
# STEP 1: Training data
# =============================================================================

print("""
STEP 1: Training Data
Train on office phrases.
Then SCORE new sentences: which sound "natural" vs "weird" to the model?
""")

train_sentences = [
    "the meeting is scheduled tomorrow",
    "the meeting is cancelled today",
    "the report is due tomorrow",
    "the report is complete today",
    "the project is delayed tomorrow",
    "the project is done today",
    "the budget is approved today",
    "the budget is rejected today",
    "the client is happy today",
    "the client is waiting today",
    "the deadline is tomorrow",
    "the team is ready today",
    "the server is down today",
    "the server is running today",
    "the feature is done today",
    "the release is scheduled tomorrow",
    "the deployment is complete today",
    "the ticket is closed today",
    "the review is pending today",
    "the review is approved today",
]

print(f"Training sentences: {len(train_sentences)}")

# =============================================================================
# STEP 2: Vocabulary and training pairs
# =============================================================================

all_words = [w for s in train_sentences for w in s.split()]
vocab = sorted(set(all_words))
vocab_size = len(vocab)
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for i, w in enumerate(vocab)}

x_list, y_list = [], []
for sentence in train_sentences:
    words = sentence.split()
    for i in range(len(words) - 1):
        x_list.append(w2i[words[i]])
        y_list.append(w2i[words[i + 1]])

x_train = torch.tensor(x_list, dtype=torch.long)
y_train = torch.tensor(y_list, dtype=torch.long)

print(f"Vocab size: {vocab_size}, Training pairs: {len(x_train)}")

# =============================================================================
# STEP 3: Train the model
# =============================================================================

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss


model = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

print("\nTraining for 600 steps...")
for step in range(600):
    _, loss = model(x_train, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 150 == 0:
        print(f"  Step {step:3d}: loss = {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.4f}")
print(f"Perplexity on training data: {math.exp(loss.item()):.2f}")

# =============================================================================
# STEP 4: Sentence scoring function
# =============================================================================

print("""
STEP 4: Score a Sentence
For each token in the sentence, get the probability of the NEXT token.
Average the log-probabilities. Compute perplexity.
""")

@torch.no_grad()
def score_sentence(model, sentence, w2i, verbose=False):
    """
    Compute the bigram log-probability and perplexity of a sentence.

    sentence  : string of space-separated words
    w2i       : word-to-index dict
    verbose   : if True, print per-token probabilities

    Returns: (log_prob, perplexity, is_in_vocab)
      log_prob    : total log probability (negative, lower = less likely)
      perplexity  : exp(-log_prob / num_tokens), lower = more natural
      is_in_vocab : False if sentence contains unknown words
    """
    words = sentence.split()

    # Check all words are known
    unknown = [w for w in words if w not in w2i]
    if unknown:
        return None, None, False

    total_log_prob = 0.0
    num_transitions = len(words) - 1   # number of bigram steps

    if num_transitions == 0:
        return 0.0, 1.0, True  # single word, trivial

    if verbose:
        print(f"\n  Scoring: '{sentence}'")
        print(f"  {'Token':<15} {'Next':<15} {'P(next|token)':<15} {'log P'}")
        print(f"  {'-'*55}")

    for i in range(num_transitions):
        current_word = words[i]
        next_word    = words[i + 1]

        # Get the model's probability distribution for the current word
        idx = torch.tensor([w2i[current_word]], dtype=torch.long)
        logits, _ = model(idx)
        log_probs = F.log_softmax(logits, dim=-1)  # log probabilities, shape (1, vocab_size)

        # Get the log-prob of the ACTUAL next word
        next_idx = w2i[next_word]
        token_log_prob = log_probs[0, next_idx].item()  # scalar

        total_log_prob += token_log_prob  # accumulate

        if verbose:
            prob = math.exp(token_log_prob)
            print(f"  {current_word:<15} {next_word:<15} {prob:<15.4f} {token_log_prob:.4f}")

    # Average log probability per token
    avg_log_prob = total_log_prob / num_transitions

    # Perplexity = exp(-avg_log_prob) = exp(cross_entropy)
    # NOTE: avg_log_prob is negative, so -avg_log_prob is positive
    perplexity = math.exp(-avg_log_prob)

    return total_log_prob, perplexity, True


# Demo: score one sentence in detail
print("Detailed scoring of 'the meeting is scheduled':")
log_p, ppl, ok = score_sentence(model, "the meeting is scheduled", w2i, verbose=True)
print(f"\n  Total log-prob     : {log_p:.4f}")
print(f"  Avg log-prob/token : {log_p/3:.4f}")
print(f"  Perplexity         : {ppl:.2f}")

# =============================================================================
# STEP 5: Compare natural vs unnatural sentences
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Comparing Sentences -- Natural vs Unnatural")
print("=" * 60)
print("""
Lower perplexity  = model finds this MORE natural (trained on similar text)
Higher perplexity = model finds this SURPRISING (pattern not seen in training)

We test:
  1. Sentences exactly from training data          (should be low perplexity)
  2. New sentences with the same pattern           (should be low perplexity)
  3. Sentences with scrambled word order           (should be high perplexity)
  4. Sentences with out-of-pattern words           (should be high perplexity)
""")

test_cases = [
    # Category, sentence
    ("TRAINING (exact match)",   "the meeting is scheduled tomorrow"),
    ("TRAINING (exact match)",   "the server is down today"),
    ("NEW (same pattern)",       "the project is complete today"),
    ("NEW (same pattern)",       "the client is ready today"),
    ("SCRAMBLED word order",     "is the meeting tomorrow scheduled"),
    ("SCRAMBLED word order",     "today server the running is"),
    ("REVERSED sentence",        "tomorrow scheduled is meeting the"),
    ("UNUSUAL pattern",          "the server tomorrow today is"),
]

print(f"\n{'Category':<30} {'Sentence':<45} {'Perplexity'}")
print("-" * 85)

results = []
for category, sentence in test_cases:
    log_p, ppl, in_vocab = score_sentence(model, sentence, w2i)
    if not in_vocab:
        print(f"{category:<30} {sentence:<45} {'(unknown words)'}")
    else:
        print(f"{category:<30} {sentence:<45} {ppl:>6.2f}")
        results.append((ppl, sentence, category))

# =============================================================================
# STEP 6: Rank a list of candidate sentences
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Ranking Candidate Completions")
print("=" * 60)
print("""
Real use case: you have a partial sentence and multiple possible completions.
Use perplexity to RANK them: lower perplexity = more natural completion.

This is how early spell-checkers and grammar-checkers worked.
GPT's beam search also scores candidate completions -- same idea, bigger model.
""")

# Given a context, which completion is most natural?
completion_tests = [
    {
        "context": "the server is",
        "candidates": ["down today", "running today", "approved today", "today running"],
    },
    {
        "context": "the release is",
        "candidates": ["scheduled tomorrow", "delayed tomorrow", "happy today", "tomorrow tomorrow"],
    },
    {
        "context": "the budget is",
        "candidates": ["approved today", "rejected today", "down today", "today approved"],
    },
]

for test in completion_tests:
    context = test["context"]
    print(f"\nContext: '{context}'")
    print(f"  Ranking completions:")

    scored = []
    for completion in test["candidates"]:
        full_sentence = f"{context} {completion}"
        _, ppl, ok = score_sentence(model, full_sentence, w2i)
        if ok:
            scored.append((ppl, completion))

    # Sort by perplexity (lowest first = most natural)
    scored.sort(key=lambda x: x[0])
    for rank, (ppl, completion) in enumerate(scored, start=1):
        marker = "<-- most natural" if rank == 1 else ""
        print(f"  {rank}. '{completion}'  (perplexity: {ppl:.2f})  {marker}")

# =============================================================================
# STEP 7: Perplexity = exp(loss) -- connecting training to evaluation
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Connecting Loss to Perplexity")
print("=" * 60)
print("""
THIS IS THE KEY INSIGHT:

During training, PyTorch reports "loss" after each step.
That loss IS cross-entropy. Perplexity IS exp(loss).

They are THE SAME THING with different names:
  Training: "loss = 1.23"
  Evaluation: "perplexity = exp(1.23) = 3.42"

Both measure: "how surprised is the model by the actual next token?"
""")

# Show this directly
print("Computing loss and perplexity on individual test sentences:\n")

eval_sentences = [
    "the meeting is scheduled tomorrow",   # in training data
    "the server is down today",             # in training data
    "the project is complete today",        # new sentence, same pattern
    "today scheduled is the meeting",       # scrambled
]

for sentence in eval_sentences:
    words = sentence.split()
    # Skip unknown words
    if any(w not in w2i for w in words):
        print(f"  '{sentence}': contains unknown word")
        continue

    # Build pairs for this sentence
    xs = torch.tensor([w2i[words[i]]   for i in range(len(words)-1)], dtype=torch.long)
    ys = torch.tensor([w2i[words[i+1]] for i in range(len(words)-1)], dtype=torch.long)

    with torch.no_grad():
        _, sentence_loss = model(xs, ys)

    sentence_ppl = math.exp(sentence_loss.item())

    print(f"  '{sentence}'")
    print(f"    cross_entropy (loss) = {sentence_loss.item():.4f}")
    print(f"    perplexity = exp({sentence_loss.item():.4f}) = {sentence_ppl:.2f}")
    print()

# =============================================================================
# STEP 8: Limitations and what GPT improves
# =============================================================================

print("=" * 60)
print("STEP 8: Bigram Perplexity Limitations")
print("=" * 60)
print("""
LIMITATION 1: Vocabulary coverage
  Any word not in training vocab gets perplexity = infinity.
  "The server is offline today" -> "offline" is unknown -> can't score.
  GPT-4 has a vocab of 100,000 tokens, covering almost all English words.

LIMITATION 2: Context window
  Bigram perplexity only uses 1 token of context.
  "The server in building A that was replaced last month is down today"
  -> Bigram ignores all context. Just sees "is" -> "down".
  -> GPT uses the WHOLE sentence as context.

LIMITATION 3: No generalization
  Bigram memorizes bigram counts. It doesn't understand MEANING.
  "The computer is broken" and "The machine is broken" are different
  bigrams to this model, even though they mean the same thing.
  GPT embeds similar words near each other, so they share predictions.

WHAT GPT IMPROVES:
  - Vocab: subword tokenization handles ANY word
  - Context: attention covers the full context window
  - Generalization: dense embeddings capture word similarity

  But the EVALUATION METRIC (perplexity) stays the same.
  GPT models are benchmarked with perplexity on standard test sets.
  Lower perplexity on a test set = better language model.

  GPT-2 (2019): ~29 perplexity on WikiText-103
  GPT-3 (2020): ~20 perplexity on many benchmarks
  LLaMA-3 70B:  ~5 perplexity on many benchmarks (excellent)
""")

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
  Perplexity  = exp(cross_entropy_loss)
              = exp(-average_log_prob_per_token)
              = "how many equally-likely choices did the model face per token?"

  Lower  = model finds the sentence natural / expected
  Higher = model finds the sentence surprising / unnatural

  Use cases:
    1. Rank candidate completions (which phrasing is more natural?)
    2. Compare two language models (which scores lower ppl on test set?)
    3. Detect out-of-distribution text (very high ppl = unusual text)
    4. Track training progress (ppl goes down as training improves)

  Relationship to training:
    Training minimizes loss (cross-entropy).
    Lower loss = lower perplexity.
    They are the same thing: loss = log(perplexity).
""")

print("=" * 60)
print("All 03-series examples complete!")
print("  03  -- character-level bigram (predict next char)")
print("  03b -- word-level bigram (phone autocomplete)")
print("  03c -- code token bigram (IDE autocomplete)")
print("  03d -- trigram (2-token context window)")
print("  03e -- perplexity (sentence scoring and evaluation)")
print()
print("Next: example_04_gpt_pytorch.py -- attention + all of the above = GPT")
print("=" * 60)
