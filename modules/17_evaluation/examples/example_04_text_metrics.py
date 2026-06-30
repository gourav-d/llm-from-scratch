"""
Example 04: Text Quality Metrics -- BLEU, ROUGE, BERTScore
Module 17: LLM Evaluation & Benchmarks

Run:  python example_04_text_metrics.py
Deps: none (pure Python)
"""

import math

print("=" * 60)
print("  Example 04: BLEU, ROUGE, and BERTScore")
print("=" * 60)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def tokenize(sentence):
    """Split sentence into lowercase tokens."""
    return sentence.lower().split()

def count_ngrams(tokens, n):
    """Count all n-grams in a token list."""
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return ngrams


# ─────────────────────────────────────────────────────────
# DEMO 1: BLEU -- n-gram Precision
# ─────────────────────────────────────────────────────────

print("\n--- DEMO 1: BLEU Score Calculation ---")
print()
print("  BLEU measures n-gram precision: of words I generated, how many match reference?")
print("  BLEU-4 = geometric mean of 1-gram, 2-gram, 3-gram, 4-gram precisions.")
print()

def clipped_ngram_precision(hypothesis, reference, n):
    """
    Compute clipped n-gram precision.
    Clipping prevents gaming by repeating common words.
    """
    hyp_ngrams = count_ngrams(hypothesis, n)
    ref_ngrams = count_ngrams(reference, n)

    matches = 0
    for gram, count in hyp_ngrams.items():
        matches += min(count, ref_ngrams.get(gram, 0))  # clip to reference count

    total = max(sum(hyp_ngrams.values()), 1)
    return matches / total

def brevity_penalty(hyp_len, ref_len):
    """Penalize very short outputs (easy to get high precision with 1 word)."""
    if hyp_len >= ref_len:
        return 1.0
    return math.exp(1 - ref_len / hyp_len)

def compute_bleu(hypothesis, reference, max_n=4):
    """
    Compute BLEU-max_n score between hypothesis and reference.
    Uses geometric mean of n-gram precisions with brevity penalty.
    """
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)

    precisions = []
    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n:
            break
        p = clipped_ngram_precision(hyp_tokens, ref_tokens, n)
        precisions.append(p)

    if not precisions or any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    bp = brevity_penalty(len(hyp_tokens), len(ref_tokens))
    return bp * math.exp(log_avg)

# Test cases showing BLEU strengths and weaknesses
test_pairs = [
    {
        "name":      "Good translation (near identical)",
        "reference": "The cat sat on the mat.",
        "hypothesis":"The cat sat on the mat.",
    },
    {
        "name":      "Good -- slightly different words",
        "reference": "The cat sat on the mat.",
        "hypothesis":"The cat sat on a rug.",
    },
    {
        "name":      "Paraphrase -- same meaning, different words",
        "reference": "The cat is sitting on the mat.",
        "hypothesis":"A feline has placed itself upon the rug.",
    },
    {
        "name":      "Poor translation -- wrong meaning",
        "reference": "The cat sat on the mat.",
        "hypothesis":"The dog ran through the park.",
    },
    {
        "name":      "Repeating one word (without clipping would score high!)",
        "reference": "The cat sat on the mat.",
        "hypothesis":"the the the the the the",
    },
]

print(f"  {'Pair':<42}  {'BLEU-4':>8}  {'Interpretation'}")
print("  " + "-" * 75)
for p in test_pairs:
    score = compute_bleu(p["hypothesis"], p["reference"])
    if score > 0.7:
        interp = "Excellent"
    elif score > 0.4:
        interp = "Good"
    elif score > 0.15:
        interp = "Fair"
    else:
        interp = "Poor"
    print(f"  {p['name']:<42}  {score:>8.3f}  {interp}")

print()
print("  Key insight: paraphrase scores poorly even when meaning is identical!")
print("  This is BLEU's main weakness. BERTScore handles paraphrase better.")


# ─────────────────────────────────────────────────────────
# DEMO 2: ROUGE -- Recall-Oriented Metrics
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 2: ROUGE Scores ---")
print()
print("  ROUGE measures recall: of words in the reference, how many did I generate?")
print("  Used mainly for summarization (did you capture the key facts?).")
print()

def rouge_n(hypothesis, reference, n):
    """Compute ROUGE-N: recall, precision, and F1."""
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)

    hyp_ngrams = count_ngrams(hyp_tokens, n)
    ref_ngrams = count_ngrams(ref_tokens, n)

    matches = 0
    for gram, count in ref_ngrams.items():
        matches += min(count, hyp_ngrams.get(gram, 0))

    recall    = matches / max(sum(ref_ngrams.values()), 1)
    precision = matches / max(sum(hyp_ngrams.values()), 1)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"recall": recall, "precision": precision, "f1": f1}

def lcs_length(seq1, seq2):
    """Longest Common Subsequence length (dynamic programming)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def rouge_l(hypothesis, reference):
    """ROUGE-L based on Longest Common Subsequence."""
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)
    lcs = lcs_length(hyp_tokens, ref_tokens)
    recall    = lcs / max(len(ref_tokens), 1)
    precision = lcs / max(len(hyp_tokens), 1)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}

reference = "The quick brown fox jumps over the lazy dog."

hyp_examples = [
    ("Identical",      "The quick brown fox jumps over the lazy dog."),
    ("Partial",        "The fox jumps over the dog."),
    ("Shorter recap",  "Fox jumps over dog."),
    ("Paraphrase",     "A swift auburn fox leaped over the sleepy canine."),
    ("Unrelated",      "The president signed the new trade agreement today."),
]

print(f"  Reference: \"{reference}\"")
print()
print(f"  {'Hypothesis':<22}  {'R-1 F1':>8}  {'R-2 F1':>8}  {'R-L F1':>8}")
print("  " + "-" * 52)
for name, hyp in hyp_examples:
    r1 = rouge_n(hyp, reference, 1)
    r2 = rouge_n(hyp, reference, 2)
    rl = rouge_l(hyp, reference)
    print(f"  {name:<22}  {r1['f1']:>8.3f}  {r2['f1']:>8.3f}  {rl['f1']:>8.3f}")

print()
print("  ROUGE-2 drops fast -- bigram overlap is strict about word order.")
print("  ROUGE-L uses LCS -- more lenient about word order than ROUGE-2.")


# ─────────────────────────────────────────────────────────
# DEMO 3: BERTScore -- Semantic Similarity (Simulated)
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 3: BERTScore -- Semantic Similarity (Simulated) ---")
print()
print("  Real BERTScore uses BERT embeddings (needs PyTorch).")
print("  Here we simulate using manually defined semantic similarity values.")
print("  Intuition: BERTScore matches each token to its semantic nearest neighbor.")
print()

def simulated_bertscore(hypothesis, reference, similarity_table):
    """
    Simulate BERTScore using a pre-defined word similarity table.
    Real BERTScore computes cosine similarity between BERT token embeddings.

    Precision: for each hyp word, find best matching ref word.
    Recall:    for each ref word, find best matching hyp word.
    F1:        harmonic mean.
    """
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)

    def word_sim(a, b):
        """Look up pre-defined similarity, default to exact match check."""
        if a == b:
            return 1.0
        key = tuple(sorted([a, b]))
        return similarity_table.get(key, 0.1)

    # Precision: for each hyp token, best match in ref
    precision_scores = []
    for hw in hyp_tokens:
        best = max(word_sim(hw, rw) for rw in ref_tokens)
        precision_scores.append(best)

    # Recall: for each ref token, best match in hyp
    recall_scores = []
    for rw in ref_tokens:
        best = max(word_sim(rw, hw) for hw in hyp_tokens)
        recall_scores.append(best)

    P = sum(precision_scores) / len(precision_scores)
    R = sum(recall_scores)    / len(recall_scores)
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return {"precision": P, "recall": R, "f1": F}

# Pre-defined semantic similarity pairs (simulates BERT embeddings)
sim_table = {
    ("car", "vehicle"):     0.92,
    ("going", "moving"):    0.88,
    ("fast", "quickly"):    0.90,
    ("cat", "feline"):      0.85,
    ("dog", "canine"):      0.84,
    ("quick", "swift"):     0.89,
    ("jumped", "leaped"):   0.87,
    ("large", "big"):       0.95,
    ("happy", "joyful"):    0.91,
    ("automobile", "car"):  0.93,
    ("sat", "rested"):      0.75,
}

bert_pairs = [
    ("Exact match",         "The car is going fast.", "The car is going fast."),
    ("Synonym-rich",        "The car is going fast.", "The vehicle is moving quickly."),
    ("Different paraphrase","The cat ran quickly.",   "The feline moved fast."),
    ("Unrelated",           "The car is going fast.", "She planted flowers in the garden."),
]

print(f"  {'Pair':<22}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
print("  " + "-" * 55)
for name, hyp, ref in bert_pairs:
    scores = simulated_bertscore(hyp, ref, sim_table)
    print(f"  {name:<22}  {scores['precision']:>10.3f}  {scores['recall']:>8.3f}  {scores['f1']:>8.3f}")

print()
print("  BERTScore sees through synonyms -- 'car' and 'vehicle' get 0.92 similarity.")
print("  BLEU would score 0 for 'The car is going fast' vs 'The vehicle is moving quickly'.")
print()
print("  METRIC SUMMARY:")
print("  BLEU      -- precision, n-gram exact match, translation standard.")
print("  ROUGE-1/2 -- recall, good for summarization (fact coverage).")
print("  ROUGE-L   -- LCS, flexible word order.")
print("  BERTScore -- semantic similarity, best for paraphrase-heavy tasks.")
