"""
Exercise 04: Text Quality Metrics -- BLEU and ROUGE
Module 17: LLM Evaluation & Benchmarks

TASKS:
  1. Implement count_ngrams(tokens, n)              -- count n-gram frequencies
  2. Implement ngram_precision(hypothesis, ref, n)  -- clipped precision
  3. Implement ngram_recall(hypothesis, ref, n)     -- recall
  4. Implement compute_bleu(hypothesis, reference)  -- BLEU-4 score

Run:  python exercise_04_text_metrics.py
Deps: none (pure Python)
"""

import math


def tokenize(sentence):
    """Split sentence into lowercase tokens (provided helper)."""
    return sentence.lower().split()


# ─────────────────────────────────────────────────────────
# TASK 1: Count N-grams
# ─────────────────────────────────────────────────────────

def count_ngrams(tokens, n):
    """
    Count all n-grams in a token list and return a frequency dictionary.

    An n-gram is a sequence of n consecutive tokens.

    Args:
        tokens: list of str tokens (already split and lowercased)
        n:      int, size of n-gram (1=unigrams, 2=bigrams, 3=trigrams, etc.)

    Returns:
        dict mapping tuple(n-gram) -> int count

    Example:
        tokens = ["the", "cat", "sat"]
        count_ngrams(tokens, 1) -> {("the",): 1, ("cat",): 1, ("sat",): 1}
        count_ngrams(tokens, 2) -> {("the", "cat"): 1, ("cat", "sat"): 1}
        count_ngrams(tokens, 3) -> {("the", "cat", "sat"): 1}
        count_ngrams(tokens, 4) -> {}   (sentence too short)

    HINT:
        result = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            result[gram] = result.get(gram, 0) + 1
        return result
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: N-gram Precision (Clipped)
# ─────────────────────────────────────────────────────────

def ngram_precision(hypothesis, reference, n):
    """
    Compute clipped n-gram precision between hypothesis and reference.

    Precision = (matching n-grams) / (total n-grams in hypothesis)
    Clipping: each n-gram match counts at most as many times as it appears in reference.
    This prevents gaming by repeating common words.

    Args:
        hypothesis: str, the generated text
        reference:  str, the reference (gold standard) text
        n:          int, n-gram size

    Returns:
        float: clipped precision (0.0 to 1.0), or 0.0 if hypothesis has no n-grams

    Example:
        hypothesis = "the cat sat on the mat"
        reference  = "the cat sat on the mat"
        ngram_precision(hypothesis, reference, 1) -> 1.0  (all unigrams match)

        hypothesis = "the the the the"
        reference  = "the cat sat on the mat"
        ngram_precision(hypothesis, reference, 1) -> 0.25
        # "the" appears 4 times in hyp but only 2 times in ref -> clipped to 2 matches
        # precision = 2/4 = 0.5  -- wait, let's recount:
        #   hyp has 4 unigrams: {("the",): 4}
        #   ref has 1 "the": clip to min(4, 1)... actually ref has 2 "the"
        #   Let's use a simpler example:

        hypothesis = "the the"
        reference  = "the cat"
        # hyp: {("the",): 2}, ref: {("the",): 1, ("cat",): 1}
        # matches for "the": min(2, 1) = 1
        # precision = 1/2 = 0.5

    HINT:
        hyp_tokens = tokenize(hypothesis)
        ref_tokens = tokenize(reference)
        hyp_ngrams = count_ngrams(hyp_tokens, n)
        ref_ngrams = count_ngrams(ref_tokens, n)
        if not hyp_ngrams:
            return 0.0
        matches = 0
        for gram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(gram, 0))
        total = sum(hyp_ngrams.values())
        return matches / total
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: N-gram Recall
# ─────────────────────────────────────────────────────────

def ngram_recall(hypothesis, reference, n):
    """
    Compute n-gram recall between hypothesis and reference.

    Recall = (matching n-grams) / (total n-grams in reference)
    This is the ROUGE perspective: how much of the reference did we cover?

    Args:
        hypothesis: str, the generated text
        reference:  str, the reference text
        n:          int, n-gram size

    Returns:
        float: recall (0.0 to 1.0), or 0.0 if reference has no n-grams

    Example:
        hypothesis = "the fox"
        reference  = "the quick brown fox"
        ngram_recall(hypothesis, reference, 1) -> 0.5
        # matching: "the", "fox" = 2 out of 4 reference unigrams

    HINT:
        hyp_tokens = tokenize(hypothesis)
        ref_tokens = tokenize(reference)
        hyp_ngrams = count_ngrams(hyp_tokens, n)
        ref_ngrams = count_ngrams(ref_tokens, n)
        if not ref_ngrams:
            return 0.0
        matches = 0
        for gram, count in ref_ngrams.items():
            matches += min(count, hyp_ngrams.get(gram, 0))
        total = sum(ref_ngrams.values())
        return matches / total
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: BLEU Score
# ─────────────────────────────────────────────────────────

def compute_bleu(hypothesis, reference, max_n=4):
    """
    Compute the BLEU-max_n score between hypothesis and reference.

    BLEU = brevity_penalty * geometric_mean(n-gram precisions for n=1..max_n)

    Brevity penalty: penalizes outputs shorter than reference.
      BP = 1.0                  if len(hypothesis) >= len(reference)
      BP = exp(1 - len(ref) / len(hyp))  otherwise

    Geometric mean: exp( (1/N) * sum(log(precision_n)) )
    Returns 0.0 if any precision is 0.

    Args:
        hypothesis: str, generated text
        reference:  str, reference text
        max_n:      int, highest n-gram order (default 4 for standard BLEU-4)

    Returns:
        float: BLEU score (0.0 to 1.0)

    Example:
        compute_bleu("the cat sat on the mat", "the cat sat on the mat") -> 1.0
        compute_bleu("the cat", "the cat sat on the mat")                -> small
        compute_bleu("totally different words", "the cat sat")           -> 0.0

    HINT:
        hyp_tokens = tokenize(hypothesis)
        ref_tokens = tokenize(reference)
        # Brevity penalty
        if len(hyp_tokens) >= len(ref_tokens):
            bp = 1.0
        else:
            bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
        # Precisions
        precisions = []
        for n in range(1, max_n + 1):
            if len(hyp_tokens) < n:
                break
            p = ngram_precision(hypothesis, reference, n)
            precisions.append(p)
        if not precisions or any(p == 0 for p in precisions):
            return 0.0
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        return bp * math.exp(log_avg)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 04: Text Quality Metrics (BLEU & ROUGE)")
    print("=" * 55)

    # Task 1: count_ngrams
    print("\n--- Task 1: count_ngrams ---")
    tokens = ["the", "cat", "sat"]
    result_1 = count_ngrams(tokens, 1)
    if result_1 is None:
        print("  NOT IMPLEMENTED YET")
    else:
        expected_1 = {("the",): 1, ("cat",): 1, ("sat",): 1}
        status = "PASS" if result_1 == expected_1 else "FAIL"
        print(f"  {status}  unigrams = {result_1}  (expected {expected_1})")

        result_2 = count_ngrams(tokens, 2)
        expected_2 = {("the", "cat"): 1, ("cat", "sat"): 1}
        status = "PASS" if result_2 == expected_2 else "FAIL"
        print(f"  {status}  bigrams  = {result_2}  (expected {expected_2})")

        result_empty = count_ngrams(tokens, 4)
        status = "PASS" if result_empty == {} else "FAIL"
        print(f"  {status}  4-grams from 3-token sentence = {result_empty}  (expected {{}})")

    # Task 2: ngram_precision
    print("\n--- Task 2: ngram_precision ---")
    if count_ngrams(["a"], 1) is not None:
        hyp = "the cat sat on the mat"
        ref = "the cat sat on the mat"
        r_exact = ngram_precision(hyp, ref, 1)
        if r_exact is None:
            print("  NOT IMPLEMENTED YET")
        else:
            status = "PASS" if abs(r_exact - 1.0) < 0.001 else "FAIL"
            print(f"  {status}  identical sentences: precision-1 = {r_exact:.3f}  (expected 1.000)")

            # Clipping test: repeated word
            hyp2 = "the the"
            ref2 = "the cat"
            r_clip = ngram_precision(hyp2, ref2, 1)
            # "the" appears 2 in hyp, 1 in ref -> clip to 1 match, 2 total -> 0.5
            status = "PASS" if abs(r_clip - 0.5) < 0.001 else "FAIL"
            print(f"  {status}  clipping 'the the' vs 'the cat': precision-1 = {r_clip:.3f}  (expected 0.500)")

            hyp3 = "the cat sat on a rug"
            ref3 = "the cat sat on the mat"
            r3 = ngram_precision(hyp3, ref3, 1)
            print(f"  INFO  'the cat sat on a rug' vs ref: precision-1 = {r3:.3f}")

    # Task 3: ngram_recall
    print("\n--- Task 3: ngram_recall ---")
    if count_ngrams(["a"], 1) is not None:
        hyp = "the fox"
        ref = "the quick brown fox"
        r_rec = ngram_recall(hyp, ref, 1)
        if r_rec is None:
            print("  NOT IMPLEMENTED YET")
        else:
            status = "PASS" if abs(r_rec - 0.5) < 0.001 else "FAIL"
            print(f"  {status}  'the fox' vs 'the quick brown fox': recall-1 = {r_rec:.3f}  (expected 0.500)")

            r_full = ngram_recall(ref, ref, 1)
            status = "PASS" if abs(r_full - 1.0) < 0.001 else "FAIL"
            print(f"  {status}  identical: recall-1 = {r_full:.3f}  (expected 1.000)")

    # Task 4: compute_bleu
    print("\n--- Task 4: compute_bleu ---")
    if ngram_precision("a b", "a b", 1) is not None:
        hyp_id = "the cat sat on the mat"
        ref_id = "the cat sat on the mat"
        r_id = compute_bleu(hyp_id, ref_id)
        if r_id is None:
            print("  NOT IMPLEMENTED YET")
        else:
            status = "PASS" if abs(r_id - 1.0) < 0.001 else "FAIL"
            print(f"  {status}  identical: BLEU-4 = {r_id:.3f}  (expected 1.000)")

            bleu_pairs = [
                ("the cat sat on a rug",                "the cat sat on the mat"),
                ("a feline has placed itself on the rug","the cat sat on the mat"),
                ("the dog ran through the park",         "the cat sat on the mat"),
            ]
            for hyp, ref in bleu_pairs:
                score = compute_bleu(hyp, ref)
                print(f"  INFO  BLEU-4 = {score:.3f}  hyp='{hyp[:35]}...'")

    # Bonus: BLEU vs ROUGE comparison
    print("\n--- BONUS: BLEU (precision) vs ROUGE (recall) ---")
    if compute_bleu("a b c", "a b c") is not None and ngram_recall("a", "a b", 1) is not None:
        reference = "The quick brown fox jumps over the lazy dog."
        hypotheses = [
            ("Full match", "the quick brown fox jumps over the lazy dog"),
            ("Subset",     "the fox jumps over the dog"),
            ("Paraphrase", "a fast red fox leaps above a sleepy canine"),
        ]
        print(f"\n  {'Hypothesis':<15}  {'BLEU-4':>8}  {'ROUGE-1 P':>10}  {'ROUGE-1 R':>10}")
        print("  " + "-" * 48)
        for name, hyp in hypotheses:
            bleu = compute_bleu(hyp, reference)
            prec = ngram_precision(hyp, reference, 1)
            rec  = ngram_recall(hyp, reference, 1)
            print(f"  {name:<15}  {bleu:>8.3f}  {prec:>10.3f}  {rec:>10.3f}")
        print()
        print("  'Subset' has high ROUGE-1 recall but penalized BLEU (brevity penalty).")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def count_ngrams(tokens, n):
#     result = {}
#     for i in range(len(tokens) - n + 1):
#         gram = tuple(tokens[i:i + n])
#         result[gram] = result.get(gram, 0) + 1
#     return result
#
# def ngram_precision(hypothesis, reference, n):
#     hyp_tokens = tokenize(hypothesis)
#     ref_tokens = tokenize(reference)
#     hyp_ngrams = count_ngrams(hyp_tokens, n)
#     ref_ngrams = count_ngrams(ref_tokens, n)
#     if not hyp_ngrams:
#         return 0.0
#     matches = 0
#     for gram, count in hyp_ngrams.items():
#         matches += min(count, ref_ngrams.get(gram, 0))
#     return matches / sum(hyp_ngrams.values())
#
# def ngram_recall(hypothesis, reference, n):
#     hyp_tokens = tokenize(hypothesis)
#     ref_tokens = tokenize(reference)
#     hyp_ngrams = count_ngrams(hyp_tokens, n)
#     ref_ngrams = count_ngrams(ref_tokens, n)
#     if not ref_ngrams:
#         return 0.0
#     matches = 0
#     for gram, count in ref_ngrams.items():
#         matches += min(count, hyp_ngrams.get(gram, 0))
#     return matches / sum(ref_ngrams.values())
#
# def compute_bleu(hypothesis, reference, max_n=4):
#     hyp_tokens = tokenize(hypothesis)
#     ref_tokens = tokenize(reference)
#     if len(hyp_tokens) >= len(ref_tokens):
#         bp = 1.0
#     else:
#         bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
#     precisions = []
#     for n in range(1, max_n + 1):
#         if len(hyp_tokens) < n:
#             break
#         p = ngram_precision(hypothesis, reference, n)
#         precisions.append(p)
#     if not precisions or any(p == 0 for p in precisions):
#         return 0.0
#     log_avg = sum(math.log(p) for p in precisions) / len(precisions)
#     return bp * math.exp(log_avg)
