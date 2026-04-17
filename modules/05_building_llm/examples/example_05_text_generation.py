"""
=============================================================================
EXAMPLE 5: Text Generation & Sampling Strategies
=============================================================================

GLOSSARY (read before the code)
---------------------------------
Autoregressive : Generating ONE token at a time, each time using ALL
                 previous tokens as context.
                 "auto" = self, "regressive" = depending on past values.

Logits         : Raw scores from the model for each possible next token.
                 These are NOT probabilities yet. Can be any number.
                 e.g. [3.2, 1.1, -0.5, 2.8, 0.9]

Probabilities  : Logits converted to values between 0 and 1 that sum to 1.
                 We use SOFTMAX to do this conversion.

Softmax        : A formula that converts any list of numbers to probabilities.
                 Formula: exp(x_i) / sum(exp(x_j) for all j)

Sampling       : Picking the next token randomly, weighted by probabilities.
                 High probability = more likely to be chosen.
                 C# analogy: like Random.NextDouble() but weighted.

Greedy         : Always pick the token with the HIGHEST probability.
                 No randomness at all. Deterministic.

Temperature    : A number that controls how "spread out" the probabilities are.
                 temperature < 1 -> more focused (top choice more dominant)
                 temperature = 1 -> unchanged (model's original probabilities)
                 temperature > 1 -> more random (all choices more equal)

Top-k          : Only consider the TOP k tokens. Ignore all others.
                 k=50 means: look at the 50 most likely tokens, ignore the rest.

Top-p          : Only consider the smallest group of tokens whose probabilities
                 add up to at least p (e.g. p=0.9 = 90%).
                 Called "nucleus sampling". Used by ChatGPT.

Repetition     : A penalty that makes the model less likely to repeat words
Penalty          it has already used. Prevents "the cat sat on the cat sat..."

=============================================================================
PART A: Showing ALL strategies with FAKE probabilities (no model needed!)
=============================================================================

The best way to understand sampling is to forget about the model entirely
and just work with fake probability tables.

Imagine a model has just processed the text "The cat" and is deciding
what comes next. It outputs these probabilities for the next word:
=============================================================================
"""

import numpy as np

print("=" * 65)
print("PART A: Sampling Strategies -- No Model Needed!")
print("=" * 65)

# =============================================================================
# Our fake "model output": probabilities for the next word.
# In a real model, these come from softmax(logits).
# We define them directly here so you can focus on the STRATEGY, not the model.
# =============================================================================

# The vocabulary (possible next words) and their probabilities
vocab  = ["sat", "ran", "slept", "ate", "drank", "sang", "jumped", "flew"]
logits = np.array([3.5, 2.1, 1.5, 1.2, 0.8, 0.3, 0.1, -0.5])  # raw scores

# Convert logits to probabilities using softmax
def softmax(x):
    """Convert raw scores to probabilities that sum to 1."""
    e = np.exp(x - x.max())    # subtract max for numerical stability
    return e / e.sum()          # divide each by the total

probs = softmax(logits)

print("\nScenario: model sees 'The cat ___' and must pick the next word.")
print()
print(f"{'Word':<10} {'Logit':>8} {'Probability':>12} {'Cumulative':>12}")
print("-" * 46)
cumulative = 0
for word, logit, prob in zip(vocab, logits, probs):
    cumulative += prob
    print(f"{word:<10} {logit:>8.2f} {prob:>11.1%}  {cumulative:>11.1%}")

# =============================================================================
# STRATEGY 1: GREEDY
# Always pick the word with the highest probability.
# =============================================================================

print("\n" + "=" * 50)
print("STRATEGY 1: Greedy")
print("=" * 50)
print("""
DEFINITION:
  Always pick the highest-probability word. No randomness.

ANALOGY:
  Like always ordering the same "most popular" dish at a restaurant.
  Safe, predictable, but gets boring fast.

WHEN TO USE:
  - Code completion (correctness matters more than creativity)
  - Factual Q&A (you want the most likely correct answer)
""")

greedy_choice = vocab[probs.argmax()]          # argmax = index of the maximum value
greedy_prob   = probs.max()

print(f"Result: '{greedy_choice}' (probability: {greedy_prob:.1%})")
print()
print("Trying 5 times -- notice it ALWAYS picks the same word:")
for i in range(5):
    print(f"  Trial {i+1}: 'The cat {vocab[probs.argmax()]}'")

print("\nProblem: 'The cat sat sat sat sat sat...' -- gets repetitive!")

# =============================================================================
# STRATEGY 2: TEMPERATURE SAMPLING
# Scale the logits before softmax to control randomness.
# =============================================================================

print("\n" + "=" * 50)
print("STRATEGY 2: Temperature Sampling")
print("=" * 50)
print("""
DEFINITION:
  Divide logits by a temperature value BEFORE applying softmax.
  This changes how spread out the probabilities are.

  Low temperature  (< 1.0) -> top word dominates even more -> focused
  Temperature = 1.0        -> original probabilities unchanged
  High temperature (> 1.0) -> all words become more equal -> creative

ANALOGY:
  Imagine probabilities as heights of bars.
  Low temp: makes the tall bars taller and short bars shorter (more extreme).
  High temp: makes all bars the same height (more equal, more random).
""")

def temperature_sample(logits, temperature):
    """Apply temperature and sample."""
    if temperature <= 0:
        return logits.argmax()                     # greedy if temp is 0
    scaled = logits / temperature                  # divide logits by temperature
    p      = softmax(scaled)                       # convert to probabilities
    return np.random.choice(len(vocab), p=p)       # sample using those probabilities

# Show how temperature changes the probabilities
print("How temperature changes the probability of 'sat' (the top word):\n")
print(f"{'Temperature':<15} {'Prob of sat':>12} {'Prob of flew':>14} {'Distribution'}")
print("-" * 65)
for temp in [0.3, 0.7, 1.0, 1.5, 2.0]:
    p_temp    = softmax(logits / temp)
    prob_sat  = p_temp[vocab.index("sat")]
    prob_flew = p_temp[vocab.index("flew")]
    bar       = "#" * int(prob_sat * 20)                      # visual bar
    print(f"{temp:<15.1f} {prob_sat:>11.1%}  {prob_flew:>12.1%}   {bar}")

print()
print("Sampling 5 times at temperature=0.3 (focused):")
for i in range(5):
    idx = temperature_sample(logits, temperature=0.3)
    print(f"  Trial {i+1}: 'The cat {vocab[idx]}'")

print()
print("Sampling 5 times at temperature=1.5 (creative):")
for i in range(5):
    idx = temperature_sample(logits, temperature=1.5)
    print(f"  Trial {i+1}: 'The cat {vocab[idx]}'")

# =============================================================================
# STRATEGY 3: TOP-K SAMPLING
# Only consider the TOP k words. Ignore all others entirely.
# =============================================================================

print("\n" + "=" * 50)
print("STRATEGY 3: Top-k Sampling")
print("=" * 50)
print("""
DEFINITION:
  1. Find the k highest-probability words.
  2. Set all other words' probabilities to zero.
  3. Re-normalize the k words so they sum to 1.
  4. Sample from only those k words.

ANALOGY:
  Like an autocomplete that shows only the top 3 suggestions.
  You pick from those 3, not from every word in the dictionary.

WHEN TO USE:
  - Code completion (GitHub Copilot uses k=10)
  - Chatbots (k=40-50)
""")

def top_k_sample(logits, k, temperature=1.0):
    """Only sample from the top-k words."""
    scaled = logits / temperature              # apply temperature first

    # Find the k-th largest value
    sorted_logits = np.sort(scaled)[::-1]      # sort descending
    threshold     = sorted_logits[k - 1]       # value of the k-th largest

    # Set everything below the threshold to -infinity
    # After softmax, -infinity becomes 0 (excluded from sampling)
    filtered = np.where(scaled >= threshold, scaled, -np.inf)

    p = softmax(filtered)                       # re-normalize among survivors
    return np.random.choice(len(vocab), p=p)

# Show what top-k keeps
print(f"Total words in vocabulary: {len(vocab)}")
print()
for k in [1, 2, 3, 5]:
    sorted_words = [w for _, w in sorted(zip(probs, vocab), reverse=True)]
    kept         = sorted_words[:k]
    print(f"  k={k}: only considers {kept}")

print()
print("Sampling 5 times with k=3 (only top-3 words allowed):")
for i in range(5):
    idx = top_k_sample(logits, k=3, temperature=0.8)
    print(f"  Trial {i+1}: 'The cat {vocab[idx]}'")

# =============================================================================
# STRATEGY 4: TOP-P (NUCLEUS) SAMPLING
# Dynamically choose the smallest group covering p% of the probability mass.
# =============================================================================

print("\n" + "=" * 50)
print("STRATEGY 4: Top-p (Nucleus) Sampling")
print("=" * 50)
print("""
DEFINITION:
  1. Sort words by probability (highest first).
  2. Keep adding words until their CUMULATIVE probability reaches p.
  3. The selected group is called the "nucleus".
  4. Sample from the nucleus only.

WHY IS THIS BETTER THAN TOP-K?
  Top-k uses a FIXED number of words.
  Top-p uses a DYNAMIC number based on the model's confidence.

  - When the model is CONFIDENT: 1-2 words cover 90% -> small nucleus
  - When the model is UNCERTAIN: 30 words needed to cover 90% -> large nucleus

  Top-p respects the model's confidence level. This is why ChatGPT uses it!

ANALOGY (from the lesson):
  Top-k = "interview only the top 10 candidates regardless of pool size"
  Top-p = "interview candidates until you've seen 90% of the talent pool"
""")

def top_p_sample(logits, p, temperature=1.0):
    """Only sample from the smallest nucleus covering probability p."""
    scaled = logits / temperature

    # Sort by probability (highest first)
    sorted_idx    = np.argsort(scaled)[::-1]   # indices in descending order
    sorted_logits = scaled[sorted_idx]
    sorted_probs  = softmax(sorted_logits)

    # Find cumulative probabilities
    cumulative    = np.cumsum(sorted_probs)

    # Cut off: keep only the words needed to reach p
    # np.searchsorted: finds the index where cumulative first exceeds p
    cutoff_idx    = np.searchsorted(cumulative, p) + 1    # +1 to include the word that pushes us over

    # Zero out everything beyond the cutoff
    filtered          = np.full_like(scaled, -np.inf)
    filtered[sorted_idx[:cutoff_idx]] = sorted_logits[:cutoff_idx]

    p_final = softmax(filtered)
    return np.random.choice(len(vocab), p=p_final)

# Show what nucleus looks like at different p values
print("Nucleus size (how many words are included) at different p values:")
print()
sorted_words = sorted(zip(probs, vocab), reverse=True)  # sort by probability
sorted_probs_only = [p for p, w in sorted_words]
sorted_word_names = [w for p, w in sorted_words]

print(f"  {'p value':<10} {'Nucleus words':>14} {'Included words'}")
print("  " + "-" * 55)
for p_val in [0.5, 0.7, 0.9, 0.95, 1.0]:
    cumul = 0
    nucleus = []
    for prob_val, word in zip(sorted_probs_only, sorted_word_names):
        if cumul >= p_val:
            break
        cumul += prob_val
        nucleus.append(word)
    # Make sure we include the word that pushes us over
    if cumul < p_val and len(nucleus) < len(sorted_word_names):
        nucleus.append(sorted_word_names[len(nucleus)])
    print(f"  {p_val:<10.2f} {len(nucleus):>4} words         {nucleus}")

print()
print("Sampling 5 times with p=0.9 (nucleus sampling):")
for i in range(5):
    idx = top_p_sample(logits, p=0.9, temperature=0.8)
    print(f"  Trial {i+1}: 'The cat {vocab[idx]}'")

# =============================================================================
# QUICK COMPARISON of all strategies
# =============================================================================

print("\n" + "=" * 65)
print("COMPARISON: All 4 strategies side by side")
print("=" * 65)

np.random.seed(42)   # fix seed for reproducible output

print(f"\n{'Strategy':<25} {'Output word':<12} {'Deterministic?'}")
print("-" * 50)

# Greedy (always the same)
g = vocab[probs.argmax()]
print(f"{'Greedy':<25} {g:<12} Yes -- always same")

# Temperature sampling
for temp, label in [(0.3, "Temperature=0.3"), (1.5, "Temperature=1.5")]:
    idx = temperature_sample(logits, temperature=temp)
    print(f"{label:<25} {vocab[idx]:<12} No -- random")

# Top-k
idx = top_k_sample(logits, k=3, temperature=0.8)
print(f"{'Top-k (k=3)':<25} {vocab[idx]:<12} No -- random among top 3")

# Top-p
idx = top_p_sample(logits, p=0.9, temperature=0.8)
print(f"{'Top-p (p=0.9)':<25} {vocab[idx]:<12} No -- nucleus random")

# =============================================================================
print("\n" + "=" * 65)
print("PART B: Full Text Generation Function")
print("=" * 65)
# =============================================================================

"""
PART B: A complete generation function that works with any GPT model.

This is the "real" version used in production:
  - Accepts any combination of strategies
  - temperature + top-k + top-p can be combined
  - Includes repetition penalty to avoid loops
"""

print("""
This shows the COMPLETE generation loop used in real chatbots.
It uses numpy to simulate a model so no PyTorch is needed here.
""")

def apply_repetition_penalty(logits, used_tokens, penalty=1.2):
    """
    Reduce the probability of tokens already used in the output.

    penalty > 1.0 -> less likely to repeat (positive logits get smaller,
                                            negative logits get more negative)
    penalty = 1.0 -> no effect

    WHY: Without this, models loop: "the cat sat on the cat sat on the cat..."
    """
    result = logits.copy()
    for token_idx in set(used_tokens):               # set() = only unique tokens
        if result[token_idx] > 0:
            result[token_idx] /= penalty             # shrink positive logits
        else:
            result[token_idx] *= penalty             # grow negative logits (more negative)
    return result


def generate_text(
    model_fn,           # function: context -> logits
    start_tokens,       # list of starting token indices
    max_new_tokens,     # how many tokens to generate
    temperature=0.8,    # controls randomness
    top_k=None,         # if set, use top-k filtering
    top_p=None,         # if set, use nucleus sampling
    repetition_penalty=1.0,  # > 1 reduces repeating
    verbose=False       # if True, print each step
):
    """
    Full text generation with all strategies combined.

    Order of operations at each step:
    1. Get logits from the model
    2. Apply repetition penalty
    3. Apply temperature
    4. Apply top-k filter (if enabled)
    5. Apply top-p filter (if enabled)
    6. Sample the next token
    7. Append and repeat
    """
    tokens = list(start_tokens)                      # copy so we don't modify the original

    for step in range(max_new_tokens):
        # Step 1: Get model output (logits) for the current context
        logits = model_fn(tokens)                    # shape: (vocab_size,)

        # Step 2: Apply repetition penalty
        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, tokens, repetition_penalty)

        # Step 3: Apply temperature (scale logits)
        logits = logits / temperature

        # Step 4: Top-k filter
        if top_k is not None:
            sorted_l = np.sort(logits)[::-1]
            threshold = sorted_l[top_k - 1]         # value of the k-th largest
            logits    = np.where(logits >= threshold, logits, -np.inf)

        # Step 5: Top-p filter
        if top_p is not None:
            sort_idx  = np.argsort(logits)[::-1]    # sort by value, descending
            cumul     = np.cumsum(softmax(logits[sort_idx]))
            cutoff    = np.searchsorted(cumul, top_p) + 1
            keep_idx  = sort_idx[:cutoff]
            mask      = np.full(len(logits), -np.inf)
            mask[keep_idx] = logits[keep_idx]
            logits    = mask

        # Step 6: Convert to probabilities and sample
        p_final    = softmax(logits)
        # Replace any NaN (from -inf columns) with 0
        p_final    = np.nan_to_num(p_final, nan=0.0)
        if p_final.sum() == 0:
            p_final = np.ones(len(p_final)) / len(p_final)  # fallback to uniform
        next_token = np.random.choice(len(p_final), p=p_final)

        if verbose:
            chosen_word = vocab[next_token]
            print(f"  Step {step+1}: chose '{chosen_word}' (prob={p_final[next_token]:.1%})")

        # Step 7: Append to sequence
        tokens.append(next_token)

    return tokens


# -------------------------------------------------------------------
# Demo: Use a fake bigram model and generate a short sentence
# -------------------------------------------------------------------

# Fake model: always returns our pre-defined logits plus a little noise
def fake_model(tokens):
    """Simulate a model. Returns logits for the next token."""
    noise  = np.random.randn(len(logits)) * 0.3    # add small random noise
    return logits + noise


print("Generating 5 words using different strategies:")
print(f"Starting with: 'The cat'  (seed: 'sat' index = {vocab.index('sat')})")
print()

seed = [vocab.index("sat")]      # pretend we started with "sat"

for label, settings in [
    ("Greedy (temp=0.1)",             {"temperature": 0.1}),
    ("Creative (temp=2.0)",           {"temperature": 2.0}),
    ("Top-k=2 (temp=0.8)",            {"temperature": 0.8, "top_k": 2}),
    ("Nucleus p=0.7 (temp=0.8)",      {"temperature": 0.8, "top_p": 0.7}),
    ("Rep. penalty=1.5",              {"temperature": 0.8, "repetition_penalty": 1.5}),
]:
    np.random.seed(99)                                      # same seed for fairness
    result = generate_text(fake_model, seed, max_new_tokens=5, **settings)
    words  = " ".join(vocab[i] for i in result)
    print(f"  {label:<30}: 'The cat {words}'")

# =============================================================================
# Decision guide
# =============================================================================

print("\n" + "=" * 65)
print("STRATEGY DECISION GUIDE")
print("=" * 65)
print("""
  What are you building?
  |
  |--- Code completion / SQL / exact output needed
  |     -> temperature=0.1, top_k=10
  |       (precision first -- creativity hurts here)
  |
  |--- Factual Q&A / summarization
  |     -> temperature=0.3, top_p=0.9
  |       (mostly accurate, small variety)
  |
  |--- Chatbot / conversational
  |     -> temperature=0.8, top_p=0.9, repetition_penalty=1.1
  |       (natural, varied, does not loop)
  |       THIS IS WHAT CHATGPT USES
  |
  |--- Creative writing / story
  |     -> temperature=1.2, top_p=0.95, repetition_penalty=1.3
  |       (surprising, diverse, avoids repetition)
  |
  \--- Translation / one "best" answer
        -> beam search (not shown here -- in Module 6)
          (explores multiple paths, picks the highest-scoring one)
""")

print("""
Real-World Settings:
  ChatGPT            -> top_p=0.9, temperature ~ 0.7-0.9
  GitHub Copilot     -> temperature ~ 0.2, top_k=10
  Google Translate   -> beam search, beam_width=4
""")

print("=" * 65)
print("Example 5 complete!")
print("Next: Run exercise_05_text_generation.py to practice")
print("=" * 65)
