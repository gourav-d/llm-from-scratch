"""
=============================================================================
EXAMPLE 05 (PyTorch Version): Text Generation Strategies
=============================================================================

GLOSSARY
---------
torch.softmax     : Converts raw logit scores → probabilities (sum to 1).
                    Same math as our NumPy softmax, but on tensors.
                    Use F.softmax(tensor, dim=-1).

torch.multinomial : Samples token indices weighted by probabilities.
                    Replaces np.random.choice(len(p), p=p) from NumPy.
                    Input : probability tensor, shape (1, vocab_size)
                    Output: sampled index tensor, shape (1, 1)

torch.topk        : Returns the TOP k values AND their indices from a tensor.
                    Replaces the np.sort + threshold approach in NumPy.
                    output.values  = the top k values
                    output.indices = which positions they came from

torch.argsort     : Returns indices that would sort a tensor (descending).
                    Replaces np.argsort(x)[::-1].

torch.cumsum      : Running cumulative sum along a dimension.
                    Replaces np.cumsum. Used to implement top-p cutoff.

masked_fill_      : In-place operation that sets tensor values to a fill
                    value wherever a boolean mask is True.
                    Replaces np.where(condition, value, -inf).

=============================================================================
HOW THIS CONNECTS TO THE NumPy VERSION (example_05_text_generation.py)
=============================================================================

Every strategy (greedy, temperature, top-k, top-p) uses the SAME logic.
The difference is just the PyTorch function used at each step.

NumPy  →  PyTorch equivalent:
  np.exp(x) / sum(...)        →   F.softmax(x, dim=-1)
  np.random.choice(n, p=p)    →   torch.multinomial(p, 1)
  np.sort(x)[::-1][:k]        →   torch.topk(x, k)
  np.cumsum(sorted_probs)      →   torch.cumsum(sorted_probs, dim=-1)
  x[x < threshold] = -inf      →   x.masked_fill_(mask, float('-inf'))

=============================================================================
"""

import torch                        # core PyTorch
import torch.nn.functional as F     # standalone functions (softmax, etc.)
import numpy as np                  # only for setting random seed in comparison

print("=" * 65)
print("TEXT GENERATION STRATEGIES — PyTorch Version")
print("=" * 65)

# =============================================================================
# Setup: fake logits (same vocabulary as NumPy version)
# =============================================================================

vocab  = ["sat", "ran", "slept", "ate", "drank", "sang", "jumped", "flew"]
vocab_size = len(vocab)

# Raw scores from a (fake) model — same values as NumPy example
# These represent what a model might output after seeing "The cat ___"
raw_logits = torch.tensor(
    [3.5, 2.1, 1.5, 1.2, 0.8, 0.3, 0.1, -0.5],
    dtype=torch.float
)

# F.softmax converts logits → probabilities
# dim=0 because our tensor is 1-D (single vector)
probs = F.softmax(raw_logits, dim=0)   # shape: (8,)

print("\nScenario: model sees 'The cat ___' and outputs these scores.")
print()
print(f"{'Word':<10} {'Logit':>8} {'Probability':>12} {'Cumulative':>12}")
print("-" * 46)
cumulative = 0.0
for word, logit, prob in zip(vocab, raw_logits.tolist(), probs.tolist()):
    cumulative += prob
    print(f"{word:<10} {logit:>8.2f} {prob:>11.1%}  {cumulative:>11.1%}")

# =============================================================================
# STRATEGY 1: GREEDY — always pick the highest-probability token
# =============================================================================

print("\n" + "=" * 50)
print("STRATEGY 1: Greedy")
print("=" * 50)
print("""
Always pick the token with the highest score. No randomness.
torch.argmax() returns the INDEX of the maximum value.
Same as np.argmax() but for tensors.
""")

# torch.argmax returns a 0-D (scalar) tensor containing the index
best_idx  = torch.argmax(probs)          # tensor containing index of max value
best_word = vocab[best_idx.item()]       # .item() extracts Python int

print(f"torch.argmax(probs) = {best_idx.item()}  →  '{best_word}'")
print()
print("Trying 5 times — always the same (no randomness in greedy):")
for i in range(5):
    idx = torch.argmax(probs).item()
    print(f"  Trial {i+1}: 'The cat {vocab[idx]}'")

print("\nProblem: deterministic — always says 'sat'. Boring!")

# =============================================================================
# STRATEGY 2: TEMPERATURE SAMPLING
# =============================================================================

print("\n" + "=" * 50)
print("STRATEGY 2: Temperature Sampling")
print("=" * 50)
print("""
Divide logits by temperature BEFORE softmax.

  temperature < 1.0  →  confident (top word dominates even more)
  temperature = 1.0  →  unchanged
  temperature > 1.0  →  creative (all words become more equal)

PyTorch: just divide the tensor:  logits / temperature
Then: F.softmax(scaled, dim=0)
Then: torch.multinomial(probs, num_samples=1) to sample

torch.multinomial is the PyTorch equivalent of np.random.choice(p=probs).
  Input : probability tensor (must sum to 1)
  Output: sampled index (as a tensor)
""")

def temperature_sample(logits, temperature):
    """Apply temperature scaling and sample one token."""
    if temperature <= 0:
        return torch.argmax(logits)           # greedy if temp = 0

    scaled = logits / temperature             # scale logits by temperature
    p      = F.softmax(scaled, dim=0)         # convert to probabilities
    # torch.multinomial samples ONE index weighted by p
    # unsqueeze(0) makes p shape (1, vocab_size) — multinomial needs 2-D input
    return torch.multinomial(p.unsqueeze(0), num_samples=1).squeeze()

print("How temperature changes probability of 'sat' (top word):")
print()
print(f"{'Temperature':<15} {'Prob(sat)':>10} {'Prob(flew)':>12} {'Bar'}")
print("-" * 55)
for temp in [0.3, 0.7, 1.0, 1.5, 2.0]:
    p_temp    = F.softmax(raw_logits / temp, dim=0)
    prob_sat  = p_temp[vocab.index("sat")].item()
    prob_flew = p_temp[vocab.index("flew")].item()
    bar       = "#" * int(prob_sat * 20)
    print(f"{temp:<15.1f} {prob_sat:>10.1%} {prob_flew:>11.1%}   {bar}")

print()
print("Sampling 5 times at temperature=0.3 (focused):")
for i in range(5):
    idx = temperature_sample(raw_logits, temperature=0.3).item()
    print(f"  Trial {i+1}: 'The cat {vocab[idx]}'")

print()
print("Sampling 5 times at temperature=1.5 (creative):")
for i in range(5):
    idx = temperature_sample(raw_logits, temperature=1.5).item()
    print(f"  Trial {i+1}: 'The cat {vocab[idx]}'")

# =============================================================================
# STRATEGY 3: TOP-K SAMPLING
# =============================================================================

print("\n" + "=" * 50)
print("STRATEGY 3: Top-k Sampling")
print("=" * 50)
print("""
Keep only the top-k tokens. Set everything else to -infinity.
After softmax, -infinity becomes 0 (impossible to sample).

PyTorch: torch.topk(tensor, k) returns the k largest values AND indices.
  output.values  → the k largest scores
  output.indices → which positions they are at

Instead of sorting and slicing (NumPy way), we use topk directly.
""")

def top_k_sample(logits, k, temperature=1.0):
    """Sample from the top-k tokens only."""
    scaled = logits / temperature

    # torch.topk returns (values, indices) of the k largest elements
    topk_result = torch.topk(scaled, k=k)
    threshold   = topk_result.values[-1]    # k-th largest value (the cutoff)

    # Set everything below the threshold to -infinity
    # masked_fill(condition, value):
    #   where condition is True, fill with value
    filtered = scaled.masked_fill(scaled < threshold, float('-inf'))

    p = F.softmax(filtered, dim=0)
    return torch.multinomial(p.unsqueeze(0), num_samples=1).squeeze()

# Show what top-k keeps at different k values
print("What each k value keeps:")
sorted_by_prob = sorted(zip(probs.tolist(), vocab), reverse=True)
for k in [1, 2, 3, 5]:
    kept = [w for _, w in sorted_by_prob[:k]]
    print(f"  k={k}: only considers {kept}")

print()
print("Sampling 5 times with k=3:")
for i in range(5):
    idx = top_k_sample(raw_logits, k=3, temperature=0.8).item()
    print(f"  Trial {i+1}: 'The cat {vocab[idx]}'")

# =============================================================================
# STRATEGY 4: TOP-P (NUCLEUS) SAMPLING
# =============================================================================

print("\n" + "=" * 50)
print("STRATEGY 4: Top-p (Nucleus) Sampling")
print("=" * 50)
print("""
Keep the SMALLEST SET of tokens whose cumulative probability reaches p.

Steps (same concept as NumPy, different functions):
  1. Sort tokens by probability (highest first)     torch.argsort
  2. Compute cumulative sum of sorted probabilities  torch.cumsum
  3. Find cutoff: first index where cumsum >= p
  4. Keep only those tokens, set rest to -infinity
  5. Sample using torch.multinomial

Key difference from NumPy:
  np.argsort(x)[::-1]    →   torch.argsort(x, descending=True)
  np.cumsum(x)            →   torch.cumsum(x, dim=0)
""")

def top_p_sample(logits, p, temperature=1.0):
    """Nucleus sampling — dynamic vocabulary based on cumulative probability."""
    scaled = logits / temperature

    # Sort from highest to lowest probability
    # torch.argsort returns INDICES that sort the tensor
    # descending=True means highest-scored tokens come first
    sorted_indices  = torch.argsort(scaled, descending=True)   # sorted positions
    sorted_logits   = scaled[sorted_indices]                    # logits in sorted order
    sorted_probs    = F.softmax(sorted_logits, dim=0)

    # Compute running cumulative sum
    # After step i: cumsum[i] = sum of sorted_probs[0..i]
    cumulative = torch.cumsum(sorted_probs, dim=0)

    # Find the cutoff: remove tokens AFTER cumulative probability exceeds p
    # We keep token i if cumulative[i-1] < p  (cumulative BEFORE this token is < p)
    # Shift cumulative by 1 position: compare previous cumsum to p
    remove_mask = (cumulative - sorted_probs) >= p   # True = remove this token

    # Set removed tokens to -infinity (they'll become 0 after softmax)
    sorted_logits = sorted_logits.masked_fill(remove_mask, float('-inf'))

    # Unsort: put logits back in the original vocabulary order
    # We create an output tensor and scatter the values back to original positions
    filtered = torch.full_like(scaled, float('-inf'))         # start with all -inf
    filtered[sorted_indices] = sorted_logits                  # restore original order

    p_final = F.softmax(filtered, dim=0)
    return torch.multinomial(p_final.unsqueeze(0), num_samples=1).squeeze()

print("Nucleus size at different p values:")
sorted_by_prob_v = sorted(zip(probs.tolist(), vocab), reverse=True)
print(f"\n  {'p value':<10} {'Nucleus size':>13} {'Included words'}")
print("  " + "-" * 50)
for p_val in [0.5, 0.7, 0.9, 0.95, 1.0]:
    cumul   = 0.0
    nucleus = []
    for prob_val, word in sorted_by_prob_v:
        if cumul >= p_val:
            break
        cumul += prob_val
        nucleus.append(word)
    if cumul < p_val and len(nucleus) < len(vocab):
        nucleus.append(sorted_by_prob_v[len(nucleus)][1])
    print(f"  {p_val:<10.2f} {len(nucleus):>4} words         {nucleus}")

print()
print("Sampling 5 times with p=0.9:")
for i in range(5):
    idx = top_p_sample(raw_logits, p=0.9, temperature=0.8).item()
    print(f"  Trial {i+1}: 'The cat {vocab[idx]}'")

# =============================================================================
# STRATEGY 5: Repetition Penalty
# =============================================================================

print("\n" + "=" * 50)
print("STRATEGY 5: Repetition Penalty")
print("=" * 50)
print("""
Reduce the score of tokens that have already been generated.
Prevents "the cat sat sat sat sat..."

If logit > 0:  logit = logit / penalty    (shrink positive scores)
If logit < 0:  logit = logit * penalty    (grow negative scores → more negative)

penalty > 1.0 makes repeated tokens less likely.
penalty = 1.0 has no effect.
""")

def apply_repetition_penalty(logits, used_token_indices, penalty=1.2):
    """Reduce logits of already-used tokens."""
    result = logits.clone()   # clone: don't modify the original tensor
    for idx in set(used_token_indices):
        if result[idx] > 0:
            result[idx] /= penalty    # shrink positive logit
        else:
            result[idx] *= penalty    # make negative logit even more negative
    return result

# Simulate: we've already generated "sat", "ran" — penalty them
used = [vocab.index("sat"), vocab.index("ran")]
penalized = apply_repetition_penalty(raw_logits, used, penalty=1.5)

print("Original logits vs penalized (sat and ran were already used):")
print(f"{'Word':<10} {'Original':>10} {'Penalized':>10} {'Change'}")
print("-" * 45)
for i, word in enumerate(vocab):
    orig = raw_logits[i].item()
    pen  = penalized[i].item()
    tag  = " <- penalized" if i in used else ""
    print(f"{word:<10} {orig:>10.2f} {pen:>10.2f}{tag}")

# =============================================================================
# Full generation loop with all strategies combined
# =============================================================================

print("\n" + "=" * 65)
print("PART B: Complete Generation Function (all strategies combined)")
print("=" * 65)

print("""
This is the real pattern used in production language models.
All 5 strategies can be combined in one generation loop.
""")

def generate_text(
    model_fn,               # function: context (list) -> logits (tensor)
    start_tokens,           # list of starting token indices
    max_new_tokens,         # how many tokens to generate
    temperature=0.8,        # controls randomness
    top_k=None,             # if set, use top-k filtering
    top_p=None,             # if set, use nucleus sampling
    repetition_penalty=1.0, # > 1 reduces repeating tokens
):
    """
    Complete generation loop with all strategies.
    Order of operations at each step:
      1. Get logits from model
      2. Apply repetition penalty
      3. Apply temperature
      4. Apply top-k (if enabled)
      5. Apply top-p (if enabled)
      6. Sample with torch.multinomial
      7. Append and repeat
    """
    tokens = list(start_tokens)   # copy — don't modify the input list

    for _ in range(max_new_tokens):
        # Step 1: Model produces logits for the current context
        logits = model_fn(tokens)    # tensor shape: (vocab_size,)

        # Step 2: Repetition penalty — reduce scores of already-used tokens
        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, tokens, repetition_penalty)

        # Step 3: Temperature — scale logits (higher temp = more random)
        logits = logits / temperature

        # Step 4: Top-k — keep only the k highest-scoring tokens
        if top_k is not None:
            topk_result = torch.topk(logits, k=top_k)
            threshold   = topk_result.values[-1]
            logits      = logits.masked_fill(logits < threshold, float('-inf'))

        # Step 5: Top-p — keep only tokens in the nucleus (smallest set summing to p)
        if top_p is not None:
            sorted_indices = torch.argsort(logits, descending=True)
            sorted_logits  = logits[sorted_indices]
            sorted_probs   = F.softmax(sorted_logits, dim=0)
            cumulative     = torch.cumsum(sorted_probs, dim=0)
            remove_mask    = (cumulative - sorted_probs) >= top_p
            sorted_logits  = sorted_logits.masked_fill(remove_mask, float('-inf'))
            filtered       = torch.full_like(logits, float('-inf'))
            filtered[sorted_indices] = sorted_logits
            logits         = filtered

        # Step 6: Convert to probabilities and sample
        p_final    = F.softmax(logits, dim=0)
        # Replace any NaN (from all-inf tensor) with uniform distribution
        if torch.isnan(p_final).any() or p_final.sum() == 0:
            p_final = torch.ones(len(p_final)) / len(p_final)
        next_token = torch.multinomial(p_final.unsqueeze(0), num_samples=1).item()

        # Step 7: Append to sequence
        tokens.append(next_token)

    return tokens


# Fake model: returns the same base logits + small random noise each call
def fake_model(tokens):
    torch.manual_seed(len(tokens))   # reproducible per position
    noise = torch.randn(vocab_size) * 0.3
    return raw_logits + noise


print("Generating 5 tokens using different strategies:")
print(f"Seed: 'sat' (index {vocab.index('sat')})")
print()

seed = [vocab.index("sat")]

strategies = [
    ("Greedy (temp=0.1)",           {"temperature": 0.1}),
    ("Creative (temp=2.0)",          {"temperature": 2.0}),
    ("Top-k=2 (temp=0.8)",           {"temperature": 0.8, "top_k": 2}),
    ("Nucleus p=0.7 (temp=0.8)",     {"temperature": 0.8, "top_p": 0.7}),
    ("Rep. penalty=1.5",             {"temperature": 0.8, "repetition_penalty": 1.5}),
]

torch.manual_seed(99)   # fix seed so output is consistent across runs
for label, settings in strategies:
    torch.manual_seed(99)
    result = generate_text(fake_model, seed, max_new_tokens=5, **settings)
    words  = " ".join(vocab[i] for i in result)
    print(f"  {label:<30}: 'The cat {words}'")

# =============================================================================
# Decision Guide (same as NumPy version)
# =============================================================================

print("\n" + "=" * 65)
print("STRATEGY DECISION GUIDE")
print("=" * 65)
print("""
  What are you building?
  |
  |--- Code completion / SQL / exact output needed
  |     -> temperature=0.1, top_k=10
  |       (precision matters, creativity hurts)
  |
  |--- Factual Q&A / summarization
  |     -> temperature=0.3, top_p=0.9
  |       (mostly accurate, slight variety)
  |
  |--- Chatbot / conversational
  |     -> temperature=0.8, top_p=0.9, repetition_penalty=1.1
  |       (natural, varied, no loops)
  |       THIS IS WHAT CHATGPT USES
  |
  |--- Creative writing / story
  |     -> temperature=1.2, top_p=0.95, repetition_penalty=1.3
  |       (surprising, diverse, avoids repetition)
  |
  \--- Translation / best answer
        -> beam search  (not shown here — in Module 6)
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 65)
print("SUMMARY: NumPy vs PyTorch Sampling")
print("=" * 65)
print("""
  NumPy (example_05):                  PyTorch (this file):
  ----------------------------          ----------------------------
  np.exp(x) / sum(np.exp(x))           F.softmax(x, dim=0)
  np.argmax(probs)                      torch.argmax(probs)
  np.random.choice(n, p=probs)          torch.multinomial(probs, 1)
  np.sort(x)[::-1][:k]                  torch.topk(x, k)
  np.argsort(x)[::-1]                   torch.argsort(x, descending=True)
  np.cumsum(x)                          torch.cumsum(x, dim=0)
  np.where(cond, x, -np.inf)            x.masked_fill(~cond, float('-inf'))
  logits.copy()                         logits.clone()

New PyTorch tools learned:
  F.softmax(x, dim)            probabilities from logits
  torch.argmax(x)              index of max value (greedy)
  torch.multinomial(p, n)      weighted random sampling
  torch.topk(x, k)             top k values and indices
  torch.argsort(x, desc=True)  sort indices descending
  torch.cumsum(x, dim)         running cumulative sum
  tensor.masked_fill(mask, v)  fill positions where mask is True
  tensor.clone()               deep copy (vs .copy() in NumPy)
""")

print("=" * 65)
print("Module 05 PyTorch examples complete!")
print("Example 04 (example_04_gpt_pytorch.py) shows the full GPT model.")
print("=" * 65)
