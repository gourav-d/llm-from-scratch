# Lesson 5.5: Text Generation & Sampling Strategies

**Control HOW your GPT generates text — creativity vs quality vs speed**

---

## 🎯 What You'll Learn

- ✅ How autoregressive generation works (one token at a time)
- ✅ Greedy decoding — always pick the highest probability token
- ✅ Temperature — dial up creativity or precision
- ✅ Top-k sampling — restrict to the top k candidates
- ✅ Top-p (nucleus) sampling — dynamically restrict by cumulative probability
- ✅ Beam search — explore multiple paths simultaneously
- ✅ Repetition penalty — stop the model looping forever
- ✅ When to use each strategy
- ✅ Real-world settings used by ChatGPT, GitHub Copilot, etc.

**Time:** 2-3 hours  
**Difficulty:** ⭐⭐⭐☆☆

---

## 🔄 How Autoregressive Generation Works

GPT doesn't generate a whole sentence at once. It generates **one token at a time**, each time using all previous tokens as context.

```
Prompt: "The cat"

Step 1:  Input = ["The", "cat"]
         Model outputs scores for every possible next token (vocab_size = 65)
         e.g. "sat": 0.42, "is": 0.18, "ran": 0.12, ...
         Sample → "sat"

Step 2:  Input = ["The", "cat", "sat"]
         Model outputs scores again
         e.g. "on": 0.55, "down": 0.20, "by": 0.10, ...
         Sample → "on"

Step 3:  Input = ["The", "cat", "sat", "on"]
         ...continues until max_tokens or end-of-sequence token
```

**This loop is called autoregressive generation.**

---

### Visual diagram

```
                    ┌─────────────────────────┐
                    │        GPT Model        │
                    └─────────────────────────┘
                              │
  "The cat" ──────────────────┤  → logits (scores for 65 tokens)
                              │  → Sample → "sat"
                              │
  "The cat sat" ──────────────┤  → logits
                              │  → Sample → "on"
                              │
  "The cat sat on" ───────────┤  → logits
                              │  → Sample → "the"
                              │
  ...continues...
```

---

### Base Generation Code (PyTorch)

```python
import torch
import torch.nn.functional as F

@torch.no_grad()                   # No gradients needed for inference
def generate(model, idx, max_new_tokens, strategy='greedy', **kwargs):
    """
    Generate tokens autoregressively

    Args:
        model:          trained GPT model
        idx:            starting token IDs, shape (1, seq_len)
        max_new_tokens: how many tokens to generate
        strategy:       'greedy', 'temperature', 'top_k', 'top_p', 'beam'
        **kwargs:       strategy-specific parameters

    Returns:
        idx: extended token IDs, shape (1, seq_len + max_new_tokens)
    """
    model.eval()

    for _ in range(max_new_tokens):
        # Crop to max sequence length
        idx_cond = idx[:, -model.max_seq_len:]

        # Forward pass → get logits
        logits, _ = model(idx_cond)

        # We only care about the LAST position (predict next token)
        logits = logits[:, -1, :]              # (1, vocab_size)

        # Apply chosen sampling strategy
        next_token = sample(logits, strategy, **kwargs)

        # Append to sequence
        idx = torch.cat([idx, next_token], dim=1)

    return idx
```

---

## 🥇 Strategy 1: Greedy Decoding

**Always pick the single highest-probability token.**

```python
def greedy(logits):
    """
    Greedy: take the argmax.

    C# analogy:
    Like always picking the highest-ranked item from a sorted list.
    """
    probs     = F.softmax(logits, dim=-1)     # Convert to probabilities
    next_token = probs.argmax(dim=-1, keepdim=True)  # Pick highest
    return next_token
```

### Example
```
Vocab:  [" ", "a", "b", "c", "sat", "cat", "on", "the"]
Logits: [0.01, 0.05, 0.02, 0.01,  0.60,  0.20, 0.08, 0.03]
                                   ↑
                             Greedy picks "sat" (0.60)
```

### Pros and Cons

| ✅ Pros | ❌ Cons |
|--------|--------|
| Fast — no randomness | Repetitive — loops on the same phrase |
| Deterministic — same input → same output | Misses good sequences that start unlikely |
| Good for factual Q&A | "The cat sat on the cat sat on the cat..." |

**Best for:** Factual completions, code, structured outputs.

---

## 🌡️ Strategy 2: Temperature Sampling

**Scale the logits before softmax to control randomness.**

```python
def temperature_sample(logits, temperature=1.0):
    """
    Temperature sampling.

    temperature < 1.0 → sharper distribution → more focused
    temperature = 1.0 → unchanged distribution (default)
    temperature > 1.0 → flatter distribution  → more random

    C# analogy:
    Like adjusting a volume knob on a probability distribution.
    Low temp = quieter, focused. High temp = louder, chaotic.
    """
    # Divide logits by temperature BEFORE softmax
    scaled_logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Sample from the distribution
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

### How Temperature Affects the Distribution

```
Original logits: [2.0,  1.0,  0.5,  0.1]

Temperature = 0.5 (focused):
  Scaled: [4.0, 2.0, 1.0, 0.2]
  Probs:  [0.72, 0.17, 0.08, 0.03]  ← top choice dominates

Temperature = 1.0 (default):
  Scaled: [2.0, 1.0, 0.5, 0.1]
  Probs:  [0.57, 0.23, 0.14, 0.06]  ← moderate spread

Temperature = 2.0 (creative):
  Scaled: [1.0, 0.5, 0.25, 0.05]
  Probs:  [0.38, 0.27, 0.21, 0.14]  ← very flat, lots of variety
```

### Visual

```
Low temperature (0.3):        High temperature (2.0):
████████░░░░░░░░░░░░          ████░░░░░░░░░░░░░░░░░░
"sat" wins almost always      Any token might be picked
```

### Real-World Temperature Settings

| Use Case | Temperature | Why |
|----------|------------|-----|
| Code completion | 0.1-0.3 | Deterministic, correct syntax matters |
| Factual Q&A | 0.3-0.5 | Accurate but not robotic |
| Chat assistant | 0.7-0.9 | Natural, slightly varied |
| Creative writing | 1.0-1.5 | Surprising, diverse ideas |
| Brainstorming | 1.5-2.0 | Wild, experimental |

---

## 🔝 Strategy 3: Top-k Sampling

**Only sample from the top k most probable tokens. Ignore the rest.**

```python
def top_k_sample(logits, k=50, temperature=1.0):
    """
    Top-k sampling: restrict to k most likely tokens, then sample.

    Prevents the model from picking very unlikely (nonsense) tokens,
    while still allowing variety among the plausible options.

    C# analogy:
    Like a search autocomplete that shows only the top 5 suggestions —
    you pick from those, not from every word in the dictionary.
    """
    # Apply temperature
    logits = logits / temperature

    # Find the k-th largest value
    top_k_values, _ = torch.topk(logits, k)
    kth_value        = top_k_values[:, -1].unsqueeze(-1)  # Smallest of the top-k

    # Set everything below k-th to -inf (removes them from sampling)
    filtered_logits = logits.masked_fill(logits < kth_value, float('-inf'))

    # Convert to probabilities and sample
    probs      = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

### Example with k=3

```
All 8 tokens and their probabilities:
"sat"  0.40  ← top 3
"on"   0.25  ← top 3
"down" 0.18  ← top 3
"by"   0.07  ← removed (below top 3)
"the"  0.05  ← removed
"a"    0.03  ← removed
"in"   0.01  ← removed
"up"   0.01  ← removed

After top-3 filtering:
"sat"  0.493  (re-normalised among the 3 survivors)
"on"   0.309
"down" 0.222

Sample from these three only.
```

### How to Choose k

| k value | Effect |
|---------|--------|
| k=1 | Same as greedy (no variety) |
| k=10-50 | Good balance — common setting |
| k=100+ | Similar to no filtering |

**GPT-3 default:** `top_k=50`  
**GitHub Copilot:** `top_k=10` (code needs precision)

---

## 🎯 Strategy 4: Top-p (Nucleus) Sampling

**Dynamically choose the smallest set of tokens whose probabilities sum to ≥ p.**

```python
def top_p_sample(logits, p=0.9, temperature=1.0):
    """
    Nucleus (top-p) sampling.

    Instead of a fixed k, we pick the minimum number of tokens
    needed to cover p% of the probability mass.

    Why better than top-k?
    - When the model is confident → few tokens cover 90% → small nucleus
    - When the model is uncertain → many tokens needed → larger nucleus
    Top-k doesn't adapt; top-p does.

    C# analogy:
    Top-k = "always interview the top 10 candidates"
    Top-p = "interview candidates until you've seen 90% of the talent pool"
    """
    # Apply temperature
    logits = logits / temperature

    # Sort tokens from most to least probable
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens once cumulative probability exceeds p
    # Shift right by 1: keep the token that pushes us over p (include it)
    remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove_mask] = float('-inf')

    # Unsort back to original order
    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    # Sample
    probs      = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

### Why Top-p Adapts Better Than Top-k

```
Sentence: "The capital of France is ___"
Model is very confident: "Paris" = 0.95, "Lyon" = 0.03, ...

Top-k=50: samples from 50 tokens (includes nonsense like "banana")
Top-p=0.9: nucleus = just "Paris"  → correct almost always ✓

Sentence: "Once upon a time there was a ___"
Model is uncertain: "princess"=0.12, "dragon"=0.11, "wizard"=0.10, ...

Top-k=50: samples from 50 tokens ✓
Top-p=0.9: nucleus = top 40 tokens (keeps variety) ✓

Top-p adapts to the model's confidence level!
```

### Top-p Settings in Practice

| p value | Effect |
|---------|--------|
| p=0.5 | Very focused, low diversity |
| p=0.9 | Good balance — **ChatGPT default** |
| p=0.95 | More creative |
| p=1.0 | No filtering (sample entire vocab) |

---

## 🔦 Strategy 5: Beam Search

**Explore multiple sequences simultaneously, keep the best ones.**

```python
def beam_search(model, idx, max_new_tokens, beam_width=5):
    """
    Beam search: maintain beam_width candidate sequences.

    At each step, expand every beam and keep the top beam_width.

    C# analogy:
    Like a BFS where you keep only the top-N nodes at each level,
    pruning unlikely branches early.
    """
    # Start with one beam: (score, sequence)
    beams = [(0.0, idx)]

    for _ in range(max_new_tokens):
        candidates = []

        for score, seq in beams:
            # Forward pass
            logits, _ = model(seq[:, -model.max_seq_len:])
            log_probs  = F.log_softmax(logits[:, -1, :], dim=-1)

            # Expand: try all vocab tokens
            top_log_probs, top_indices = log_probs.topk(beam_width)

            for i in range(beam_width):
                new_score = score + top_log_probs[0, i].item()
                new_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq   = torch.cat([seq, new_token], dim=1)
                candidates.append((new_score, new_seq))

        # Keep only the top beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    # Return the best sequence
    best_score, best_seq = beams[0]
    return best_seq
```

### Visualisation: beam_width = 3

```
Step 0:                "The"
                      /  |  \
Step 1:          "cat" "dog" "bird"
                 / \     |     |
Step 2:       "sat""is" "ran" "flew"
               |         |
Step 3:      "on"       "away"
              |
            "the"

Best path: "The" → "cat" → "sat" → "on" → "the"
Score = sum of log probabilities along the path
```

### Beam Search vs Sampling

| | Beam Search | Temperature/Top-p |
|--|-------------|------------------|
| **Deterministic?** | Yes | No |
| **Diversity** | Low (finds "best") | High |
| **Speed** | Slow (beam_width × slower) | Fast |
| **Best for** | Translation, summarisation | Chat, creative writing |
| **Used by** | Machine translation | ChatGPT |

---

## 🔁 Repetition Penalty

**Penalise tokens the model has already used — breaks repetition loops.**

```python
def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """
    Reduce probability of tokens already in the sequence.

    Without this: "The cat sat on the cat sat on the cat sat on..."
    With this:    "The cat sat on the mat and looked around."

    penalty > 1.0: reduces repeated token probability
    penalty = 1.0: no effect
    penalty < 1.0: increases repeated tokens (rarely useful)
    """
    for token_id in set(generated_ids[0].tolist()):
        # If logit is positive: divide (reduces it)
        # If logit is negative: multiply (makes it more negative)
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty

    return logits
```

---

## 🧩 Putting It All Together

```python
@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    temperature     = 1.0,
    top_k           = None,
    top_p           = None,
    repetition_penalty = 1.0
):
    """
    Full generation function with all strategies combined.

    Common production settings:
    - Chat bot:       temperature=0.8, top_p=0.9, repetition_penalty=1.1
    - Code:           temperature=0.2, top_k=10
    - Creative:       temperature=1.2, top_p=0.95, repetition_penalty=1.3
    - Deterministic:  temperature=0.1 (near-greedy)
    """
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]              # Last position only

        # 1. Repetition penalty
        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, idx, repetition_penalty)

        # 2. Temperature
        logits = logits / temperature

        # 3. Top-k filter
        if top_k is not None:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_k_values[:, -1:]] = float('-inf')

        # 4. Top-p filter
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        # 5. Sample
        probs      = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 6. Append and continue
        idx = torch.cat([idx, next_token], dim=1)

    return idx
```

---

## 🎛️ Strategy Decision Guide

```
What are you building?
        │
        ├─ Code completion / SQL / structured output
        │     → temperature=0.1-0.3, top_k=10
        │       (precision matters, creativity hurts)
        │
        ├─ Factual Q&A / summarisation
        │     → temperature=0.3-0.5, top_p=0.9
        │       (mostly accurate, small variety)
        │
        ├─ Chat / conversational assistant
        │     → temperature=0.7-0.9, top_p=0.9, repetition_penalty=1.1
        │       (natural, varied, non-repetitive)
        │
        ├─ Creative writing / story generation
        │     → temperature=1.0-1.5, top_p=0.95, repetition_penalty=1.3
        │       (surprising, diverse, avoids loops)
        │
        └─ Translation / beam-based tasks
              → beam_search(beam_width=4-5)
                (find best single output, not diverse)
```

---

## 🌍 Real-World Settings

| System | Strategy |
|--------|---------|
| **ChatGPT** | top_p=0.9, temperature≈0.7-0.9 |
| **GitHub Copilot** | temperature≈0.2, top_k=10 |
| **GPT-3 (default API)** | temperature=1.0, top_p=1.0 |
| **Google Translate** | Beam search, beam_width=4 |
| **Stable Diffusion (text)** | temperature=0.7, top_k=50 |

---

## 📝 Quiz

### Question 1
**What does temperature = 0.0 effectively give you?**

<details>
<summary>Click to see answer</summary>

Temperature = 0.0 would divide logits by 0 (undefined mathematically), but as temperature → 0, the distribution approaches a **one-hot** — the highest logit gets all the probability.

In practice, very low temperature (e.g., 0.01) = **greedy decoding** — the top token wins almost every time. This is why people say "set temperature to 0 for deterministic output" — they mean "nearly 0" or they handle it by using argmax directly.
</details>

---

### Question 2
**Why does top-p outperform top-k for chat applications?**

<details>
<summary>Click to see answer</summary>

Top-k uses a fixed number regardless of the model's confidence:

- If model is 99% sure of one token and k=50, it still samples from 50 tokens including nonsense.
- If model is evenly uncertain across 200 tokens and k=50, it cuts off 150 plausible options.

Top-p adapts:
- Confident model → small nucleus (maybe just 2-3 tokens cover 90%)
- Uncertain model → large nucleus (maybe 80 tokens needed to cover 90%)

Top-p respects the model's natural confidence level, giving better outputs in conversation where topics shift unpredictably.
</details>

---

### Question 3
**You're building a medical report generator. Which strategy should you use and why?**

<details>
<summary>Click to see answer</summary>

Use **low temperature (0.1-0.3) with top-k (k=10-20)**.

**Why:**
- Medical reports require factual accuracy — creativity is dangerous.
- Low temperature keeps the model focused on high-probability (likely correct) completions.
- Top-k filters out any low-probability tokens that could introduce hallucinations.
- You want deterministic, repeatable outputs so doctors can verify the system's behaviour.

Avoid: high temperature, nucleus sampling with large p, or beam search (which can still hallucinate just more fluently).

Combine with: retrieval-augmented generation (RAG) to ground outputs in real patient data.
</details>

---

### Question 4
**What's the difference between `torch.argmax` (greedy) and `torch.multinomial` (sampling)?**

<details>
<summary>Click to see answer</summary>

```python
probs = torch.tensor([0.6, 0.3, 0.1])

# Greedy — always the max
torch.argmax(probs)            # Always returns index 0 (prob=0.6)

# Multinomial — random draw weighted by probs
torch.multinomial(probs, 1)   # 60% chance index 0
                               # 30% chance index 1
                               # 10% chance index 2
```

`argmax` is deterministic — same input, same output every time.  
`multinomial` is stochastic — introduces controlled randomness.

Temperature + top-k/top-p modify the `probs` **before** we call `multinomial`, shaping the distribution to control how much randomness we allow.
</details>

---

## 🧪 Exercises

### Exercise 1: Compare Outputs
Generate the same prompt with 5 different settings and compare the output quality:

```python
prompt = "To be or not to be"
settings = [
    {"temperature": 0.1},                       # near-greedy
    {"temperature": 1.0},                       # default
    {"temperature": 2.0},                       # chaotic
    {"temperature": 0.8, "top_k": 10},          # top-k focused
    {"temperature": 0.8, "top_p": 0.9},         # nucleus
]
for s in settings:
    output = generate(model, encode(prompt), max_new_tokens=50, **s)
    print(f"\n{s}:")
    print(decode(output[0].tolist()))
```

### Exercise 2: Implement Greedy as Temperature → 0
Without using `torch.argmax`, implement greedy decoding purely via temperature:

```python
def greedy_via_temperature(logits, epsilon=1e-6):
    """Use temperature → 0 to achieve greedy behaviour"""
    # YOUR CODE HERE
    # Hint: divide by a very small number before softmax
    pass
```

### Exercise 3: Implement Repetition Penalty and Test It
```python
# Generate 100 tokens WITHOUT repetition penalty, observe loops
output_no_penalty = generate(model, start, 100, temperature=1.0)

# Generate 100 tokens WITH repetition penalty
output_penalised  = generate(model, start, 100, temperature=1.0, repetition_penalty=1.3)

print("Without penalty:", decode(output_no_penalty[0].tolist()))
print("With penalty:   ", decode(output_penalised[0].tolist()))
# Notice: penalised version should have more variety
```

---

## 🎓 Key Takeaways

| Strategy | When | Key Parameter |
|----------|------|--------------|
| **Greedy** | Factual, code | None |
| **Temperature** | All tasks | `temperature` (0.1–2.0) |
| **Top-k** | Code, chat | `k` (10–50) |
| **Top-p** | Chat, creative | `p` (0.9–0.95) |
| **Beam** | Translation | `beam_width` (4–5) |
| **Rep. penalty** | Long generation | `penalty` (1.1–1.5) |

**Most production systems use: temperature + top-p + repetition_penalty combined.**

---

## 🏁 Module 05 Complete!

You now understand the full pipeline for Building Your LLM:

```
Lesson 5.1: Tokenization           → text → token IDs
Lesson 5.2: Word Embeddings        → token IDs → dense vectors
Lesson 5.3: nanoGPT                → build GPT in 200 lines (NumPy)
Lesson 5.4: GPT with PyTorch       → build GPT the production way
Lesson 5.5: Text Generation        → control HOW it generates ← you are here
```

**You can now build and run your own language model end-to-end!**

**Next:** Module 06 — Training & Fine-tuning (how to train on real data, fine-tune, RLHF)
