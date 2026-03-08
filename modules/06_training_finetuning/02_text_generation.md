# Lesson 6.2: Text Generation & Sampling Strategies

**Make your GPT model generate creative, coherent text!**

---

## 🎯 Learning Objectives

By the end of this lesson, you will:
- ✅ Understand autoregressive text generation
- ✅ Implement greedy sampling (deterministic)
- ✅ Control randomness with temperature
- ✅ Use top-k sampling to limit choices
- ✅ Use top-p (nucleus) sampling like GPT-3
- ✅ Understand beam search for quality
- ✅ Balance creativity vs coherence
- ✅ Generate text with your GPT model

**Time Required:** 3-4 hours

---

## 📚 What is Text Generation?

### Autoregressive Generation

**Autoregressive** means the model generates text one token at a time, using its own output as input for the next prediction.

```
Start with prompt: "The cat"
    ↓
Model predicts: "sat" (most likely next word)
    ↓
New input: "The cat sat"
    ↓
Model predicts: "on" (next word)
    ↓
New input: "The cat sat on"
    ↓
Model predicts: "the" (next word)
    ↓
New input: "The cat sat on the"
    ↓
... continue until max length or special token
```

### The Core Loop

```python
def generate_text(model, prompt, max_length=50):
    """
    Generate text autoregressively.

    Think of this like StringBuilder.Append() in C#,
    but we're appending one token at a time based on
    model predictions!
    """
    # Tokenize the prompt
    tokens = tokenize(prompt)

    # Generate tokens one by one
    for _ in range(max_length):
        # Get model predictions
        logits = model.forward(tokens)

        # Get logits for last position (next token)
        next_token_logits = logits[-1]  # Shape: (vocab_size,)

        # Sample next token (different strategies!)
        next_token = sample(next_token_logits)

        # Append to sequence
        tokens.append(next_token)

        # Stop if we hit end-of-sequence token
        if next_token == EOS_TOKEN:
            break

    # Convert tokens back to text
    text = detokenize(tokens)
    return text
```

**C#/.NET Analogy:**
```csharp
// Similar to:
var sb = new StringBuilder(prompt);
for (int i = 0; i < maxLength; i++)
{
    var prediction = model.Predict(sb.ToString());
    sb.Append(prediction);

    if (prediction == endToken) break;
}
return sb.ToString();
```

---

## 🎲 Sampling Strategy 1: Greedy Sampling

### Always Pick the Most Likely Token

**Greedy sampling** = Always choose the token with highest probability.

```python
def greedy_sampling(logits):
    """
    Greedy sampling: Pick the most likely token.

    Args:
        logits: Raw model outputs, shape (vocab_size,)

    Returns:
        token_id: Most likely token
    """
    # Convert logits to probabilities
    probs = softmax(logits)

    # Pick token with highest probability
    token_id = np.argmax(probs)

    return token_id
```

**Example:**
```python
# Model outputs probabilities:
# "the":   0.45  ← Highest!
# "a":     0.25
# "this":  0.15
# "that":  0.10
# ... (other tokens have < 0.05)

# Greedy sampling picks: "the" (always!)
```

### Pros and Cons

**Pros:**
- ✅ Deterministic (same input = same output)
- ✅ Fast (no randomness needed)
- ✅ Often grammatically correct
- ✅ Good for short sequences

**Cons:**
- ❌ Repetitive (gets stuck in loops)
- ❌ Not creative
- ❌ Boring output
- ❌ "The cat sat on the mat. The cat sat on the mat. The cat sat..."

**When to use:**
- Summarization (want predictable output)
- Translation (want most likely translation)
- Short completions

---

## 🌡️ Sampling Strategy 2: Temperature Sampling

### Control Randomness with Temperature

**Temperature** controls how "random" or "confident" the model is.

```python
def temperature_sampling(logits, temperature=1.0):
    """
    Temperature sampling: Control randomness.

    Args:
        logits: Raw model outputs, shape (vocab_size,)
        temperature: Controls randomness
                    - Low (0.1): More confident, less random
                    - 1.0: Use probabilities as-is
                    - High (2.0): More random, less confident

    Returns:
        token_id: Sampled token
    """
    # Apply temperature scaling
    # Divide logits by temperature BEFORE softmax
    logits = logits / temperature

    # Convert to probabilities
    probs = softmax(logits)

    # Sample from probability distribution
    token_id = np.random.choice(len(probs), p=probs)

    return token_id
```

**How temperature works:**

```python
# Original probabilities (temperature = 1.0):
# "the":   0.45
# "a":     0.25
# "this":  0.15
# "that":  0.10
# "my":    0.05

# Low temperature (0.5): Makes distribution sharper
# "the":   0.68  ← Even more likely!
# "a":     0.18  ← Less likely
# "this":  0.08
# "that":  0.04
# "my":    0.02
# Result: More confident, less creative

# High temperature (2.0): Makes distribution flatter
# "the":   0.28  ← Less dominant
# "a":     0.22  ← More likely
# "this":  0.18
# "that":  0.16
# "my":    0.16
# Result: More random, more creative
```

**Visual representation:**

```
Temperature = 0.1 (Very confident)
████████████████████████ "the" (95%)
█ "a" (3%)
█ "this" (1%)
  (other tokens: ~0%)

Temperature = 1.0 (Normal)
█████████ "the" (45%)
█████ "a" (25%)
███ "this" (15%)
██ "that" (10%)
█ "my" (5%)

Temperature = 2.0 (Very random)
█████ "the" (28%)
████ "a" (22%)
████ "this" (18%)
███ "that" (16%)
███ "my" (16%)
```

### The Math Behind Temperature

```python
# Why does dividing by temperature work?

# Softmax formula:
# prob[i] = exp(logit[i]) / sum(exp(logit[j]))

# With temperature:
# prob[i] = exp(logit[i] / T) / sum(exp(logit[j] / T))

# When T < 1 (e.g., 0.5):
# - Exponentiation amplifies differences
# - High logits become even higher
# - Distribution becomes sharper (more confident)

# When T > 1 (e.g., 2.0):
# - Exponentiation reduces differences
# - High and low logits become closer
# - Distribution becomes flatter (more random)
```

**Example generation:**

```python
prompt = "Once upon a time"

# Temperature = 0.2 (Conservative)
"Once upon a time, there was a young girl who lived in a small village."
# Predictable, grammatical, boring

# Temperature = 0.8 (Balanced)
"Once upon a time, in a mystical forest, an ancient dragon discovered a magical stone."
# Creative, coherent, interesting

# Temperature = 1.5 (Wild)
"Once upon a time, the moonlight whispered secrets to dancing shadows beneath the ocean."
# Very creative, might lose coherence
```

---

## 🔝 Sampling Strategy 3: Top-k Sampling

### Limit to k Most Likely Tokens

**Top-k sampling** = Only consider the k most likely tokens, then sample from those.

```python
def top_k_sampling(logits, k=40, temperature=1.0):
    """
    Top-k sampling: Sample from k most likely tokens.

    Args:
        logits: Raw model outputs, shape (vocab_size,)
        k: Number of top tokens to consider
        temperature: Temperature for sampling

    Returns:
        token_id: Sampled token
    """
    # Apply temperature
    logits = logits / temperature

    # Get top-k token indices
    # np.argpartition is faster than full sort
    top_k_indices = np.argpartition(logits, -k)[-k:]

    # Get top-k logits
    top_k_logits = logits[top_k_indices]

    # Convert to probabilities
    top_k_probs = softmax(top_k_logits)

    # Sample from top-k
    # np.random.choice returns index into top_k_indices
    sample_index = np.random.choice(len(top_k_probs), p=top_k_probs)

    # Get actual token ID
    token_id = top_k_indices[sample_index]

    return token_id
```

**How it works:**

```python
# All tokens sorted by probability:
1.  "the"      0.30  ← Top-5 starts here
2.  "a"        0.20  │
3.  "this"     0.15  │
4.  "that"     0.10  │
5.  "my"       0.08  ← Top-5 ends here
6.  "our"      0.05  ← Ignored with k=5
7.  "your"     0.04  │
8.  "their"    0.03  │
9.  "some"     0.02  │
... (40,000+ more tokens) ← All ignored!

# With k=5, we only sample from top 5
# Renormalize probabilities:
"the":   0.361  (0.30 / 0.83)
"a":     0.241  (0.20 / 0.83)
"this":  0.181  (0.15 / 0.83)
"that":  0.120  (0.10 / 0.83)
"my":    0.096  (0.08 / 0.83)
```

### Why Top-k?

**Problems with pure temperature sampling:**
- Might sample very unlikely tokens
- "The cat sat on the quantum" (quantum has 0.001% probability!)
- Leads to nonsense

**Top-k solves this:**
- ✅ Prevents sampling garbage tokens
- ✅ Maintains quality
- ✅ Still allows creativity
- ✅ Widely used in practice (k=40 is common)

**Choosing k:**
```python
k = 1      # Equivalent to greedy (no randomness)
k = 10     # Very conservative
k = 40     # Balanced (GPT-2 default)
k = 100    # More creative
k = 500    # Very creative (might get weird)
```

---

## 🎯 Sampling Strategy 4: Top-p (Nucleus) Sampling

### Sample from Smallest Set with Cumulative Probability ≥ p

**Top-p (nucleus) sampling** = Keep adding tokens until cumulative probability reaches p.

**Why better than top-k?**
- Top-k always uses exactly k tokens
- But sometimes we're confident (use fewer tokens), sometimes uncertain (use more tokens)
- Top-p adapts!

```python
def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    Top-p (nucleus) sampling: Sample from smallest set with cumulative prob ≥ p.

    This is what GPT-3 uses!

    Args:
        logits: Raw model outputs, shape (vocab_size,)
        p: Cumulative probability threshold (typically 0.9 or 0.95)
        temperature: Temperature for sampling

    Returns:
        token_id: Sampled token
    """
    # Apply temperature
    logits = logits / temperature

    # Convert to probabilities
    probs = softmax(logits)

    # Sort probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Calculate cumulative probabilities
    cumsum_probs = np.cumsum(sorted_probs)

    # Find cutoff index where cumsum >= p
    # This is the "nucleus" of the distribution
    cutoff_index = np.searchsorted(cumsum_probs, p)

    # Include one more token to ensure cumsum >= p
    cutoff_index = cutoff_index + 1

    # Get nucleus tokens and probabilities
    nucleus_indices = sorted_indices[:cutoff_index]
    nucleus_probs = sorted_probs[:cutoff_index]

    # Renormalize probabilities
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    # Sample from nucleus
    sample_index = np.random.choice(len(nucleus_probs), p=nucleus_probs)

    # Get actual token ID
    token_id = nucleus_indices[sample_index]

    return token_id
```

**Example 1: High confidence (model is sure)**

```python
# Probabilities (sorted):
"the":    0.70  ← Cumsum: 0.70
"a":      0.15  ← Cumsum: 0.85
"this":   0.10  ← Cumsum: 0.95 ← Hit p=0.9! Nucleus = 3 tokens
"that":   0.03  ← Ignored
"my":     0.02  ← Ignored
...

# With p=0.9, nucleus = ["the", "a", "this"]
# Only 3 tokens! Model is confident.
```

**Example 2: Low confidence (model is uncertain)**

```python
# Probabilities (sorted):
"the":    0.15  ← Cumsum: 0.15
"a":      0.12  ← Cumsum: 0.27
"this":   0.10  ← Cumsum: 0.37
"that":   0.09  ← Cumsum: 0.46
"my":     0.08  ← Cumsum: 0.54
"our":    0.07  ← Cumsum: 0.61
"your":   0.06  ← Cumsum: 0.67
"their":  0.05  ← Cumsum: 0.72
"some":   0.05  ← Cumsum: 0.77
"one":    0.04  ← Cumsum: 0.81
"two":    0.04  ← Cumsum: 0.85
"three":  0.04  ← Cumsum: 0.89
"any":    0.03  ← Cumsum: 0.92 ← Hit p=0.9! Nucleus = 13 tokens
...

# With p=0.9, nucleus = 13 tokens
# More tokens! Model is uncertain, so we allow more options.
```

**This is GPT-3's secret sauce!**

### Top-p vs Top-k

| Scenario | Top-k (k=40) | Top-p (p=0.9) |
|----------|-------------|---------------|
| **Confident prediction** | Uses 40 tokens (maybe overkill) | Uses 3 tokens (efficient!) |
| **Uncertain prediction** | Uses 40 tokens (maybe not enough) | Uses 100 tokens (adaptive!) |
| **Highly uncertain** | Still 40 tokens | Uses 500+ tokens |

**Top-p adapts to model confidence!**

---

## 🔍 Sampling Strategy 5: Beam Search

### Explore Multiple Paths Simultaneously

**Beam search** = Keep track of multiple candidate sequences, always keep the k most likely.

```python
def beam_search(model, prompt, beam_width=5, max_length=50):
    """
    Beam search: Explore multiple hypotheses.

    Args:
        model: GPT model
        prompt: Initial text
        beam_width: Number of beams (hypotheses to track)
        max_length: Maximum generation length

    Returns:
        best_sequence: Most likely sequence
    """
    # Initialize: One beam with the prompt
    beams = [(tokenize(prompt), 0.0)]  # (sequence, log_probability)

    for _ in range(max_length):
        new_beams = []

        # For each current beam
        for sequence, score in beams:
            # Get model predictions
            logits = model.forward(sequence)
            next_token_logits = logits[-1]

            # Convert to log probabilities
            log_probs = log_softmax(next_token_logits)

            # Get top beam_width tokens
            top_indices = np.argsort(log_probs)[-beam_width:]

            # Create new beams
            for token_id in top_indices:
                new_sequence = sequence + [token_id]
                new_score = score + log_probs[token_id]
                new_beams.append((new_sequence, new_score))

        # Keep only top beam_width beams
        # Sort by score (higher is better)
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    # Return best beam
    best_sequence, best_score = beams[0]
    return detokenize(best_sequence)
```

**Visualization:**

```
Start: "The cat"
    ↓
Beam 1: "The cat sat" (score: -1.2)
Beam 2: "The cat ran" (score: -1.5)
Beam 3: "The cat is"  (score: -1.8)
Beam 4: "The cat was" (score: -2.0)
Beam 5: "The cat ate" (score: -2.1)
    ↓
Expand each beam with top-5 tokens...
    ↓
Keep top-5 overall:
Beam 1: "The cat sat on"  (score: -2.1)
Beam 2: "The cat sat in"  (score: -2.4)
Beam 3: "The cat ran to"  (score: -2.5)
Beam 4: "The cat is a"    (score: -2.6)
Beam 5: "The cat sat by"  (score: -2.7)
    ↓
Continue until max_length...
    ↓
Return best: "The cat sat on the mat."
```

**Pros and Cons:**

**Pros:**
- ✅ More thorough search
- ✅ Often higher quality
- ✅ Good for tasks with "correct" answers (translation, summarization)

**Cons:**
- ❌ Slower (beam_width × model calls)
- ❌ Still can be repetitive
- ❌ Less creative than sampling
- ❌ Tends toward generic text

---

## ⚖️ Comparing All Strategies

### Side-by-Side Comparison

Prompt: "Once upon a time"

**Greedy Sampling:**
```
"Once upon a time, there was a little girl who lived in a small house.
She was a very good girl."
```
→ Boring, repetitive, grammatical

**Temperature = 0.5:**
```
"Once upon a time, there was a young prince who lived in a beautiful castle.
One day, he decided to go on an adventure."
```
→ Safe, coherent, predictable

**Temperature = 1.0:**
```
"Once upon a time, in a mysterious forest, lived a wise old owl who guarded
ancient secrets. Many travelers sought his wisdom."
```
→ Creative, coherent, interesting

**Temperature = 1.5:**
```
"Once upon a time, beneath shimmering stars, whispered forgotten melodies
danced through crystalline chambers of endless possibility."
```
→ Very creative, might lose coherence

**Top-k (k=40, T=0.8):**
```
"Once upon a time, there lived a curious young inventor who dreamed of building
flying machines. Her workshop was filled with strange contraptions."
```
→ Creative, coherent, avoids nonsense

**Top-p (p=0.9, T=0.8):**
```
"Once upon a time, in a land where magic and science intertwined, a brave
explorer discovered a portal to another dimension."
```
→ Adaptive, creative, high quality (GPT-3 uses this!)

**Beam Search (width=5):**
```
"Once upon a time, there was a beautiful princess who lived in a magnificent
castle. She was the most beautiful princess in all the land."
```
→ Generic, repetitive, "safe" output

---

## 🎛️ Combining Strategies

### The Optimal Approach

**Best practice:** Combine temperature + top-p

```python
def advanced_sampling(logits, temperature=0.8, top_p=0.9):
    """
    Combine temperature and top-p for best results.

    This is what production systems use!
    """
    # Step 1: Apply temperature
    logits = logits / temperature

    # Step 2: Apply top-p (nucleus sampling)
    token_id = top_p_sampling(logits, p=top_p, temperature=1.0)

    return token_id
```

**Common configurations:**

```python
# Conservative (factual, safe)
temperature = 0.3
top_p = 0.9
top_k = 20
# Use case: Customer support, factual Q&A

# Balanced (default for most uses)
temperature = 0.7
top_p = 0.9
top_k = 40
# Use case: General text generation, chatbots

# Creative (storytelling, brainstorming)
temperature = 1.0
top_p = 0.95
top_k = 100
# Use case: Creative writing, idea generation

# Wild (experimental, artistic)
temperature = 1.5
top_p = 0.95
top_k = 200
# Use case: Poetry, experimental fiction
```

---

## 🔧 Complete Generation Function

### Production-Ready Text Generator

```python
def generate_text(
    model,
    prompt,
    max_length=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    stop_tokens=None
):
    """
    Generate text with full control over sampling.

    Args:
        model: GPT model
        prompt: Initial text string
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (0.1 to 2.0)
        top_k: Top-k filtering (None to disable)
        top_p: Top-p (nucleus) filtering (None to disable)
        stop_tokens: List of token IDs that stop generation

    Returns:
        generated_text: Complete text (prompt + generated)
    """
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)

    # Default stop tokens
    if stop_tokens is None:
        stop_tokens = [tokenizer.eos_token_id]

    # Generation loop
    for _ in range(max_length):
        # Get model predictions
        # Input: (batch_size=1, current_length)
        token_tensor = np.array([tokens])
        logits, _ = model.forward(token_tensor)

        # Get logits for last position (next token prediction)
        next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

        # Apply sampling strategy
        next_token = sample_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # Append to sequence
        tokens.append(next_token)

        # Check for stop tokens
        if next_token in stop_tokens:
            break

    # Convert back to text
    generated_text = tokenizer.decode(tokens)

    return generated_text


def sample_token(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Sample next token with all filtering options.
    """
    # Step 1: Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Step 2: Apply top-k filtering
    if top_k is not None:
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = -np.inf

    # Step 3: Apply top-p filtering
    if top_p is not None:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]

        cumsum_probs = np.cumsum(softmax(sorted_logits))
        cutoff = np.searchsorted(cumsum_probs, top_p) + 1

        indices_to_remove = np.ones_like(logits, dtype=bool)
        indices_to_remove[sorted_indices[:cutoff]] = False
        logits[indices_to_remove] = -np.inf

    # Step 4: Sample from filtered distribution
    probs = softmax(logits)
    token_id = np.random.choice(len(probs), p=probs)

    return token_id
```

---

## 🎨 Controlling Generation Quality

### Tips and Tricks

**1. Repetition Penalty**
```python
def apply_repetition_penalty(logits, tokens, penalty=1.2):
    """
    Penalize tokens that already appeared.

    Prevents: "The cat sat on the mat. The cat sat on the mat."
    """
    for token in set(tokens):
        logits[token] /= penalty
    return logits
```

**2. Length Normalization**
```python
# For beam search: Prefer longer sequences
score = log_prob / (len(sequence) ** alpha)  # alpha = 0.6 typical
```

**3. Stop Sequences**
```python
# Stop when generating certain phrases
stop_sequences = ["\n\n", "THE END", "<|endoftext|>"]
```

**4. Prompt Engineering**
```python
# Better prompt = better generation
bad_prompt = "Story"
good_prompt = "Write a creative short story about a time traveler:\n\n"
```

---

## 📊 Evaluation Metrics

### How to Measure Generation Quality?

**1. Perplexity (lower is better)**
```python
# How surprised is the model by the generated text?
perplexity = np.exp(loss)
# Good: < 20, Okay: 20-50, Poor: > 50
```

**2. BLEU Score (for translation/summarization)**
```python
# How similar to reference text?
from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu(reference, generated)
# Range: 0 to 1 (1 = perfect match)
```

**3. Human Evaluation (gold standard)**
- Coherence: Does it make sense?
- Fluency: Is it grammatical?
- Relevance: Is it on-topic?
- Creativity: Is it interesting?

---

## ✅ Summary

### What You Learned

1. **Autoregressive generation** - One token at a time
2. **Greedy sampling** - Deterministic, boring
3. **Temperature** - Control randomness (0.1 to 2.0)
4. **Top-k** - Limit to k most likely tokens
5. **Top-p** - Adaptive nucleus sampling (GPT-3's method!)
6. **Beam search** - Explore multiple paths

### Key Insights

```python
# The magic formula:
temperature = 0.8  # Control randomness
top_p = 0.9        # Adaptive filtering
top_k = 40         # Safety net

# This combination gives:
# ✅ Creative text
# ✅ Coherent output
# ✅ Prevents nonsense
# ✅ Production-ready quality
```

### Best Practices

1. **Start with defaults** - temperature=0.8, top_p=0.9
2. **Adjust temperature** for creativity vs safety
3. **Use top-p** for adaptive behavior
4. **Add repetition penalty** to prevent loops
5. **Engineer prompts** for better output

---

## 🎯 Practice Exercises

### Exercise 1: Implement Temperature Sampling
Write temperature_sampling() from scratch and test with different temperatures.

### Exercise 2: Compare Strategies
Generate text with all 5 strategies and compare quality.

### Exercise 3: Find Optimal Parameters
Experiment to find best temperature/top_p for your use case.

---

## 📚 Next Steps

✅ **You can now generate text with GPT!**

**What's next:**
1. Train your GPT on custom data
2. Fine-tune for specific tasks
3. Build a complete chatbot
4. Deploy as an API

**Congratulations!** You've completed the core GPT implementation! 🎉

---

**You now have all the tools to build production-quality language models!** 🚀
