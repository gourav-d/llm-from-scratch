# Module 16 — Concepts: Modern LLM Architectures

---

# Lesson 1 — Auto-Regressive Recap + Limitations

## What You Already Know

In Module 05, you built a GPT-style model. Here is a quick recap of how it works:

```
Input:  "The cat sat"
Step 1: Predict → "on"       (sees "The cat sat")
Step 2: Predict → "the"      (sees "The cat sat on")
Step 3: Predict → "mat"      (sees "The cat sat on the")
Step 4: Predict → "."        (sees "The cat sat on the mat")
```

This is called **auto-regressive** generation.
- "Auto" = self
- "Regressive" = predicting from past values

**C# analogy:**
Auto-regressive is like building a `StringBuilder` — you append one character at a time. You cannot go back and change an earlier character once appended.

---

## The Math Behind It

Auto-regressive models learn the probability:

```
P(token_5 | token_1, token_2, token_3, token_4)
```

Which reads: "Given tokens 1-4, what is the probability of token 5?"

The full sentence probability is a chain:

```
P(sentence) = P(t1) × P(t2|t1) × P(t3|t1,t2) × P(t4|t1,t2,t3) × ...
```

**This is the Chain Rule of Probability.**

---

## What Auto-Regressive Does Well

- Simple and effective — GPT-2, GPT-3, LLaMA all use this
- Theoretically clean — chain rule gives exact probability
- Easy to train — just predict next token, compare with cross-entropy loss

---

## The Problems

### Problem 1: Sequential Generation = Slow

```
Generate 100 tokens = 100 separate forward passes through the model

Step 1 → forward pass → token 1
Step 2 → forward pass → token 2
...
Step 100 → forward pass → token 100
```

Each step depends on the previous. **You cannot parallelize this at inference time.**

KV Cache (Module 14) helps, but it does not eliminate the sequential bottleneck.

### Problem 2: No Revision

```
Generated: "The capital of France is Berlin"
                                        ^^^^^
                                        WRONG — but cannot go back!
```

Once "Berlin" is generated, the model cannot revise it. It must continue from the mistake.

Humans revise while writing. GPT cannot.

### Problem 3: Left-to-Right Bias

```
"The [BLANK] is very fast"
```

Auto-regressive models can only use LEFT context to fill the blank.
But "fast" on the right is a huge clue! Bidirectional models (BERT) use both sides — but BERT cannot generate text.

---

## Visual Summary

```
Auto-Regressive Generation Timeline:

t=1  [The] ???  ???  ???  ???
t=2  [The][cat] ???  ???  ???
t=3  [The][cat][sat] ???  ???
t=4  [The][cat][sat][on] ???
t=5  [The][cat][sat][on][mat]

→ Sequential. Each step blocks the next.
→ No going back. First token locked forever.
```

---

## Quiz 1

**Q1.** Why can't auto-regressive generation be parallelized at inference time?
- A) The model is too large
- B) Each token depends on all previous tokens
- C) The vocabulary is too big
- D) Attention is too slow

**Q2.** What does "auto-regressive" mean in simple terms?
- A) The model uses regression trees
- B) The model generates tokens one at a time using previous tokens
- C) The model is trained automatically
- D) The model uses recursive functions

**Q3.** A GPT model generates "Paris is the capital of Germany." What is the key limitation shown?
- A) The model is too slow
- B) The model cannot revise the wrong token once generated
- C) The model does not use attention
- D) The vocabulary is too small

**Answers:** Q1=B, Q2=B, Q3=B

---

---

# Lesson 2 — Diffusion Process for Text

## What Is a Diffusion Model?

You may have heard of **Stable Diffusion** or **DALL-E** — AI image generators. They use diffusion.

The core idea:
1. **Forward process:** Gradually add noise to data until it becomes pure noise
2. **Reverse process:** Train a neural network to remove noise step by step
3. **Generation:** Start with pure noise, run reverse process → get real data

```
Forward (destroy):  [Real Image] → [Noisy] → [More Noisy] → [Pure Noise]
Reverse (create):   [Pure Noise] → [Less Noisy] → [Almost Real] → [Real Image]

The neural network learns the REVERSE process.
```

---

## The Problem: Text Is Discrete

Images are continuous — pixel values are floats (0.0 to 1.0).
You can add Gaussian noise (random float) to a pixel easily.

Text is **discrete** — tokens are integers (0, 1, 2, ..., 50000).
You CANNOT add 0.3 to token 4821. That is not a valid token.

**C# analogy:**
Images are like `float[]` — you can add 0.3 anywhere.
Text tokens are like `int[]` or `enum` values — you cannot add 0.3 to an enum.

---

## Solution: Masking Instead of Gaussian Noise

For text, "adding noise" = **replacing tokens with [MASK]**

```
Forward process (add noise = mask tokens):
Step 0: "The cat sat on the mat"       ← original text
Step 1: "The cat [M] on the mat"       ← 1 token masked
Step 2: "The cat [M] on [M] mat"       ← 2 tokens masked
Step 3: "[M] cat [M] [M] [M] mat"      ← many tokens masked
Step 4: "[M] [M] [M] [M] [M] [M]"     ← fully masked (pure noise)

Reverse process (remove noise = predict masked tokens):
Step 4: "[M] [M] [M] [M] [M] [M]"     ← start from fully masked
Step 3: "[M] cat [M] [M] [M] mat"      ← unmask some tokens
Step 2: "The cat [M] on [M] mat"       ← unmask more
Step 1: "The cat sat on [M] mat"       ← almost done
Step 0: "The cat sat on the mat"       ← fully reconstructed
```

---

## How the Model Learns

At each reverse step, the model sees:
- Partially masked sentence
- The timestep `t` (how noisy is this?)

And predicts: **what were the masked tokens?**

```
Input:  "[M] cat [M] on the mat",  t=2
Output: probability distribution over vocabulary for each [M]
        → Position 0: P("The"=0.7, "A"=0.1, "One"=0.05, ...)
        → Position 2: P("sat"=0.8, "ran"=0.1, "lay"=0.05, ...)
```

This is called **denoising** — removing noise from a noisy input.

---

## Noise Schedule

How fast do we add noise?

```
Masking probability at each timestep:

t=0  (clean):    0% masked
t=1              20% masked
t=2              40% masked
t=3              60% masked
t=4              80% masked
t=T (pure noise): 100% masked
```

This schedule controls the difficulty of each training step.

---

## Key Insight: Parallel Generation

Unlike GPT, diffusion generates ALL tokens simultaneously at each step:

```
GPT (sequential):
  Step 1: generate token 1
  Step 2: generate token 2
  ...
  Step N: generate token N
  Total: N steps, each depends on previous

Diffusion (parallel):
  Step 1: denoise ALL tokens at once  (parallel!)
  Step 2: denoise ALL tokens at once  (parallel!)
  ...
  Step T: final ALL tokens at once    (parallel!)
  Total: T steps, but each step is FULLY PARALLEL
```

If T < N (T diffusion steps < N tokens), diffusion is faster.
Typical: T=10-50 steps, N=100-1000 tokens → 10-100x speedup.

---

## Visual Summary

```
Diffusion Process for Text:

FORWARD (training — destroy information):
"The quick brown fox"
       ↓ mask 25%
"The [M] brown fox"
       ↓ mask 50%
"[M] [M] brown fox"
       ↓ mask 75%
"[M] [M] [M] fox"
       ↓ mask 100%
"[M] [M] [M] [M]"

REVERSE (generation — create information):
"[M] [M] [M] [M]"          ← start from noise
       ↓ model predicts
"The [M] [M] fox"           ← confident tokens appear first
       ↓ model predicts
"The quick [M] fox"
       ↓ model predicts
"The quick brown fox"       ← done!
```

---

## C# Analogy for the Whole Process

```csharp
// Auto-regressive (GPT) = StringBuilder, append only
var sb = new StringBuilder();
sb.Append(Predict(sb.ToString())); // token 1
sb.Append(Predict(sb.ToString())); // token 2
// ...

// Diffusion = iterative refinement (like Photoshop blur/sharpen)
string[] tokens = new string[N]; // start: all [MASK]
for (int step = T; step >= 0; step--)
{
    tokens = Model.Denoise(tokens, step); // refine ALL tokens at once
}
// tokens is now your generated text
```

---

## Quiz 2

**Q1.** Why can't standard Gaussian noise be applied to text tokens?
- A) Text tokens are too large
- B) Text tokens are discrete integers — you can't add a float to an integer token
- C) The model is too slow
- D) Gaussian noise only works in 2D

**Q2.** In text diffusion, what does "adding noise" mean?
- A) Adding random words to the sentence
- B) Shuffling the word order
- C) Replacing tokens with [MASK]
- D) Increasing the vocabulary size

**Q3.** Why is diffusion potentially faster than GPT at inference?
- A) Diffusion uses a smaller model
- B) Diffusion generates all tokens in parallel at each step
- C) Diffusion skips the attention layer
- D) Diffusion uses fewer parameters

**Answers:** Q1=B, Q2=C, Q3=B

---

---

# Lesson 3 — Diffusion Language Models

## Overview

Now that you understand diffusion for text, meet the actual models that implement it.

Three major diffusion language models:

| Model | Full Name | Key Idea |
|-------|-----------|----------|
| MDLM | Masked Diffusion Language Model | Absorbing state masking |
| SEDD | Score Entropy Discrete Diffusion | Score-based approach |
| Plaid | Plaid Language Diffusion | Combines discrete diffusion + LM tricks |

---

## MDLM — Masked Diffusion Language Model

**Core idea:** Each token can either stay as-is or be "absorbed" into a [MASK] state.
Once masked, it stays masked until the reverse process unmasks it.

```
Forward process for one token:
  token "cat"
    → with prob β_t: becomes [MASK]   (absorbed)
    → with prob 1-β_t: stays "cat"    (unchanged)

This is an "absorbing state" — [MASK] is a one-way door.
Once absorbed, stays absorbed until model reverses it.
```

**Why absorbing?**
- Simple math — only two states: original or [MASK]
- Easy to compute probabilities at any timestep t without running all steps 1..t
- Can jump directly to any noise level: `q(x_t | x_0)` has closed form

**Training:**
At each step, randomly mask some fraction of tokens.
Model sees masked sentence + timestep t.
Model predicts original tokens for each [MASK].
Loss = cross-entropy between predicted tokens and true tokens.

---

## SEDD — Score Entropy Discrete Diffusion

**Core idea:** Instead of masking, use a general noise process over the full vocabulary.
A token can transition to ANY other token, not just [MASK].

```
Forward process:
  token "cat" (id=4821)
    → can become "dog" (id=1432) with some probability
    → can become "the" (id=263) with some probability
    → can become [MASK] (id=50256) with some probability
    → can stay "cat" with high probability at small t
```

**Score function:**
SEDD learns a "score" — the gradient of the log-probability.
This tells the model: "in which direction should I move each token to make the sentence more likely?"

**Why SEDD?**
- More expressive than masking — vocabulary-level transitions
- Theoretically grounded in score-based generative modeling
- Better for capturing complex token correlations

---

## Plaid — Practical Diffusion LM

**Core idea:** Take MDLM and make it production-ready.
Add tricks from standard LLM training (scale, data, efficiency).

Key contributions:
- Shows diffusion LMs can scale like GPT (more data + compute = better)
- Demonstrates competitive performance with auto-regressive models on benchmarks
- Uses efficient training with modern hardware (Flash Attention, bf16)

**Why Plaid matters:**
Prior work showed diffusion works for text.
Plaid shows diffusion works at **scale** — it is not just a research toy.

---

## Comparison Table

```
┌──────────────────┬──────────────┬───────────────┬────────────────┐
│ Property         │ MDLM         │ SEDD          │ Plaid          │
├──────────────────┼──────────────┼───────────────┼────────────────┤
│ Noise type       │ Masking only │ Full vocab    │ Masking        │
│ Math complexity  │ Simple       │ Complex       │ Simple + tricks│
│ Generation speed │ Fast         │ Moderate      │ Fast           │
│ Quality          │ Good         │ Better        │ Best (scaled)  │
│ Best for         │ Learning     │ Research      │ Production     │
└──────────────────┴──────────────┴───────────────┴────────────────┘
```

---

## How Generation Works at Inference

```
Step 1: Start with fully masked sentence
        "[M] [M] [M] [M] [M] [M]"

Step 2: Feed through model at timestep T
        Model predicts probability for each [M]
        Sample high-confidence tokens, keep rest as [M]
        "[M] cat [M] on [M] mat"

Step 3: Feed through model at timestep T-1
        More tokens get filled in
        "The cat [M] on [M] mat"

Step 4: Continue until no [M] remain
        "The cat sat on the mat"
```

Key decision at each step: **which tokens to commit vs keep as [M]**
- Greedy: always take argmax → fast but lower quality
- Sampling: sample from probability distribution → better diversity
- Confidence threshold: only commit tokens above 90% confidence → best quality

---

## Visual: How MDLM Fills in Tokens

```
                    ← Timestep t (noise level) ←
High noise                                    Low noise
t=T                                           t=0

[M][M][M][M][M][M]
        ↓ step 1
[M][cat][M][M][M][M]        ← high-confidence tokens appear first
        ↓ step 2
[The][cat][M][on][M][M]     ← more tokens commit
        ↓ step 3
[The][cat][sat][on][M][mat] ← almost done
        ↓ step 4
[The][cat][sat][on][the][mat] ← complete!
```

Notice: tokens appear in order of **confidence**, not position.
"cat" might appear before "the" because the model is more sure about it.

---

## C# Analogy

```csharp
// Diffusion generation = iterative form filling
// Like a form where you fill in the easy fields first, then the hard ones

var form = new string[6] { "[M]", "[M]", "[M]", "[M]", "[M]", "[M]" };

// Each round: fill in fields you're most confident about
for (int step = numSteps; step >= 0; step--)
{
    var predictions = model.Predict(form, step);
    foreach (var (pos, token, confidence) in predictions)
    {
        if (confidence > threshold || step == 0)
            form[pos] = token; // commit
        // else: leave as [M], try again next step
    }
}
```

---

## Quiz 3

**Q1.** In MDLM, what does "absorbing state" mean?
- A) The model absorbs GPU memory
- B) Once a token is masked, it stays masked until the model unmasks it
- C) Tokens absorb information from neighbors
- D) The [MASK] token absorbs gradients

**Q2.** What is the main difference between MDLM and SEDD?
- A) MDLM is faster
- B) SEDD allows tokens to transition to any vocabulary token, not just [MASK]
- C) MDLM uses transformers, SEDD uses RNNs
- D) SEDD requires more training data

**Q3.** In diffusion generation, why might "cat" appear before "the" in the output?
- A) "cat" has a lower token ID
- B) Tokens appear in order of model confidence, not position
- C) "cat" is a more common word
- D) The attention mask prioritizes nouns

**Answers:** Q1=B, Q2=B, Q3=B

---

---

# Lesson 4 — Diffusion Loss Function

## Why We Need a Special Loss

For GPT, the loss is simple:

```
Loss = CrossEntropy(predicted_token, actual_next_token)

At each position, compare predicted vs actual. Average over all positions.
```

For diffusion models, the loss is more complex because:
1. We are not predicting the NEXT token — we are predicting MASKED tokens
2. We need to account for the ENTIRE noising/denoising process
3. The model must work well at ALL timesteps t (not just one)

---

## ELBO — Evidence Lower Bound

**The goal:** Maximize the probability of the real data: `log P(x)`

**The problem:** `log P(x)` is intractable to compute directly.
(It requires integrating over all possible noise trajectories.)

**The solution:** Maximize a lower bound instead — the **ELBO**.

```
ELBO ≤ log P(x)

Maximizing ELBO → indirectly maximizes log P(x)
```

**C# analogy:**
You want to maximize a function `f(x)` but `f(x)` is too expensive to compute.
So you maximize `g(x)` where `g(x) ≤ f(x)` always, and `g(x)` is cheap.
When `g(x)` is high, `f(x)` is high too.

This is like getting a lower bound on your performance metric and optimizing that.

---

## Breaking Down the ELBO

The ELBO decomposes into two terms:

```
ELBO = Reconstruction Term - KL Divergence Term

ELBO = E[log P(x_0 | x_1)]           ← how well does step 1 reconstruct original?
       -
       KL(q(x_T | x_0) || P(x_T))    ← how close is fully-noised to our prior?
       -
       Σ KL(q(x_{t-1} | x_t, x_0) || P(x_{t-1} | x_t))  ← denoising at each step
```

**In plain English:**

| Term | Meaning |
|------|---------|
| Reconstruction | Final denoising step: can model recover original text from step-1 noise? |
| Prior KL | Is fully-noised text close to what we assume (uniform distribution)? |
| Denoising KL | At each step t, how well does model denoise one step? |

---

## VLB — Variational Lower Bound

**VLB is the same as ELBO.** Different papers use different names.

- **ELBO** = Evidence Lower BOund (used in variational autoencoders, VAEs)
- **VLB** = Variational Lower Bound (used in diffusion papers)
- **NELBO** = Negative ELBO (used as loss, since we minimize losses but maximize ELBO)

```
Training loss = -ELBO = NELBO
Minimizing NELBO = Maximizing ELBO = Approximating maximum likelihood
```

---

## Denoising Score Matching

**Score = gradient of log-probability of data**

```
Score(x) = ∇_x log P(x)

This gradient points in the direction that makes x more likely.
Think of it as: "which direction should I nudge x to make it more realistic?"
```

**Score matching** trains a neural network `s_θ(x)` to predict this score.

For diffusion, we use **denoising score matching**:

```
Loss = E[ ||s_θ(x_noisy, t) - score_of_true_data||² ]

Train model to predict the score (direction toward real data) from noisy data.
```

**C# analogy:**
Imagine you have a noisy GPS signal. The "score" is the direction to the true destination.
Score matching trains a model to say: "from this noisy position, walk THIS direction to get to real data."

---

## Simplified Loss for MDLM

For masked diffusion (MDLM), the loss simplifies greatly:

```python
# At timestep t, some tokens are masked
# Model predicts probability for each masked token

loss = 0
for each masked position i:
    predicted_probs = model.predict(masked_sequence, t)[i]  # shape: [vocab_size]
    true_token = original_tokens[i]
    loss += cross_entropy(predicted_probs, true_token)

# Weight by timestep (earlier = harder = higher weight)
loss = loss * weight(t)
```

This is just **weighted cross-entropy** — similar to GPT's loss, but:
- Only computed on MASKED positions (not all positions)
- Weighted by timestep t

---

## Full Training Objective

```
L_diffusion = E_{t, x_0, x_t} [ Σ_{masked positions} -log P_θ(x_0[i] | x_t, t) ]

Where:
  x_0 = original text
  x_t = noised text at timestep t
  P_θ = model's predicted probability
  i   = masked position indices
```

In simpler terms:
- Sample a real sentence x_0
- Sample a random timestep t
- Mask tokens according to schedule (get x_t)
- Model predicts original tokens for masked positions
- Loss = negative log probability of correct tokens

---

## Comparison: GPT vs Diffusion Loss

```
GPT Loss:
  For each position i:
    Predict token_i given token_0..token_{i-1}
    Loss += CrossEntropy(predicted, token_i)
  Average over all positions

Diffusion Loss:
  Sample random noise level t
  Mask tokens according to t
  For each MASKED position i:
    Predict original_token_i given all visible tokens + t
    Loss += CrossEntropy(predicted, original_token_i)
  Weight by t (harder timesteps get higher weight)
```

Key difference: diffusion loss averages over **random noise levels**, not positions.

---

## Visual: Loss Computation

```
Original:  "The  cat  sat  on  the  mat"
            [0]  [1]  [2]  [3]  [4]  [5]

At t=2 (40% masked):
Masked:    "The  [M]  sat  [M]  the  mat"
                  ^         ^
                  Masked    Masked

Model predicts:
  Position 1: P("cat"=0.85, "dog"=0.05, ...) → cross-entropy with true "cat"
  Position 3: P("on"=0.79, "in"=0.08, ...)   → cross-entropy with true "on"

Loss = -log(0.85) + (-log(0.79))
     = 0.163 + 0.236
     = 0.399

Multiply by weight(t=2) → final loss for this sample
```

---

## Quiz 4

**Q1.** Why can't we directly maximize `log P(x)` in diffusion models?
- A) The computation is too slow for GPUs
- B) It requires integrating over all possible noise trajectories, which is intractable
- C) The vocabulary is too large
- D) The gradient is always zero

**Q2.** What is the relationship between ELBO and VLB?
- A) ELBO is always larger than VLB
- B) They are different names for the same quantity
- C) VLB applies to images, ELBO applies to text
- D) ELBO maximizes while VLB minimizes

**Q3.** In MDLM, the training loss is computed only on which positions?
- A) All positions equally
- B) Only the last token (like GPT)
- C) Only masked positions
- D) Only the first and last tokens

**Q4.** What does the "score" represent in score matching?
- A) A metric for model accuracy
- B) The gradient of log-probability — the direction to move x to make it more likely
- C) The model's confidence on each token
- D) The loss value at each training step

**Answers:** Q1=B, Q2=B, Q3=C, Q4=B

---

---

# Lesson 5 — Multi-Token Prediction (MTP)

## Background: One Token at a Time

Standard GPT-style training:

```
Input:   "The cat sat on the mat"
Target:  "cat sat on the mat ."

At each position, predict the NEXT single token.
```

This means at position 0, the model only knows:
"The next word after 'The' should be 'cat'"

It never explicitly learns: "after 'The cat', 'sat' comes" OR "after 'The cat sat', 'on' comes"
(These are learned only indirectly, at those positions.)

---

## What Is Multi-Token Prediction?

**MTP trains the model to predict the next N tokens simultaneously.**

Introduced by Meta AI in the paper:
**"Better & Faster Large Language Models via Multi-Token Prediction"** (2024)

```
Standard (predict 1):
  Position 0: predict token 1 only
  Position 1: predict token 2 only
  ...

MTP (predict 4):
  Position 0: predict tokens 1, 2, 3, 4 simultaneously
  Position 1: predict tokens 2, 3, 4, 5 simultaneously
  ...
```

---

## Architecture: How MTP Works

Standard GPT has one output head:

```
Transformer → [one output head] → P(next_token | context)
```

MTP adds multiple output heads:

```
Transformer → [head 1] → P(token at t+1 | context)
           → [head 2] → P(token at t+2 | context)
           → [head 3] → P(token at t+3 | context)
           → [head 4] → P(token at t+4 | context)
```

Each head is a small linear layer + softmax on top of the shared transformer.

```
┌─────────────────────────────┐
│    Shared Transformer       │ ← same backbone, trained once
│    (all layers)             │
└─────────────┬───────────────┘
              │
     ┌────────┼────────┐────────┐
     ↓        ↓        ↓        ↓
  [Head 1] [Head 2] [Head 3] [Head 4]
  t+1 pred  t+2 pred  t+3 pred  t+4 pred
```

**C# analogy:**
Like having a single `IAnalyzer` interface implemented by one concrete class,
but exposing 4 different methods that each return a different prediction horizon.
All methods share the same internal computation, but have different output layers.

---

## Training

Loss = sum of cross-entropy losses from all heads:

```python
# Standard GPT loss
loss = cross_entropy(head1_output, next_token)

# MTP loss (N=4)
loss = cross_entropy(head1_output, token_t1)   # next 1
     + cross_entropy(head2_output, token_t2)   # next 2
     + cross_entropy(head3_output, token_t3)   # next 3
     + cross_entropy(head4_output, token_t4)   # next 4

# Optionally weighted:
loss = w1 * ce(head1, t1) + w2 * ce(head2, t2) + ...
```

The total loss is larger, but the gradients provide **richer signal** to the shared transformer.

---

## Why Does This Help?

### Benefit 1: Richer Training Signal

Standard training: each position provides 1 gradient signal.
MTP: each position provides N gradient signals.

More signal = model learns better representations faster.

```
Standard:  "The" → loss from predicting "cat"
MTP N=4:   "The" → loss from predicting "cat" + "sat" + "on" + "the"

The model must learn richer features to satisfy all 4 heads simultaneously.
```

### Benefit 2: Better Long-Range Planning

With standard training, the model optimizes greedily — best next token.
With MTP, the model must choose tokens that are good for the next 4 steps.

```
Greedy (standard):
  "The quick" → "brown" (highest probability next token)
  But maybe "brown fox jumps" leads to a better sentence than "brown cat runs"?

MTP:
  "The quick" → predicts "brown", "fox", "jumps", "over" together
  Must choose "brown" such that "fox", "jumps", "over" are also likely
  → Better long-range coherence
```

### Benefit 3: Faster Inference (Speculative Decoding)

Extra heads can be used at inference time for **speculative decoding**:

```
Standard inference: generate 1 token per forward pass

Speculative decoding with MTP:
  Step 1: Run forward pass → head 1 gives t+1, head 4 gives t+4
  Step 2: Verify tokens t+1..t+4 with the main head (much cheaper)
  Step 3: Accept verified tokens, reject rest
  
Result: potentially 2-3x fewer forward passes for same output
```

---

## Meta's Results

From the MTP paper:

| Metric | Standard | MTP (N=4) | Improvement |
|--------|----------|-----------|-------------|
| HumanEval (code) | Baseline | +12% | Significant |
| GSM8K (math) | Baseline | +8% | Significant |
| Training efficiency | Baseline | Same FLOP, better result | Better |
| Inference speed | 1x | Up to 3x with speculative decode | Faster |

MTP particularly helps on tasks requiring **multi-step reasoning** (code, math).
Makes sense — these tasks require planning ahead.

---

## MTP vs Diffusion: When to Use Which?

```
┌────────────────────┬──────────────────────┬───────────────────────┐
│ Property           │ MTP                  │ Diffusion LM          │
├────────────────────┼──────────────────────┼───────────────────────┤
│ Architecture change│ Small (add heads)    │ Large (new training)  │
│ Training cost      │ ~same                │ Higher                │
│ Inference speed    │ 2-3x with spec-decode│ T steps (parallel)    │
│ Quality gain       │ +5-15% on code/math  │ Competitive with GPT  │
│ Revision ability   │ No (still AR)        │ Yes (iterative)       │
│ Adoption           │ Easy (drop-in)       │ New paradigm          │
│ Best for           │ Reasoning tasks      │ Any generation        │
└────────────────────┴──────────────────────┴───────────────────────┘
```

**Key insight:** MTP is an **enhancement** to auto-regressive models.
Diffusion is a **replacement** for auto-regressive models.

Both aim to generate better text faster. Different approaches.

---

## Summary of the Three Paradigms

```
1. Auto-Regressive (GPT, LLaMA):
   → Generate tokens: left to right, one at a time
   → Simple, well-understood, dominant today

2. Diffusion LM (MDLM, SEDD, Plaid):
   → Start with all tokens masked
   → Iteratively unmask in parallel (T steps)
   → Can revise, bidirectional context, parallel

3. Multi-Token Prediction (MTP):
   → Still auto-regressive base
   → Predict N future tokens per step (during training)
   → Better representations, faster inference via spec-decode
   → Drop-in improvement on top of existing AR models
```

---

## Visual: Three Paradigms Side-by-Side

```
AUTO-REGRESSIVE:
  Input: [The] [cat] [sat]
  Output at each step: one token
  
  t=1: The  ___  ___  ___  ___
  t=2: The  cat  ___  ___  ___
  t=3: The  cat  sat  ___  ___
  t=4: The  cat  sat  on   ___
  t=5: The  cat  sat  on   mat


DIFFUSION:
  t=T: [M]  [M]  [M]  [M]  [M]   ← all masked
  t=3: The  [M]  [M]  [M]  mat   ← confident tokens appear
  t=2: The  cat  [M]  on   mat   ← more fill in
  t=1: The  cat  sat  on   mat   ← complete
  
  Note: ALL positions updated SIMULTANEOUSLY at each step


MTP (N=3, training time):
  Input: The  cat  sat  on   mat
  At position "The":
    Head 1 → predicts "cat"         (t+1)
    Head 2 → predicts "sat"         (t+2)
    Head 3 → predicts "on"          (t+3)
  All three losses backprop through the SAME transformer
```

---

## Quiz 5

**Q1.** MTP adds extra output heads to the transformer. What do these heads share?
- A) Nothing — each head is an independent model
- B) The same transformer backbone (all layers)
- C) Only the embedding layer
- D) The attention weights

**Q2.** Why does MTP improve performance on coding and math tasks specifically?
- A) Code and math use fewer tokens
- B) These tasks require planning ahead — MTP trains the model to predict multiple future steps
- C) MTP uses a different tokenizer for code
- D) Math tasks have smaller vocabularies

**Q3.** What is "speculative decoding" in the context of MTP?
- A) The model speculates about user intent
- B) Using extra heads to predict multiple tokens, then verifying them cheaply to accelerate inference
- C) Decoding with a probability threshold
- D) Running two models in parallel and taking the best output

**Q4.** What is the key difference between MTP and Diffusion LMs?
- A) MTP uses masking, diffusion does not
- B) MTP enhances auto-regressive generation; diffusion replaces it entirely
- C) MTP requires more parameters
- D) Diffusion only works for images

**Answers:** Q1=B, Q2=B, Q3=B, Q4=B

---

---

# Module 16 — Full Summary

## What You Learned

### Lesson 1: Auto-Regressive Limits
- GPT generates left-to-right, one token at a time
- Cannot revise earlier tokens
- Sequential generation = slow (cannot parallelize)

### Lesson 2: Diffusion for Text
- "Adding noise" = masking tokens (not Gaussian noise — text is discrete)
- Forward process: progressively mask more tokens
- Reverse process: model learns to unmask tokens
- Parallel generation: all tokens updated simultaneously at each step

### Lesson 3: Diffusion Language Models
- **MDLM**: absorbing state masking, simple closed-form, good baseline
- **SEDD**: vocabulary-level transitions, score-based, theoretically rich
- **Plaid**: production-scale diffusion, competitive with GPT

### Lesson 4: Diffusion Loss
- Cannot directly maximize `log P(x)` — intractable
- ELBO/VLB: optimize a lower bound instead
- Denoising score matching: predict direction toward real data from noisy data
- MDLM simplifies to weighted cross-entropy on masked positions only

### Lesson 5: Multi-Token Prediction
- Train with N output heads, each predicting t+1, t+2, ..., t+N
- Richer gradient signal → better representations
- Forces model to plan ahead → better coherence
- Enables speculative decoding → 2-3x inference speedup

---

## Architecture Decision Tree

```
Need to improve your LLM?

→ Drop-in improvement, keep AR paradigm?
   → Use MTP (add extra heads, same architecture)

→ Want parallel generation, revision ability?
   → Use Diffusion LM (MDLM to start)

→ Best quality at scale, production use?
   → AR + MTP (Meta LLaMA approach)
   → Plaid (if willing to switch paradigm)
```

---

## Key Equations to Remember

```
1. Chain rule (auto-regressive):
   P(x) = ∏ P(x_t | x_1..x_{t-1})

2. ELBO (diffusion):
   ELBO = E[log P(x_0|x_1)] - KL(q(x_T|x_0) || P(x_T)) - Σ KL terms

3. MDLM loss:
   L = E_{t,x_0,x_t} [ -Σ_{masked i} log P_θ(x_0[i] | x_t, t) ]

4. MTP loss:
   L = Σ_{k=1}^{N} w_k × CrossEntropy(head_k(x), x_{t+k})
```

---

## Next Module

**Module 17 — LLM Evaluation & Benchmarks**
- How to measure if your LLM is good
- Perplexity, MMLU, GSM8K, HumanEval, BLEU, ROUGE
- Running evaluation harness, reading leaderboards

After M17: **Capstone Project** — Chat with Codebase (offline RAG app)
