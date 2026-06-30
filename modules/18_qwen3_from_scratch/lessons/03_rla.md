# Lesson 3: Recurrent Linear Attention (RLA)

## What Problem Does This Solve?

Standard attention — the kind you built in Module 04 — has a fundamental scaling problem.

When generating a token at position N, the model computes attention over ALL N past tokens:

```
Attention scores shape: [N, N]

For N = 1,000 tokens:    1,000 × 1,000    = 1 million values
For N = 10,000 tokens:  10,000 × 10,000   = 100 million values
For N = 100,000 tokens: 100,000 × 100,000 = 10 billion values
```

This is the **O(N²) problem**. Both time and memory grow with the square of sequence length.

```
+=========================================================================+
|  THE O(N^2) WALL                                                        |
+=========================================================================+
|                                                                         |
|  Standard softmax attention:                                            |
|    time:   O(N^2 * d)   — compute N×N score matrix                     |
|    memory: O(N^2)       — store N×N score matrix                        |
|                                                                         |
|  At 128K tokens (Qwen3.5 context limit):                                |
|    score matrix = 128,000 × 128,000 = 16 billion values                |
|    At 2 bytes each = 32 GB — just for ONE attention layer!              |
|                                                                         |
|  Flash Attention (Module 15) fixes the MEMORY problem by tiling.       |
|  But the TIME complexity (N^2 operations) remains.                     |
|                                                                         |
|  For very long contexts, you need something fundamentally different.    |
+=========================================================================+
```

**Recurrent Linear Attention (RLA)** reduces both time AND memory to O(N).
The key insight: replace the N×N score matrix with a fixed-size hidden state.

---

## The Core Idea: From Matrix to Hidden State

Standard attention computes:

```
Output_t = sum over all past tokens j of:
    softmax( Q_t · K_j / sqrt(d) ) * V_j
```

This requires comparing Q_t against every past K_j — hence O(N) per token = O(N²) total.

Linear attention replaces the softmax comparison with a trick:

```
+------------------------------------------------------------------+
|  THE LINEAR ATTENTION TRICK                                     |
+------------------------------------------------------------------+
|                                                                  |
|  Standard:                                                       |
|    Output_t = (sum_j softmax(Q_t · K_j) * V_j) / Z             |
|    Requires: look at all j for each t  →  O(N^2)               |
|                                                                  |
|  Linear (replace softmax with kernel phi):                      |
|    Output_t = phi(Q_t) · (sum_j phi(K_j) ⊗ V_j) / Z           |
|                         [------- this is S_t -------]          |
|                                                                  |
|  Key observation:                                                |
|    S_t = sum of phi(K_j) outer-product V_j  for all j <= t     |
|    S_t = S_{t-1} + phi(K_t) ⊗ V_t                              |
|                                                                  |
|  So we can ACCUMULATE S_t one token at a time!                  |
|    S_0 = zeros                                                   |
|    S_1 = S_0 + phi(K_1) ⊗ V_1                                  |
|    S_2 = S_1 + phi(K_2) ⊗ V_2                                  |
|    ...                                                           |
|    S_t = S_{t-1} + phi(K_t) ⊗ V_t                              |
|                                                                  |
|    Output_t = phi(Q_t) · S_t                                    |
|                                                                  |
+------------------------------------------------------------------+
```

`S_t` is the **hidden state** — a fixed-size matrix that summarizes all past tokens.
No N×N matrix ever created. O(1) memory per token. O(N) total.

---

## What Is phi (the Kernel Function)?

The kernel function `phi` replaces softmax. It maps a vector into a feature space
where the dot product approximates the original attention similarity.

```
+------------------------------------------------------------------+
|  KERNEL FUNCTION phi                                            |
+------------------------------------------------------------------+
|                                                                  |
|  Goal: phi(q) · phi(k) ≈ exp(q · k / sqrt(d))  (approximate    |
|                            the softmax numerator)               |
|                                                                  |
|  Simple option (used in many papers):                           |
|    phi(x) = ELU(x) + 1                                         |
|    (ELU = exponential linear unit; the +1 keeps values >= 0)   |
|                                                                  |
|  Why non-negative?                                              |
|    The hidden state S accumulates outer products phi(K) ⊗ V.   |
|    If phi can be negative, additions can cancel each other out  |
|    and information is lost. Non-negative phi avoids this.       |
|                                                                  |
|  Qwen3.5 uses a more sophisticated kernel called HGRN2          |
|  (Hierarchical Gated Recurrent Network), but the principle      |
|  is the same: fixed-size hidden state updated per token.        |
+------------------------------------------------------------------+
```

---

## Recurrent Form: Token by Token

This is why it is called "recurrent" — it processes tokens one at a time
like an RNN, maintaining a hidden state between steps.

```
RECURRENT LINEAR ATTENTION — INFERENCE LOOP

  S = zeros(d_k, d_v)     # hidden state, shape: [key_dim, value_dim]
  z = zeros(d_k)          # normalizer, shape: [key_dim]

  For each token t in sequence:

    q = phi(Q_t)           # feature vector for query
    k = phi(K_t)           # feature vector for key
    v = V_t                # value (no phi needed)

    S = S + outer(k, v)    # update hidden state: add K⊗V contribution
    z = z + k              # update normalizer

    Output_t = (q @ S) / (q @ z + epsilon)   # query the hidden state


VISUAL:

  token 1:  S = 0 + k1⊗v1 = k1⊗v1
  token 2:  S = k1⊗v1 + k2⊗v2
  token 3:  S = k1⊗v1 + k2⊗v2 + k3⊗v3
  ...
  token N:  S = sum of all k_j⊗v_j for j=1..N

  At each step, we query S with q_t and get the output.
  S is FIXED SIZE regardless of how many tokens have passed.
```

---

## Memory Comparison: Standard vs Linear Attention

```
+------------------------------------------------------------------+
|  MEMORY AT INFERENCE (generating token number 10,000)          |
+------------------------------------------------------------------+
|                                                                  |
|  Standard softmax attention:                                     |
|    Must store K and V for all 10,000 past tokens in KV cache   |
|    KV cache grows with sequence length: O(N)                    |
|    Plus computing N scores per step: O(N) compute per token     |
|    Total over whole sequence: O(N^2) compute                    |
|                                                                  |
|  Linear (recurrent) attention:                                   |
|    Hidden state S: fixed size [d_k, d_v]                        |
|    Does NOT grow with sequence length                            |
|    One matrix multiply per token: O(d_k * d_v) per step        |
|    Total over whole sequence: O(N * d_k * d_v) = O(N)          |
|                                                                  |
|  Example (d_k = d_v = 128):                                      |
|    S size = 128 × 128 = 16,384 values — fixed forever           |
|    At 128K tokens: standard = 128,000 KV vectors stored         |
|                    linear   = still just 16,384 values          |
+------------------------------------------------------------------+
```

---

## The Quality Tradeoff

Linear attention is faster and uses less memory, but there is a cost:

```
+------------------------------------------------------------------+
|  WHY LINEAR ATTENTION LOSES SOME QUALITY                       |
+------------------------------------------------------------------+
|                                                                  |
|  Standard softmax attention:                                     |
|    Can "focus" sharply on ONE token (softmax = sharp peak)      |
|    Each output depends on the EXACT past it needs               |
|    Perfect recall of specific tokens                            |
|                                                                  |
|  Linear attention:                                               |
|    Hidden state S is a COMPRESSION of all past tokens           |
|    Like a running average — information can blur together        |
|    Hard to recall one specific token from long ago              |
|                                                                  |
|  Analogy:                                                        |
|    Standard attention = random access memory (read any address) |
|    Linear attention   = RAM with lossy compression              |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Qwen3.5 Hybrid: Best of Both Worlds

Qwen3.5 does not choose one or the other. It uses **both** in a hybrid design:

```
QWEN3.5 HYBRID ATTENTION STACK (per block):

  Layer 1:  Recurrent Linear Attention  (RLA)  ← O(N), fast, memory efficient
  Layer 2:  Recurrent Linear Attention  (RLA)
  Layer 3:  Recurrent Linear Attention  (RLA)
  Layer 4:  Full Softmax Attention (GQA)        ← precise recall, used sparingly
  Layer 5:  Recurrent Linear Attention  (RLA)
  Layer 6:  Recurrent Linear Attention  (RLA)
  Layer 7:  Recurrent Linear Attention  (RLA)
  Layer 8:  Full Softmax Attention (GQA)
  ...

  Pattern: every 4th layer uses full softmax attention.
  The rest use RLA.

WHY THIS WORKS:
  The full GQA layers restore precision where it matters.
  The RLA layers handle the bulk of computation efficiently.
  Together: near-full-attention quality at a fraction of the cost.
```

---

## Visual: Parallel vs Recurrent Mode

Linear attention has two computation modes:

```
PARALLEL MODE (training — process all tokens at once):

  All tokens available simultaneously.
  Compute S = sum of all K⊗V in one batched operation.
  Efficient on GPU — high parallelism.

  token1  token2  token3  ... tokenN
    |       |       |             |
    v       v       v             v
  K1⊗V1  K2⊗V2  K3⊗V3  ...  KN⊗VN
    \       \       \    ...    /
     \_______\_______\________/
                    S (sum)
    /       /       /    ...    \
  Q1·S    Q2·S    Q3·S  ...  QN·S


RECURRENT MODE (inference — one token at a time):

  S = 0
  Token arrives → update S → produce output → next token

  token1 → S1 = K1⊗V1    → output1 = Q1·S1
  token2 → S2 = S1+K2⊗V2  → output2 = Q2·S2
  token3 → S3 = S2+K3⊗V3  → output3 = Q3·S3
  ...

  KEY: S has constant size. Inference uses O(1) memory per step.
```

Same math — two different execution strategies depending on context.

---

## C# Analogy: Streaming Aggregation

```csharp
// Standard softmax attention is like SQL with a full table scan:
//
//   SELECT weighted_average(V)
//   FROM past_tokens
//   WHERE similarity(Q_current, K_j) is high
//   ORDER BY similarity DESC;
//
//   At token 10,000: you scan all 10,000 rows — O(N) per query.
//   Total over all tokens: O(N^2).
//
//
// Linear attention is like a running aggregate (streaming LINQ):
//
//   // Hidden state S = accumulated outer product
//   var hiddenState = Matrix.Zeros(keyDim, valueDim);
//
//   foreach (var token in tokenStream)          // one token at a time
//   {
//       var k = Phi(token.Key);                 // kernel feature
//       var v = token.Value;
//
//       hiddenState += OuterProduct(k, v);      // update hidden state — O(d^2)
//
//       var q = Phi(token.Query);
//       var output = q * hiddenState;           // query hidden state — O(d^2)
//       yield return output;
//   }
//
//   // hiddenState is always the same size: [keyDim, valueDim]
//   // It does NOT grow with the number of tokens.
//   // Like IEnumerable<T> with yield return — true streaming, O(1) memory.
//
//
// The tradeoff:
//   SQL full scan = perfect recall, expensive
//   Running aggregate = lossy compression, cheap
//   Qwen3.5 = mix both, use full scan sparingly (every 4 layers)
```

---

## Quiz Questions

**Q1**: What is the time complexity of standard softmax attention over a sequence of N tokens?
        a) O(N)
        b) O(N log N)
        c) O(N^2)
        d) O(N^3)

**Q2**: In recurrent linear attention, the hidden state S has what size?
        a) Grows with sequence length N
        b) Fixed size [d_k, d_v] — does not grow with N
        c) Fixed size N × N regardless of token count
        d) Size equals the number of attention heads

**Q3**: Why does Qwen3.5 use a HYBRID of RLA and full softmax attention?
        a) RLA is used at training time, full attention at inference time
        b) Full attention handles short sequences, RLA handles long ones
        c) RLA layers are fast and memory-efficient; occasional full attention layers restore precision
        d) They produce identical outputs, so the choice is arbitrary

**Q4**: What does the kernel function phi replace in linear attention?
        a) The value (V) matrix
        b) The softmax normalization in the attention score
        c) The query (Q) projection weight matrix
        d) The positional encoding

*(Answers: Q1=c, Q2=b, Q3=c, Q4=b)*

---

## Key Takeaways

1. Standard attention is O(N²) — the attention matrix grows with sequence length squared
2. Linear attention replaces softmax with a kernel phi, enabling O(N) computation
3. The hidden state S = accumulated sum of phi(K) outer-product V — fixed size regardless of N
4. Recurrent form: S_t = S_{t-1} + phi(K_t) ⊗ V_t — one update per token at inference
5. Parallel form: compute all K⊗V sums at once — used during training for GPU efficiency
6. Quality tradeoff: linear attention cannot "focus" as sharply as softmax attention
7. Qwen3.5 hybrid solution: RLA every layer except every 4th, which uses full GQA

---

*Next: Lesson 4 — KV Cache Management (storing, reusing, and evicting past key-value pairs)*
