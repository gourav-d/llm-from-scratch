# Lesson 2: Group-Query Attention (GQA)

## What Problem Does This Solve?

In Module 04, you learned Multi-Head Attention (MHA).
Each attention head has its own Query, Key, and Value matrices.

```
Standard Multi-Head Attention with 8 heads:

  Head 1:  Q1, K1, V1
  Head 2:  Q2, K2, V2
  Head 3:  Q3, K3, V3
  Head 4:  Q4, K4, V4
  Head 5:  Q5, K5, V5
  Head 6:  Q6, K6, V6
  Head 7:  Q7, K7, V7
  Head 8:  Q8, K8, V8
```

During training, this is fine. But during **inference** (generating text token by token),
something expensive happens: the **KV Cache**.

---

## The KV Cache Problem

When generating token number 100, the model needs to attend to all 99 previous tokens.
It must recompute K and V for every past token — OR store them in memory.

Storing them is called the **KV Cache**. (You learned this in Module 14, Lesson 6.)

```
+=========================================================================+
|  KV CACHE MEMORY COST                                                   |
+=========================================================================+
|                                                                         |
|  Formula:                                                               |
|  KV cache size = 2 × layers × n_kv_heads × seq_len × head_dim × bytes  |
|                                                                         |
|  Example: Qwen3.5-7B (32 layers, 8 KV heads, head_dim=128, bf16=2B)    |
|  At 32K sequence length:                                                |
|    = 2 × 32 × 8 × 32768 × 128 × 2 bytes                                |
|    = 2 × 32 × 8 × 32768 × 256                                          |
|    = ~4.3 GB just for the KV cache                                      |
|                                                                         |
|  With STANDARD MHA (32 KV heads instead of 8):                         |
|    = 2 × 32 × 32 × 32768 × 128 × 2 bytes                               |
|    = ~17 GB just for the KV cache                                       |
|                                                                         |
|  That is 4× MORE memory for the same model, same sequence.              |
|  On a 24 GB GPU, 17 GB for KV cache leaves almost nothing for weights.  |
|                                                                         |
+=========================================================================+
```

This is the problem GQA solves: **reduce KV cache memory without hurting quality**.

---

## Three Attention Variants

There are three ways to design the relationship between Q heads and KV heads:

```
+=========================================================================+
|  THREE ATTENTION DESIGNS                                                |
+=========================================================================+
|                                                                         |
|  1. MHA — Multi-Head Attention (standard)                               |
|     Every Q head has its OWN K head and V head.                         |
|     n_q_heads = n_kv_heads = 8 (or 32, or 64)                          |
|                                                                         |
|     Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8                                     |
|     |   |   |   |   |   |   |   |                                      |
|     K1  K2  K3  K4  K5  K6  K7  K8                                     |
|     V1  V2  V3  V4  V5  V6  V7  V8                                     |
|                                                                         |
|  2. MQA — Multi-Query Attention (extreme)                               |
|     ALL Q heads share ONE single K head and V head.                     |
|     n_q_heads = 8, n_kv_heads = 1                                       |
|                                                                         |
|     Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8                                     |
|      \  |  / \ /  \ / \ / \ /  /                                       |
|             K1                                                          |
|             V1                                                          |
|                                                                         |
|  3. GQA — Group-Query Attention (balanced)                              |
|     Q heads are divided into GROUPS. Each group shares one KV head.    |
|     n_q_heads = 8, n_kv_heads = 2, group_size = 4                      |
|                                                                         |
|     Group 1:          Group 2:                                          |
|     Q1  Q2  Q3  Q4    Q5  Q6  Q7  Q8                                   |
|      \  |  / \  /      \  |  / \  /                                    |
|          K1                  K2                                         |
|          V1                  V2                                         |
|                                                                         |
+=========================================================================+
```

**MHA**: best quality, worst memory  
**MQA**: best memory, quality can drop  
**GQA**: balanced — nearly same quality as MHA, much less memory than MHA

---

## How GQA Works in Detail

The core operation: each query head attends to the K/V of its group.

```
SETUP:
  n_q_heads   = 32    (number of query heads)
  n_kv_heads  = 8     (number of KV heads)
  group_size  = 32 / 8 = 4   (each KV head serves 4 Q heads)

FORWARD PASS:
  Q shape: [batch, seq_len, 32, head_dim]
  K shape: [batch, seq_len,  8, head_dim]
  V shape: [batch, seq_len,  8, head_dim]

FOR EACH KV HEAD k (k = 0 to 7):
  Serving Q heads: k*4, k*4+1, k*4+2, k*4+3

  For each of these 4 Q heads q:
    scores = Q[..., q, :] @ K[..., k, :].T / sqrt(head_dim)
    weights = softmax(scores)
    output[q] = weights @ V[..., k, :]

RESULT:
  Each of the 32 Q heads produces an output.
  But K and V had only 8 copies — 4x less KV memory.
```

The key implementation trick is **repeat_kv**: expand K and V from 8 heads to 32 heads
by repeating each 4 times, then run standard attention.

```python
# Pseudocode — expand KV heads to match Q heads
K_expanded = K.repeat_interleave(group_size, dim=2)  # [batch, seq, 32, head_dim]
V_expanded = V.repeat_interleave(group_size, dim=2)  # [batch, seq, 32, head_dim]

# Now standard attention works as usual
scores = Q @ K_expanded.transpose(-2, -1) / sqrt(head_dim)
output = softmax(scores) @ V_expanded
```

This repeat happens DURING COMPUTATION, not in the KV cache.
The cache stores only 8 KV heads. The expansion is a cheap operation at attention time.

---

## Memory Comparison: MHA vs GQA vs MQA

Using Qwen3.5-7B as the example:
- 32 layers
- head_dim = 128
- Sequence length = 32,768 tokens
- bf16 (2 bytes per value)

```
+------------------------------------------------------------------+
|  KV CACHE SIZE COMPARISON (Qwen3.5-7B at 32K context)          |
+------------------------------------------------------------------+
|                                                                  |
|  Variant  | KV heads | KV cache size | vs MHA                   |
|           |          |               |                          |
|  MHA      |    32    |    17.2 GB    | 1x (baseline)            |
|  GQA (8)  |     8    |     4.3 GB    | 4x smaller               |
|  MQA      |     1    |     0.5 GB    | 32x smaller              |
|                                                                  |
|  Formula:                                                        |
|  2 × 32 layers × n_kv × 32768 seq × 128 head_dim × 2 bytes     |
|                                                                  |
+------------------------------------------------------------------+
```

Why not always use MQA (only 1 KV head)?

```
+------------------------------------------------------------------+
|  QUALITY VS MEMORY TRADEOFF                                     |
+------------------------------------------------------------------+
|                                                                  |
|  MHA  (32 KV heads):  Best quality.  Many diverse attention     |
|                        patterns. Each head specializes.         |
|                        Expensive at inference.                  |
|                                                                  |
|  MQA  (1 KV head):    Worst quality at large scale. All Q       |
|                        heads forced to use same K/V.            |
|                        Less expressive. Hard to specialize.     |
|                                                                  |
|  GQA  (8 KV heads):   Near-MHA quality. 4 Q heads share 1 KV   |
|                        head — enough diversity. Big memory win. |
|                                                                  |
|  Papers show: GQA with 8 KV heads matches MHA quality with 32  |
|  KV heads, on benchmarks. LLaMA 3, Qwen3.5, Mistral all use GQA|
+------------------------------------------------------------------+
```

---

## Where GQA Lives in the Model

Every transformer block has one attention layer. GQA replaces MHA in that layer:

```
TRANSFORMER BLOCK (Qwen3.5 style):

  Input
    |
    v
  RMSNorm
    |
    v
  +---------------------------+
  |  Group-Query Attention    |
  |                           |
  |  W_q: [d, n_q * head_d]  |  <- 32 query heads
  |  W_k: [d, n_kv * head_d] |  <- 8 KV heads  (4x smaller than MHA)
  |  W_v: [d, n_kv * head_d] |  <- 8 KV heads  (4x smaller than MHA)
  |  W_o: [n_q * head_d, d]  |  <- output projection
  |                           |
  |  RoPE applied to Q and K  |  <- from Lesson 1!
  +---------------------------+
    |
    v
  Residual add
    |
    v
  RMSNorm
    |
    v
  SwiGLU FFN
    |
    v
  Residual add
    |
    v
  Output
```

Notice: **RoPE from Lesson 1 is applied inside the GQA layer**, to Q and K
before the dot product. Both lessons connect here.

---

## The Weight Matrix Size Difference

The reduced KV heads means W_k and W_v are physically smaller matrices:

```
WEIGHT MATRICES: MHA vs GQA

MODEL DIMENSION d = 4096
n_q_heads = 32, head_dim = 128
n_kv_heads (MHA) = 32
n_kv_heads (GQA) = 8

MHA weight sizes:
  W_q: 4096 × (32 × 128) = 4096 × 4096 = 16.7M parameters
  W_k: 4096 × (32 × 128) = 4096 × 4096 = 16.7M parameters
  W_v: 4096 × (32 × 128) = 4096 × 4096 = 16.7M parameters
  Total QKV: 50.3M per layer

GQA weight sizes:
  W_q: 4096 × (32 × 128) = 4096 × 4096 = 16.7M parameters  (same)
  W_k: 4096 × (8 × 128)  = 4096 × 1024 =  4.2M parameters  (4x smaller!)
  W_v: 4096 × (8 × 128)  = 4096 × 1024 =  4.2M parameters  (4x smaller!)
  Total QKV: 25.1M per layer  (50% less than MHA)
```

GQA reduces both: (1) model weights and (2) KV cache at inference.
A double saving.

---

## Visual: What Each Head Learns

Think of attention heads as specialists. GQA lets some specialization happen
without the full cost of MHA:

```
GQA with 8 KV heads, 32 Q heads (group_size = 4):

  KV Head 1 (syntax):
    Q1 — looks for subject-verb agreement
    Q2 — looks for noun phrases
    Q3 — looks for verb tense
    Q4 — looks for article-noun match

  KV Head 2 (semantics):
    Q5 — looks for topic words
    Q6 — looks for sentiment
    Q7 — looks for named entities
    Q8 — looks for coreference

  ... (8 groups, 4 Q heads each) ...

  The 4 Q heads in a group use the SAME K/V to attend —
  but each has a different W_q, so they ask different questions
  of the same key-value memory.
```

The K and V determine WHAT information is stored from each token.
The Q determines WHAT question is being asked.
GQA separates these: 32 questions, but only 8 memory stores.

---

## C# Analogy: Shared Read-Only Cache

```csharp
// Imagine 32 worker threads (32 Q heads) that all need to look up
// information from a database (K/V memory).
//
// MHA approach: each worker has its OWN private database connection.
//   - 32 connections open simultaneously
//   - Each connection maintains its own cursor and buffer (KV cache)
//   - Expensive: 32x memory for connections
//
//   class AttentionHead {
//       private SqlConnection _connection;   // private K/V per head
//       private SqlDataReader _cache;        // private KV cache
//   }
//   // 32 instances = 32 connections = 32x memory
//
// GQA approach: workers are grouped. Each group of 4 shares ONE connection.
//   - 8 connections for 32 workers (group_size = 4)
//   - Each group has a shared SqlConnection (shared KV head)
//   - Workers in the same group each have their own WHERE clause (W_q)
//   - They ask DIFFERENT questions of the SAME database connection
//
//   class AttentionGroup {
//       private SqlConnection _sharedConnection;  // shared K/V for the group
//   }
//   class AttentionHead {
//       private AttentionGroup _group;  // reference to shared connection
//       private Matrix _queryWeights;   // private W_q — different per head
//   }
//   // 8 connections for 32 workers = 4x memory savings
//
// The shared SqlConnection = the KV head
// The private WHERE clause  = the W_q matrix (query weights)
// The connection buffer      = the KV cache entry
//
// Quality insight: each worker still asks its own specialized question.
// They just share the underlying memory storage, not the query logic.
```

---

## GQA in Real Models

| Model | n_q_heads | n_kv_heads | Group Size | Approach |
|-------|-----------|------------|------------|----------|
| GPT-2 124M | 12 | 12 | 1 | MHA |
| LLaMA 2 7B | 32 | 32 | 1 | MHA |
| LLaMA 3 8B | 32 | 8 | 4 | GQA |
| Mistral 7B | 32 | 8 | 4 | GQA |
| Qwen3.5 7B | 32 | 8 | 4 | GQA |
| Qwen3.5 0.5B | 16 | 8 | 2 | GQA |

Notice: most modern 7B models converged on **32 Q heads, 8 KV heads, group size 4**.
This is the sweet spot: 4x memory saving, near-MHA quality.

---

## Summary: MHA vs GQA Decision

```
+------------------------------------------------------------------+
|  WHEN TO USE WHICH                                              |
+------------------------------------------------------------------+
|                                                                  |
|  Use MHA if:                                                     |
|    - Training a small model (< 1B params)                       |
|    - Memory is not a constraint                                  |
|    - Sequence lengths are short (< 2048 tokens)                 |
|    - Maximum quality is the only goal                            |
|                                                                  |
|  Use GQA if:                                                     |
|    - Model is 1B+ parameters (inference memory matters)         |
|    - Long context (4K+ tokens) — KV cache grows fast            |
|    - Deploying to consumer hardware (24 GB GPU)                 |
|    - Production LLM — this is the industry standard choice      |
|                                                                  |
|  Use MQA if:                                                     |
|    - Extreme memory constraint (mobile, edge devices)            |
|    - Speed is critical and some quality loss is acceptable       |
|    - Encoder-only models (quality loss is less noticeable)      |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Quiz Questions

**Q1**: In GQA with 32 Q heads and 8 KV heads, how many Q heads share each KV head?
        a) 1
        b) 2
        c) 4
        d) 8

**Q2**: Why does reducing KV heads save more memory at inference than at training?
        a) The KV cache only exists during inference — it stores K/V for all past tokens
        b) Gradient computation requires more memory during training
        c) The optimizer state uses the KV heads during training
        d) KV heads are larger during training due to batch size

**Q3**: A Qwen3.5-7B model has 32 Q heads and 8 KV heads at 32K context, bf16.
        Compared to an MHA version (32 KV heads), GQA uses approximately:
        a) Same KV cache memory
        b) 2x less KV cache memory
        c) 4x less KV cache memory
        d) 8x less KV cache memory

**Q4**: In the `repeat_kv` trick, what exactly is being repeated?
        a) The Q weights, to match the number of KV heads
        b) The K and V tensors, expanding from n_kv_heads to n_q_heads
        c) The attention scores, once per group
        d) The output projection, once per group

*(Answers: Q1=c, Q2=a, Q3=c, Q4=b)*

---

## Key Takeaways

1. MHA gives every Q head its own KV head — expensive at inference (large KV cache)
2. GQA groups Q heads and shares one KV head per group — 4x less KV memory, near-MHA quality
3. MQA is the extreme: all Q heads share one KV head — smallest memory, biggest quality drop
4. The industry standard today is GQA with group_size = 4 (32 Q heads, 8 KV heads)
5. GQA also shrinks W_k and W_v weight matrices — fewer model parameters, not just less cache
6. `repeat_kv` expands KV heads to match Q heads at compute time — cache stays compact
7. RoPE (Lesson 1) is applied to Q and K inside the GQA layer — the two lessons connect

---

*Next: Lesson 3 — Recurrent Linear Attention (how Qwen3.5 handles very long contexts in O(N) memory)*
