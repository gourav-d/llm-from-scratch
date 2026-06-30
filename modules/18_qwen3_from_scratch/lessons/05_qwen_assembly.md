# Lesson 5: Full Qwen3.5 Decoder Assembly

## What Problem Does This Solve?

In Lessons 1 through 4, you built four separate components:

| Lesson | Component | What It Does |
|--------|-----------|-------------|
| L1 | RoPE | Encodes token position by rotating Q and K |
| L2 | GQA | Reduces KV heads to cut memory at inference |
| L3 | RLA | Replaces softmax attention with O(N) recurrence |
| L4 | KV Cache | Stores past K/V so they are not recomputed |

This lesson assembles all four into a working Qwen3.5-style decoder.
You will see where each component lives, how they connect, and what the
full forward pass looks like from input tokens to output logits.

---

## Architecture Overview

Qwen3.5 is a **decoder-only transformer** — the same family as GPT.
Text goes in, next-token probabilities come out.

The full model has three sections:

```
+=========================================================================+
|  QWEN3.5 FULL MODEL STRUCTURE                                           |
+=========================================================================+
|                                                                         |
|  INPUT                                                                  |
|  Token IDs: [15496, 428, 318, ...]    (integers)                        |
|       |                                                                 |
|       v                                                                 |
|  [EMBEDDING LAYER]                                                      |
|  Token Embedding table: [vocab_size, d_model]                           |
|  Lookup token ID → dense vector                                         |
|  Output: [batch, seq_len, d_model]                                      |
|       |                                                                 |
|       v                                                                 |
|  [DECODER BLOCKS] × N_layers                                            |
|  Each block: RMSNorm → Attention → Residual → RMSNorm → FFN → Residual  |
|  Attention alternates: RLA (3 of 4 blocks) / GQA (1 of 4 blocks)       |
|  RoPE applied inside every attention layer                              |
|  KV cache used in GQA blocks, hidden state used in RLA blocks          |
|  Output: [batch, seq_len, d_model]                                      |
|       |                                                                 |
|       v                                                                 |
|  [OUTPUT HEAD]                                                          |
|  Final RMSNorm                                                          |
|  Linear: [d_model, vocab_size]   (no bias)                             |
|  Output: [batch, seq_len, vocab_size]   (logits)                        |
|                                                                         |
+=========================================================================+
```

---

## Inside One Decoder Block

Every decoder block has two sub-layers: Attention and Feed-Forward Network (FFN).
Both use a residual (skip) connection and RMSNorm.

```
+------------------------------------------------------------------+
|  ONE DECODER BLOCK (detailed)                                   |
+------------------------------------------------------------------+
|                                                                  |
|  Input x: [batch, seq_len, d_model]                             |
|       |                                                          |
|       +------ residual connection ---------+                    |
|       |                                   |                    |
|       v                                   |                    |
|  RMSNorm(x)                               |                    |
|       |                                   |                    |
|       v                                   |                    |
|  Attention Layer (GQA or RLA):            |                    |
|    IF this is a GQA layer:                |                    |
|      Q = x @ W_q   [d → n_q * head_dim]  |                    |
|      K = x @ W_k   [d → n_kv * head_dim] |                    |
|      V = x @ W_v   [d → n_kv * head_dim] |                    |
|      Apply RoPE to Q and K                |                    |
|      Update KV cache with new K, V        |                    |
|      GQA attention (repeat_kv trick)      |                    |
|      attn_out = attention @ W_o           |                    |
|    IF this is an RLA layer:               |                    |
|      Same Q, K, V projections             |                    |
|      Apply RoPE to Q and K                |                    |
|      Update hidden state S (no KV cache)  |                    |
|      Output = phi(Q) @ S                  |                    |
|      attn_out = output @ W_o              |                    |
|       |                                   |                    |
|       v                                   |                    |
|  attn_out + residual  <-------------------+                    |
|       |                                                          |
|       +------ residual connection ---------+                    |
|       |                                   |                    |
|       v                                   |                    |
|  RMSNorm                                  |                    |
|       |                                   |                    |
|       v                                   |                    |
|  SwiGLU FFN:                              |                    |
|    gate  = x @ W_gate   [d → d_ffn]      |                    |
|    up    = x @ W_up     [d → d_ffn]      |                    |
|    down  = silu(gate) * up @ W_down       |                    |
|    ffn_out = down   [d_ffn → d]           |                    |
|       |                                   |                    |
|       v                                   |                    |
|  ffn_out + residual  <--------------------+                    |
|       |                                                          |
|  Output x: [batch, seq_len, d_model]                            |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Two New Components: RMSNorm and SwiGLU

These replace components you know from GPT (LayerNorm and GELU FFN).

### RMSNorm

LayerNorm from Module 04 subtracts the mean before normalizing:

```
LayerNorm(x) = (x - mean(x)) / sqrt(variance(x) + eps) * gamma + beta
```

RMSNorm skips the mean subtraction. Simpler and faster:

```
+------------------------------------------------------------------+
|  RMSNORM                                                        |
+------------------------------------------------------------------+
|                                                                  |
|  RMSNorm(x) = x / RMS(x) * gamma                                |
|                                                                  |
|  Where:                                                          |
|    RMS(x) = sqrt( mean(x^2) + eps )   (root mean square)       |
|    gamma   = learned scale parameter (one per dimension)        |
|    No beta (bias) term — simpler than LayerNorm                  |
|    No mean subtraction                                          |
|                                                                  |
|  Why use RMSNorm?                                               |
|    Slightly faster (skips mean computation).                    |
|    Empirically similar quality to LayerNorm.                    |
|    Standard in LLaMA, Qwen, Mistral.                            |
+------------------------------------------------------------------+
```

### SwiGLU FFN

Standard GPT FFN uses two linear layers with a GELU activation:

```
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
```

SwiGLU uses three linear layers with a gating mechanism:

```
+------------------------------------------------------------------+
|  SWIGLU FFN                                                     |
+------------------------------------------------------------------+
|                                                                  |
|  gate = x @ W_gate   [d_model → d_ffn]                         |
|  up   = x @ W_up     [d_model → d_ffn]                         |
|                                                                  |
|  hidden = silu(gate) * up                                       |
|                                                                  |
|  out = hidden @ W_down   [d_ffn → d_model]                     |
|                                                                  |
|  Where silu(z) = z * sigmoid(z)   (smooth gating function)     |
|                                                                  |
|  Why SwiGLU?                                                    |
|    The gate controls WHICH information flows forward.           |
|    gate acts as a learned filter on the up projection.          |
|    Empirically: better quality than standard GELU FFN.          |
|    Used in LLaMA, Qwen, PaLM, Gemma.                           |
|                                                                  |
|  Note: SwiGLU uses 3 weight matrices (W_gate, W_up, W_down)     |
|        Standard FFN uses 2 (W1, W2).                            |
|        To keep total params equal, d_ffn is smaller:            |
|        d_ffn = (2/3) * 4 * d_model  (common choice)            |
+------------------------------------------------------------------+
```

---

## Qwen3.5 Hyperparameters by Model Size

```
+------------------------------------------------------------------+
|  QWEN3.5 MODEL CONFIGS                                          |
+------------------------------------------------------------------+
|                                                                  |
|  Name      | d_model | layers | n_q | n_kv | d_ffn  | params   |
|  ----------+--------+--------+-----+------+--------+----------  |
|  0.5B      |   896  |   24   |  14 |   2  |  4864  |  0.5B    |
|  1.7B      |  1536  |   28   |  12 |   2  |  8960  |  1.7B    |
|  4B        |  2048  |   36   |  16 |   8  | 11008  |  4.0B    |
|  7B        |  3584  |   28   |  28 |   4  | 18944  |  7.6B    |
|  14B       |  5120  |   40   |  40 |   8  | 13824  | 14.7B    |
|  32B       |  5120  |   64   |  64 |   8  | 27648  | 32.5B    |
|                                                                  |
|  All sizes use:                                                  |
|    head_dim = 128                                               |
|    RoPE base theta = 1,000,000                                   |
|    Hybrid attention: 1 GQA per 4 blocks                         |
|    SwiGLU activation in FFN                                     |
|    RMSNorm (no bias)                                            |
|    Tied embeddings: output head weight = embedding weight       |
+------------------------------------------------------------------+
```

---

## Token Generation: Full Forward Pass

Here is the complete sequence from prompt to generated token:

```
PROMPT: "The sky is"

STEP 1: Tokenize
  "The sky is" → [464, 6766, 318]   (token IDs)

STEP 2: Embed
  [464, 6766, 318] → embedding lookup → [3, d_model] matrix
  Each row is a dense vector representing that token

STEP 3: Pass through N decoder blocks
  Block 1 (RLA):  update hidden state S1, compute attention output
  Block 2 (RLA):  update hidden state S2
  Block 3 (RLA):  update hidden state S3
  Block 4 (GQA):  compute full attention, store K/V in cache
  Block 5 (RLA):  update hidden state S5
  ...
  Block N (GQA):  final GQA layer
  Output: [3, d_model]

STEP 4: Final RMSNorm
  Normalize the output

STEP 5: Output head (linear projection)
  [3, d_model] @ [d_model, vocab_size] → [3, vocab_size]   (logits)
  We only care about the LAST row: logits for what comes after "is"

STEP 6: Sample next token
  Apply softmax to last row → probability distribution
  Sample (or greedy pick) → e.g., token 4171 = "blue"
  Append to sequence: [464, 6766, 318, 4171]

STEP 7: Generate next token
  Feed ONLY the new token [4171] into the model
  KV cache for GQA layers already holds K/V for tokens 1,2,3
  RLA hidden states already updated through token 1,2,3
  Output: logits for token 5 → sample → "."

  Sequence: "The sky is blue."  Done.
```

---

## Mini-Qwen vs NanoGPT (Module 05): Comparison

```
+------------------------------------------------------------------+
|  NANOGPT (MODULE 05) vs MINI-QWEN (MODULE 18)                   |
+------------------------------------------------------------------+
|                                                                  |
|  Feature              | NanoGPT (M05)        | Mini-Qwen (M18)  |
|  ---------------------+----------------------+------------------  |
|  Positional encoding  | Learned embedding     | RoPE             |
|  Attention type       | Multi-Head (MHA)      | GQA + RLA hybrid |
|  KV heads             | n_heads = n_kv_heads  | n_kv << n_q      |
|  KV cache             | None                  | Yes (GQA layers) |
|  Normalization        | LayerNorm             | RMSNorm          |
|  FFN activation       | GELU                  | SwiGLU           |
|  Context limit        | Fixed (trained size)  | Extrapolates     |
|  Memory at inference  | O(N^2) attention      | O(N) hybrid      |
|  Generation speed     | Slow (recompute past) | Fast (cache)     |
|                                                                  |
|  Same:                                                           |
|    Decoder-only (causal mask)                                   |
|    Token embedding + output head                                |
|    Residual connections around attention and FFN                |
|    Cross-entropy loss on next-token prediction                  |
|    Autoregressive generation (one token at a time)              |
+------------------------------------------------------------------+
```

The differences are engineering improvements — the underlying idea (predict next token) is identical.

---

## Parameter Count Formula

Understanding how many parameters each part contributes:

```
For one decoder block with d_model=D, n_q heads, n_kv heads, head_dim=H, d_ffn=F:

  Attention weights:
    W_q:    D × (n_q * H)
    W_k:    D × (n_kv * H)    ← smaller because n_kv < n_q (GQA)
    W_v:    D × (n_kv * H)    ← same
    W_o:    (n_q * H) × D

  FFN weights (SwiGLU has 3 matrices):
    W_gate: D × F
    W_up:   D × F
    W_down: F × D

  Norm weights (RMSNorm):
    gamma_attn: D
    gamma_ffn:  D

TOTAL per block = D*(n_q*H + 2*n_kv*H + n_q*H) + D*(3*F) + 2*D

For Mini-Qwen (d=256, n_q=8, n_kv=2, H=32, F=683, 6 layers):
  Attention per block: 256*(8*32 + 2*32 + 2*32 + 8*32) = 256*640 = 163,840
  FFN per block:       256*3*683 = 524,544
  Norm per block:      2*256 = 512
  Per block total:     ~689,000
  6 blocks:            ~4.1M
  Embedding:           vocab_size * 256 (e.g., 50257*256 = 12.9M)
  Total:               ~17M parameters  (small enough to train on CPU)
```

---

## Visual: Full Model Stack

```
Input Token IDs
       |
       v
[Embedding Table]    50257 × 256
       |
       v
[Decoder Block 1]  -- RLA  (hidden state S1, no KV cache)
       |
[Decoder Block 2]  -- RLA  (hidden state S2)
       |
[Decoder Block 3]  -- RLA  (hidden state S3)
       |
[Decoder Block 4]  -- GQA  (KV cache, full attention)  ← precise recall
       |
[Decoder Block 5]  -- RLA  (hidden state S5)
       |
[Decoder Block 6]  -- RLA  (hidden state S6)   ← last block if 6 layers
       |
       v
[Final RMSNorm]
       |
       v
[Output Head]      256 × 50257   (tied with embedding)
       |
       v
Logits [batch, seq, 50257]
       |
       v
Sample next token
```

---

## C# Analogy: Assembling a Microservices Pipeline

```csharp
// Building a Qwen3.5 decoder is like building a distributed request pipeline
// where each microservice adds value to the data as it flows through.
//
// Full pipeline analogy:
//
//   Tokenizer = URL router (converts raw input → structured request)
//
//   Embedding layer = service discovery registry
//     (token ID → feature vector, like resolving a service URL to an IP)
//
//   Decoder blocks = middleware pipeline (like ASP.NET Core middleware):
//     RMSNorm     = request normalization (normalize headers/body)
//     RLA block   = stateful stream processor (keeps running aggregation)
//     GQA block   = SQL lookup with cache (precise but memory-heavy)
//     RoPE        = request timestamp injector (position info added in-flight)
//     SwiGLU FFN  = feature transformer (enrich the representation)
//     Residual    = pass-through: output = processed + original
//
//   KV cache = Redis cache for GQA blocks
//     (computed once per token, read many times during generation)
//
//   RLA hidden state = Kafka consumer group offset
//     (small fixed state that remembers everything seen so far)
//
//   Output head = response serializer
//     (converts internal representation → vocabulary probabilities)
//
// The assembly pattern:
//   public class Qwen3Decoder
//   {
//       private EmbeddingLayer _embed;
//       private List<DecoderBlock> _blocks;   // mix of RLA and GQA
//       private RMSNorm _finalNorm;
//       private LinearLayer _outputHead;
//
//       public Tensor Forward(int[] tokenIds, KVCache cache)
//       {
//           var x = _embed.Forward(tokenIds);
//           foreach (var block in _blocks)
//               x = block.Forward(x, cache);   // each block refines x
//           x = _finalNorm.Forward(x);
//           return _outputHead.Forward(x);      // logits
//       }
//   }
//
// Each decoder block is a middleware that takes x and returns a refined x.
// The KV cache is a shared service injected into GQA blocks.
// RLA blocks use their own private state (hidden state S) instead.
```

---

## Putting It All Together: What to Build in the Project

The Mini-Qwen project assembles everything from this module into one file:

```
MINI-QWEN BUILD PLAN:

  1. RoPEEmbedding class
       compute_freqs(seq_len, head_dim, base=10000)
       apply(q, k, position_ids)
       Uses: Lesson 1

  2. GroupQueryAttention class (GQA block)
       __init__(n_q_heads, n_kv_heads, d_model, head_dim)
       forward(x, rope, kv_cache)  -> output, updated_kv_cache
       Uses: Lesson 1 (RoPE) + Lesson 2 (GQA) + Lesson 4 (KV cache)

  3. LinearAttention class (RLA block)
       __init__(n_heads, d_model, head_dim)
       forward(x, rope, hidden_state)  -> output, updated_hidden_state
       Uses: Lesson 1 (RoPE) + Lesson 3 (RLA)

  4. SwiGLUFFN class
       forward(x)  -> x
       Uses: Lesson 5

  5. DecoderBlock class
       contains: RMSNorm, one attention layer (GQA or RLA), SwiGLUFFN, residuals
       forward(x, ...)  -> x
       Uses: all above

  6. Qwen3Decoder class
       contains: embedding, N decoder blocks, final RMSNorm, output head
       forward(token_ids, kv_cache, hidden_states)  -> logits
       Uses: all above

  7. generate() function
       tokenize prompt → forward pass → sample → repeat
       Uses: full model

  8. Train on Shakespeare (~1MB, free)
       compare val loss and generated text vs nanoGPT from Module 05
```

---

## Quiz Questions

**Q1**: In Qwen3.5's hybrid architecture, roughly what fraction of decoder blocks
        use recurrent linear attention (RLA)?
        a) All blocks use RLA
        b) Half the blocks use RLA, half use GQA
        c) About 3 out of every 4 blocks use RLA
        d) RLA is only used in the first and last blocks

**Q2**: RMSNorm differs from LayerNorm (used in original GPT) primarily because:
        a) RMSNorm uses a learned scale, LayerNorm does not
        b) RMSNorm skips the mean subtraction step
        c) RMSNorm applies after the FFN, LayerNorm applies before
        d) RMSNorm uses fp16, LayerNorm uses fp32

**Q3**: SwiGLU FFN uses three weight matrices (W_gate, W_up, W_down).
        The gate and up projections combine as:
        a) gate + up (additive)
        b) silu(gate) * up (multiplicative gating)
        c) softmax(gate) @ up (attention-style)
        d) gate * up + bias (linear combination)

**Q4**: In Qwen3.5, the output head weight matrix is "tied" with the embedding table.
        This means:
        a) The output head has no parameters — it uses zeros
        b) The embedding table and output head are the SAME weight matrix (shared)
        c) The output head is initialized from the embedding table but trained separately
        d) The output head is applied before the embedding

*(Answers: Q1=c, Q2=b, Q3=b, Q4=b)*

---

## Key Takeaways

1. Qwen3.5 is a decoder-only transformer: Embedding → N Decoder Blocks → RMSNorm → Output Head
2. Each decoder block: RMSNorm → Attention (RLA or GQA) → Residual → RMSNorm → SwiGLU FFN → Residual
3. RMSNorm = LayerNorm without mean subtraction — simpler and nearly same quality
4. SwiGLU FFN = 3 matrices (gate, up, down). gate controls what information flows through
5. Hybrid attention: every 4th block is GQA (precise, KV cache), the rest are RLA (fast, hidden state)
6. RoPE is applied inside EVERY attention block, regardless of whether it is RLA or GQA
7. Token generation: process full prompt once → generate tokens one at a time using caches
8. Mini-Qwen: 6 layers, d=256, ~17M params — trainable on CPU, achieves better perplexity than nanoGPT
9. All components from Lessons 1–4 snap together at the DecoderBlock level

---

*Module 18 lessons complete. Next: build examples (code) and the Mini-Qwen project.*
