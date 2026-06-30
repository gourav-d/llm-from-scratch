# Lesson 1: RoPE — Rotary Position Embeddings

## What Problem Does This Solve?

In Module 04, you learned about positional encoding.
The transformer has no built-in sense of order — it sees all tokens at once.
To tell it "this token is at position 3, that one is at position 7", you add
a positional embedding to the input.

GPT uses a simple learned table:

```
position 0  →  vector [0.1, -0.3, 0.7, ...]
position 1  →  vector [0.4,  0.2, 0.1, ...]
...
position N  →  vector [...]

These vectors are LEARNED during training and ADDED to the token embeddings.
```

This works, but has two problems:

```
+=========================================================================+
|  PROBLEMS WITH LEARNED POSITIONAL EMBEDDINGS (GPT-style)               |
+=========================================================================+
|                                                                         |
|  Problem 1 — Fixed context length:                                      |
|  If you train on sequences up to 512 tokens, position 513 has           |
|  NO learned embedding. The model breaks on longer inputs.               |
|                                                                         |
|  Problem 2 — No relative position sense:                                |
|  The model learns "token at position 5" separately from "token at       |
|  position 6". It does not automatically learn that positions 5 and 6    |
|  are adjacent. It must discover this from the data, which wastes        |
|  capacity.                                                              |
|                                                                         |
|  What we want:                                                          |
|  The model should naturally know that position 5 and 6 are close,      |
|  and that position 5 and 500 are far apart —                           |
|  WITHOUT needing to learn a separate vector for every position.         |
+=========================================================================+
```

**RoPE solves both problems** by encoding position through rotation,
not through a learned lookup table.

---

## The Core Idea: Rotate, Don't Add

Instead of adding a position vector to the input, RoPE **rotates** the Query
and Key vectors based on position.

```
Old approach (additive):
  token_embedding + position_embedding → input to attention

RoPE approach (rotational):
  Q_rotated = Rotate(Q, position_of_token)
  K_rotated = Rotate(K, position_of_token)
  attention = softmax(Q_rotated · K_rotated^T / sqrt(d))
```

The magic happens when you compute `Q · K`:

```
+------------------------------------------------------------------+
|  WHY ROTATION CAPTURES RELATIVE POSITION                        |
+------------------------------------------------------------------+
|                                                                  |
|  Let Q be at position m.  Let K be at position n.               |
|                                                                  |
|  Q_rotated = R(m) * Q                                           |
|  K_rotated = R(n) * K                                           |
|                                                                  |
|  Dot product:                                                    |
|    Q_rotated · K_rotated = (R(m) * Q) · (R(n) * K)             |
|                           = Q · (R(m-n) * K)    [rotation math] |
|                           = Q · R(m-n) · K                      |
|                                                                  |
|  The dot product ONLY depends on (m - n) — the relative         |
|  distance between positions — not on the absolute values of     |
|  m or n separately.                                             |
|                                                                  |
|  Result: the model naturally learns relative distances.         |
|  Position 5 attending to position 3 uses the same rotation      |
|  as position 105 attending to position 103. (Both are 2 apart.) |
+------------------------------------------------------------------+
```

This is the key insight. The attention score between any two tokens depends
only on how far apart they are — not where they are absolutely.

---

## What Is a Rotation Matrix?

Before the RoPE math, you need to understand what a rotation matrix does.

In 2D space, rotating a vector `[x, y]` by angle `θ` gives:

```
+------------------------------------------------------------------+
|  2D ROTATION MATRIX                                             |
|                                                                  |
|  Original vector:  [x, y]                                       |
|                                                                  |
|  Rotation by angle θ:                                           |
|                                                                  |
|  [x']   [cos(θ)  -sin(θ)] [x]                                  |
|  [y'] = [sin(θ)   cos(θ)] [y]                                   |
|                                                                  |
|  x' = x*cos(θ) - y*sin(θ)                                       |
|  y' = x*sin(θ) + y*cos(θ)                                       |
|                                                                  |
|  Key property: the LENGTH of the vector stays the same.         |
|  Rotation only changes DIRECTION, not magnitude.                |
+------------------------------------------------------------------+
```

RoPE applies this 2D rotation to pairs of dimensions in Q and K.
If the embedding dimension is 128, there are 64 pairs: (dim 0, dim 1),
(dim 2, dim 3), ..., (dim 126, dim 127).
Each pair gets rotated by a different angle.

---

## The RoPE Angles: The Theta Formula

Different dimension pairs rotate at different speeds.
Lower dimensions rotate slowly (good for capturing long-range relationships).
Higher dimensions rotate quickly (good for capturing short-range relationships).

```
+=========================================================================+
|  ROPE ANGLE FORMULA                                                     |
+=========================================================================+
|                                                                         |
|  For dimension pair i (where i goes from 0 to d/2 - 1):               |
|                                                                         |
|  theta_i = 1 / (10000 ^ (2i / d))                                      |
|                                                                         |
|  Where:                                                                 |
|    i   = index of the dimension pair (0, 1, 2, ..., d/2 - 1)           |
|    d   = total embedding dimension (e.g., 128)                          |
|    10000 = base frequency (Qwen uses 1,000,000 for long context)        |
|                                                                         |
|  At position m, dimension pair i rotates by:                           |
|    angle = m * theta_i                                                  |
|                                                                         |
+=========================================================================+
```

Let us compute a few values with d = 4 (so 2 pairs) and base = 10000:

```
Dimension pair 0 (dims 0, 1):
  theta_0 = 1 / (10000 ^ (0/4)) = 1 / 1 = 1.0
  At position 1: angle = 1 × 1.0 = 1.0 radian  (fast rotation)

Dimension pair 1 (dims 2, 3):
  theta_1 = 1 / (10000 ^ (2/4)) = 1 / 100 = 0.01
  At position 1: angle = 1 × 0.01 = 0.01 radian  (slow rotation)

At position 100:
  Pair 0: angle = 100 × 1.0   = 100.0 radians (has rotated many times)
  Pair 1: angle = 100 × 0.01  = 1.0 radian    (still only 1 full rotation)
```

This is exactly like the original sinusoidal positional encoding from Module 04 —
but instead of ADDING sine/cosine values, RoPE USES them to rotate the vectors.

---

## Visual: What Happens to a Query Vector

```
BEFORE ROPE (standard Q vector at any position):

  Q = [q0, q1, q2, q3, q4, q5, q6, q7]
         |       |       |       |
        pair0   pair1   pair2   pair3


AFTER ROPE at position m:

  Pair 0 (q0, q1) → rotated by angle: m * theta_0
  Pair 1 (q2, q3) → rotated by angle: m * theta_1
  Pair 2 (q4, q5) → rotated by angle: m * theta_2
  Pair 3 (q6, q7) → rotated by angle: m * theta_3

  Q_rotated = [q0', q1', q2', q3', q4', q5', q6', q7']


VISUAL (2D example, one pair):

                     ^ y
                     |
              q1'  * |          ← q after rotation by θ
                 *   |
              *   θ  |
  -----------*--------→ x
              q0  original

  The vector [q0, q1] has been rotated by angle θ.
  Its length is unchanged. Only its direction changed.
```

---

## The Full RoPE Computation (Step by Step)

Here is exactly what happens in code:

```
INPUT:
  Q matrix shape:   [batch, seq_len, n_heads, head_dim]
  K matrix shape:   [batch, seq_len, n_heads, head_dim]
  positions:        [0, 1, 2, 3, ..., seq_len-1]

STEP 1: Compute theta frequencies
  theta = [1/10000^(0/d), 1/10000^(2/d), ..., 1/10000^((d-2)/d)]
  shape: [d/2]

STEP 2: Compute angles for each position
  angles[pos, i] = pos * theta[i]
  shape: [seq_len, d/2]

STEP 3: Compute cos and sin of angles
  cos_angles = cos(angles)    shape: [seq_len, d/2]
  sin_angles = sin(angles)    shape: [seq_len, d/2]

STEP 4: Split Q into pairs
  Q_even = Q[..., 0::2]    (dimensions 0, 2, 4, ...)   shape: [..., d/2]
  Q_odd  = Q[..., 1::2]    (dimensions 1, 3, 5, ...)   shape: [..., d/2]

STEP 5: Apply rotation formula to each pair
  Q_rotated_even = Q_even * cos - Q_odd * sin
  Q_rotated_odd  = Q_even * sin + Q_odd * cos

STEP 6: Interleave back
  Q_rotated = interleave(Q_rotated_even, Q_rotated_odd)
  shape: [..., d]

Repeat steps 4-6 for K.
Use Q_rotated and K_rotated in the attention computation.
```

---

## Why RoPE Extrapolates to Longer Sequences

Recall the problem with learned positional embeddings: position 513 has no embedding
if training only went up to 512.

With RoPE, position 513 simply produces a new rotation angle:
```
  angle = 513 * theta_i
```
This is just a number. No table lookup needed. The math works for any position.

The model may not generalize PERFECTLY to positions much longer than training,
but it degrades gracefully instead of breaking completely.

Qwen3.5 extends this further by using **YaRN** (Yet another RoPE extensioN) to scale
the base frequency (from 10,000 to 1,000,000), which makes the model handle up to
128K or 1M token contexts. But the core RoPE idea is exactly what you learned above.

---

## RoPE vs Sinusoidal (Module 04) — Side by Side

```
+------------------------------------------------------------------+
|                  | SINUSOIDAL (GPT/BERT)    | ROPE               |
+------------------+--------------------------+--------------------+
|  How applied     | Added to token embedding | Rotates Q and K    |
|  Learned?        | No (fixed formula)       | No (fixed formula) |
|  Relative pos?   | Not directly             | Yes, by design     |
|  Extrapolates?   | Poor (breaks beyond N)   | Good (graceful)    |
|  Memory          | Lookup table in RAM      | Computed on-the-fly|
|  Used in         | GPT-1/2, BERT, early T5  | LLaMA, Qwen, Mistral|
+------------------------------------------------------------------+
```

Both use sine and cosine. The difference is WHERE the math is applied:
- Sinusoidal: add to the embedding before attention
- RoPE: multiply into Q and K inside attention

---

## C# Analogy: Encoding Position as Phase

```csharp
// Imagine you have a list of events in a distributed system.
// You want to encode "how far apart" two events are — not "what time they happened".
//
// Old approach (additive, like sinusoidal PE):
//   Each event gets a timestamp added to its payload.
//   event_with_position = event_data + position_vector[t]
//
//   Problem: if a new event arrives at t=10000 but your table only goes to t=9999,
//   you have no entry. System breaks.
//
// RoPE approach (rotational):
//   Each event gets its data ROTATED by a function of its timestamp.
//   event_rotated = Rotate(event_data, angle: t * frequency)
//
//   To compare two events A (at time m) and B (at time n):
//   Dot(A_rotated, B_rotated) = Dot(A, Rotate(B, angle: (m-n) * frequency))
//   The comparison ONLY depends on (m - n) — the gap.
//
//   A new event at t=10000 just computes angle = 10000 * frequency.
//   No table needed. Math works for any t.
//
// In C# terms:
//   Sinusoidal PE = DateTime lookup in a Dictionary<int, Vector> — breaks at max key
//   RoPE          = compute TimeSpan(t1 - t2) on the fly — always works
```

---

## Quiz Questions

**Q1**: What is the main problem with learned positional embeddings (used in GPT-2)?
        a) They use too much GPU memory during training
        b) They break when the model sees sequences longer than those seen in training
        c) They make attention computation slower
        d) They cannot represent positions beyond 128

**Q2**: When RoPE is applied to Q at position m and K at position n,
        the attention score Q_rotated · K_rotated depends on:
        a) The absolute positions m and n separately
        b) Only the position m of the query
        c) Only the relative distance (m - n) between the two tokens
        d) Neither position — RoPE removes position information

**Q3**: What does the theta formula `theta_i = 1 / (10000 ^ (2i/d))` control?
        a) How many attention heads are used per layer
        b) How fast each dimension pair rotates relative to position index
        c) The learning rate schedule during training
        d) The size of the KV cache

**Q4**: Compared to sinusoidal positional encoding, RoPE is applied:
        a) Before the embedding layer
        b) Added to the token embedding, same as sinusoidal
        c) Inside the attention mechanism, by rotating Q and K
        d) Only during inference, not during training

*(Answers: Q1=b, Q2=c, Q3=b, Q4=c)*

---

## Key Takeaways

1. Sinusoidal PE adds a position vector to the input — RoPE rotates Q and K instead
2. Rotation is applied per dimension pair, each at a different angular frequency
3. The dot product Q_rotated · K_rotated depends only on relative position (m - n)
4. RoPE extrapolates to longer sequences because it computes angles, not table lookups
5. Theta formula: `theta_i = 1 / (base ^ (2i/d))` — low i rotates fast, high i rotates slow
6. Qwen3.5, LLaMA 3, Mistral — all use RoPE for positional encoding
7. Qwen uses a larger base (1,000,000) to support 128K+ context lengths

---

*Next: Lesson 2 — Group-Query Attention (how Qwen3.5 slashes KV memory during inference)*
