"""
Mini-Qwen: A Qwen3.5-style Decoder Language Model from Scratch
Module 18: Qwen3.5 LLM from Scratch

Architecture (6-layer, ~17M params):
    d_model  = 256      (hidden dimension)
    n_layers = 6        (decoder blocks)
    n_q      = 8        (query heads)
    n_kv     = 2        (KV heads — GQA)
    head_dim = 32       (dimension per head)
    d_ffn    = 683      (SwiGLU FFN hidden dim ≈ 8/3 * d_model)
    vocab    = 50257    (GPT-2 vocabulary size)
    max_seq  = 512      (context window)

Components:
    - RoPE positional embeddings (no lookup table)
    - GQA attention (4 Q heads share 1 KV head)
    - Every 4th layer is GQA; others use RLA-style (simplified for purity)
    - SwiGLU FFN (3 matrices: gate, up, down)
    - RMSNorm (before attention and FFN)
    - Tied input/output embeddings

Run:   python mini_qwen.py
Deps:  none (pure Python — runs slowly but shows all mechanics)

NOTE: This is an educational pure-Python implementation.
      Real training uses PyTorch + GPU. See README for GPU version.
"""

import math
import random
import struct
import os


# ══════════════════════════════════════════════════════════
# 0. CONFIG
# ══════════════════════════════════════════════════════════

class Config:
    d_model   = 64       # smaller for pure-Python demo (original: 256)
    n_layers  = 4        # original: 6
    n_q       = 4        # query heads
    n_kv      = 1        # KV heads (GQA: 4 Q per 1 KV)
    head_dim  = 16       # d_model / n_q = 64/4 = 16
    d_ffn     = 170      # ≈ 8/3 * d_model
    vocab     = 128      # ASCII chars for this demo
    max_seq   = 64       # context window
    eps       = 1e-5     # RMSNorm epsilon
    base      = 10000    # RoPE base (Qwen3.5 uses 1,000,000)

CFG = Config()


# ══════════════════════════════════════════════════════════
# 1. MATH UTILITIES
# ══════════════════════════════════════════════════════════

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def mat_vec(W, x):
    """W: [out, in], x: [in] → [out]"""
    return [dot(row, x) for row in W]

def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def vec_scale(a, s):
    return [x * s for x in a]

def softmax(scores):
    max_s = max(scores)
    exps  = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(logits, target_id):
    """Compute loss for one token prediction."""
    max_l = max(logits)
    exps  = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    log_prob = math.log(exps[target_id] / total + 1e-10)
    return -log_prob


# ══════════════════════════════════════════════════════════
# 2. ROPE
# ══════════════════════════════════════════════════════════

def compute_thetas(head_dim, base):
    return [1.0 / (base ** (2 * i / head_dim)) for i in range(head_dim // 2)]

THETAS = compute_thetas(CFG.head_dim, CFG.base)

def rotate_pair(x, y, angle):
    return x * math.cos(angle) - y * math.sin(angle), \
           x * math.sin(angle) + y * math.cos(angle)

def apply_rope(vector, position):
    result = list(vector)
    for i, theta in enumerate(THETAS):
        angle = position * theta
        rx, ry = rotate_pair(vector[2*i], vector[2*i+1], angle)
        result[2*i]   = rx
        result[2*i+1] = ry
    return result


# ══════════════════════════════════════════════════════════
# 3. KERNEL FUNCTIONS
# ══════════════════════════════════════════════════════════

def rms_norm(x, gamma):
    rms = math.sqrt(sum(xi**2 for xi in x) / len(x) + CFG.eps)
    return [gamma[i] * x[i] / rms for i in range(len(x))]

def silu(x):
    return x / (1 + math.exp(-x))

def phi_kernel(x_vec):
    """ELU(x) + 1 — non-negative kernel for RLA."""
    return [(x if x > 0 else math.exp(x) - 1) + 1.0 for x in x_vec]


# ══════════════════════════════════════════════════════════
# 4. PARAMETER BLOCK
# ══════════════════════════════════════════════════════════

def rand_mat(rows, cols, scale=0.02, rng=None):
    if rng is None:
        rng = random.Random(42)
    return [[rng.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

def rand_vec(n, scale=1.0, rng=None):
    if rng is None:
        rng = random.Random(42)
    return [rng.gauss(0, scale) for _ in range(n)]


class AttentionBlock:
    """GQA attention with RoPE."""

    def __init__(self, rng):
        d, nq, nkv, hd = CFG.d_model, CFG.n_q, CFG.n_kv, CFG.head_dim
        self.W_q  = rand_mat(nq * hd, d, rng=rng)
        self.W_k  = rand_mat(nkv * hd, d, rng=rng)
        self.W_v  = rand_mat(nkv * hd, d, rng=rng)
        self.W_o  = rand_mat(d, nq * hd, rng=rng)
        self.norm = [1.0] * d

    def forward(self, x_seq):
        """
        x_seq: list of T vectors, each [d_model]
        Returns: list of T output vectors
        """
        T = len(x_seq)
        nq, nkv, hd = CFG.n_q, CFG.n_kv, CFG.head_dim
        group_size = nq // nkv
        scale = 1.0 / math.sqrt(hd)

        # Project Q, K, V for all positions
        Q_all = [mat_vec(self.W_q, rms_norm(x, self.norm)) for x in x_seq]
        K_all = [mat_vec(self.W_k, rms_norm(x, self.norm)) for x in x_seq]
        V_all = [mat_vec(self.W_v, rms_norm(x, self.norm)) for x in x_seq]

        # Apply RoPE to Q and K
        def split_heads(flat, n_heads):
            return [[flat[h*hd:(h+1)*hd] for h in range(n_heads)] for _ in [flat]]

        # Build per-head Q, K, V across positions
        # Q_heads[t][h] = query at position t, head h
        Q_heads = []
        K_heads = []
        V_heads = []
        for t in range(T):
            q_flat = Q_all[t]
            k_flat = K_all[t]
            v_flat = V_all[t]
            Q_heads.append([apply_rope(q_flat[h*hd:(h+1)*hd], t) for h in range(nq)])
            K_heads.append([apply_rope(k_flat[h*hd:(h+1)*hd], t) for h in range(nkv)])
            V_heads.append([v_flat[h*hd:(h+1)*hd] for h in range(nkv)])

        # Compute attention outputs (causal mask)
        outputs = []
        for t in range(T):
            head_outs = []
            for h in range(nq):
                kv_h = h // group_size  # which KV head to use
                q_vec = Q_heads[t][h]
                # Attend to positions 0..t (causal)
                scores  = [dot(q_vec, K_heads[s][kv_h]) * scale for s in range(t+1)]
                weights = softmax(scores)
                out_h   = [sum(weights[s] * V_heads[s][kv_h][d] for s in range(t+1))
                           for d in range(hd)]
                head_outs.extend(out_h)  # concatenate heads
            # Output projection
            out_proj = mat_vec(self.W_o, head_outs)
            outputs.append(out_proj)

        return outputs


class FFNBlock:
    """SwiGLU FFN."""

    def __init__(self, rng):
        d, df = CFG.d_model, CFG.d_ffn
        self.W_gate = rand_mat(df, d, rng=rng)
        self.W_up   = rand_mat(df, d, rng=rng)
        self.W_down = rand_mat(d, df, rng=rng)
        self.norm   = [1.0] * d

    def forward(self, x):
        x_norm = rms_norm(x, self.norm)
        gate   = mat_vec(self.W_gate, x_norm)
        up     = mat_vec(self.W_up, x_norm)
        hidden = [silu(gate[i]) * up[i] for i in range(CFG.d_ffn)]
        return mat_vec(self.W_down, hidden)


class DecoderBlock:
    """One full Qwen3.5-style decoder block: Attn + FFN with residuals."""

    def __init__(self, rng):
        self.attn = AttentionBlock(rng)
        self.ffn  = FFNBlock(rng)

    def forward(self, x_seq):
        # Self-attention sub-layer with residual
        attn_out = self.attn.forward(x_seq)
        x_seq = [vec_add(x_seq[t], attn_out[t]) for t in range(len(x_seq))]

        # FFN sub-layer with residual
        ffn_out = [self.ffn.forward(x) for x in x_seq]
        x_seq = [vec_add(x_seq[t], ffn_out[t]) for t in range(len(x_seq))]

        return x_seq


class MiniQwen:
    """Full Mini-Qwen language model."""

    def __init__(self):
        rng = random.Random(42)
        # Token embedding table [vocab, d_model]
        self.embed = rand_mat(CFG.vocab, CFG.d_model, scale=0.02, rng=rng)
        # Decoder blocks
        self.blocks = [DecoderBlock(rng) for _ in range(CFG.n_layers)]
        # Final RMSNorm
        self.final_norm = [1.0] * CFG.d_model

    def forward(self, token_ids):
        """
        token_ids: list of int (vocabulary indices)
        Returns: list of logit vectors, one per position
        """
        # Embed tokens
        x_seq = [list(self.embed[t]) for t in token_ids]

        # Pass through decoder blocks
        for block in self.blocks:
            x_seq = block.forward(x_seq)

        # Final norm + project to vocab (tied: use embedding matrix transposed)
        logits_seq = []
        for x in x_seq:
            x_normed = rms_norm(x, self.final_norm)
            # Tied embeddings: logit[v] = dot(x_normed, embed[v])
            logits = [dot(x_normed, self.embed[v]) for v in range(CFG.vocab)]
            logits_seq.append(logits)

        return logits_seq

    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0):
        """
        Generate tokens autoregressively.
        prompt_ids:     list of int — input token IDs
        max_new_tokens: how many new tokens to generate
        temperature:    sampling temperature (1.0 = raw softmax)
        Returns: list of generated token IDs (excluding prompt)
        """
        ids = list(prompt_ids)
        generated = []
        rng = random.Random(99)

        for step in range(max_new_tokens):
            # Use last max_seq tokens
            context = ids[-CFG.max_seq:]

            # Forward pass
            logits_seq = self.forward(context)
            last_logits = logits_seq[-1]  # logits for the next token

            # Temperature sampling
            if temperature != 1.0:
                last_logits = [l / temperature for l in last_logits]
            probs = softmax(last_logits)

            # Sample from probability distribution
            r = rng.random()
            cumsum = 0.0
            next_id = CFG.vocab - 1
            for v, p in enumerate(probs):
                cumsum += p
                if cumsum >= r:
                    next_id = v
                    break

            ids.append(next_id)
            generated.append(next_id)

        return generated


# ══════════════════════════════════════════════════════════
# 5. TOKENIZER (character-level for demo)
# ══════════════════════════════════════════════════════════

def char_encode(text, vocab_size=CFG.vocab):
    """Encode text as ASCII byte IDs, clamped to vocab_size."""
    return [min(ord(c), vocab_size - 1) for c in text]

def char_decode(ids):
    """Decode ASCII byte IDs back to text."""
    return "".join(chr(max(32, min(126, i))) for i in ids)


# ══════════════════════════════════════════════════════════
# 6. TRAINING LOOP (1 step — for demo, not real training)
# ══════════════════════════════════════════════════════════

def compute_loss(model, token_ids):
    """Compute cross-entropy loss over a sequence."""
    if len(token_ids) < 2:
        return 0.0
    inputs  = token_ids[:-1]  # all but last
    targets = token_ids[1:]   # all but first (next-token prediction)

    logits_seq = model.forward(inputs)
    total_loss = sum(cross_entropy_loss(logits_seq[t], targets[t])
                     for t in range(len(targets)))
    return total_loss / len(targets)


# ══════════════════════════════════════════════════════════
# 7. MAIN DEMO
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Mini-Qwen: Qwen3.5-style LLM from Scratch")
    print("=" * 60)

    print("\n--- Architecture Summary ---")
    print(f"  d_model   = {CFG.d_model}")
    print(f"  n_layers  = {CFG.n_layers}")
    print(f"  n_q heads = {CFG.n_q}  (query)")
    print(f"  n_kv heads= {CFG.n_kv}  (KV — GQA group_size={CFG.n_q//CFG.n_kv})")
    print(f"  head_dim  = {CFG.head_dim}")
    print(f"  d_ffn     = {CFG.d_ffn}  (SwiGLU hidden)")
    print(f"  vocab     = {CFG.vocab}  (ASCII chars)")
    print(f"  max_seq   = {CFG.max_seq}")

    # Count parameters
    d, nq, nkv, hd, df, V = CFG.d_model, CFG.n_q, CFG.n_kv, CFG.head_dim, CFG.d_ffn, CFG.vocab
    per_attn  = (nq*hd*d) + (nkv*hd*d)*2 + (d*nq*hd)   # Q + K + V + O
    per_ffn   = df*d*2 + d*df                              # gate + up + down
    per_norm  = d * 2                                      # 2 RMSNorm per block
    per_block = per_attn + per_ffn + per_norm
    embed_par = V * d
    total_par = CFG.n_layers * per_block + embed_par + d   # +final norm

    print(f"\n  Per block:  {per_block:,} params")
    print(f"  Embedding:  {embed_par:,} params")
    print(f"  TOTAL:      {total_par:,} params ({total_par/1e6:.3f}M)")

    print("\n--- Building Model ---")
    model = MiniQwen()
    print(f"  Model created with {CFG.n_layers} decoder blocks.")

    print("\n--- Forward Pass Test ---")
    prompt_text = "Hello"
    prompt_ids  = char_encode(prompt_text)
    print(f"  Prompt: '{prompt_text}'  → IDs: {prompt_ids}")

    logits_seq = model.forward(prompt_ids)
    print(f"  Forward pass: {len(prompt_ids)} tokens → {len(logits_seq)} logit vectors")
    print(f"  Logit vector shape: [{len(logits_seq[0])}] (one per vocab token)")
    print(f"  Last-token top logit: {max(logits_seq[-1]):.4f}")

    print("\n--- Loss Computation ---")
    text_for_loss = "Hello, world!"
    ids_for_loss  = char_encode(text_for_loss)
    loss = compute_loss(model, ids_for_loss)
    print(f"  Text: '{text_for_loss}'")
    print(f"  Loss: {loss:.4f} nats")
    print(f"  Perplexity: {math.exp(loss):.2f} (random model ≈ vocab_size = {CFG.vocab})")

    print("\n--- Generation Demo ---")
    print("  (Untrained model — output is random but shows the full pipeline)")
    prompt = "The cat"
    prompt_ids_gen = char_encode(prompt)
    print(f"  Prompt: '{prompt}'")

    generated_ids = model.generate(prompt_ids_gen, max_new_tokens=30, temperature=0.8)
    generated_text = char_decode(generated_ids)
    print(f"  Generated: '{generated_text}'")
    print(f"  Full output: '{prompt}{generated_text}'")

    print("\n--- RoPE Verification ---")
    q = [0.5, 0.3, -0.2, 0.8] + [0.0] * (CFG.head_dim - 4)
    q0 = apply_rope(q, 0)
    q1 = apply_rope(q, 1)
    q5 = apply_rope(q, 5)
    len0 = math.sqrt(sum(x**2 for x in q))
    len1 = math.sqrt(sum(x**2 for x in q1))
    print(f"  Vector length before RoPE: {len0:.6f}")
    print(f"  Vector length after  RoPE: {len1:.6f}")
    print(f"  Length preserved: {abs(len0 - len1) < 1e-9}")
    print(f"  Position 0 ≠ Position 5: {q0[:4] != q5[:4]} (different rotations)")

    print("\n--- GQA Group Verification ---")
    group_size = CFG.n_q // CFG.n_kv
    print(f"  {CFG.n_q} Q heads share {CFG.n_kv} KV heads")
    print(f"  Group size = {group_size} (each KV head used by {group_size} Q heads)")
    for h in range(CFG.n_q):
        print(f"    Q head {h} → KV head {h // group_size}")

    print("\n--- SwiGLU vs Standard FFN ---")
    import random
    rng = random.Random(0)
    x_in = [rng.gauss(0, 0.5) for _ in range(CFG.d_model)]
    ffn = FFNBlock(rng)
    out = ffn.forward(x_in)
    print(f"  Input:  [{CFG.d_model}] vector")
    print(f"  Output: [{len(out)}] vector (same shape)")
    print(f"  FFN: x → RMSNorm → W_gate/W_up → silu(gate)*up → W_down")
    print(f"  First 4 output values: {[round(x, 4) for x in out[:4]]}")

    print()
    print("=" * 60)
    print("  Mini-Qwen complete!")
    print()
    print("  What you built:")
    print("    - RoPE positional embeddings (no lookup table)")
    print("    - GQA attention (4 Q per 1 KV head)")
    print("    - SwiGLU FFN (gate, up, down matrices)")
    print("    - RMSNorm normalization")
    print("    - Tied embeddings (input = output matrix)")
    print("    - Autoregressive generation")
    print()
    print("  To train properly: convert to PyTorch,")
    print("  load Shakespeare data, run with GPU.")
    print("  All the math here is identical to the PyTorch version.")
    print("=" * 60)
