# =============================================================================
# step_06_kv_cache.py
# =============================================================================
# STEP 6: KV CACHE — FAST TEXT GENERATION
#
# WHAT'S THE PROBLEM WITHOUT KV CACHE?
#
#   When we generate text token by token, the NAIVE approach recomputes
#   K and V for ALL past tokens at EVERY step:
#
#   Generating token 1: compute K,V for 1 token  (1 computation)
#   Generating token 2: compute K,V for 2 tokens (2 computations) ← recomputes token 1!
#   Generating token 3: compute K,V for 3 tokens (3 computations) ← recomputes 1 and 2!
#   ...
#   Generating token T: compute K,V for T tokens (T computations)
#
#   Total work: 1 + 2 + 3 + ... + T = T(T+1)/2 = O(T²) — QUADRATIC!
#
#   For T=1000 tokens: ~500,000 computations
#   For T=4000 tokens: ~8,000,000 computations  (16× more work!)
#
# WHAT IS THE KV CACHE?
#
#   Cache the K and V tensors from previous tokens. Never recompute them.
#
#   Generating token 1: compute K1, V1. CACHE them.
#   Generating token 2: compute K2, V2. APPEND to cache. Reuse K1, V1.
#   Generating token 3: compute K3, V3. APPEND to cache. Reuse K1-K2, V1-V2.
#   ...
#   Generating token T: compute KT, VT. APPEND to cache. Reuse all past K,V.
#
#   Total work: T × (one K,V computation + one matmul) = O(T) — LINEAR!
#
#   For T=1000 tokens: 1,000 computations  (500× faster than naive!)
#   For T=4000 tokens: 4,000 computations  (2000× faster than naive!)
#
# VISUAL COMPARISON:
#
#   WITHOUT CACHE (at token step t=3):
#   ┌──────────────────────────────────────────┐
#   │ Recompute K,V for positions: [0, 1, 2, 3]│  ← 4 tokens × full attention
#   └──────────────────────────────────────────┘
#
#   WITH CACHE (at token step t=3):
#   ┌────────────────────────┬─────────────────┐
#   │ Read cached K,V:  [0,1,2] │ Compute NEW K,V: [3] │
#   └────────────────────────┴─────────────────┘
#   Only 1 new computation! The rest is a simple memory read.
#
# C# ANALOGY:
#   KV cache = Dictionary<int, (float[] Key, float[] Value)>
#   where int = token position.
#
#   Like MEMOIZATION in C#:
#     private readonly Dictionary<int, float[]> _cache = new();
#
#     float[] GetKey(int position, float[] input) {
#         if (!_cache.TryGetValue(position, out var cached)) {
#             cached = ComputeKey(input);     // expensive operation
#             _cache[position] = cached;      // store for reuse
#         }
#         return cached;
#     }
#
#   The cache grows as you generate more tokens, but each position is
#   computed EXACTLY ONCE — never recomputed.
#
# IMPLEMENTATION APPROACH:
#   We add a `past_kv` parameter to the attention forward pass.
#   - past_kv = None: normal training mode (no cache)
#   - past_kv = (K_cache, V_cache): generation mode (use cache)
#
#   During generation:
#     new_k = k_proj(current_token)            # just 1 token
#     new_v = v_proj(current_token)
#     k_full = torch.cat([past_k, new_k], dim=1)  # append to cache
#     v_full = torch.cat([past_v, new_v], dim=1)
#     # attend using full K/V (all past + current)
#     # return output + updated (k_full, v_full) as new cache
# =============================================================================

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from shared import (
    load_data,
    build_vocab,
    split_data,
    run_training,
    encode,
    decode,
)


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
CONFIG = {
    "block_size"    : 64,
    "batch_size"    : 64,
    "max_iters"     : 5000,
    "eval_interval" : 500,
    "lr"            : 3e-4,
    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
}

N_EMBD     = 128
N_HEADS    = 4
N_KV_HEADS = 2   # GQA from step_05
N_LAYERS   = 3
DROPOUT    = 0.1


# =============================================================================
# HELPER: repeat_kv (same as step_05)
# =============================================================================
def repeat_kv(kv, n_repeat):
    """Expand K/V from n_kv_heads to n_heads by repeating each KV head."""
    if n_repeat == 1:
        return kv
    B, T, n_kv, hs = kv.shape
    return kv.unsqueeze(3).expand(B, T, n_kv, n_repeat, hs).reshape(B, T, n_kv * n_repeat, hs)


# =============================================================================
# MODULE: CachedGQA — GQA with KV Cache support
# =============================================================================
class CachedGQA(nn.Module):
    """
    Group-Query Attention with optional KV cache for fast generation.

    The KEY difference from step_05's GQA:
      forward() now accepts and returns past_kv (the KV cache).

    During TRAINING:
      past_kv = None → compute attention normally over full sequence
      No cache storage needed — training uses the full sequence at once anyway
      (the causal mask handles causality during training)

    During GENERATION (one token at a time):
      past_kv = (K_past, V_past) → append new K,V to cache, use all for attention
      Returns updated (K_full, V_full) to be stored as new cache

    C# ANALOGY:
      Think of past_kv as a ConcurrentDictionary<int, (float[], float[])>
      that you pass around, adding to it each generation step.
      Like passing a StringBuilder through a method chain instead of
      creating a new string at each step.
    """

    def __init__(self, n_embd, n_heads, n_kv_heads, block_size, dropout):
        super().__init__()

        assert n_heads % n_kv_heads == 0

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_repeat   = n_heads // n_kv_heads
        self.head_size  = n_embd // n_heads
        self.scale      = math.sqrt(self.head_size)

        # Q: all n_heads query projections
        self.q_proj = nn.Linear(n_embd, n_heads    * self.head_size, bias=False)
        # K,V: fewer n_kv_heads projections (GQA)
        self.k_proj = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)

        # Causal mask for training (full sequence)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional KV cache.

        PARAMETERS:
          x: (B, T_new, n_embd)
             During training:   T_new = block_size (full sequence)
             During generation: T_new = 1          (one new token)

          past_kv: Optional[(K_past, V_past)]
             None during training (no cache)
             (K_past, V_past) during generation:
               K_past shape: (B, T_past, n_kv_heads, head_size)
               V_past shape: (B, T_past, n_kv_heads, head_size)

        RETURNS:
          (output, new_kv)
            output: (B, T_new, n_embd)
            new_kv: (K_full, V_full) — updated cache to store for next step
        """
        B, T_new, C = x.shape

        # ---- Project Q, K, V for the NEW token(s) only ----
        # During generation, x contains just 1 new token, so T_new=1
        # We project only the new token — this is the KEY efficiency gain!
        Q_new = self.q_proj(x)   # (B, T_new, n_heads * head_size)
        K_new = self.k_proj(x)   # (B, T_new, n_kv_heads * head_size)  ← small!
        V_new = self.v_proj(x)   # (B, T_new, n_kv_heads * head_size)  ← small!

        # ---- Reshape into per-head tensors ----
        Q_new = Q_new.view(B, T_new, self.n_heads,    self.head_size)
        K_new = K_new.view(B, T_new, self.n_kv_heads, self.head_size)
        V_new = V_new.view(B, T_new, self.n_kv_heads, self.head_size)

        # ---- Append new K,V to the cache ----
        if past_kv is not None:
            K_past, V_past = past_kv   # Retrieve cached K,V from previous steps

            # Concatenate: [past tokens | new token]
            # dim=1 = sequence dimension (T)
            # K_past: (B, T_past, n_kv_heads, hs)
            # K_new:  (B, 1,      n_kv_heads, hs)
            # K_full: (B, T_past+1, n_kv_heads, hs)
            K_full = torch.cat([K_past, K_new], dim=1)   # (B, T_total, n_kv_heads, hs)
            V_full = torch.cat([V_past, V_new], dim=1)   # (B, T_total, n_kv_heads, hs)
        else:
            # First call or training mode — no cache to append to
            K_full = K_new   # (B, T_new, n_kv_heads, hs)
            V_full = V_new   # (B, T_new, n_kv_heads, hs)

        # T_total = total sequence length (past + new)
        T_total = K_full.shape[1]

        # ---- Expand K,V from n_kv_heads to n_heads (GQA) ----
        K_exp = repeat_kv(K_full, self.n_repeat)   # (B, T_total, n_heads, hs)
        V_exp = repeat_kv(V_full, self.n_repeat)   # (B, T_total, n_heads, hs)

        # ---- Transpose for batched matmul: (B, n_heads, T, hs) ----
        Q = Q_new.transpose(1, 2)  # (B, n_heads, T_new, hs)
        K = K_exp.transpose(1, 2)  # (B, n_heads, T_total, hs)
        V = V_exp.transpose(1, 2)  # (B, n_heads, T_total, hs)

        # ---- Attention scores ----
        # Q: (B, n_heads, T_new, hs)
        # K: (B, n_heads, hs, T_total)   ← transposed
        # scores: (B, n_heads, T_new, T_total)
        scores = Q @ K.transpose(-2, -1) / self.scale

        # ---- Causal mask ----
        # During training: T_new = T_total, use standard lower-triangular mask
        # During generation: T_new = 1, the single query token can see all past tokens
        if T_new == T_total:
            # Training: standard causal mask
            scores = scores.masked_fill(
                self.mask[:T_total, :T_total] == 0,
                float("-inf")
            )
        # During generation (T_new=1): no mask needed!
        # The single new query naturally attends to all past K/V (all are in the past)

        # ---- Softmax and weighted sum ----
        weights = F.softmax(scores, dim=-1)           # (B, n_heads, T_new, T_total)
        weights = self.dropout(weights)
        out     = weights @ V                          # (B, n_heads, T_new, hs)

        # ---- Reassemble ----
        out = out.transpose(1, 2).contiguous()        # (B, T_new, n_heads, hs)
        out = out.view(B, T_new, self.n_heads * self.head_size)  # (B, T_new, n_embd)
        out = self.dropout(self.out_proj(out))        # (B, T_new, n_embd)

        # Return both the output AND the updated cache
        # The caller will store (K_full, V_full) and pass it next time
        return out, (K_full, V_full)


# =============================================================================
# MODULE: FeedForward and CachedBlock
# =============================================================================
class FeedForward(nn.Module):
    """Same as step_04/05."""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class CachedBlock(nn.Module):
    """
    Transformer block with KV-cache-aware GQA.
    Passes past_kv through to CachedGQA and returns updated cache.
    """
    def __init__(self, n_embd, n_heads, n_kv_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.gqa = CachedGQA(n_embd, n_heads, n_kv_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x, past_kv=None):
        """
        Forward with optional cache.
        Returns: (x_out, new_kv) where new_kv = updated (K_full, V_full)
        """
        # Apply attention and get updated cache
        attn_out, new_kv = self.gqa(self.ln1(x), past_kv)
        x = x + attn_out              # residual
        x = x + self.ffn(self.ln2(x))  # feedforward + residual
        return x, new_kv


# =============================================================================
# MODEL: KVCacheModel
# =============================================================================
class KVCacheModel(nn.Module):
    """
    GQA-based transformer that supports KV cache during generation.
    During training: same as step_05 (no cache, full sequence attention).
    During generation: uses KV cache for O(T) instead of O(T²) computation.
    """

    def __init__(self, vocab_size, n_embd, n_heads, n_kv_heads, n_layers, block_size, dropout):
        super().__init__()

        self.block_size  = block_size
        self.n_layers    = n_layers

        self.token_embd  = nn.Embedding(vocab_size, n_embd)
        self.pos_embd    = nn.Embedding(block_size, n_embd)

        # Use CachedBlock (instead of nn.Sequential, since we need to pass cache through)
        # nn.ModuleList: a list of modules that PyTorch tracks (like ModuleList in C#)
        self.blocks = nn.ModuleList([
            CachedBlock(n_embd, n_heads, n_kv_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        self.final_ln    = nn.LayerNorm(n_embd)
        self.output_proj = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None, past_kvs=None):
        """
        Forward pass with optional KV cache.

        PARAMETERS:
          x: (B, T_new)
          targets: (B, T_new) or None
          past_kvs: list of (K, V) tuples, one per layer. None = no cache.

        RETURNS:
          (logits, loss, new_past_kvs)
          new_past_kvs: updated list of (K, V) tuples for next generation step
        """
        B, T_new = x.shape

        # Determine starting position for positional embedding
        # If using cache, we've already processed T_past tokens
        # New tokens start at position T_past
        if past_kvs is not None and past_kvs[0] is not None:
            T_past = past_kvs[0][0].shape[1]   # Past K has shape (B, T_past, n_kv, hs)
        else:
            T_past = 0   # No past tokens — start from position 0

        # Token embeddings for new tokens
        tok = self.token_embd(x)   # (B, T_new, n_embd)

        # Positional embeddings: start from T_past, not 0
        # T_past + T_new might exceed block_size during long generation
        # We clamp to block_size - 1 to stay valid
        positions = torch.arange(T_past, T_past + T_new, device=x.device)
        positions = positions.clamp(max=self.block_size - 1)  # safety clamp
        pos = self.pos_embd(positions)   # (T_new, n_embd)

        h = tok + pos   # (B, T_new, n_embd)

        # Run through blocks, collecting updated KV caches
        new_past_kvs = []   # Will hold updated (K, V) for each layer

        for i, block in enumerate(self.blocks):
            # Get this layer's past KV (or None if no cache)
            layer_past = past_kvs[i] if past_kvs is not None else None

            # Forward through block
            h, new_kv = block(h, layer_past)

            # Collect the updated cache for this layer
            new_past_kvs.append(new_kv)

        h = self.final_ln(h)
        logits = self.output_proj(h)   # (B, T_new, vocab_size)

        if targets is None:
            loss = None
        else:
            B2, T2, V = logits.shape
            loss = F.cross_entropy(logits.view(B2 * T2, V), targets.view(B2 * T2))

        return logits, loss, new_past_kvs


# =============================================================================
# GENERATION FUNCTIONS — with and without cache
# =============================================================================

def generate_without_cache(model, seed_ids, max_new, block_size, device, temperature=0.8):
    """
    Generate tokens WITHOUT KV cache — recomputes all K,V at every step.
    O(T²) time complexity. Correct but slow.

    Used for benchmarking to show the speedup FROM KV cache.
    """
    model.eval()

    # Context starts as the seed tokens
    ctx = torch.tensor(seed_ids, dtype=torch.long).unsqueeze(0).to(device)
    generated = list(seed_ids)

    with torch.no_grad():
        for _ in range(max_new):
            # Crop to block_size (model can't process longer sequences)
            ctx_cropped = ctx[:, -block_size:]

            # Full forward pass — recomputes K,V for ALL tokens every time!
            logits, _loss, _kvs = model(ctx_cropped, past_kvs=None)

            # Take last token's logits
            logits_last = logits[:, -1, :] / temperature
            probs       = F.softmax(logits_last, dim=-1)
            next_id     = torch.multinomial(probs, 1)

            ctx = torch.cat([ctx, next_id], dim=1)
            generated.append(next_id.item())

    model.train()
    return generated


def generate_with_cache(model, seed_ids, max_new, block_size, device, temperature=0.8):
    """
    Generate tokens WITH KV cache — computes K,V only for the new token each step.
    O(T) time complexity. Faster!

    STRATEGY:
    1. Process seed tokens all at once (warm up the cache)
    2. For each new token:
       a. Feed only the 1 new token to the model
       b. Pass past_kvs (the cache) to skip recomputation
       c. Get output for the 1 new token
       d. Update the cache with the new K,V
    """
    model.eval()

    generated = list(seed_ids)

    with torch.no_grad():
        # ---- Phase 1: Process the seed sequence ----
        # Feed all seed tokens at once to warm up the cache
        seed_tensor = torch.tensor(seed_ids, dtype=torch.long).unsqueeze(0).to(device)
        _logits, _loss, past_kvs = model(seed_tensor, past_kvs=None)
        # past_kvs now contains K,V for all seed tokens

        # Get the last token's logits to predict the first NEW token
        last_logits = _logits[:, -1, :] / temperature
        probs       = F.softmax(last_logits, dim=-1)
        next_id     = torch.multinomial(probs, 1)
        generated.append(next_id.item())

        # ---- Phase 2: Generate new tokens one by one ----
        for _ in range(max_new - 1):
            # Feed ONLY the 1 new token (not the whole sequence!)
            # C# analogy: we're only processing the delta, not the full state
            x_single = next_id   # (1, 1) — batch=1, seq_len=1

            # Total sequence so far
            total_len = len(generated)

            # Don't go beyond block_size
            if total_len > block_size:
                # Truncate past_kvs to fit within block_size
                # For simplicity, skip generation beyond block_size
                break

            # Forward with cache — only computes K,V for the 1 new token!
            logits, _loss, past_kvs = model(x_single, past_kvs=past_kvs)

            # Get the prediction for next token
            last_logits = logits[:, -1, :] / temperature
            probs       = F.softmax(last_logits, dim=-1)
            next_id     = torch.multinomial(probs, 1)
            generated.append(next_id.item())

    model.train()
    return generated


# =============================================================================
# BENCHMARK
# =============================================================================
def benchmark_generation(model, char2idx, idx2char, device, block_size, n_tokens=200):
    """
    Compare generation speed: with vs without KV cache.
    Prints tokens/sec for each method and the speedup ratio.
    """
    seed = "The "
    seed_ids = encode(seed, char2idx)

    print(f"\n  Benchmarking: generating {n_tokens} tokens from seed '{seed}'")
    print()

    # ---- Without cache ----
    print("  Running WITHOUT KV cache (slow, O(T²))...")
    t0 = time.time()
    ids_no_cache = generate_without_cache(model, seed_ids, n_tokens, block_size, device)
    t_no_cache   = time.time() - t0
    tps_no_cache = n_tokens / t_no_cache  # tokens per second

    # ---- With cache ----
    print("  Running WITH KV cache (fast, O(T))...")
    t0 = time.time()
    ids_cache = generate_with_cache(model, seed_ids, n_tokens, block_size, device)
    t_cache   = time.time() - t0
    tps_cache = n_tokens / t_cache

    speedup = t_no_cache / t_cache  # How many times faster?

    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │  GENERATION BENCHMARK RESULTS                     │")
    print("  ├──────────────────────────────────────────────────┤")
    print(f"  │  Without KV cache: {t_no_cache:.2f}s  ({tps_no_cache:.1f} tokens/sec)   │")
    print(f"  │  With    KV cache: {t_cache:.2f}s  ({tps_cache:.1f} tokens/sec)   │")
    print(f"  │  Speedup         : {speedup:.1f}x faster with cache!          │")
    print("  └──────────────────────────────────────────────────┘")
    print()

    # Show that both methods generate equivalent text
    print("  Text generated WITHOUT cache (first 200 chars):")
    print(f"  {decode(ids_no_cache, idx2char)[:200]}")
    print()
    print("  Text generated WITH cache (first 200 chars):")
    print(f"  {decode(ids_cache, idx2char)[:200]}")
    print()

    return speedup


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  STEP 6: KV Cache                                        ║")
    print("║  What's new   : Cache K,V → O(T) generation             ║")
    print("║  Architecture : Same GQA model, smarter generation loop ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    print("WHAT THIS STEP TEACHES:")
    print("  - Without KV cache: generation is O(T²) — quadratic time")
    print("  - With KV cache: generation is O(T) — linear time")
    print("  - Cache the K and V from past tokens, only compute K,V for new token")
    print("  - ALL production LLMs (GPT-4, LLaMA, Gemini) use KV cache!")
    print()
    print("C# ANALOGY:")
    print("  KV cache = Dictionary<int, (float[] Key, float[] Value)>")
    print("  Like memoization: compute once, reuse many times.")
    print("  Same pattern as .memoize() in functional C# libraries.")
    print()

    # ---- Data ----
    print("[ Loading data ]")
    text = load_data(max_chars=10_000_000)

    print()
    print("[ Building vocabulary ]")
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(idx2char)

    print()
    print("[ Splitting data ]")
    train_data, val_data = split_data(text, char2idx)

    # ---- Model ----
    print()
    print("[ Creating model ]")
    model = KVCacheModel(
        vocab_size  = vocab_size,
        n_embd      = N_EMBD,
        n_heads     = N_HEADS,
        n_kv_heads  = N_KV_HEADS,
        n_layers    = N_LAYERS,
        block_size  = CONFIG["block_size"],
        dropout     = DROPOUT,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  KV cache: stores K,V for each of {N_LAYERS} layers as tokens are generated")

    # ---- Train ----
    # Training uses the standard full-sequence forward pass (past_kvs=None)
    # The KV cache is only used during generation
    print()
    print("[ Training ]")
    # We use run_training from shared.py — it calls model(xb, yb) which calls forward()
    # with past_kvs=None by default (training mode)
    final_val_loss = run_training(
        model_name = "KVCache",
        model      = model,
        train_data = train_data,
        val_data   = val_data,
        char2idx   = char2idx,
        idx2char   = idx2char,
        config     = CONFIG,
    )

    # ---- Benchmark! ----
    print()
    print("[ Benchmarking KV Cache Speedup ]")
    device = CONFIG["device"]
    speedup = benchmark_generation(
        model      = model,
        char2idx   = char2idx,
        idx2char   = idx2char,
        device     = device,
        block_size = CONFIG["block_size"],
        n_tokens   = 100,   # Generate 100 tokens (fast enough on CPU)
    )

    # ---- Summary ----
    print("=" * 60)
    print(f"  FINAL VAL LOSS: {final_val_loss:.4f}")
    print(f"  KV Cache Speedup: ~{speedup:.1f}x faster generation")
    print()
    print("  KEY INSIGHT:")
    print("  KV cache doesn't change model quality AT ALL.")
    print("  It only makes GENERATION faster.")
    print("  Training still uses full-sequence attention (no cache needed).")
    print("  This is exactly how ChatGPT, Claude, LLaMA work in production!")
    print()
    print("=" * 60)
    print("  Next step: python step_07_rope.py")
    print("  What's next: RoPE — better positional encoding than learned embeddings")
    print("=" * 60)
