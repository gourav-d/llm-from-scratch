"""
Module 19 - Mixture of Experts
Example 05: Mini MoE GPT -- Complete Working Model

This is the MAIN PROJECT of Module 19.
You will build a complete GPT-style language model where every
transformer block's FFN is replaced by a Mixture of Experts layer.

GLOSSARY
--------
Expert       : Standard FFN layer (Linear -> GELU -> Linear).
               There are N of these, but only K run per token.
               Identical to the FFN in M05 and M18.

Router       : One Linear layer that assigns tokens to experts.
               Output: probability distribution over N experts.
               Learns WHICH expert is best for WHICH token type.

MoELayer     : Router + N Experts combined.
               Drop-in replacement for the standard FFN in a transformer block.
               Input/output shape: [batch, seq, d_model] (same as FFN).

CausalSelfAttention: Multi-head attention with causal mask.
               Same as the attention in M05 and M18.
               "Causal" = each token can only look at PREVIOUS tokens (left-to-right).

MoETransformerBlock: One transformer block with MoELayer instead of standard FFN.
               Components: LayerNorm -> Attention (residual) -> LayerNorm -> MoELayer (residual).

MoEGPT       : Full language model: embeddings -> N MoETransformerBlocks -> output head.
               Character-level: predicts the next character given previous characters.

Auxiliary loss: Extra loss term that penalizes unequal expert usage.
               total_loss = cross_entropy + alpha * aux_loss
               Forces the router to spread tokens across all N experts.

Character-level LM: Language model that predicts one character at a time.
               Vocabulary = 27 characters (a-z and space).
               Easy to train quickly on tiny data (no tokenizer needed).
"""

import torch                          # PyTorch: main deep learning library
import torch.nn as nn                 # nn: building blocks like Linear, LayerNorm, etc.
import torch.nn.functional as F       # F: functions like softmax, gelu, cross_entropy
import time                           # time: measure how long training takes

torch.manual_seed(42)                 # fixed random seed -- same result every run

print("=" * 65)
print("Example 05: Mini MoE GPT -- Full Working Model")
print("=" * 65)
print()

# ============================================================
# CONFIGURATION
# Tiny numbers so model trains on CPU in under 60 seconds
# ============================================================
VOCAB_SIZE   = 27     # 26 letters + 1 space character
D_MODEL      = 64     # embedding dimension (each token = 64-element vector)
NUM_HEADS    = 4      # attention heads (head_dim = 64 / 4 = 16)
D_FF         = 128    # hidden dim inside each expert (2x D_MODEL)
NUM_LAYERS   = 2      # number of MoE transformer blocks stacked
NUM_EXPERTS  = 4      # experts per MoE layer
TOP_K        = 2      # experts activated per token
MAX_SEQ_LEN  = 32     # maximum sequence length (context window)
BATCH_SIZE   = 8      # sequences per training batch
NUM_STEPS    = 200    # training steps (increase for better quality)
LR           = 3e-3   # learning rate (Adam optimizer)
ALPHA        = 0.01   # weight of auxiliary load-balancing loss

print(f"Model config: d_model={D_MODEL}, {NUM_LAYERS} layers, "
      f"{NUM_EXPERTS} experts, top-{TOP_K} routing")
print(f"Training: {NUM_STEPS} steps, batch={BATCH_SIZE}, lr={LR}")
print()

# ============================================================
# STEP 1: Build Toy Dataset (Character-Level Text)
# ============================================================
print("--- Step 1: Preparing Character-Level Dataset ---")

# Vocabulary: 26 lowercase letters + space
# Using ASCII only (cp1252 safe -- no Unicode)
CHARS = "abcdefghijklmnopqrstuvwxyz "    # 27 characters
assert len(CHARS) == VOCAB_SIZE           # confirm 27 characters
char_to_idx = {c: i for i, c in enumerate(CHARS)}  # dict: character -> integer index
idx_to_char = {i: c for i, c in enumerate(CHARS)}  # dict: integer -> character

# Training text: short sentences repeated many times so the model can memorize patterns
# A real model trains on gigabytes of text; this tiny example shows the MECHANISM
raw_text = (
    "hello world this is a test of the mixture of experts model "
    "the quick brown fox jumps over the lazy dog "
    "learning to build llms from scratch is fun "
    "experts help language models scale efficiently "
) * 20   # repeat 20 times so we have enough data for training

# Keep only characters in our vocabulary (discard anything else)
clean_text = "".join(c for c in raw_text.lower() if c in char_to_idx)

# Convert text to integer indices (like tokenization, but at character level)
text_ids = [char_to_idx[c] for c in clean_text]   # list of integers 0..26
text_tensor = torch.tensor(text_ids, dtype=torch.long)  # convert to PyTorch tensor

print(f"Vocabulary: '{CHARS}'")
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Training text length: {len(text_ids)} characters")
print(f"Sample encoding: 'hello' -> {[char_to_idx[c] for c in 'hello']}")
print()

def get_batch(text_tensor, batch_size, seq_len):
    """
    Randomly sample a batch of (input, target) pairs from the text.
    Input:  [batch_size, seq_len] -- characters at positions i..i+seq_len-1
    Target: [batch_size, seq_len] -- characters at positions i+1..i+seq_len
    The target is the input shifted by one position (next-character prediction).
    """
    max_start = len(text_tensor) - seq_len - 1        # latest valid start index
    starts = torch.randint(0, max_start, (batch_size,))  # random start positions
    # x: input sequences (positions i to i+seq_len-1)
    x = torch.stack([text_tensor[s:s+seq_len]   for s in starts])  # [B, S]
    # y: target sequences (positions i+1 to i+seq_len), shifted by 1
    y = torch.stack([text_tensor[s+1:s+seq_len+1] for s in starts])  # [B, S]
    return x, y

# ============================================================
# STEP 2: Build Model Components
# ============================================================
print("--- Step 2: Building Model Components ---")
print()

# ---- Component 1: Expert (Standard FFN) ----
class Expert(nn.Module):
    """
    One expert FFN: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model).
    This is IDENTICAL to the FFN in standard transformers (M05, M18).
    The only thing special: there are N of these with INDEPENDENT weights.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)    # expand: 64 -> 128
        self.fc2 = nn.Linear(d_ff, d_model)    # contract: 128 -> 64

    def forward(self, x):
        x = F.gelu(self.fc1(x))               # expand + GELU nonlinearity
        x = self.fc2(x)                        # contract back to d_model
        return x                               # shape: same as input

# ---- Component 2: Router ----
class Router(nn.Module):
    """
    Router: one Linear layer that learns which experts handle each token.
    Input:  token embeddings [batch, seq, d_model]
    Output: top-K expert probabilities + indices, plus full probs for aux_loss
    """
    def __init__(self, d_model, num_experts):
        super().__init__()
        # gate: learns to score each expert for each token
        # bias=False: standard practice in MoE routers
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x, top_k):
        logits  = self.gate(x)                                    # [B, S, num_experts]
        probs   = F.softmax(logits, dim=-1)                       # [B, S, num_experts], sums to 1
        topk_w, topk_idx = torch.topk(probs, k=top_k, dim=-1)    # top-K selection
        # renormalize: K selected weights must sum to 1 for valid weighted average
        topk_w  = topk_w / topk_w.sum(dim=-1, keepdim=True)      # renorm over K dim
        return topk_w, topk_idx, probs  # all_probs needed for auxiliary loss

# ---- Component 3: MoELayer ----
class MoELayer(nn.Module):
    """
    MoE Layer: combines Router + N Experts.
    Replaces the standard FFN in each transformer block.
    Input/output shape: [batch, seq_len, d_model] (drop-in FFN replacement).
    """
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts                 # N total experts
        self.top_k = top_k                            # K experts to run per token
        # N independent expert FFNs stored in a ModuleList (PyTorch tracks all params)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_experts)]
        )
        self.router = Router(d_model, num_experts)    # routing decision maker

    def forward(self, x):
        B, S, D = x.shape                             # batch, seq_len, d_model

        # Routing: get which experts handle each token + their weights
        topk_weights, topk_indices, all_probs = self.router(x, self.top_k)
        # topk_weights:  [B, S, K] -- normalized weights for selected experts
        # topk_indices:  [B, S, K] -- which expert indices were selected
        # all_probs:     [B, S, N] -- full softmax output (for aux_loss)

        # Flatten batch and sequence dims for expert dispatch
        x_flat   = x.view(B * S, D)                  # [B*S, D] -- all tokens as rows
        w_flat   = topk_weights.view(B * S, self.top_k)   # [B*S, K]
        idx_flat = topk_indices.view(B * S, self.top_k)   # [B*S, K]

        # Output accumulator: starts at zero, we add weighted expert outputs
        output = torch.zeros_like(x_flat)             # [B*S, D]

        # For each of the K expert slots, for each expert, find its tokens and run it
        for k_pos in range(self.top_k):              # k_pos = 0, 1, ..., K-1
            for eid in range(self.num_experts):      # eid = 0, 1, ..., N-1
                # Boolean mask: which tokens chose expert eid at slot k_pos?
                mask = (idx_flat[:, k_pos] == eid)   # shape [B*S], True for matching tokens
                if not mask.any():                   # if no tokens chose this expert here, skip
                    continue
                tokens_in = x_flat[mask]             # [num_selected, D] -- tokens for this expert
                expert_out = self.experts[eid](tokens_in)  # run expert on selected tokens
                weight = w_flat[mask, k_pos].unsqueeze(-1) # [num_selected, 1] -- scale factor
                output[mask] += weight * expert_out  # weighted add to accumulator

        return output.view(B, S, D), all_probs       # reshape + return probs for aux_loss

# ---- Component 4: Causal Self-Attention ----
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    "Causal" = each position can only attend to positions BEFORE it (left-to-right).
    This is the SAME attention you built in M05 and M18 -- no change for MoE.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads                    # H attention heads
        self.head_dim  = d_model // num_heads         # dimension per head

        # Combined Q, K, V projection (more efficient than 3 separate Linear layers)
        self.qkv  = nn.Linear(d_model, 3 * d_model)  # project to Q+K+V at once
        self.proj = nn.Linear(d_model, d_model)       # output projection

    def forward(self, x):
        B, S, D = x.shape                            # batch, seq_len, d_model

        # Project to Q, K, V all at once then split
        qkv = self.qkv(x)                           # [B, S, 3*D]
        q, k, v = qkv.split(D, dim=-1)              # each shape: [B, S, D]

        # Reshape to multi-head format: split D into (num_heads, head_dim)
        # then move heads before seq_len for batched matmul
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, Hd]
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, Hd]
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, Hd]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5                # 1/sqrt(head_dim) -- prevents large dot products
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, S, S] -- all query-key pairs

        # Causal mask: prevent attending to FUTURE tokens
        # triu = upper triangular matrix (True above diagonal = future positions)
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device), diagonal=1
        ).bool()                                     # True = position is in the future
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))  # future -> -inf

        attn_weights = F.softmax(attn_scores, dim=-1)   # [B, H, S, S] -- softmax over keys

        # Weighted sum of values
        out = attn_weights @ v                       # [B, H, S, Hd]

        # Concatenate heads: (B, H, S, Hd) -> (B, S, H*Hd) = (B, S, D)
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        out = self.proj(out)                         # final linear projection [B, S, D]
        return out

# ---- Component 5: MoE Transformer Block ----
class MoETransformerBlock(nn.Module):
    """
    One transformer block with MoE instead of standard FFN.
    Structure:
      x -> LayerNorm -> Attention -> + x (residual) -> LayerNorm -> MoE -> + x (residual)
    The ONLY difference from a standard transformer block: MoELayer replaces FFN.
    """
    def __init__(self, d_model, num_heads, d_ff, num_experts, top_k):
        super().__init__()
        self.norm1     = nn.LayerNorm(d_model)                         # pre-attention norm
        self.attn      = CausalSelfAttention(d_model, num_heads)       # self-attention
        self.norm2     = nn.LayerNorm(d_model)                         # pre-MoE norm
        self.moe       = MoELayer(d_model, d_ff, num_experts, top_k)  # MoE replaces FFN

    def forward(self, x):
        # Pre-norm attention with residual connection
        x = x + self.attn(self.norm1(x))              # norm -> attn -> add back to x

        # Pre-norm MoE with residual connection
        moe_out, all_probs = self.moe(self.norm2(x))  # norm -> MoE (returns probs too)
        x = x + moe_out                               # add MoE output back to x

        return x, all_probs    # return all_probs for auxiliary loss in training loop

# ---- Component 6: Full MoE GPT Model ----
class MoEGPT(nn.Module):
    """
    Complete GPT-style language model with MoE layers.
    Takes: token indices [batch, seq_len]
    Returns: logits [batch, seq_len, vocab_size] and router probs for aux_loss
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, num_experts, top_k, max_seq_len):
        super().__init__()

        # Token embedding: converts each token index to a d_model-dimensional vector
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Positional embedding: adds position information (learned, like in GPT-2)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)

        # Stack of MoE transformer blocks
        self.blocks    = nn.ModuleList([
            MoETransformerBlock(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])

        # Final layer normalization (before output head)
        self.final_norm = nn.LayerNorm(d_model)

        # Output head: convert d_model representation to vocabulary logits
        # bias=False: common practice for weight-tied output heads
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        B, S = token_ids.shape                        # batch size, sequence length

        # Create position indices: [0, 1, 2, ..., S-1] for each item in batch
        positions = torch.arange(S, device=token_ids.device)  # [S]

        # Embed tokens and add positional encoding
        x = self.token_emb(token_ids)                # [B, S, D] -- token embeddings
        x = x + self.pos_emb(positions)              # [B, S, D] -- add position info

        # Pass through all transformer blocks
        all_router_probs = []                         # collect router probs from each block
        for block in self.blocks:
            x, block_probs = block(x)                # forward through one block
            all_router_probs.append(block_probs)     # save router probs for aux_loss

        # Final normalization
        x = self.final_norm(x)                       # [B, S, D]

        # Convert to vocabulary logits
        logits = self.output_head(x)                 # [B, S, vocab_size]

        return logits, all_router_probs              # return logits + all router probs

# ============================================================
# STEP 3: Auxiliary Loss Function
# ============================================================

def compute_aux_loss(all_router_probs, num_experts):
    """
    Compute Switch Transformer auxiliary loss for load balancing.
    Called once per training step using router probs from ALL blocks.

    Args:
        all_router_probs : list of tensors, one per block, shape [B, S, N]
        num_experts      : N (number of experts)

    Returns:
        aux_loss : scalar tensor (differentiable through P_i)
    """
    total_aux = 0.0                                  # accumulate aux loss across blocks

    for probs in all_router_probs:                   # one probs tensor per transformer block
        B, S, N = probs.shape                        # batch, seq, num_experts

        # P_i: average router probability for expert i (soft, differentiable)
        P = probs.mean(dim=[0, 1])                   # [N] -- average over batch and seq

        # f_i: hard fraction of tokens routed to each expert (non-differentiable)
        # We use argmax (primary expert per token) to compute hard counts
        primary_expert = probs.argmax(dim=-1)        # [B, S] -- which expert got highest prob
        primary_flat   = primary_expert.view(-1)     # [B*S] -- flatten for counting

        # Count fraction of tokens for each expert
        f = torch.zeros(N, device=probs.device)      # [N]
        for i in range(N):
            f[i] = (primary_flat == i).float().mean()  # fraction of tokens -> expert i

        # aux_loss for this layer: N * sum(f_i * P_i)
        # Penalizes when BOTH f_i is large (many tokens) AND P_i is large (high confidence)
        layer_aux = N * (f * P).sum()                # scalar
        total_aux = total_aux + layer_aux            # accumulate

    # Average across layers (so aux_loss scale doesn't depend on num_layers)
    return total_aux / len(all_router_probs)         # scalar tensor

# ============================================================
# STEP 4: Create Model and Optimizer
# ============================================================
print("--- Step 4: Creating Model and Optimizer ---")

model = MoEGPT(
    vocab_size   = VOCAB_SIZE,
    d_model      = D_MODEL,
    num_heads    = NUM_HEADS,
    d_ff         = D_FF,
    num_layers   = NUM_LAYERS,
    num_experts  = NUM_EXPERTS,
    top_k        = TOP_K,
    max_seq_len  = MAX_SEQ_LEN,
)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())   # sum all params
# Count parameters per expert (for active params calculation)
params_per_expert = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
# Active params: K experts per block + everything else
expert_params_all = sum(p.numel() for p in model.blocks[0].moe.experts.parameters())
non_expert_params = total_params - NUM_LAYERS * expert_params_all  # embedding, attn, norms, head
active_expert_params = NUM_LAYERS * TOP_K * params_per_expert       # K active experts per layer

print(f"Total parameters:  {total_params:,}")
print(f"Active params/tok: {non_expert_params + active_expert_params:,}  "
      f"(non-expert: {non_expert_params:,} + active experts: {active_expert_params:,})")
print(f"Inactive params:   {total_params - (non_expert_params + active_expert_params):,}  "
      f"(exist in memory but do NOT run per token)")
print()

# Adam optimizer: standard choice for transformer training
# weight_decay=0.01 is mild regularization (like L2 in .NET ML)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

# ============================================================
# STEP 5: Training Loop
# ============================================================
print("--- Step 5: Training Loop ---")
print()
print(f"{'Step':>6} | {'main_loss':>10} | {'aux_loss':>10} | {'total_loss':>11} | {'elapsed':>8}")
print("-" * 58)

start_time = time.time()    # record training start time

losses = []                  # track losses for final plot (ASCII)

for step in range(NUM_STEPS):
    model.train()            # set model to training mode (enables dropout if any)

    # Sample a random batch from the training text
    x, y = get_batch(text_tensor, BATCH_SIZE, MAX_SEQ_LEN)
    # x: [batch, seq_len] -- input token indices
    # y: [batch, seq_len] -- target token indices (shifted by 1)

    # Forward pass
    logits, all_router_probs = model(x)
    # logits: [batch, seq_len, vocab_size]

    # Main loss: cross-entropy (predict next character)
    # Reshape: [B, S, V] -> [B*S, V] and [B, S] -> [B*S]
    main_loss = F.cross_entropy(
        logits.view(-1, VOCAB_SIZE),   # [B*S, V] -- predictions for every position
        y.view(-1)                     # [B*S]    -- true next character
    )

    # Auxiliary loss: load balancing across experts
    aux_loss = compute_aux_loss(all_router_probs, NUM_EXPERTS)

    # Total loss: main task + load balancing
    total_loss = main_loss + ALPHA * aux_loss

    # Backward pass: compute gradients
    optimizer.zero_grad()             # clear gradients from previous step
    total_loss.backward()             # compute gradients for all parameters

    # Gradient clipping: prevent exploding gradients (common in transformer training)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update parameters
    optimizer.step()                  # apply gradients (move parameters in direction of lower loss)

    # Record and print progress
    losses.append(main_loss.item())   # save main loss for later analysis

    if step % 20 == 0 or step == NUM_STEPS - 1:    # print every 20 steps + final step
        elapsed = time.time() - start_time           # seconds since start
        print(f"{step:>6} | {main_loss.item():>10.4f} | {aux_loss.item():>10.4f} | "
              f"{total_loss.item():>11.4f} | {elapsed:>7.1f}s")

# ============================================================
# STEP 6: Evaluate -- Generate Text
# ============================================================
print()
print("--- Step 6: Text Generation (Greedy Decoding) ---")
print()

model.eval()                          # set model to evaluation mode (no dropout, etc.)

def generate(model, start_text, max_new_chars=40):
    """
    Generate new characters autoregressively.
    Greedy decoding: always pick the character with the highest logit.

    Args:
        model         : trained MoEGPT model
        start_text    : initial "prompt" string (must be in CHARS vocabulary)
        max_new_chars : how many new characters to generate

    Returns:
        generated string
    """
    with torch.no_grad():              # no gradient computation during generation
        # Convert start text to indices
        context = [char_to_idx.get(c, char_to_idx[' ']) for c in start_text.lower()]
        context = torch.tensor(context, dtype=torch.long).unsqueeze(0)  # [1, len]

        for _ in range(max_new_chars):
            # Truncate context to max_seq_len if too long
            ctx = context[:, -MAX_SEQ_LEN:]              # [1, min(len, MAX_SEQ_LEN)]

            # Get model predictions
            logits, _ = model(ctx)                       # logits: [1, S, V]
            last_logits = logits[0, -1, :]               # [V] -- prediction for NEXT char
            next_idx = last_logits.argmax().item()       # greedy: pick highest logit
            next_char = idx_to_char[next_idx]            # convert index back to character

            # Append new character to context
            next_tensor = torch.tensor([[next_idx]], dtype=torch.long)  # [1, 1]
            context = torch.cat([context, next_tensor], dim=1)          # extend context

        # Decode full context (including original start text) to string
        all_indices = context[0].tolist()
        return "".join(idx_to_char[i] for i in all_indices)

# Generate with different starting texts
prompts = ["hello", "the ", "learn"]
for prompt in prompts:
    generated = generate(model, prompt, max_new_chars=35)
    print(f"Prompt '{prompt}' -> '{generated}'")

print()

# ============================================================
# STEP 7: Routing Statistics -- Which Experts Are Used?
# ============================================================
print("--- Step 7: Expert Routing Statistics ---")
print()

model.eval()

# Run model on a sample batch to inspect routing
sample_x, _ = get_batch(text_tensor, batch_size=4, seq_len=16)  # small sample
with torch.no_grad():
    _, all_probs = model(sample_x)    # get router probabilities

print(f"Analyzing routing for {4 * 16} tokens across {NUM_LAYERS} blocks...")
print()

for layer_idx, probs in enumerate(all_probs):    # one set of probs per block
    # probs shape: [B, S, num_experts]
    primary = probs.argmax(dim=-1)               # [B, S] -- primary expert per token
    primary_flat = primary.view(-1)              # [B*S]

    # Compute usage fraction per expert
    usage = [(primary_flat == i).float().mean().item() for i in range(NUM_EXPERTS)]

    print(f"Block {layer_idx} expert usage:")
    for eid, frac in enumerate(usage):
        bar = "#" * int(frac * 30)               # ASCII bar: 30 chars = 100%
        print(f"  Expert {eid}: |{bar:<30}| {frac:.1%}")
    print()

print("If experts show roughly equal usage (~25% each), load balancing is working!")
print()

# ============================================================
# STEP 8: ASCII Loss Curve
# ============================================================
print("--- Step 8: Loss Curve (ASCII) ---")
print()
print("main_loss over training (sampled every 20 steps):")
print()

# Print a simple ASCII loss plot
sampled_losses = losses[::20] + [losses[-1]]    # every 20 steps + final
max_loss = max(sampled_losses)                   # for scaling
min_loss = min(sampled_losses)

for i, loss in enumerate(sampled_losses):
    step_num = min(i * 20, NUM_STEPS - 1)       # which training step this is
    bar_len  = int((loss - min_loss) / (max_loss - min_loss + 1e-8) * 30)  # scale to 30
    bar = "#" * bar_len                          # ASCII bar (longer = higher loss)
    print(f"  Step {step_num:>4}: {loss:.4f} |{bar:<30}|")

print()
print(f"Starting loss: {losses[0]:.4f}")
print(f"Final loss:    {losses[-1]:.4f}")
if losses[-1] < losses[0]:                      # check loss actually decreased
    print(f"Loss decreased by {losses[0] - losses[-1]:.4f} -- training worked!")
else:
    print("Warning: loss did not decrease. Try more steps or a higher learning rate.")

print()
total_time = time.time() - start_time
print(f"Total training time: {total_time:.1f} seconds")

print()
print("=" * 65)
print("Example 05 complete!")
print("You built a full Mini MoE GPT from scratch:")
print("  Expert -> Router -> MoELayer -> MoETransformerBlock -> MoEGPT")
print("Trained on character-level text with auxiliary load balancing loss.")
print("The model predicts the next character using only K of N experts.")
print("=" * 65)
