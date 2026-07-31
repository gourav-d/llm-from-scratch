"""
Module 19 - Mixture of Experts
Example 04: MoE vs Dense -- Parameters, Compute, and Capacity

GLOSSARY
--------
Dense FFN    : Single Feed-Forward Network where ALL tokens use ALL weights.
               Active params per token = 100% of total params.
               Simple, no routing overhead, no load balancing needed.

MoE FFN      : N expert FFNs where each token uses only K of them.
               Total params = N * per_expert_params.
               Active params per token = K * per_expert_params.
               Compute per token same as K dense FFNs.

FLOP count   : Number of floating-point multiply-accumulate operations.
               Linear(in, out) with input [batch, seq, in]:
                 FLOPs = batch * seq * in * out (one matmul).
               Lower FLOPs = cheaper computation.

Parameter efficiency: Quality achieved per unit of active computation.
               MoE achieves high quality with fewer active params
               because it has more TOTAL params to store knowledge.

Expert capacity: Hard limit on tokens per expert per batch.
               capacity = (total_tokens / num_experts) * capacity_factor
               Tokens over capacity are dropped (use residual = identity).

Capacity factor: Buffer above the "fair share" of tokens.
               1.0 = exactly equal share (strict, some tokens may drop).
               1.25 = 25% buffer (common, reduces dropped tokens).
               2.0 = 100% buffer (generous, almost no drops).

Token dropping: When an expert is full, extra tokens bypass it.
               Their output = their input (residual connection).
               The model is robust to occasional drops during training.
"""

import torch                        # PyTorch: deep learning framework
import torch.nn as nn               # nn: neural network building blocks
import torch.nn.functional as F     # F: stateless functions

torch.manual_seed(42)               # reproducible results

print("=" * 60)
print("Example 04: MoE vs Dense -- Parameters, Compute, Capacity")
print("=" * 60)
print()

# ============================================================
# SECTION 1: Parameter Count Comparison
# ============================================================
print("--- SECTION 1: Parameter Count Comparison ---")
print()

# Use realistic small-model dimensions
d_model      = 256    # embedding dimension
d_ff         = 1024   # hidden dimension in FFN (4x d_model)
num_experts  = 8      # 8 experts in MoE
top_k        = 2      # top-2 routing (K=2)

print(f"Configuration: d_model={d_model}, d_ff={d_ff}")
print(f"MoE: num_experts={num_experts}, top_k={top_k}")
print()

# ---- Dense FFN ----
# Two linear layers: fc1 (d_model -> d_ff) and fc2 (d_ff -> d_model)
# Each nn.Linear has weight (out x in) and bias (out)
class DenseFFN(nn.Module):
    """Standard dense FFN: all tokens use the same weights."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # expand
        self.fc2 = nn.Linear(d_ff, d_model)   # contract

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))  # fc1 -> GELU -> fc2


dense_ffn = DenseFFN(d_model, d_ff)          # create one dense FFN

# Count parameters
dense_total_params = sum(p.numel() for p in dense_ffn.parameters())
dense_active_params = dense_total_params     # always 100% active

print(f"Dense FFN:")
print(f"  fc1 params: {d_ff} x {d_model} + {d_ff} = {d_ff * d_model + d_ff:,}")
print(f"  fc2 params: {d_model} x {d_ff} + {d_model} = {d_model * d_ff + d_model:,}")
print(f"  Total params:  {dense_total_params:,}")
print(f"  Active params: {dense_active_params:,}  (100% -- all tokens use all weights)")
print()

# ---- MoE FFN ----
class Expert(nn.Module):
    """One expert FFN: identical structure to DenseFFN."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # same as dense fc1
        self.fc2 = nn.Linear(d_ff, d_model)   # same as dense fc2

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))  # same computation


# Create all N experts
expert_list = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
# Router: one extra Linear layer
router_linear = nn.Linear(d_model, num_experts, bias=False)  # d_model -> num_experts

# Count parameters
params_per_expert = sum(p.numel() for p in expert_list[0].parameters())
moe_expert_params = sum(p.numel() for p in expert_list.parameters())  # all N experts
moe_router_params = sum(p.numel() for p in router_linear.parameters())
moe_total_params  = moe_expert_params + moe_router_params
moe_active_params = top_k * params_per_expert + moe_router_params  # K experts + router

print(f"MoE FFN (N={num_experts} experts, K={top_k}):")
print(f"  Params per expert: {params_per_expert:,}")
print(f"  Total expert params ({num_experts} experts): {moe_expert_params:,}")
print(f"  Router params:     {moe_router_params:,}")
print(f"  Total MoE params:  {moe_total_params:,}")
print(f"  Active params/token: {moe_active_params:,}  (only K={top_k} experts run)")
print()

# Summary comparison
print("--- COMPARISON SUMMARY ---")
print(f"{'Metric':<25} {'Dense FFN':>15} {'MoE FFN':>15} {'Ratio':>10}")
print("-" * 67)
print(f"{'Total params':<25} {dense_total_params:>15,} {moe_total_params:>15,} "
      f"{moe_total_params / dense_total_params:>9.1f}x")
print(f"{'Active params/token':<25} {dense_active_params:>15,} {moe_active_params:>15,} "
      f"{moe_active_params / dense_active_params:>9.1f}x")
print(f"{'Active ratio':<25} {'100%':>15} "
      f"{moe_active_params / moe_total_params:>14.1%} {'':>10}")
print()

# ============================================================
# SECTION 2: FLOP Count Comparison
# ============================================================
print("--- SECTION 2: FLOP Count (Compute) Comparison ---")
print()

# FLOPs for a Linear(in, out) layer applied to [batch, seq, in]:
# = batch * seq * in * out  multiply-adds
# We count forward pass FLOPs only (not backward)

batch_size = 4    # batch of 4 sequences
seq_len    = 64   # each sequence has 64 tokens
num_tokens = batch_size * seq_len   # total tokens per batch

def count_linear_flops(in_dim, out_dim, num_tokens):
    """Count FLOPs for Linear(in_dim, out_dim) applied to num_tokens."""
    return num_tokens * in_dim * out_dim    # one multiply-add per (token, in, out)

# Dense FFN FLOPs (both layers, all tokens)
dense_fc1_flops = count_linear_flops(d_model, d_ff, num_tokens)    # expand
dense_fc2_flops = count_linear_flops(d_ff, d_model, num_tokens)    # contract
dense_total_flops = dense_fc1_flops + dense_fc2_flops

# MoE FFN FLOPs
# Router: all tokens go through the router
moe_router_flops = count_linear_flops(d_model, num_experts, num_tokens)
# Expert computation: only K experts run, each on a SUBSET of tokens
# On average, each expert gets (top_k / num_experts) * num_tokens tokens
tokens_per_expert = (top_k / num_experts) * num_tokens   # expected tokens per expert
moe_expert_fc1_flops = count_linear_flops(d_model, d_ff, top_k * num_tokens)  # K*tokens total
moe_expert_fc2_flops = count_linear_flops(d_ff, d_model, top_k * num_tokens)  # K*tokens total
moe_total_flops = moe_router_flops + moe_expert_fc1_flops + moe_expert_fc2_flops

print(f"Batch: {batch_size} sequences x {seq_len} tokens = {num_tokens} total tokens")
print()
print(f"Dense FFN FLOPs:")
print(f"  fc1 ({d_model}->{d_ff}): {dense_fc1_flops:,}")
print(f"  fc2 ({d_ff}->{d_model}): {dense_fc2_flops:,}")
print(f"  Total:          {dense_total_flops:,}")
print()
print(f"MoE FFN FLOPs:")
print(f"  router ({d_model}->{num_experts}): {moe_router_flops:,}")
print(f"  {top_k}x fc1 ({d_model}->{d_ff}):   {moe_expert_fc1_flops:,}  (only K={top_k} experts run)")
print(f"  {top_k}x fc2 ({d_ff}->{d_model}):   {moe_expert_fc2_flops:,}")
print(f"  Total:           {moe_total_flops:,}")
print()
print(f"FLOP comparison:")
print(f"  Dense:  {dense_total_flops:,}")
print(f"  MoE:    {moe_total_flops:,}")
print(f"  Ratio:  {moe_total_flops / dense_total_flops:.2f}x")
print(f"  MoE uses roughly {top_k}/{num_experts} = {top_k/num_experts:.0%} of the expert compute.")
print(f"  (Plus a tiny router overhead: {moe_router_flops / dense_total_flops:.2%} of dense FLOPs)")
print()

# ============================================================
# SECTION 3: Forward Pass -- Same Input Through Dense vs MoE
# ============================================================
print("--- SECTION 3: Forward Pass Comparison ---")
print()

# Create test input: [batch=2, seq=4, d_model=256]
test_input = torch.randn(2, 4, d_model)    # random token embeddings

# Dense forward pass
dense_output = dense_ffn(test_input)       # all 2*4=8 tokens use the dense FFN
print(f"Dense FFN:")
print(f"  Input shape:  {test_input.shape}")
print(f"  Output shape: {dense_output.shape}")
print(f"  All {2*4} tokens processed by the SAME FFN weights")
print()

# MoE forward pass (simplified -- using the router + expert dispatch)
class SimpleMoE(nn.Module):
    """Minimal MoE layer for demonstration."""
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router  = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        B, S, D = x.shape                                  # unpack dimensions
        logits = self.router(x)                            # [B, S, num_experts]
        probs  = F.softmax(logits, dim=-1)                 # [B, S, num_experts]
        topk_w, topk_idx = probs.topk(self.top_k, dim=-1) # top-K selection
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)# renormalize

        x_flat = x.view(B * S, D)                         # flatten for dispatch
        w_flat = topk_w.view(B * S, self.top_k)           # flatten weights
        i_flat = topk_idx.view(B * S, self.top_k)         # flatten indices

        out = torch.zeros_like(x_flat)                    # accumulate outputs

        # Track which expert processed how many tokens (for reporting)
        expert_token_counts = [0] * self.num_experts       # count per expert

        for k in range(self.top_k):                        # for each k position
            for eid in range(self.num_experts):            # for each expert
                mask = (i_flat[:, k] == eid)               # tokens routed here
                if not mask.any():
                    continue
                tokens_here = x_flat[mask]                 # subset of tokens
                expert_token_counts[eid] += mask.sum().item()  # count them
                expert_out = self.experts[eid](tokens_here)    # run expert
                w = w_flat[mask, k].unsqueeze(-1)              # weights
                out[mask] += w * expert_out                    # accumulate

        return out.view(B, S, D), probs, expert_token_counts  # reshape back

moe_layer = SimpleMoE(d_model, d_ff, num_experts, top_k)
moe_output, moe_probs, token_counts = moe_layer(test_input)

print(f"MoE FFN:")
print(f"  Input shape:  {test_input.shape}")
print(f"  Output shape: {moe_output.shape}")
print(f"  Tokens per expert (across {top_k} selections): {token_counts}")
print(f"  (Total = {sum(token_counts)}, = {2*4} tokens x {top_k} experts each = {2*4*top_k})")
print()

# ============================================================
# SECTION 4: Expert Capacity and Token Dropping
# ============================================================
print("--- SECTION 4: Expert Capacity and Token Dropping ---")
print()

def apply_expert_capacity(router_probs, num_experts, capacity_factor=1.0):
    """
    Simulate token dropping when an expert exceeds its capacity.

    Args:
        router_probs    : shape [batch, seq, num_experts]
        num_experts     : N
        capacity_factor : buffer above fair share (1.0 = strict, 1.25 = 25% buffer)

    Returns:
        expert_assignments : list of token indices per expert (after capacity cap)
        dropped_tokens     : number of tokens that were dropped
    """
    B, S, NE = router_probs.shape
    num_tokens = B * S                    # total tokens in this batch

    # capacity = tokens an expert is ALLOWED to process
    capacity = int((num_tokens / num_experts) * capacity_factor)
    print(f"  Total tokens:    {num_tokens}")
    print(f"  Fair share/expert: {num_tokens}/{num_experts} = {num_tokens // num_experts}")
    print(f"  Capacity factor: {capacity_factor}")
    print(f"  Capacity/expert: {capacity} tokens (= fair_share * {capacity_factor})")

    # Flatten and get primary expert for each token
    probs_flat = router_probs.view(num_tokens, num_experts)  # [N_tokens, N_experts]
    # sort tokens by their score for each expert (highest first)
    # topk_idx: which expert each token prefers most
    _, topk_idx = probs_flat.topk(1, dim=-1)   # primary expert per token
    primary_flat = topk_idx[:, 0]               # [num_tokens]

    # For each expert, collect which tokens want to go there
    # Sort by router probability (highest score = token gets priority)
    expert_assignments = {i: [] for i in range(num_experts)}  # expert -> list of token ids
    dropped_tokens = 0

    # For simplicity, process tokens in order (in practice: sort by score)
    for token_id in range(num_tokens):
        expert_id = primary_flat[token_id].item()   # which expert does this token want?
        if len(expert_assignments[expert_id]) < capacity:
            expert_assignments[expert_id].append(token_id)  # expert has room
        else:
            dropped_tokens += 1                             # expert is full, token dropped

    return expert_assignments, dropped_tokens

# Test with a badly balanced router (many tokens want Expert 0)
# Simulate the overloaded case
# Create biased probs: Expert 0 gets all traffic
B_test, S_test = 2, 8                                  # small batch for clarity
biased_probs = torch.zeros(B_test, S_test, num_experts)
biased_probs[:, :, 0] = 0.95  # expert 0 overwhelmingly preferred
biased_probs[:, :, 1] = 0.02
biased_probs[:, :, 2] = 0.02
biased_probs[:, :, 3] = 0.01
# (using 4 experts for this demo, ignoring 4-7)
four_expert_probs = biased_probs[:, :, :4]   # [2, 8, 4]

print("Scenario: Overloaded Expert 0 (all tokens prefer Expert 0)")
print()
print("With capacity_factor = 1.0 (strict):")
assignments_strict, dropped_strict = apply_expert_capacity(
    four_expert_probs, num_experts=4, capacity_factor=1.0
)
print(f"  Expert 0 gets: {len(assignments_strict[0])} tokens "
      f"(max={int(B_test*S_test/4*1.0)}, dropped {dropped_strict} tokens)")
print()

print("With capacity_factor = 1.25 (25% buffer):")
assignments_buf, dropped_buf = apply_expert_capacity(
    four_expert_probs, num_experts=4, capacity_factor=1.25
)
print(f"  Expert 0 gets: {len(assignments_buf[0])} tokens "
      f"(max={int(B_test*S_test/4*1.25)}, dropped {dropped_buf} tokens)")
print()

print("With capacity_factor = 2.0 (100% buffer):")
assignments_gen, dropped_gen = apply_expert_capacity(
    four_expert_probs, num_experts=4, capacity_factor=2.0
)
print(f"  Expert 0 gets: {len(assignments_gen[0])} tokens "
      f"(max={int(B_test*S_test/4*2.0)}, dropped {dropped_gen} tokens)")
print()

print("KEY INSIGHT: Higher capacity_factor = fewer dropped tokens,")
print("but less efficient GPU memory usage (larger buffers needed).")
print()

# ============================================================
# SECTION 5: When to Use MoE vs Dense (Decision Summary)
# ============================================================
print("--- SECTION 5: Decision Summary ---")
print()
print("Use DENSE when:")
print("  - Model size < 7B params (MoE overhead not worth it)")
print("  - Edge / mobile deployment (can't shard experts)")
print("  - Simple deployment needed (no expert parallelism)")
print("  - Fine-tuning a narrow single-task model")
print()
print("Use MoE when:")
print("  - Model size > 30B params (MoE saves massive compute)")
print("  - Multi-GPU server inference (can shard experts)")
print("  - Diverse tasks (code + math + language + science)")
print("  - Cost efficiency at large scale is critical")
print()

print("=" * 60)
print("Example 04 complete!")
print("You compared Dense FFN and MoE on: total params, active params,")
print("FLOP count, forward pass, expert capacity, and token dropping.")
print("=" * 60)
