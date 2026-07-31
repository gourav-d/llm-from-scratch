"""
Module 19 - Mixture of Experts
Exercise 04: MoE vs Dense -- Parameters, Compute, and Efficiency

GLOSSARY
--------
Dense FFN    : Standard FFN where ALL tokens use ALL weights.
               One Linear(d_model, d_ff) and one Linear(d_ff, d_model).
               100% of params active per token -- no routing, no experts.

MoE FFN      : N expert FFNs, each token uses only K of them.
               Total params = N * per_expert_params + router_params.
               Active params per token = K * per_expert_params + router_params.

FLOP         : Floating Point Operation. One multiply + one add in a matrix multiply.
               Linear(in, out) applied to num_tokens: FLOPs = num_tokens * in * out.
               MoE uses far fewer FLOPs than dense at the same total param count.

Parameter efficiency: How well a model uses its parameters.
               MoE is parameter-efficient: 8x more total params, same FLOPs.
               Dense grows FLOPs linearly with params; MoE does not.

p.numel()    : PyTorch method: number of elements in a tensor.
               p.numel() for a [16, 8] weight matrix = 16 * 8 = 128.
               sum(p.numel() for p in model.parameters()) = total param count.

Capacity factor: Multiplier for expert token capacity buffer.
               capacity = (total_tokens / N) * capacity_factor
               Factor = 1.0: strict. Factor = 1.25: 25% overflow buffer.

Expert load  : How many tokens each expert actually processed in a batch.
               Measured as fraction of total_tokens (e.g., 30% for expert 2).
               Ideal: 1/N = 25% each (for N=4 experts).
"""

import torch                           # PyTorch: deep learning framework
import torch.nn as nn                  # nn: neural network modules
import torch.nn.functional as F        # F: stateless functions

torch.manual_seed(42)                  # reproducible results

print("=" * 60)
print("Exercise 04: MoE vs Dense -- Parameter and Compute Analysis")
print("=" * 60)
print()

# Configuration
d_model      = 64     # embedding dimension
d_ff         = 128    # FFN hidden dimension (2x d_model)
num_experts  = 4      # 4 experts in MoE
top_k        = 2      # top-2 routing

# Helper classes (complete, not exercises)
class DenseFFN(nn.Module):
    """Standard dense FFN -- all tokens use this one set of weights."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)    # expand
        self.fc2 = nn.Linear(d_ff, d_model)    # contract

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Expert(nn.Module):
    """One MoE expert -- identical structure to DenseFFN."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

# ============================================================
#  EXERCISE 1
#  Topic: Count parameters in a dense FFN
#
#  Background:
#    Every parameter takes memory. Counting params helps us understand
#    how much memory a model needs and how it scales with dimensions.
#    PyTorch tensors: weight shape [out_features, in_features], bias shape [out_features].
#    p.numel() gives total elements in a parameter tensor.
#
#  Your Task:
#    Complete count_dense_params(d_model, d_ff) to:
#      - Create a DenseFFN instance
#      - Count total parameters using p.numel()
#      - Return total_params (int)
#
#  C# Analogy:
#    int CountDenseParams(int dModel, int dFf) {
#        var ffn = new DenseFFN(dModel, dFf);
#        return ffn.Parameters().Sum(p => p.NumberOfElements());
#    }
# ============================================================

def count_dense_params(d_model, d_ff):
    """
    Count total parameters in a Dense FFN.

    Args:
        d_model : embedding dimension
        d_ff    : hidden dimension

    Returns:
        total_params : int -- total number of learnable parameters
    """
    # TODO: create DenseFFN(d_model, d_ff)
    # TODO: sum p.numel() for p in ffn.parameters()
    # TODO: return total_params
    pass


# --- Test Exercise 1 ---
print("--- Exercise 1 Test ---")
dense_params = count_dense_params(d_model, d_ff)

if dense_params is not None:
    # Manual calculation: fc1 (d_model*d_ff + d_ff) + fc2 (d_ff*d_model + d_model)
    expected = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    print(f"Dense FFN parameters: {dense_params:,}")
    print(f"  fc1: {d_model} x {d_ff} weights + {d_ff} bias = {d_model*d_ff + d_ff}")
    print(f"  fc2: {d_ff} x {d_model} weights + {d_model} bias = {d_ff*d_model + d_model}")
    print(f"  Total (manual): {expected:,}")
    assert dense_params == expected, f"Expected {expected}, got {dense_params}"
    print("PASS: Dense FFN parameter count is correct.")
else:
    print("Hint: sum(p.numel() for p in model.parameters())")
print()

# ============================================================
#  EXERCISE 2
#  Topic: Count total and active params in MoE
#
#  Background:
#    MoE has N expert FFNs + 1 router (Linear, no bias).
#    Total params = N * expert_params + router_params
#    Active params = K * expert_params + router_params  (only K experts run per token)
#    Key insight: total params grows with N, but active params only grows with K.
#
#  Your Task:
#    Complete count_moe_params(d_model, d_ff, num_experts, top_k) to:
#      - Create N Expert instances and 1 router Linear layer
#      - Count: total_params, active_params, params_per_expert, router_params
#      - Return (total_params, active_params, params_per_expert, router_params)
# ============================================================

def count_moe_params(d_model, d_ff, num_experts, top_k):
    """
    Count total and active parameters in a MoE FFN.

    Returns:
        (total_params, active_params, params_per_expert, router_params) -- all ints
    """
    # TODO: create one Expert(d_model, d_ff) and count its params
    # params_per_expert = sum(p.numel() for p in expert.parameters())

    # TODO: count router params: router = nn.Linear(d_model, num_experts, bias=False)
    # router_params = sum(p.numel() for p in router.parameters())

    # TODO: total_params = num_experts * params_per_expert + router_params
    # TODO: active_params = top_k * params_per_expert + router_params

    # TODO: return (total_params, active_params, params_per_expert, router_params)
    pass


# --- Test Exercise 2 ---
print("--- Exercise 2 Test ---")
moe_result = count_moe_params(d_model, d_ff, num_experts, top_k)

if moe_result is not None:
    total, active, per_expert, router_p = moe_result
    print(f"MoE FFN parameters ({num_experts} experts, top-{top_k}):")
    print(f"  Per-expert params: {per_expert:,}")
    print(f"  Router params:     {router_p:,}  (= d_model * num_experts = {d_model}*{num_experts})")
    print(f"  Total params:      {total:,}  ({num_experts} experts + router)")
    print(f"  Active params:     {active:,}  ({top_k} experts + router)")

    if dense_params is not None:
        print()
        print(f"Comparison vs Dense FFN ({dense_params:,} params):")
        print(f"  Total params ratio:   {total / dense_params:.1f}x  (MoE stores more)")
        print(f"  Active params ratio:  {active / dense_params:.1f}x  (MoE runs less)")
else:
    print("Hint: Return (total, active, per_expert, router_params)")
print()

# ============================================================
#  EXERCISE 3
#  Topic: Compute FLOP count for Dense vs MoE
#
#  Background:
#    FLOPs = compute cost = how many arithmetic operations per forward pass.
#    Linear(in, out) applied to [num_tokens, in]: FLOPs = num_tokens * in * out
#    Dense FFN FLOPs: num_tokens * (d_model*d_ff + d_ff*d_model) = 2 * N * d_model * d_ff
#    MoE FFN FLOPs: (router FLOPs) + (top_k * expert FLOPs)
#                   expert FLOPs use only (top_k/N fraction) of all tokens
#
#  Your Task:
#    Complete compare_flops(d_model, d_ff, num_experts, top_k, num_tokens) to:
#      - Compute FLOPs for dense FFN
#      - Compute FLOPs for MoE FFN (router + K experts on all tokens)
#      - Return (dense_flops, moe_flops, ratio)
#        ratio = moe_flops / dense_flops
# ============================================================

def compare_flops(d_model, d_ff, num_experts, top_k, num_tokens):
    """
    Compare FLOPs between Dense FFN and MoE FFN.

    Args:
        num_tokens : total tokens in this batch (batch_size * seq_len)

    Returns:
        (dense_flops, moe_flops, ratio) -- ints and float
    """
    # Dense FFN: all tokens go through both fc1 and fc2
    # fc1 FLOPs: num_tokens * d_model * d_ff
    # fc2 FLOPs: num_tokens * d_ff * d_model

    # TODO: compute dense_flops = 2 * num_tokens * d_model * d_ff

    # MoE FFN FLOPs:
    # router FLOPs: all tokens go through router Linear(d_model, num_experts)
    # expert FLOPs: only top_k experts run; each processes on average
    #   (top_k / num_experts) * num_tokens tokens
    # For simplicity: total expert forward tokens = top_k * num_tokens (spread across K experts)
    # expert FLOPs: 2 * top_k * num_tokens * d_model * d_ff

    # TODO: compute moe_router_flops = num_tokens * d_model * num_experts
    # TODO: compute moe_expert_flops = 2 * top_k * num_tokens * d_model * d_ff
    # TODO: moe_flops = moe_router_flops + moe_expert_flops
    # TODO: ratio = moe_flops / dense_flops

    # TODO: return (dense_flops, moe_flops, ratio)
    pass


# --- Test Exercise 3 ---
print("--- Exercise 3 Test ---")
num_tokens_test = 4 * 16   # batch=4, seq=16

flop_result = compare_flops(d_model, d_ff, num_experts, top_k, num_tokens_test)

if flop_result is not None:
    dense_flops, moe_flops, ratio = flop_result
    print(f"Number of tokens: {num_tokens_test}")
    print(f"Dense FFN FLOPs:  {dense_flops:,}")
    print(f"MoE FFN FLOPs:    {moe_flops:,}")
    print(f"MoE/Dense ratio:  {ratio:.3f}  (< 1.0 means MoE is cheaper per token)")
    if ratio < 1.0:
        print(f"MoE uses only {ratio:.1%} of Dense FLOPs -- {1/ratio:.1f}x cheaper!")
    print()
    print(f"Key: top_k/num_experts = {top_k}/{num_experts} = {top_k/num_experts:.0%} of experts active.")
    print(f"MoE expert FLOPs = {top_k/num_experts:.0%} of what dense would use.")
else:
    print("Hint: dense = 2*tokens*d_model*d_ff. MoE = router + top_k*expert FLOPs")
print()

# ============================================================
#  EXERCISE 4
#  Topic: Compute parameter efficiency ratio
#
#  Background:
#    "Parameter efficiency" = how much compute you get per parameter.
#    Dense: increasing params directly increases FLOPs (1:1 ratio).
#    MoE: increasing total params does NOT increase FLOPs (uses K/N of experts).
#    Efficiency ratio: (quality_per_FLOP) / (quality_per_total_param)
#    We approximate this as: FLOP_density = active_params / total_params
#    Higher FLOP_density = each stored param is more "computationally active."
#
#  Your Task:
#    Complete parameter_efficiency(dense_total, dense_active, moe_total, moe_active) to:
#      - Compute FLOP_density_dense = dense_active / dense_total (should be 1.0)
#      - Compute FLOP_density_moe   = moe_active / moe_total
#      - Compute capacity_multiplier = moe_total / dense_total
#        (how many more params MoE stores vs dense)
#      - Return (flop_density_dense, flop_density_moe, capacity_multiplier)
# ============================================================

def parameter_efficiency(dense_total, dense_active, moe_total, moe_active):
    """
    Compute parameter efficiency metrics for Dense vs MoE.

    Returns:
        (flop_density_dense, flop_density_moe, capacity_multiplier) -- all floats
    """
    # TODO: flop_density_dense = dense_active / dense_total
    # TODO: flop_density_moe = moe_active / moe_total
    # TODO: capacity_multiplier = moe_total / dense_total
    # TODO: return all three
    pass


# --- Test Exercise 4 ---
print("--- Exercise 4 Test ---")
if moe_result is not None and dense_params is not None:
    total_moe, active_moe, _, _ = moe_result
    eff_result = parameter_efficiency(
        dense_params, dense_params,    # dense: all params are active
        total_moe, active_moe          # moe: total vs active
    )

    if eff_result is not None:
        fd_dense, fd_moe, cap_mult = eff_result
        print(f"Dense FFN parameter efficiency:")
        print(f"  FLOP density (active/total): {fd_dense:.2f}  (100% -- all params always active)")
        print()
        print(f"MoE FFN parameter efficiency:")
        print(f"  FLOP density (active/total): {fd_moe:.2f}  ({fd_moe:.0%} -- only K experts active)")
        print(f"  Capacity multiplier:         {cap_mult:.1f}x  (MoE stores {cap_mult:.1f}x more params)")
        print()
        print(f"Trade-off: MoE stores {cap_mult:.1f}x more params (needs more GPU memory),")
        print(f"but only activates {fd_moe:.0%} of them per token (much cheaper compute).")
    else:
        print("Hint: Return (active/total for dense, active/total for moe, moe_total/dense_total)")
else:
    print("Skipped (depends on earlier exercises)")
print()

# ============================================================
#  EXERCISE 5
#  Topic: Simulate routing with N experts and measure actual expert load
#
#  Background:
#    In a real training batch, we want to know: how many tokens went to each expert?
#    This lets us check if the router is collapsed or balanced.
#    Simulate: create random router probs, apply top-K, count tokens per expert.
#
#  Your Task:
#    Complete simulate_routing(num_tokens, num_experts, top_k) to:
#      - Create random router probabilities [num_tokens, num_experts]
#      - Apply softmax and top-K selection
#      - Count how many of the (num_tokens * top_k) expert slots went to each expert
#      - Return expert_loads: list of ints (one per expert)
# ============================================================

def simulate_routing(num_tokens, num_experts, top_k):
    """
    Simulate routing for a batch of tokens.

    Args:
        num_tokens   : total tokens in this batch
        num_experts  : N
        top_k        : K experts per token

    Returns:
        expert_loads : list of ints -- how many (token, slot) pairs each expert received
                       Total = num_tokens * top_k (each token uses K slots)
    """
    # TODO: create random logits [num_tokens, num_experts] and apply softmax
    # probs = F.softmax(torch.randn(num_tokens, num_experts), dim=-1)

    # TODO: apply topk: topk_idx shape [num_tokens, top_k]
    # _, topk_idx = probs.topk(top_k, dim=-1)

    # TODO: flatten topk_idx to [num_tokens * top_k] and count per expert
    # idx_flat = topk_idx.view(-1)
    # loads = [(idx_flat == i).sum().item() for i in range(num_experts)]

    # TODO: return loads
    pass


# --- Test Exercise 5 ---
print("--- Exercise 5 Test ---")
sim_tokens = 256   # 256 tokens
loads = simulate_routing(sim_tokens, num_experts, top_k)

if loads is not None:
    print(f"Simulating routing: {sim_tokens} tokens, {num_experts} experts, top-{top_k}")
    print(f"Total expert slots: {sim_tokens * top_k}  (each token uses {top_k} slots)")
    print()
    total_load = sum(loads)
    fair_share = sim_tokens * top_k / num_experts    # ideal load per expert
    print(f"Expert load distribution (fair share = {fair_share:.0f} per expert):")
    for i, load in enumerate(loads):
        bar = "#" * int(load / (sim_tokens * top_k) * 40)  # ASCII bar
        pct = load / total_load
        print(f"  Expert {i}: {load:>4} tokens ({pct:.1%}) |{bar:<40}|")
    print(f"  Total:    {total_load} (should be {sim_tokens * top_k})")
    assert total_load == sim_tokens * top_k, f"Loads should sum to {sim_tokens * top_k}!"
    print("PASS: Total load matches expected (num_tokens * top_k).")
else:
    print("Hint: topk gives [num_tokens, top_k] indices. Flatten and count per expert.")

print()
print("=" * 60)
print("Exercise 04 complete!")
print("You counted Dense FFN params, MoE total/active params,")
print("compared FLOPs, computed parameter efficiency, and")
print("simulated routing to measure actual expert load distribution.")
print("=" * 60)
