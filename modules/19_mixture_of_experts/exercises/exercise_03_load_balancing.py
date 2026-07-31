"""
Module 19 - Mixture of Experts
Exercise 03: Load Balancing -- Computing Auxiliary Loss

GLOSSARY
--------
Expert collapse  : All tokens route to 1-2 experts; others never train.
                   Happens without load balancing because "rich get richer."
                   A started advantage compounds: more tokens -> more gradient -> better.

Auxiliary loss   : Extra loss term forcing balanced expert usage.
                   total_loss = main_loss + alpha * aux_loss
                   aux_loss is minimized when all experts process equal fractions of tokens.

f_i (hard usage): Fraction of tokens that were routed to expert i.
                   f_i = count(tokens -> expert i) / total_tokens
                   NON-DIFFERENTIABLE: computed from argmax/topk (discrete, no gradient).

P_i (soft usage): Average router probability for expert i across all tokens.
                   P_i = mean(router_probs[:, :, i]) over batch and seq dimensions.
                   DIFFERENTIABLE: can backpropagate through router Linear weights.

Switch aux_loss  : N * sum(f_i * P_i) for i in 0..N-1
                   Penalizes when an expert has high BOTH f_i AND P_i (overloaded).
                   Minimum value = 1.0 (when perfectly balanced).

Expert capacity  : Maximum tokens an expert processes per batch.
                   capacity = (total_tokens / num_experts) * capacity_factor
                   Tokens over capacity are "dropped" (use residual = identity output).

Capacity factor  : Buffer multiplier for expert capacity.
                   1.0 = strict (exactly fair share). 1.25 = 25% buffer.
                   Higher factor = fewer drops but more memory needed per expert.

alpha            : Coefficient for aux_loss in total_loss = main + alpha * aux.
                   Typically 0.01. Too large: model prioritizes balance over quality.
"""

import torch                           # PyTorch: deep learning framework
import torch.nn as nn                  # nn: neural network building blocks
import torch.nn.functional as F        # F: softmax, cross_entropy, etc.

torch.manual_seed(42)                  # reproducible results

print("=" * 60)
print("Exercise 03: Load Balancing -- Auxiliary Loss")
print("=" * 60)
print()

# Setup: small MoE parameters for exercises
num_experts  = 4      # 4 experts
batch_size   = 3      # 3 sequences per batch
seq_len      = 8      # 8 tokens per sequence
total_tokens = batch_size * seq_len   # 24 total tokens

# Provide some sample router probability tensors for exercises
# Collapsed case: Expert 0 gets ~90% of traffic
collapsed_probs = torch.zeros(batch_size, seq_len, num_experts)  # [3, 8, 4]
collapsed_probs[:, :, 0] = 0.91    # expert 0 gets ~91%
collapsed_probs[:, :, 1] = 0.04    # expert 1 gets ~4%
collapsed_probs[:, :, 2] = 0.03    # expert 2 gets ~3%
collapsed_probs[:, :, 3] = 0.02    # expert 3 gets ~2%

# Balanced case: all experts get ~25%
balanced_probs = torch.full((batch_size, seq_len, num_experts), 0.25)  # 1/4 each

# ============================================================
#  EXERCISE 1
#  Topic: Compute f_i -- fraction of tokens routed to each expert
#
#  Background:
#    f_i = fraction of tokens that "primarily" went to expert i.
#    "Primary" = the expert with the HIGHEST router probability for that token.
#    Computed via argmax: for each token, which expert has the highest prob?
#    f_i = count(argmax_expert == i) / total_tokens
#
#  Your Task:
#    Complete compute_f(router_probs, num_experts) to:
#      - Find the primary expert for each token using argmax
#      - Count fraction of tokens going to each expert
#      - Return f: tensor of shape [num_experts], values in [0,1], sum = 1.0
#
#  C# Analogy:
#    float[] ComputeF(Tensor probs, int numExperts) {
#        var primary = probs.Argmax(dim: -1).Flatten();  // primary expert per token
#        return Enumerable.Range(0, numExperts)
#            .Select(i => (float)primary.Count(x => x == i) / primary.Count)
#            .ToArray();
#    }
# ============================================================

def compute_f(router_probs, num_experts):
    """
    Compute f_i: fraction of tokens routed to each expert.

    Args:
        router_probs : [batch, seq_len, num_experts] -- softmax probabilities
        num_experts  : N

    Returns:
        f : [num_experts] -- fraction for each expert, sums to 1.0
    """
    # TODO: get primary expert per token using argmax over last dim
    # primary_expert = router_probs.argmax(dim=-1)  # [batch, seq_len]

    # TODO: flatten to [batch*seq_len]
    # primary_flat = primary_expert.view(-1)

    # TODO: compute fraction for each expert i
    # f = torch.zeros(num_experts)
    # for i in range(num_experts): f[i] = (primary_flat == i).float().mean()

    # TODO: return f
    pass


# --- Test Exercise 1 ---
print("--- Exercise 1 Test ---")
f_collapsed = compute_f(collapsed_probs, num_experts)
f_balanced  = compute_f(balanced_probs,  num_experts)

if f_collapsed is not None:
    print("Collapsed router -- f_i (fraction of tokens to each expert):")
    for i, fi in enumerate(f_collapsed):
        print(f"  Expert {i}: {fi.item():.3f}  ({fi.item()*100:.1f}% of tokens)")
    print(f"  Sum: {f_collapsed.sum().item():.3f}  (must be 1.0)")

    print()
    print("Balanced router -- f_i:")
    for i, fi in enumerate(f_balanced):
        print(f"  Expert {i}: {fi.item():.3f}")
    print(f"  Sum: {f_balanced.sum().item():.3f}")
else:
    print("Hint: argmax(dim=-1) gives primary expert per token. Count fractions.")
print()

# ============================================================
#  EXERCISE 2
#  Topic: Compute P_i -- average router probability for each expert
#
#  Background:
#    P_i = mean of router_probs[:, :, i] over ALL tokens.
#    This is the SOFT (differentiable) version of expert usage.
#    Unlike f_i (discrete argmax), P_i has gradients and can be backpropagated.
#    In PyTorch: probs.mean(dim=[0, 1]) averages over batch (dim 0) and seq (dim 1).
#
#  Your Task:
#    Complete compute_P(router_probs) to:
#      - Average router_probs over batch and sequence dimensions
#      - Return P: tensor of shape [num_experts], values in [0,1], sum = 1.0
#
#  C# Analogy:
#    float[] ComputeP(Tensor probs) => probs.Mean(axes: new[]{0, 1}).ToArray();
# ============================================================

def compute_P(router_probs):
    """
    Compute P_i: average router probability for each expert.

    Args:
        router_probs : [batch, seq_len, num_experts] -- softmax probabilities

    Returns:
        P : [num_experts] -- average probability per expert, sums to 1.0
    """
    # TODO: average router_probs over batch (dim 0) and seq (dim 1)
    # Use router_probs.mean(dim=[0, 1])
    pass


# --- Test Exercise 2 ---
print("--- Exercise 2 Test ---")
P_collapsed = compute_P(collapsed_probs)
P_balanced  = compute_P(balanced_probs)

if P_collapsed is not None:
    print("Collapsed router -- P_i (average router probability per expert):")
    for i, pi in enumerate(P_collapsed):
        print(f"  Expert {i}: P_{i} = {pi.item():.4f}")
    print(f"  Sum: {P_collapsed.sum().item():.4f}  (must be ~1.0)")

    print()
    print("Balanced router -- P_i:")
    for i, pi in enumerate(P_balanced):
        print(f"  Expert {i}: P_{i} = {pi.item():.4f}")
else:
    print("Hint: router_probs.mean(dim=[0, 1]) averages over batch and seq")
print()

# ============================================================
#  EXERCISE 3
#  Topic: Compute the Switch Transformer auxiliary loss
#
#  Background:
#    aux_loss = N * sum(f_i * P_i)  for i in 0..N-1
#    When balanced (f_i = P_i = 1/N): aux_loss = N * N * (1/N)^2 = 1.0
#    When collapsed (f_0=1, P_0=1): aux_loss = N * (1 * 1 + 0 + 0 + 0) = N = 4
#    We MINIMIZE aux_loss: pushes toward balance (minimum = 1.0).
#    Note: f_i is non-differentiable, but P_i carries the gradient.
#
#  Your Task:
#    Complete compute_aux_loss(router_probs, num_experts) to:
#      - Compute f_i using compute_f (from Exercise 1)
#      - Compute P_i using compute_P (from Exercise 2)
#      - Return N * sum(f_i * P_i)
#
#  C# Analogy:
#    float AuxLoss(float[] f, float[] P, int N) =>
#        N * Enumerable.Range(0, N).Sum(i => f[i] * P[i]);
# ============================================================

def compute_aux_loss(router_probs, num_experts):
    """
    Compute Switch Transformer auxiliary load balancing loss.

    Args:
        router_probs : [batch, seq_len, num_experts]
        num_experts  : N

    Returns:
        aux_loss : scalar tensor. Minimum = 1.0 (perfect balance). Larger = more imbalanced.
    """
    # TODO: compute f using compute_f
    # TODO: compute P using compute_P
    # TODO: return num_experts * (f * P).sum()
    pass


# --- Test Exercise 3 ---
print("--- Exercise 3 Test ---")
aux_collapsed = compute_aux_loss(collapsed_probs, num_experts)
aux_balanced  = compute_aux_loss(balanced_probs,  num_experts)

if aux_collapsed is not None:
    print(f"Collapsed aux_loss: {aux_collapsed.item():.4f}  (LARGE = penalized = BAD)")
    print(f"Balanced  aux_loss: {aux_balanced.item():.4f}  (close to 1.0 = minimum = GOOD)")
    print()
    print(f"Minimum possible aux_loss (perfect balance) = 1.0")
    print(f"Higher aux_loss = more imbalanced = more penalized during training")
    if aux_collapsed > aux_balanced:
        print("PASS: Collapsed aux_loss is larger than balanced aux_loss (correct!)")
else:
    print("Hint: N * (f * P).sum() where f and P are both [num_experts] tensors")
print()

# ============================================================
#  EXERCISE 4
#  Topic: Implement expert capacity check (drop tokens over capacity)
#
#  Background:
#    Expert capacity limits how many tokens an expert processes per batch.
#    capacity = (total_tokens / num_experts) * capacity_factor
#    If more tokens want to go to expert i than capacity:
#      - Process only the first 'capacity' tokens (drop the rest)
#      - "Dropped" tokens use their original embedding (residual = identity)
#    This prevents one expert from being overwhelmed while others idle.
#
#  Your Task:
#    Complete check_capacity(router_probs, num_experts, capacity_factor) to:
#      - Compute expert capacity
#      - For each expert, count how many tokens are assigned (using argmax)
#      - Count how many tokens would be DROPPED (excess over capacity)
#      - Return (capacity, expert_loads, num_dropped)
#        expert_loads: list of ints -- how many tokens each expert gets (capped at capacity)
#        num_dropped: int -- total tokens dropped across all experts
#
#  C# Analogy:
#    Like a connection pool: each expert (connection pool) has a max size.
#    Requests over the limit are rejected (dropped) with a fallback (residual).
# ============================================================

def check_capacity(router_probs, num_experts, capacity_factor=1.0):
    """
    Check expert capacity and count dropped tokens.

    Args:
        router_probs     : [batch, seq_len, num_experts]
        num_experts      : N
        capacity_factor  : buffer above fair share (1.0 = strict, 1.25 = 25% buffer)

    Returns:
        capacity     : int -- max tokens per expert
        expert_loads : list of ints -- actual tokens sent to each expert (capped)
        num_dropped  : int -- total dropped tokens
    """
    B, S, N = router_probs.shape
    total_tokens = B * S               # total tokens in this batch

    # TODO: compute capacity = int((total_tokens / num_experts) * capacity_factor)

    # TODO: get primary expert per token (argmax over last dim, then flatten)
    # primary_flat = router_probs.argmax(dim=-1).view(-1)

    # TODO: for each expert i, count tokens assigned to it
    # requested = [(primary_flat == i).sum().item() for i in range(num_experts)]

    # TODO: cap each expert load at capacity, sum up dropped tokens
    # expert_loads = [min(req, capacity) for req in requested]
    # num_dropped = sum(max(0, req - capacity) for req in requested)

    # TODO: return (capacity, expert_loads, num_dropped)
    pass


# --- Test Exercise 4 ---
print("--- Exercise 4 Test ---")
print("Checking capacity for COLLAPSED router (Expert 0 gets all tokens):")
print()
for cf in [1.0, 1.25, 2.0]:          # test different capacity factors
    result = check_capacity(collapsed_probs, num_experts, capacity_factor=cf)
    if result is not None:
        cap, loads, dropped = result
        print(f"  capacity_factor={cf}: capacity={cap}/expert, loads={loads}, dropped={dropped}")
    else:
        print(f"  capacity_factor={cf}: Hint: capacity = int(total_tokens/N * factor)")

print()
print("Expected: higher capacity_factor -> fewer dropped tokens")
print(f"Total tokens = {total_tokens}, with {num_experts} experts")
print(f"Fair share = {total_tokens // num_experts} per expert")
print()

# ============================================================
#  EXERCISE 5
#  Topic: Measure load balance before and after adding aux_loss
#
#  Background:
#    We can measure "load balance quality" using a simple metric:
#      imbalance = std(f_i) across experts
#      std = 0.0 means perfectly balanced (all equal fractions)
#      std > 0.0 means some experts get more than others
#    With aux_loss: imbalance decreases over training.
#    Without aux_loss: imbalance increases (collapse).
#
#  Your Task:
#    Complete measure_imbalance(router_probs, num_experts) to:
#      - Compute f_i (fraction per expert)
#      - Return the standard deviation of f_i values
#      - Low std = balanced. High std = collapsed.
# ============================================================

def measure_imbalance(router_probs, num_experts):
    """
    Measure expert load imbalance as standard deviation of f_i.

    Args:
        router_probs : [batch, seq_len, num_experts]
        num_experts  : N

    Returns:
        imbalance : scalar float (std of f_i values)
                    0.0 = perfect balance, higher = more imbalanced
    """
    # TODO: compute f using compute_f
    # TODO: return f.std().item()  -- standard deviation of expert fractions
    pass


# --- Test Exercise 5 ---
print("--- Exercise 5 Test ---")
imbalance_collapsed = measure_imbalance(collapsed_probs, num_experts)
imbalance_balanced  = measure_imbalance(balanced_probs,  num_experts)

if imbalance_collapsed is not None:
    print(f"Collapsed router imbalance (std of f_i): {imbalance_collapsed:.4f}")
    print(f"Balanced  router imbalance (std of f_i): {imbalance_balanced:.4f}")
    print()
    if imbalance_collapsed > imbalance_balanced:
        print("PASS: Collapsed router has higher imbalance than balanced (correct!).")
    print()
    print("During training with aux_loss, imbalance should approach 0.0.")
    print("That means all f_i are equal (~1/N each) -- perfect balance.")
else:
    print("Hint: compute f using compute_f, then return f.std().item()")

print()
print("=" * 60)
print("Exercise 03 complete!")
print("You implemented f_i (hard usage), P_i (soft probability),")
print("the Switch Transformer aux_loss formula, expert capacity,")
print("and a load imbalance metric. These are the tools needed")
print("to train a balanced MoE without expert collapse.")
print("=" * 60)
