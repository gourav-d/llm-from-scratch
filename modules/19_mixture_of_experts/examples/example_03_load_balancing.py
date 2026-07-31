"""
Module 19 - Mixture of Experts
Example 03: Load Balancing -- Preventing Expert Collapse

GLOSSARY
--------
Expert collapse : When the router sends ALL tokens to 1-2 experts. Other experts
                 stop receiving training signal and become useless.
                 This is a positive feedback loop: better expert -> more tokens
                 -> more training -> even better -> even more tokens -> ...

Auxiliary loss  : An extra training loss that penalizes unequal expert usage.
                 Added to the main language modeling loss with a small coefficient.
                 Formula (Switch Transformer): N * sum(f_i * P_i) for i in 0..N-1

f_i             : Fraction of tokens in this batch that were routed to expert i.
                 f_i = count(tokens routed to expert i) / total_tokens
                 This is HARD (non-differentiable -- computed via argmax/topk).

P_i             : Average ROUTER PROBABILITY assigned to expert i across all tokens.
                 P_i = mean of router_probs[:, :, i] over batch and seq dims.
                 This is SOFT (differentiable -- can backpropagate through it).

Expert capacity : Maximum number of tokens an expert will process per batch.
                 capacity = (total_tokens / num_experts) * capacity_factor
                 Tokens over capacity are "dropped" (use residual = identity).

Load balance    : The ideal state: each expert processes 1/N of all tokens.
                 When balanced, aux_loss reaches its minimum value of 1.0.

alpha           : Coefficient for aux_loss in total_loss = main_loss + alpha * aux.
                 Typical values: 0.001 to 0.01. Too large -> quality suffers.
"""

import torch                         # PyTorch: deep learning framework
import torch.nn as nn                # nn: neural network building blocks
import torch.nn.functional as F      # F: stateless functions (softmax, gelu, etc.)

torch.manual_seed(42)                # fixed seed for reproducibility

print("=" * 60)
print("Example 03: Load Balancing and Expert Collapse")
print("=" * 60)
print()

# ============================================================
# SECTION 1: Demonstrate Expert Collapse (Without Aux Loss)
# ============================================================
print("--- SECTION 1: Expert Collapse Without Auxiliary Loss ---")
print()

# Simulate what happens when a router has NO incentive to balance experts.
# We will manually create a "collapsed" router probability distribution.

num_experts = 4      # 4 experts available
batch_size  = 3      # 3 sequences in this batch
seq_len     = 8      # each sequence has 8 tokens
# total tokens in batch:
total_tokens = batch_size * seq_len   # 3 * 8 = 24

print(f"Batch:  {batch_size} sequences x {seq_len} tokens = {total_tokens} total tokens")
print(f"Experts: {num_experts}")
print()

# COLLAPSED router: Expert 0 gets almost all the probability mass
# This simulates what happens after expert collapse
collapsed_probs = torch.tensor([
    [0.92, 0.03, 0.03, 0.02],  # token A: almost all mass on expert 0
    [0.91, 0.04, 0.02, 0.03],  # token B: same
    [0.93, 0.02, 0.03, 0.02],  # token C: same
    [0.90, 0.05, 0.02, 0.03],  # ... (illustrating collapse)
]).unsqueeze(0)                 # add batch dim: shape [1, 4, num_experts]

print("Collapsed router probabilities (first 4 tokens of one sequence):")
print("  Expert 0  Expert 1  Expert 2  Expert 3")
for i in range(4):              # print each token's distribution
    row = collapsed_probs[0, i]
    print(f"  {row[0]:.2f}      {row[1]:.2f}      {row[2]:.2f}      {row[3]:.2f}"
          f"  <- token {i}")

print()
print("Problem: Expert 0 gets ~92% of traffic. Experts 1,2,3 are starved.")
print("They receive almost no gradient -> they stop learning -> useless.")
print()

# Show what "perfect collapse" looks like
print("Extreme collapse example (top-1 routing without balance):")
# Create argmax selection for those collapsed probs
argmax_idx = collapsed_probs.argmax(dim=-1)   # [1, 4] -- which expert for each token
print(f"  Argmax expert per token: {argmax_idx[0].tolist()}")
print("  (All tokens go to Expert 0 -- the other 3 experts never train)")
print()

# ============================================================
# SECTION 2: The Auxiliary Loss Formula (Step by Step)
# ============================================================
print("--- SECTION 2: Switch Transformer Auxiliary Loss ---")
print()
print("Formula: aux_loss = N * sum(f_i * P_i)  for i in 0..N-1")
print()
print("Where:")
print("  N   = num_experts")
print("  f_i = fraction of tokens routed to expert i  (hard, non-differentiable)")
print("  P_i = average router probability for expert i (soft, differentiable)")
print()

def compute_aux_loss(router_probs, num_experts, top_k=1):
    """
    Compute Switch Transformer auxiliary loss for load balancing.

    Args:
        router_probs : shape [batch, seq_len, num_experts] -- softmax output
        num_experts  : total number of experts N
        top_k        : number of experts selected per token (used to compute f_i)

    Returns:
        aux_loss : scalar tensor
    """
    batch, seq, ne = router_probs.shape          # unpack dimensions

    # --- Compute f_i: fraction of tokens routed to expert i ---
    # f_i is based on the hard (argmax/topk) routing decisions.
    # For simplicity we use argmax (top-1) to determine "primary" expert.
    # This gives a discrete assignment: each token "belongs to" one expert.
    primary_expert = router_probs.argmax(dim=-1)  # [batch, seq] -- expert index per token
    primary_flat   = primary_expert.view(-1)      # [batch*seq] -- flatten for counting

    # Count how many tokens went to each expert
    f = torch.zeros(ne, device=router_probs.device)   # [num_experts]
    for i in range(ne):
        f[i] = (primary_flat == i).float().mean()     # fraction for expert i
    # f sums to 1.0 (each token goes to exactly one primary expert)

    # --- Compute P_i: average router probability for expert i ---
    # P_i is the mean of router_probs over all tokens (both batch and seq dims)
    # dim=[0,1] means average over batch dimension AND sequence dimension
    P = router_probs.mean(dim=[0, 1])    # [num_experts] -- soft average probability
    # P also sums to 1.0 (since router_probs sum to 1 for each token)

    # --- Compute auxiliary loss ---
    # The product f_i * P_i is large when expert i is OVERLOADED:
    #   f_i large = many tokens go there (hard usage)
    #   P_i large = router gives it high probability (soft usage)
    # Both being large = the router is confidently overloading this expert.
    # Penalizing this product discourages overloading.
    aux_loss = ne * (f * P).sum()    # scale by N for numerical stability
    # When perfectly balanced: f_i = P_i = 1/N
    #   sum(f_i * P_i) = N * (1/N)^2 = 1/N
    #   aux_loss = N * (1/N) = 1.0   <- minimum value when balanced

    return aux_loss, f, P    # return all three for inspection


# Example with collapsed distribution
# Simulate a full [batch=3, seq=8, num_experts=4] tensor
# Most tokens route to Expert 0 (simulating collapse)
collapsed_full = torch.zeros(batch_size, seq_len, num_experts)  # start with zeros
collapsed_full[:, :, 0] = 0.93   # expert 0 gets 93% probability for all tokens
collapsed_full[:, :, 1] = 0.03   # expert 1 gets 3%
collapsed_full[:, :, 2] = 0.02   # expert 2 gets 2%
collapsed_full[:, :, 3] = 0.02   # expert 3 gets 2%
# These sum to 1.0 per token

collapsed_loss, f_vals, P_vals = compute_aux_loss(collapsed_full, num_experts)

print("COLLAPSED router (Expert 0 gets ~93% of tokens):")
print(f"  f_i (actual token fractions): {[round(v.item(), 3) for v in f_vals]}")
print(f"  P_i (avg probabilities):      {[round(v.item(), 3) for v in P_vals]}")
print(f"  aux_loss = {collapsed_loss.item():.4f}")
print()

# Example with BALANCED distribution (ideal)
balanced_full = torch.full((batch_size, seq_len, num_experts), 1.0 / num_experts)
# Each expert gets exactly 1/4 = 0.25 probability for every token

balanced_loss, f_vals_b, P_vals_b = compute_aux_loss(balanced_full, num_experts)

print("BALANCED router (each expert gets 25% of tokens):")
print(f"  f_i (actual token fractions): {[round(v.item(), 3) for v in f_vals_b]}")
print(f"  P_i (avg probabilities):      {[round(v.item(), 3) for v in P_vals_b]}")
print(f"  aux_loss = {balanced_loss.item():.4f}  <- minimum possible (= 1.0 when balanced)")
print()

print(f"Comparison:")
print(f"  Collapsed aux_loss:  {collapsed_loss.item():.4f}  (LARGE = penalized)")
print(f"  Balanced  aux_loss:  {balanced_loss.item():.4f}  (SMALL = minimum)")
print(f"  The training optimizer MINIMIZES aux_loss, pushing toward balance.")
print()

# ============================================================
# SECTION 3: Simulate Training WITH and WITHOUT Aux Loss
# ============================================================
print("--- SECTION 3: Simulating Training With vs Without Aux Loss ---")
print()
print("We will train a tiny router for 60 steps and track expert usage.")
print()

d_model     = 16        # small embedding dimension
num_experts = 4         # 4 experts
top_k       = 2         # top-2 routing

# Create a very simple router to observe collapse behavior
class TrainableRouter(nn.Module):
    """Simple router with optional noisy initialization to trigger collapse."""
    def __init__(self, d_model, num_experts, biased=False):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=True)  # include bias
        if biased:
            # Initialize bias so Expert 0 starts with slight advantage
            # This simulates the "small head start" that triggers collapse
            with torch.no_grad():                      # don't track this init in gradients
                self.gate.bias.data[0] = 1.0           # boost expert 0 at start

    def forward(self, x):
        logits  = self.gate(x)                         # [B, S, num_experts]
        probs   = F.softmax(logits, dim=-1)            # probabilities
        topk_w, topk_idx = probs.topk(k=top_k, dim=-1)# top-K selection
        topk_w  = topk_w / topk_w.sum(dim=-1, keepdim=True)  # renormalize
        return topk_w, topk_idx, probs

def track_expert_usage(router, inputs):
    """Run router on inputs and return fraction of tokens to each expert."""
    with torch.no_grad():                              # no gradient tracking (inspection only)
        _, topk_idx, _ = router(inputs)                # get routing decisions
        primary = topk_idx[:, :, 0].view(-1)           # primary expert (k=0 position)
        fractions = [(primary == i).float().mean().item() for i in range(num_experts)]
    return fractions

# Generate fixed random "token" inputs for consistent measurement
# [batch=4, seq=8, d_model=16]
dummy_inputs = torch.randn(4, 8, d_model)   # random token embeddings

print("Training WITHOUT auxiliary loss (collapse expected):")
print()

# Create biased router (Expert 0 starts slightly stronger)
router_no_aux = TrainableRouter(d_model, num_experts, biased=True)
optimizer_no_aux = torch.optim.Adam(router_no_aux.parameters(), lr=0.05)

print(f"{'Step':>5} | {'E0':>6} | {'E1':>6} | {'E2':>6} | {'E3':>6} | {'Note'}")
print("-" * 55)

for step in range(61):                        # 61 steps (0 to 60)
    # Fake "main loss" -- we just want to see routing evolve
    # In a real model, this would be cross-entropy language modeling loss
    topk_w, topk_idx, all_probs = router_no_aux(dummy_inputs)
    # Fake loss: try to make all outputs use the primary expert heavily
    fake_loss = -all_probs[:, :, 0].mean()   # artificially favors expert 0

    optimizer_no_aux.zero_grad()             # clear previous gradients
    fake_loss.backward()                     # compute gradients
    optimizer_no_aux.step()                  # update router weights

    if step % 15 == 0:                       # print every 15 steps
        usage = track_expert_usage(router_no_aux, dummy_inputs)
        note = "COLLAPSED!" if usage[0] > 0.8 else ""
        print(f"{step:>5} | {usage[0]:>5.0%} | {usage[1]:>5.0%} | "
              f"{usage[2]:>5.0%} | {usage[3]:>5.0%} | {note}")

print()
print("Training WITH auxiliary loss (balance expected):")
print()

# Same biased start, but now add aux_loss to fight collapse
router_with_aux = TrainableRouter(d_model, num_experts, biased=True)
optimizer_with_aux = torch.optim.Adam(router_with_aux.parameters(), lr=0.05)
alpha = 0.5   # stronger alpha for demonstration (normally 0.01)

print(f"{'Step':>5} | {'E0':>6} | {'E1':>6} | {'E2':>6} | {'E3':>6} | {'aux_loss':>10}")
print("-" * 62)

for step in range(61):
    topk_w, topk_idx, all_probs = router_with_aux(dummy_inputs)
    # Fake main loss (favors expert 0 -- simulates collapse pressure)
    fake_loss = -all_probs[:, :, 0].mean()
    # Auxiliary loss: penalize imbalance
    aux, _, _ = compute_aux_loss(all_probs, num_experts)
    # Total loss: balance the two objectives
    total = fake_loss + alpha * aux

    optimizer_with_aux.zero_grad()
    total.backward()
    optimizer_with_aux.step()

    if step % 15 == 0:
        usage = track_expert_usage(router_with_aux, dummy_inputs)
        aux_val, _, _ = compute_aux_loss(
            router_with_aux(dummy_inputs)[2], num_experts
        )
        print(f"{step:>5} | {usage[0]:>5.0%} | {usage[1]:>5.0%} | "
              f"{usage[2]:>5.0%} | {usage[3]:>5.0%} | {aux_val.item():>10.4f}")

print()

# ============================================================
# SECTION 4: ASCII Histogram of Expert Usage
# ============================================================
print("--- SECTION 4: ASCII Histograms of Expert Usage ---")
print()

def ascii_histogram(fractions, label, bar_width=30):
    """Print an ASCII bar chart of expert usage fractions."""
    print(f"{label}:")
    for i, f in enumerate(fractions):
        bar_len = int(f * bar_width)           # proportional bar length
        bar = "#" * bar_len                    # draw bar with # symbols
        print(f"  Expert {i}: |{bar:<{bar_width}}| {f:.1%}")
    print()

# Get final usage for both routers
usage_no_aux = track_expert_usage(router_no_aux, dummy_inputs)
usage_with_aux = track_expert_usage(router_with_aux, dummy_inputs)

ascii_histogram(usage_no_aux,   "WITHOUT aux_loss (after 60 steps)")
ascii_histogram(usage_with_aux, "WITH aux_loss (after 60 steps)")

print("The aux_loss forces the router to spread tokens more evenly.")
print("Perfect balance = each expert gets 25% of tokens (4 experts).")

print()
print("=" * 60)
print("Example 03 complete!")
print("You saw expert collapse, learned the Switch Transformer")
print("auxiliary loss formula (f_i * P_i), and saw that adding")
print("aux_loss prevents collapse and keeps experts balanced.")
print("=" * 60)
