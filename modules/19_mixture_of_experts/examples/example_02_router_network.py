"""
Module 19 - Mixture of Experts
Example 02: The Router Network

GLOSSARY
--------
Router       : A small Linear layer that decides which experts handle each token.
               Input: token embedding [d_model].
               Output: probability for each expert [num_experts].
               Architecture: Linear(d_model, num_experts) -> softmax.
               C# analogy: a learned load balancer that routes HTTP requests.

Top-K gating : Keep only the K highest router probabilities. Zero out the rest.
               K=1 is "hard routing" (Switch Transformer).
               K=2 is most common (Mixtral, Qwen3 MoE).
               The top-K operation is NON-DIFFERENTIABLE (like argmax).

Renormalization: After top-K selection, the K remaining weights no longer sum to 1.
               Apply softmax AGAIN over just the K values to renormalize.
               Now: topk_weights.sum() == 1.0, valid for weighted average.

Softmax      : Converts raw scores (logits) into probabilities summing to 1.
               Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
               Always positive, always sums to 1.

Weighted sum : output = w1 * expert1(token) + w2 * expert2(token)
               where w1 + w2 = 1.0. A weighted average of expert outputs.

Router noise : Small random noise added to logits DURING TRAINING to encourage
               exploration. Without it, router may always pick the same experts.
               Not used during inference.

Token routing: For each token, the router independently picks K experts.
               Different tokens in the SAME sequence can route to DIFFERENT experts.
"""

import torch                         # PyTorch: deep learning framework
import torch.nn as nn                # nn: neural network modules (like C# base classes)
import torch.nn.functional as F      # F: stateless functions (softmax, gelu, etc.)

torch.manual_seed(42)                # fixed seed for reproducible results

print("=" * 60)
print("Example 02: The Router Network")
print("=" * 60)
print()

# ============================================================
# SECTION 1: Build the Expert class (from Example 01)
# ============================================================
# We need experts to combine with the router later

class Expert(nn.Module):
    """Standard FFN expert: Linear -> GELU -> Linear."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)    # expand: d_model -> d_ff
        self.fc2 = nn.Linear(d_ff, d_model)    # contract: d_ff -> d_model

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))   # compact one-liner: fc1 -> GELU -> fc2

# ============================================================
# SECTION 2: Build the Router class
# ============================================================
print("--- SECTION 2: The Router Architecture ---")
print()

class Router(nn.Module):
    """
    Router: decides which experts handle each token.
    Architecture: Linear(d_model, num_experts) -> softmax -> top-K
    """
    def __init__(self, d_model, num_experts):
        super().__init__()
        # one Linear layer: maps d_model features to num_experts scores
        # bias=False: most MoE implementations omit router bias
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x, k, add_noise=False):
        """
        Args:
            x          : token embeddings, shape [batch, seq_len, d_model]
            k          : number of experts to select (top-K)
            add_noise  : if True, add random noise to logits (for training exploration)

        Returns:
            topk_weights: normalized weights for selected experts, shape [batch, seq, k]
            topk_indices: which experts were selected, shape [batch, seq, k]
            all_probs   : full softmax distribution, shape [batch, seq, num_experts]
                          (needed for auxiliary loss computation)
        """
        # Step 1: compute raw logits (unnormalized scores for each expert)
        logits = self.gate(x)               # [batch, seq_len, num_experts]

        # Optional: add noise during training to encourage exploration
        # Without noise, router can get "stuck" always picking the same experts
        if add_noise:
            # noise scaled by 1/num_experts (standard from Switch Transformer paper)
            noise = torch.randn_like(logits) * (1.0 / logits.shape[-1])
            logits = logits + noise         # add small random perturbation

        # Step 2: softmax over expert dimension -> probabilities summing to 1
        all_probs = F.softmax(logits, dim=-1)   # [batch, seq_len, num_experts]

        # Step 3: top-K selection -- keep K highest probabilities
        # torch.topk returns (values, indices) of the K largest elements
        topk_weights, topk_indices = torch.topk(all_probs, k=k, dim=-1)
        # topk_weights:  [batch, seq_len, k]  -- the K highest probabilities
        # topk_indices:  [batch, seq_len, k]  -- which expert (0..num_experts-1)

        # Step 4: renormalize the K selected weights so they sum to 1
        # Before renorm: sum of K weights < 1 (we discarded N-K experts)
        # After renorm:  sum of K weights == 1 (valid for weighted average)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        # keepdim=True: preserve the last dimension so broadcasting works

        return topk_weights, topk_indices, all_probs


# Test the router
d_model     = 16     # embedding dimension (tiny for demonstration)
num_experts = 4      # 4 experts to choose from
top_k       = 2      # select top-2 experts per token

router = Router(d_model, num_experts)    # create router instance

# Create 5 different "tokens" (random embeddings)
# In a real model, these come from the embedding table + positional encoding
num_tokens = 5                               # 5 tokens to demonstrate
tokens = torch.randn(1, num_tokens, d_model) # shape: [batch=1, seq=5, d_model=16]

print(f"Input shape:      {tokens.shape}  [batch=1, seq={num_tokens}, d_model={d_model}]")
print(f"Router gate shape: {list(router.gate.weight.shape)}  [num_experts x d_model]")
print()

# Run the router (without noise, for clean demonstration)
topk_weights, topk_indices, all_probs = router(tokens, k=top_k, add_noise=False)

print(f"Output shapes:")
print(f"  topk_weights  : {topk_weights.shape}   [batch=1, seq=5, k=2]")
print(f"  topk_indices  : {topk_indices.shape}   [batch=1, seq=5, k=2]")
print(f"  all_probs     : {all_probs.shape}   [batch=1, seq=5, num_experts=4]")
print()

# ============================================================
# SECTION 3: Show routing decisions for each token
# ============================================================
print("--- SECTION 3: Routing Decisions for Each Token ---")
print()
print(f"{'Token':>6} | {'Expert probs':>40} | {'Top-2 experts':>14} | {'Weights':>16}")
print("-" * 80)

for i in range(num_tokens):                  # loop over each of the 5 tokens
    probs_i = all_probs[0, i]                # shape: [num_experts] -- probs for token i
    experts_i = topk_indices[0, i]           # shape: [k] -- selected expert indices
    weights_i = topk_weights[0, i]           # shape: [k] -- normalized weights

    # format for printing
    prob_str   = str([round(p.item(), 3) for p in probs_i])   # all 4 expert probs
    expert_str = str(experts_i.tolist())                       # e.g. [2, 0]
    weight_str = str([round(w.item(), 3) for w in weights_i]) # e.g. [0.6, 0.4]

    print(f"Token {i:>2}  | {prob_str:>40} | {expert_str:>14} | {weight_str:>16}")

print()
print("Observe: different tokens route to DIFFERENT experts.")
print("The weights for each token sum to 1.0 (after renormalization).")
print()

# Verify weights sum to 1 for each token
for i in range(num_tokens):
    w_sum = topk_weights[0, i].sum().item()     # sum of K weights for token i
    print(f"Token {i} top-K weights sum: {w_sum:.6f}  (should be 1.0)")

print()

# ============================================================
# SECTION 4: Show weighted combination of expert outputs
# ============================================================
print("--- SECTION 4: Weighted Combination of Expert Outputs ---")
print()

d_ff = 32    # expert hidden dimension

# Create 4 experts
experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

# Focus on Token 0 to show the full routing + combination
token_0 = tokens[:, 0:1, :]         # shape: [1, 1, d_model] -- just token 0
expert_picks_0 = topk_indices[0, 0] # which experts token 0 routes to
weights_0 = topk_weights[0, 0]      # their weights

print(f"Token 0 routes to experts: {expert_picks_0.tolist()}")
print(f"With weights:              {[round(w.item(), 4) for w in weights_0]}")
print()

# Run each selected expert on token 0
output_combined = torch.zeros(1, 1, d_model)  # start with zero tensor

for k_pos in range(top_k):                    # for each of the K=2 selected experts
    expert_id = expert_picks_0[k_pos].item()  # which expert to run (integer)
    weight    = weights_0[k_pos].item()       # its weight (float)

    expert_output = experts[expert_id](token_0)   # run this expert on token 0
    output_combined = output_combined + weight * expert_output  # add weighted output

    print(f"  Expert {expert_id} output (first 4 values): "
          f"{expert_output[0, 0, :4].detach().numpy().round(3)}")
    print(f"  Contribution (weight={weight:.4f}):          "
          f"{(weight * expert_output)[0, 0, :4].detach().numpy().round(3)}")
    print()

print(f"Final combined output (first 4 values): "
      f"{output_combined[0, 0, :4].detach().numpy().round(3)}")
print("The combined output is the WEIGHTED AVERAGE of the two expert outputs.")
print()

# ============================================================
# SECTION 5: Full MoELayer combining router + experts
# ============================================================
print("--- SECTION 5: Full MoELayer (Router + N Experts) ---")
print()

class MoELayer(nn.Module):
    """
    Full MoE Layer: Router + N Expert FFNs.
    Drop-in replacement for a standard FFN in a transformer block.
    """
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts      # total number of experts N
        self.top_k = top_k                 # experts to activate per token K
        # create N independent expert FFNs
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        # create router: maps d_model -> num_experts probabilities
        self.router = Router(d_model, num_experts)

    def forward(self, x):
        batch, seq_len, dim = x.shape      # unpack input dimensions

        # get routing decisions for all tokens at once
        topk_weights, topk_indices, all_probs = self.router(x, self.top_k)
        # topk_weights:  [batch, seq_len, K]
        # topk_indices:  [batch, seq_len, K]
        # all_probs:     [batch, seq_len, num_experts]  -- for aux_loss

        # flatten batch and seq dims for simpler dispatch
        x_flat   = x.view(batch * seq_len, dim)            # [B*S, d_model]
        w_flat   = topk_weights.view(batch * seq_len, self.top_k)  # [B*S, K]
        idx_flat = topk_indices.view(batch * seq_len, self.top_k)  # [B*S, K]

        # initialize output tensor with zeros
        output = torch.zeros_like(x_flat)   # [B*S, d_model]

        # for each k position (0, 1, ..., K-1) and each expert,
        # find tokens assigned to that expert at that k position
        for k_pos in range(self.top_k):            # loop over K expert slots
            for eid in range(self.num_experts):    # loop over all N experts
                # create boolean mask: which tokens chose expert eid at position k_pos?
                mask = (idx_flat[:, k_pos] == eid)  # shape: [B*S], True/False
                if not mask.any():                   # skip if no tokens chose this expert
                    continue
                # extract only the tokens assigned to this expert
                token_subset = x_flat[mask]          # [num_masked_tokens, d_model]
                # run expert on those tokens
                expert_out = self.experts[eid](token_subset)  # [num_masked_tokens, d_model]
                # get the weight for this (token, k_pos) combination
                weight = w_flat[mask, k_pos].unsqueeze(-1)    # [num_masked_tokens, 1]
                # add weighted expert output to accumulator
                output[mask] += weight * expert_out            # broadcast weight over d_model

        # reshape back to [batch, seq_len, d_model]
        output = output.view(batch, seq_len, dim)

        return output, all_probs    # return all_probs for auxiliary loss computation


# Test the full MoELayer
moe_layer = MoELayer(d_model=16, d_ff=32, num_experts=4, top_k=2)

# Input: batch of 2 sequences, each 6 tokens long, embedding dim 16
test_input = torch.randn(2, 6, 16)          # [batch=2, seq=6, d_model=16]
print(f"MoELayer input shape:  {test_input.shape}")

moe_output, router_probs = moe_layer(test_input)    # forward pass

print(f"MoELayer output shape: {moe_output.shape}")  # [2, 6, 16] -- same as input
print(f"Router probs shape:    {router_probs.shape}") # [2, 6, 4] -- all expert probs

# Verify output shape matches input shape (MoELayer is a drop-in replacement)
assert moe_output.shape == test_input.shape, "Output shape must match input shape!"
print("Shape check PASSED: output shape == input shape (MoELayer is a drop-in FFN replacement)")

print()
print("=" * 60)
print("Example 02 complete!")
print("You built a Router (Linear->softmax->top-K->renorm),")
print("saw how different tokens route to different experts,")
print("computed weighted combinations, and built a full MoELayer.")
print("=" * 60)
