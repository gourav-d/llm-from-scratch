"""
Module 19 - Mixture of Experts
Exercise 02: The Router Network

GLOSSARY
--------
Router       : A small Linear layer that maps token embeddings to expert probabilities.
               One Linear(d_model, num_experts) layer -- no hidden layers.
               Output is raw logits; apply softmax to get probabilities.

Softmax      : Converts raw scores (logits) into probabilities.
               formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
               Output: always positive, always sums to 1.0.
               In PyTorch: F.softmax(x, dim=-1) over the last dimension.

Top-K gating : Keep only the K highest probabilities; zero out the rest.
               torch.topk(probs, k, dim=-1) returns (values, indices).
               NON-DIFFERENTIABLE (like argmax) -- but we need it for routing.

Renormalization: After top-K, the K selected weights no longer sum to 1.
               Apply softmax AGAIN over the K values: they now sum to 1.
               This makes the weighted sum of experts a valid weighted average.

weighted sum : Final MoE output = sum(weight_i * expert_i(token)) for i in top-K.
               Weights are the renormalized top-K probabilities.
               Like a weighted average of expert opinions.

Router params: A tiny fraction of total model params.
               d_model=512, num_experts=8: router has 512*8 = 4,096 params.
               Experts have 8M+ params each. Router controls but is much smaller.

Routing collapse: When the router always picks the same K experts for ALL tokens.
               This is the load balancing problem fixed in Exercise 03.
"""

import torch                           # PyTorch: deep learning framework
import torch.nn as nn                  # nn: neural network building blocks
import torch.nn.functional as F        # F: functions like softmax, gelu

torch.manual_seed(0)                   # fixed seed for reproducibility

print("=" * 60)
print("Exercise 02: The Router Network")
print("=" * 60)
print()

# Hyperparameters
d_model     = 16    # embedding dimension
num_experts = 4     # number of experts
top_k       = 2     # top-K selection

# Expert class (from Exercise 01 -- provided here for completeness)
class Expert(nn.Module):
    """Standard FFN expert: Linear -> GELU -> Linear."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)     # expand
        self.fc2 = nn.Linear(d_ff, d_model)     # contract

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))    # fc1 -> GELU -> fc2

# ============================================================
#  EXERCISE 1
#  Topic: Build the Router class (Linear only, no softmax yet)
#
#  Background:
#    The router's first job: compute a raw score (logit) for each expert.
#    Just one Linear layer: d_model inputs -> num_experts outputs.
#    No bias is standard practice for MoE routers.
#
#  Your Task:
#    Complete the Router class:
#      __init__: create self.gate = nn.Linear(d_model, num_experts, bias=False)
#      forward:  return self.gate(x)  -- just the raw logits, no softmax yet
#
#  C# Analogy:
#    public class Router {
#        private Linear gate;
#        public Router(int dModel, int numExperts) {
#            gate = new Linear(dModel, numExperts, useBias: false);
#        }
#        public Tensor Forward(Tensor x) => gate.Forward(x);  // raw logits
#    }
# ============================================================

class Router(nn.Module):
    """Router: maps token embeddings to expert logits."""
    def __init__(self, d_model, num_experts):
        super().__init__()
        # TODO: create self.gate = nn.Linear(d_model, num_experts, bias=False)
        pass

    def forward(self, x):
        # TODO: return raw logits from self.gate applied to x
        # x shape: [batch, seq_len, d_model]
        # output shape: [batch, seq_len, num_experts]
        pass


# --- Test Exercise 1 ---
print("--- Exercise 1 Test ---")
router = Router(d_model, num_experts)    # create router
tokens = torch.randn(2, 5, d_model)     # [batch=2, seq=5, d_model=16]
logits = router(tokens)                  # raw scores for each expert

if logits is not None:
    print(f"Router input shape:  {tokens.shape}")
    print(f"Router output shape: {logits.shape}  (expected: [2, 5, 4])")
    assert logits.shape == (2, 5, num_experts), "Wrong shape! Should be [batch, seq, num_experts]"
    print("PASS: Router output shape is correct.")
    print(f"Sample logits for token 0: {logits[0, 0].detach().numpy().round(3)}")
else:
    print("Hint: self.gate is nn.Linear(d_model, num_experts, bias=False)")
print()

# ============================================================
#  EXERCISE 2
#  Topic: Apply softmax to get expert probabilities
#
#  Background:
#    Logits are raw scores -- can be any value (negative, large positive, etc.)
#    Softmax converts them to probabilities: always positive, always sum to 1.
#    We apply softmax over the LAST dimension (dim=-1) = over the expert scores.
#    After softmax: each token has a probability distribution over all N experts.
#
#  Your Task:
#    Complete get_expert_probs(logits) to:
#      - Apply softmax over the last dimension (dim=-1)
#      - Return the result (same shape as input, values in [0,1], sum=1 per token)
#
#  C# Analogy:
#    Tensor GetExpertProbs(Tensor logits) => Softmax(logits, dim: -1);
# ============================================================

def get_expert_probs(logits):
    """
    Convert router logits to probabilities using softmax.

    Args:
        logits : raw scores [batch, seq_len, num_experts]

    Returns:
        probs : probabilities [batch, seq_len, num_experts], sums to 1 per token
    """
    # TODO: apply F.softmax(logits, dim=-1) and return the result
    pass


# --- Test Exercise 2 ---
print("--- Exercise 2 Test ---")
probs = get_expert_probs(logits) if logits is not None else None

if probs is not None:
    print(f"Probs shape:         {probs.shape}  (same as logits)")
    print(f"Sample probs token 0: {probs[0, 0].detach().numpy().round(4)}")
    prob_sum = probs[0, 0].sum().item()
    print(f"Sum of probs for token 0: {prob_sum:.6f}  (must be 1.0)")
    assert abs(prob_sum - 1.0) < 1e-5, "Probs must sum to 1.0!"
    print("PASS: Probabilities sum to 1.0.")
else:
    print("Hint: F.softmax(logits, dim=-1) gives probabilities summing to 1")
print()

# ============================================================
#  EXERCISE 3
#  Topic: Top-K selection -- pick K experts with highest probability
#
#  Background:
#    From N expert probabilities, we select only K (e.g., K=2).
#    Use torch.topk(probs, k, dim=-1) which returns (values, indices).
#    values:  the K highest probabilities
#    indices: which expert indices (0..N-1) were selected
#    After top-K, the K selected values do NOT sum to 1 anymore.
#
#  Your Task:
#    Complete top_k_gate(probs, k) to:
#      - Select top-K probabilities and their indices
#      - Return (topk_values, topk_indices)
#        topk_values:   [batch, seq, k] -- the K highest probs
#        topk_indices:  [batch, seq, k] -- which experts (0..N-1)
#
#  C# Analogy:
#    var (topK, indices) = probs.TopK(k, dim: -1);
# ============================================================

def top_k_gate(probs, k):
    """
    Select top-K experts by probability.

    Args:
        probs : [batch, seq, num_experts] -- softmax probabilities
        k     : number of experts to select

    Returns:
        topk_values  : [batch, seq, k] -- top K probabilities
        topk_indices : [batch, seq, k] -- expert indices (0..N-1)
    """
    # TODO: use torch.topk(probs, k=k, dim=-1) to select top-K
    # TODO: return (topk_values, topk_indices)
    pass


# --- Test Exercise 3 ---
print("--- Exercise 3 Test ---")
topk_result = top_k_gate(probs, k=top_k) if probs is not None else None

if topk_result is not None:
    topk_vals, topk_idx = topk_result
    print(f"topk_values  shape: {topk_vals.shape}  (expected: [2, 5, 2])")
    print(f"topk_indices shape: {topk_idx.shape}  (expected: [2, 5, 2])")
    print(f"Top-2 experts for token 0:   {topk_idx[0, 0].tolist()}")
    print(f"Their probabilities:          {topk_vals[0, 0].detach().numpy().round(4)}")
    vals_sum = topk_vals[0, 0].sum().item()
    print(f"Sum of top-K probs (token 0): {vals_sum:.4f}  (< 1.0, not renormalized yet)")
else:
    print("Hint: torch.topk(probs, k=k, dim=-1) returns (values, indices)")
print()

# ============================================================
#  EXERCISE 4
#  Topic: Renormalize top-K weights so they sum to 1
#
#  Background:
#    After top-K, the K selected probs sum to LESS than 1.0.
#    Example: if all_probs = [0.5, 0.3, 0.1, 0.1] and we pick top-2:
#      top-2 values = [0.5, 0.3]  -> sum = 0.8 (not 1.0)
#    We need them to sum to 1.0 for a valid weighted average.
#    FIX: divide each value by the sum of the K values:
#      renorm = topk_values / topk_values.sum(dim=-1, keepdim=True)
#    Result: [0.5/0.8, 0.3/0.8] = [0.625, 0.375] -> sum = 1.0
#
#  Your Task:
#    Complete renormalize(topk_values) to:
#      - Divide topk_values by their sum along the last dimension
#      - Return renormalized weights (same shape, now sums to 1.0 per token)
#
#  C# Analogy:
#    Tensor Renormalize(Tensor topkValues) =>
#        topkValues / topkValues.Sum(dim: -1, keepDim: true);
# ============================================================

def renormalize(topk_values):
    """
    Renormalize top-K probabilities so they sum to 1.0.

    Args:
        topk_values : [batch, seq, k] -- raw top-K probabilities

    Returns:
        weights : [batch, seq, k] -- same values but now sum to 1.0 per token
    """
    # TODO: divide topk_values by topk_values.sum(dim=-1, keepdim=True)
    # keepdim=True preserves the last dimension for correct broadcasting
    pass


# --- Test Exercise 4 ---
print("--- Exercise 4 Test ---")
if topk_result is not None:
    topk_vals, topk_idx = topk_result
    weights = renormalize(topk_vals)

    if weights is not None:
        print(f"Before renorm (token 0): {topk_vals[0,0].detach().numpy().round(4)}")
        print(f"After renorm  (token 0): {weights[0,0].detach().numpy().round(4)}")
        w_sum = weights[0, 0].sum().item()
        print(f"Sum after renorm: {w_sum:.6f}  (must be 1.0)")
        assert abs(w_sum - 1.0) < 1e-5, "Renormalized weights must sum to 1.0!"
        print("PASS: Weights sum to 1.0 after renormalization.")
    else:
        print("Hint: topk_values / topk_values.sum(dim=-1, keepdim=True)")
else:
    print("Skipped (depends on Exercise 3)")
print()

# ============================================================
#  EXERCISE 5
#  Topic: Build full MoELayer.forward using router + experts + weighted sum
#
#  Background:
#    Putting it all together. The MoELayer forward pass:
#      1. Run router: get probabilities for all experts
#      2. Top-K gate: select K experts and their weights
#      3. Renormalize weights
#      4. For each selected expert: run it on the tokens assigned to it
#      5. Return weighted sum of expert outputs
#    This is the complete MoE computation for one layer.
#
#  Your Task:
#    Complete moe_forward(x, experts, router, top_k) to:
#      - Run the router to get probs
#      - Apply top-K gating and renormalize
#      - Dispatch tokens to experts and accumulate weighted outputs
#      - Return final output tensor (same shape as x)
#
#  C# Analogy:
#    Tensor MoeForward(Tensor x, List<Expert> experts, Router router, int topK) {
#        var probs = Softmax(router.Forward(x));
#        var (weights, indices) = TopK(probs, topK);
#        weights = Renormalize(weights);
#        return WeightedExpertSum(x, experts, weights, indices);
#    }
# ============================================================

def moe_forward(x, experts, router_module, top_k):
    """
    Full MoE layer forward pass.

    Args:
        x             : token embeddings [batch, seq_len, d_model]
        experts       : nn.ModuleList of Expert instances
        router_module : Router instance
        top_k         : number of experts to select per token

    Returns:
        output : [batch, seq_len, d_model] -- weighted sum of expert outputs
    """
    B, S, D = x.shape                              # unpack dimensions
    num_exp = len(experts)                         # number of experts N

    # TODO Step 1: get logits from router_module, apply softmax to get probs
    # probs shape: [B, S, num_experts]

    # TODO Step 2: top-K selection
    # topk_weights, topk_indices = top_k_gate(probs, top_k)

    # TODO Step 3: renormalize
    # topk_weights = renormalize(topk_weights)

    # TODO Step 4: flatten, dispatch to experts, accumulate
    # Hint: flatten [B, S, D] to [B*S, D], loop over k positions and expert ids

    # TODO Step 5: return output reshaped to [B, S, D]
    pass


# --- Test Exercise 5 ---
print("--- Exercise 5 Test ---")
if probs is not None:
    d_ff_test = 32
    experts_test = nn.ModuleList([Expert(d_model, d_ff_test) for _ in range(num_experts)])
    router_test  = Router(d_model, num_experts)

    test_x = torch.randn(2, 5, d_model)           # [batch=2, seq=5, d_model=16]
    output = moe_forward(test_x, experts_test, router_test, top_k)

    if output is not None:
        print(f"MoE input  shape: {test_x.shape}")
        print(f"MoE output shape: {output.shape}  (must match input shape)")
        assert output.shape == test_x.shape, "MoELayer output must match input shape!"
        print("PASS: MoE output shape matches input -- it is a drop-in FFN replacement.")
    else:
        print("Hint: Combine router -> softmax -> topk -> renorm -> expert dispatch -> output")
else:
    print("Skipped (depends on earlier exercises)")

print()
print("=" * 60)
print("Exercise 02 complete!")
print("You built every piece of the router: Linear, softmax,")
print("top-K gating, renormalization, and full MoE dispatch.")
print("=" * 60)
