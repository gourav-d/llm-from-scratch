"""
Module 19 - Mixture of Experts
Exercise 01: MoE Concept -- Building Experts and Manual Routing

GLOSSARY
--------
Expert       : One FFN (Feed-Forward Network) layer in a MoE model.
               Architecture: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model).
               Identical to the standard transformer FFN you know from M05/M18.

d_model      : Embedding dimension. Every token is represented as a vector of d_model numbers.
               Example: d_model=8 means each token is 8 numbers.

d_ff         : Hidden dimension INSIDE the expert FFN. Usually 2x or 4x d_model.
               The expert expands from d_model -> d_ff -> d_model.

nn.ModuleList: A PyTorch list that properly registers all contained nn.Modules.
               Like C# List<NeuralNetworkLayer>, but PyTorch tracks all their parameters.
               Use this when you have N experts: nn.ModuleList([Expert(...) for _ in range(N)]).

Total params : Sum of ALL expert parameters across ALL N experts.
               total_params = N * params_per_expert
               Even idle experts count toward total params (they are stored in memory).

Active params: Parameters that are ACTUALLY USED for a given token.
               active_params = K * params_per_expert  (K = top-K, e.g. K=2)
               Active params determine COMPUTE COST (FLOPs).

Weighted sum : The final output of a MoE layer:
               output = w1 * expert1(token) + w2 * expert2(token)
               where w1 + w2 = 1.0. The weights come from the router (softmax scores).

Manual routing: For learning purposes only -- in real MoE, the ROUTER decides.
               Here we manually choose which expert each token goes to.
"""

import torch                          # PyTorch: deep learning framework
import torch.nn as nn                 # nn: neural network building blocks
import torch.nn.functional as F       # F: functions like gelu, softmax

torch.manual_seed(0)                  # fixed seed -- same results every run

print("=" * 60)
print("Exercise 01: MoE Concept -- Build Expert and Route Manually")
print("=" * 60)
print()

# Hyperparameters (small for clarity)
d_model = 8     # each token = 8-element vector
d_ff    = 16    # hidden dimension inside expert (2x d_model)

# ============================================================
#  EXERCISE 1
#  Topic: Build a single Expert (FFN) layer
#
#  Background:
#    In MoE, each "expert" is just a standard FFN:
#      Linear(d_model, d_ff) -> GELU activation -> Linear(d_ff, d_model)
#    This is IDENTICAL to the FFN you built in M05 and M18.
#    The only difference: MoE has N of these with INDEPENDENT weights.
#
#  Your Task:
#    Complete the Expert class:
#      __init__: create self.fc1 (d_model -> d_ff) and self.fc2 (d_ff -> d_model)
#      forward:  fc1(x) -> gelu -> fc2, return the result
#
#  C# Analogy:
#    Like a simple pipeline processor class:
#      public class Expert {
#          private Linear fc1, fc2;
#          public Expert(int dModel, int dFf) {
#              fc1 = new Linear(dModel, dFf);
#              fc2 = new Linear(dFf, dModel);
#          }
#          public Tensor Forward(Tensor x) => fc2(Gelu(fc1(x)));
#      }
# ============================================================

class Expert(nn.Module):
    """One expert FFN layer: Linear -> GELU -> Linear."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO: create self.fc1: Linear from d_model to d_ff
        # TODO: create self.fc2: Linear from d_ff to d_model
        pass  # remove this line when you add your code

    def forward(self, x):
        # TODO: apply fc1, then gelu, then fc2. Return the result.
        pass  # replace this with your implementation


# --- Test Exercise 1 ---
print("--- Exercise 1 Test ---")
expert = Expert(d_model, d_ff)          # create one expert
token  = torch.randn(1, 1, d_model)    # fake token [batch=1, seq=1, d_model=8]
output = expert(token)                  # run expert

if output is not None:                  # graceful handling if not yet implemented
    print(f"Expert input  shape: {token.shape}")
    print(f"Expert output shape: {output.shape}")
    # Both must be [1, 1, 8] -- expert preserves shape
    assert output.shape == token.shape, "Expert must preserve input shape!"
    print("PASS: Expert output shape matches input shape.")
else:
    print("Hint: Implement Expert.__init__ and Expert.forward, then run again.")
print()

# ============================================================
#  EXERCISE 2
#  Topic: Create N experts and count total vs active parameters
#
#  Background:
#    MoE has N independent expert FFNs. Each has its OWN weights.
#    Total params = N * params_per_expert  (all stored in GPU memory)
#    Active params = K * params_per_expert (only K experts run per token)
#    This "conditional computation" is what makes MoE efficient.
#
#  Your Task:
#    Complete create_n_experts(n, d_model, d_ff) to return:
#      - an nn.ModuleList of n Expert instances
#      - total parameter count across all experts
#      - active parameter count (assume top_k=2)
#
#  C# Analogy:
#    List<Expert> experts = Enumerable.Range(0, n)
#        .Select(_ => new Expert(dModel, dFf))
#        .ToList();
# ============================================================

def create_n_experts(n, d_model, d_ff, top_k=2):
    """
    Create n independent Expert FFN layers.

    Args:
        n       : number of experts to create
        d_model : embedding dimension
        d_ff    : hidden dimension inside each expert
        top_k   : number of active experts per token (for active param count)

    Returns:
        experts        : nn.ModuleList of n Expert instances
        total_params   : total parameters across all n experts (int)
        active_params  : parameters used per token = top_k * params_per_expert (int)
    """
    # TODO: create nn.ModuleList of n Expert instances
    # TODO: count total params (sum of all params across all experts)
    # TODO: count active params (top_k * params for one expert)
    # TODO: return (experts, total_params, active_params)
    pass


# --- Test Exercise 2 ---
print("--- Exercise 2 Test ---")
result = create_n_experts(4, d_model, d_ff, top_k=2)

if result is not None:
    experts_list, total, active = result
    print(f"Number of experts: {len(experts_list)}")
    print(f"Total params:      {total}")
    print(f"Active params:     {active}  (for top_k=2)")
    print(f"Inactive params:   {total - active}  (stored but not run per token)")

    # Verify experts are independent (different weights)
    if len(experts_list) >= 2 and hasattr(experts_list[0], 'fc1'):
        same = torch.allclose(experts_list[0].fc1.weight, experts_list[1].fc1.weight)
        print(f"Expert 0 == Expert 1 weights? {same}  (expected: False -- independent)")
else:
    print("Hint: Return (nn.ModuleList, total_params, active_params)")
print()

# ============================================================
#  EXERCISE 3
#  Topic: Manually route a token to a specific expert
#
#  Background:
#    In a real MoE, the ROUTER decides which expert to use.
#    Here we do it manually to understand the basic flow:
#      1. Pick which expert to use (e.g., expert_id = 2)
#      2. Run the token through that expert
#      3. See the output
#
#  Your Task:
#    Complete manual_route(token, experts, expert_id) to:
#      - Run token through the expert at experts[expert_id]
#      - Return the output
#
#  C# Analogy:
#    Tensor ManualRoute(Tensor token, List<Expert> experts, int expertId) {
#        return experts[expertId].Forward(token);
#    }
# ============================================================

def manual_route(token, experts, expert_id):
    """
    Manually route a token to a specific expert.

    Args:
        token     : tensor [batch, seq_len, d_model]
        experts   : nn.ModuleList of Expert instances
        expert_id : which expert to use (integer index 0..N-1)

    Returns:
        output : tensor [batch, seq_len, d_model] -- expert's output
    """
    # TODO: run token through experts[expert_id] and return the result
    pass


# --- Test Exercise 3 ---
print("--- Exercise 3 Test ---")
if result is not None:
    experts_list, _, _ = result         # reuse experts from Exercise 2
    cat_token = torch.randn(1, 1, d_model)   # fake token for "cat"

    out0 = manual_route(cat_token, experts_list, expert_id=0)  # send to expert 0
    out2 = manual_route(cat_token, experts_list, expert_id=2)  # send to expert 2

    if out0 is not None and out2 is not None:
        print(f"Same token through Expert 0: {out0[0,0,:4].detach().numpy().round(3)}")
        print(f"Same token through Expert 2: {out2[0,0,:4].detach().numpy().round(3)}")
        same_out = torch.allclose(out0, out2, atol=1e-4)
        print(f"Same output from Expert 0 and Expert 2? {same_out}  (expected: False)")
    else:
        print("Hint: Return experts[expert_id](token)")
else:
    print("Skipped (depends on Exercise 2)")
print()

# ============================================================
#  EXERCISE 4
#  Topic: Compute weighted sum of two expert outputs
#
#  Background:
#    MoE final output = weighted sum of K expert outputs.
#    The router gives a weight (probability) to each selected expert.
#    All K weights sum to 1.0.
#    output = w1 * expert1_out + w2 * expert2_out  (where w1 + w2 = 1.0)
#
#  Your Task:
#    Complete weighted_expert_sum(expert_outputs, weights) to:
#      - Compute and return the weighted sum
#      - expert_outputs: list of tensors (each [batch, seq, d_model])
#      - weights: list of floats (must sum to 1.0)
#
#  C# Analogy:
#    Tensor WeightedExpertSum(List<Tensor> outputs, List<float> weights) {
#        Tensor result = Tensor.Zeros(outputs[0].Shape);
#        for (int i = 0; i < outputs.Count; i++)
#            result += weights[i] * outputs[i];
#        return result;
#    }
# ============================================================

def weighted_expert_sum(expert_outputs, weights):
    """
    Compute weighted sum of expert outputs.

    Args:
        expert_outputs : list of tensors, each shape [batch, seq, d_model]
        weights        : list of floats that sum to 1.0

    Returns:
        combined : tensor [batch, seq, d_model] -- weighted average of expert outputs
    """
    # TODO: initialize a zero tensor same shape as expert_outputs[0]
    # TODO: loop over outputs and weights, add weight * output to accumulator
    # TODO: return the combined tensor
    pass


# --- Test Exercise 4 ---
print("--- Exercise 4 Test ---")
if result is not None and out0 is not None and out2 is not None:
    # Use outputs from Exercise 3 (out0 from Expert 0, out2 from Expert 2)
    w1, w2 = 0.7, 0.3                       # weights summing to 1.0
    combined = weighted_expert_sum([out0, out2], [w1, w2])

    if combined is not None:
        print(f"Expert 0 output (first 3): {out0[0,0,:3].detach().numpy().round(4)}")
        print(f"Expert 2 output (first 3): {out2[0,0,:3].detach().numpy().round(4)}")
        print(f"Combined 0.7*E0+0.3*E2:  {combined[0,0,:3].detach().numpy().round(4)}")
        print(f"Combined shape: {combined.shape}  (should be {out0.shape})")
        # Verify: 0.7 * out0 + 0.3 * out2 should match
        expected = 0.7 * out0 + 0.3 * out2
        assert torch.allclose(combined, expected, atol=1e-5), "Weighted sum is incorrect!"
        print("PASS: Weighted sum is correct.")
    else:
        print("Hint: accumulate weight * output for each (output, weight) pair")
else:
    print("Skipped (depends on earlier exercises)")
print()

# ============================================================
#  EXERCISE 5
#  Topic: Compare output shapes -- single expert vs weighted combination
#
#  Background:
#    Key property of MoE: the OUTPUT SHAPE is the SAME as the INPUT SHAPE.
#    Whether you use 1 expert or K experts with weighted sum:
#      input:  [batch, seq, d_model]
#      output: [batch, seq, d_model]
#    This means MoE is a DROP-IN replacement for the standard FFN.
#
#  Your Task:
#    Complete compare_shapes(token, experts_list, expert_id_1, expert_id_2, w1, w2) to:
#      1. Run token through expert_id_1 and expert_id_2
#      2. Compute weighted sum with weights w1 and w2
#      3. Return (single_output, combined_output)
#         where single_output = experts[expert_id_1](token)
#         and combined_output = w1 * experts[expert_id_1](token) + w2 * experts[expert_id_2](token)
# ============================================================

def compare_shapes(token, experts_list, expert_id_1, expert_id_2, w1, w2):
    """
    Run token through two experts and show both single and combined outputs.

    Returns:
        single_output   : output from expert_id_1 alone
        combined_output : w1 * expert1(token) + w2 * expert2(token)
    """
    # TODO: get output from expert_id_1
    # TODO: get output from expert_id_2
    # TODO: compute weighted sum
    # TODO: return (single_output, combined_output)
    pass


# --- Test Exercise 5 ---
print("--- Exercise 5 Test ---")
if result is not None:
    test_token = torch.randn(2, 4, d_model)   # [batch=2, seq=4, d_model=8]
    shape_result = compare_shapes(test_token, experts_list,
                                  expert_id_1=0, expert_id_2=1,
                                  w1=0.6, w2=0.4)

    if shape_result is not None:
        single_out, combined_out = shape_result
        print(f"Input shape:            {test_token.shape}")
        print(f"Single expert output:   {single_out.shape}")
        print(f"Combined (2 experts):   {combined_out.shape}")
        print(f"All shapes equal: {test_token.shape == single_out.shape == combined_out.shape}")
        print("KEY POINT: MoE is a drop-in FFN replacement -- shapes always match!")
    else:
        print("Hint: Return (output_from_expert1, weighted_sum_of_both)")
else:
    print("Skipped (depends on earlier exercises)")

print()
print("=" * 60)
print("Exercise 01 complete!")
print("You built the Expert class, created N independent experts,")
print("manually routed tokens, and computed weighted sum outputs.")
print("These are the building blocks of every MoE model in production.")
print("=" * 60)
