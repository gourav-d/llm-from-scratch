"""
Module 19 - Mixture of Experts
Example 01: MoE Concept and Intuition

GLOSSARY
--------
Expert       : One FFN (Feed-Forward Network) inside a MoE model.
               Like a specialist doctor: handles one domain very well.
               Architecture: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model).
               This is IDENTICAL to the FFN you built in M05 and M18.

MoE (Mixture of Experts): A model where N expert FFNs exist but only K run per token.
               N experts total, K experts active per token.
               Total params = N * expert_params. Active params = K * expert_params.

Dense FFN    : The standard FFN where ALL tokens use the SAME weights.
               100% of parameters are always active.
               The FFN you built in M05 -- nothing new.

Conditional computation: Parameters that EXIST in memory but are NOT always used.
               MoE exploits this: store N experts, run only K per token.
               Same compute as K experts, same storage as N experts.

FLOPS        : Floating Point Operations Per Second -- a measure of compute cost.
               Dense model: params doubled -> FLOPs doubled.
               MoE model: experts doubled -> total params doubled, FLOPs UNCHANGED.

d_model      : Embedding dimension. Every token is a vector of size d_model.
               In this example: d_model = 8 (tiny, for clarity).

d_ff         : Hidden dimension inside the FFN (usually 4 * d_model).
               In this example: d_ff = 16.
"""

import torch                        # PyTorch: deep learning framework
import torch.nn as nn               # nn: neural network building blocks
import torch.nn.functional as F     # F: stateless functions (gelu, softmax, etc.)

# Set random seed so results are reproducible (same every run)
torch.manual_seed(42)               # like C#: Random random = new Random(42);

print("=" * 60)                     # print separator line (60 dashes)
print("Example 01: MoE Concept and Intuition")
print("=" * 60)
print()                             # blank line for readability

# ============================================================
# SECTION 1: Build a single Expert (FFN layer)
# ============================================================
print("--- SECTION 1: One Expert = One FFN Layer ---")
print()

# Define hyperparameters (small values so output is easy to read)
d_model = 8     # each token is a vector of 8 numbers
d_ff = 16       # hidden dimension inside the FFN (2x d_model here)

# Build one expert: Linear -> GELU -> Linear
# This is EXACTLY the FFN you built in M05. No difference.
class Expert(nn.Module):                         # inherit from nn.Module (like C# abstract base class)
    """One expert FFN: Linear -> GELU -> Linear."""
    def __init__(self, d_model, d_ff):           # constructor receives dimensions
        super().__init__()                       # must call parent constructor (C#: base())
        self.fc1 = nn.Linear(d_model, d_ff)     # first linear: expand d_model -> d_ff
        self.fc2 = nn.Linear(d_ff, d_model)     # second linear: contract d_ff -> d_model

    def forward(self, x):                        # forward pass: how data flows through expert
        x = F.gelu(self.fc1(x))                 # fc1 -> GELU activation
        x = self.fc2(x)                          # fc2 -> final projection back to d_model
        return x                                 # output same shape as input


# Create one expert instance
expert = Expert(d_model, d_ff)                   # create expert (randomly initialized)

# Count its parameters
num_params = sum(p.numel() for p in expert.parameters())  # sum up all parameter counts
print(f"One expert has {num_params} parameters.")         # numel() = number of elements
# Expected: fc1 (8*16 + 16=144) + fc2 (16*8 + 8=136) = 280 params

# Create a fake "token" -- a random vector of size d_model
# In a real model, this comes from the embedding + position encoding
token = torch.randn(1, 1, d_model)   # shape: [batch=1, seq_len=1, d_model=8]
print(f"Input token shape: {token.shape}")  # [1, 1, 8]

# Run the token through the expert
output = expert(token)                # forward pass
print(f"Expert output shape: {output.shape}")  # [1, 1, 8] -- same shape as input
print(f"Input  (first 4 values): {token[0, 0, :4].detach().numpy().round(3)}")
print(f"Output (first 4 values): {output[0, 0, :4].detach().numpy().round(3)}")
print("Notice: input and output have the same shape but DIFFERENT values.")
print("The expert has TRANSFORMED the token representation.")
print()

# ============================================================
# SECTION 2: Build N experts and show they have different weights
# ============================================================
print("--- SECTION 2: N Independent Experts ---")
print()

num_experts = 4                       # we will have 4 experts

# Create a list of N experts -- each has its own INDEPENDENT weights
# This is like N separate FFN layers with no shared parameters
experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
# nn.ModuleList is like C# List<Expert> but PyTorch knows about the parameters

# Check: how many TOTAL parameters does the MoE have?
total_moe_params = sum(p.numel() for p in experts.parameters())
print(f"Number of experts: {num_experts}")
print(f"Params per expert: {num_params}")
print(f"Total MoE params:  {total_moe_params}  (= {num_experts} x {num_params})")
print(f"Dense FFN params:  {num_params}           (just 1 FFN)")
print(f"MoE has {num_experts}x more TOTAL params than a single dense FFN.")
print()

# Show that experts have DIFFERENT (independent) weights
# Compare fc1 weights of Expert 0 vs Expert 1
w0 = experts[0].fc1.weight          # shape: [d_ff, d_model] = [16, 8]
w1 = experts[1].fc1.weight          # shape: [d_ff, d_model] = [16, 8]
are_same = torch.allclose(w0, w1)   # check if all values are equal
print(f"Expert 0 fc1 weights == Expert 1 fc1 weights? {are_same}")
print("(Expected: False -- each expert has independently initialized weights)")
print()

# ============================================================
# SECTION 3: Manual routing -- send same token to different experts
# ============================================================
print("--- SECTION 3: Same Token, Different Experts -> Different Outputs ---")
print()

# Create a token that represents the word "cat" (in reality, embeddings come
# from a lookup table, but here we just use a fixed random vector)
cat_token = torch.randn(1, 1, d_model)   # fake embedding for "cat"

# Send the cat token through Expert 0
output_from_expert_0 = experts[0](cat_token)

# Send the SAME cat token through Expert 2 (different expert)
output_from_expert_2 = experts[2](cat_token)

print(f"'cat' token input:          {cat_token[0, 0].detach().numpy().round(3)}")
print(f"Output via Expert 0:        {output_from_expert_0[0, 0].detach().numpy().round(3)}")
print(f"Output via Expert 2:        {output_from_expert_2[0, 0].detach().numpy().round(3)}")
print()
print("KEY INSIGHT: The SAME token produces DIFFERENT outputs depending on")
print("which expert processes it. After training, each expert specializes.")
print("Expert 0 might be better at math tokens, Expert 2 at language tokens.")
print()

# ============================================================
# SECTION 4: Weighted combination of two expert outputs
# ============================================================
print("--- SECTION 4: Weighted Sum of Expert Outputs ---")
print()

# In a real MoE, the router assigns WEIGHTS to each selected expert.
# The final output is a weighted average of the selected experts' outputs.

# Fake router weights for illustration (in real MoE, these come from softmax)
# Say we pick Expert 0 with weight 0.7 and Expert 2 with weight 0.3
weight_0 = 0.7    # expert 0 is more confident for this token
weight_2 = 0.3    # expert 2 is less confident
# Note: weights sum to 1.0 (like probabilities after softmax)
print(f"Router picks Expert 0 (weight={weight_0}) and Expert 2 (weight={weight_2})")
print(f"Weights sum to: {weight_0 + weight_2}  (must be 1.0)")

# Compute weighted sum
combined_output = (weight_0 * output_from_expert_0 +    # expert 0 contribution
                   weight_2 * output_from_expert_2)     # expert 2 contribution

print(f"Expert 0 output:            {output_from_expert_0[0, 0].detach().numpy().round(3)}")
print(f"Expert 2 output:            {output_from_expert_2[0, 0].detach().numpy().round(3)}")
print(f"Combined (0.7*E0 + 0.3*E2): {combined_output[0, 0].detach().numpy().round(3)}")
print()
print("The final output is a weighted blend of Expert 0 and Expert 2.")
print("Experts 1 and 3 are COMPLETELY SKIPPED for this token (no computation).")
print()

# ============================================================
# SECTION 5: Parameter count comparison -- Dense vs MoE
# ============================================================
print("--- SECTION 5: Dense FFN vs MoE Parameter Comparison ---")
print()

# Compare a SINGLE dense FFN vs a 4-expert MoE with same per-expert size
d_model_big = 512    # realistic embedding dimension for a small LLM
d_ff_big    = 2048   # typical FFN hidden dim (4x d_model)
num_experts_big = 4  # 4 experts in MoE
top_k = 2            # activate K=2 experts per token

# Dense FFN parameter count
# Two linear layers: fc1 (d_model x d_ff) + fc2 (d_ff x d_model)
dense_params = (d_model_big * d_ff_big +    # fc1 weights: 512 * 2048
                d_ff_big +                  # fc1 bias
                d_ff_big * d_model_big +    # fc2 weights: 2048 * 512
                d_model_big)               # fc2 bias
print(f"Dense FFN (d_model={d_model_big}, d_ff={d_ff_big}):")
print(f"  Total params   = {dense_params:,}")         # format with commas
print(f"  Active params  = {dense_params:,}  (100% always used)")
print()

# MoE FFN: same per-expert size, 4 experts, top-2
moe_total_params = num_experts_big * dense_params     # 4 experts
moe_active_params = top_k * dense_params              # only 2 run per token
print(f"MoE FFN (num_experts={num_experts_big}, top_k={top_k}, same expert size):")
print(f"  Total params   = {moe_total_params:,}  ({num_experts_big}x more than dense)")
print(f"  Active params  = {moe_active_params:,}  ({top_k}/{num_experts_big} = {top_k/num_experts_big:.0%} of total)")
print()

# Express the key ratios
print(f"Total params ratio (MoE / Dense):   {moe_total_params / dense_params:.1f}x")
print(f"Active params ratio (MoE / Dense):  {moe_active_params / dense_params:.1f}x")
print()
print("KEY TAKEAWAY:")
print(f"  MoE has {num_experts_big}x MORE TOTAL params -> can store {num_experts_big}x more knowledge.")
print(f"  MoE uses {top_k/num_experts_big:.0%} of params per token -> SAME compute as {top_k} dense FFNs.")
print(f"  This is the 'free lunch' of MoE: more knowledge, same compute cost.")

# ============================================================
# SECTION 6: Real-world MoE scale numbers
# ============================================================
print()
print("--- SECTION 6: Real-World MoE Scale ---")
print()

# DeepSeek-V3 approximate numbers
ds_total = 671e9    # 671 billion total parameters
ds_active = 37e9    # 37 billion active per token
ds_ratio = ds_total / ds_active    # compute saving factor

print("DeepSeek-V3:")
print(f"  Total params:  {ds_total/1e9:.0f}B")
print(f"  Active params: {ds_active/1e9:.0f}B per token")
print(f"  Compute ratio: {ds_ratio:.1f}x FEWER FLOPs than a dense {ds_total/1e9:.0f}B model")

# Mixtral 8x7B approximate numbers
mx_total = 47e9     # 47 billion total
mx_active = 13e9    # 13 billion active
mx_ratio = mx_total / mx_active

print()
print("Mixtral 8x7B:")
print(f"  Total params:  {mx_total/1e9:.0f}B")
print(f"  Active params: {mx_active/1e9:.0f}B per token")
print(f"  Compute ratio: {mx_ratio:.1f}x FEWER FLOPs than a dense {mx_total/1e9:.0f}B model")

print()
print("=" * 60)
print("Example 01 complete!")
print("You built an Expert (FFN), created 4 independent experts,")
print("manually routed a token, computed weighted sum outputs,")
print("and compared dense vs MoE parameter counts.")
print("=" * 60)
