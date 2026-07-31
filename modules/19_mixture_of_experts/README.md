# Module 19: Mixture of Experts (MoE)

**One-line description**: Replace the dense FFN in each transformer block
with N specialized "expert" FFN layers, routing each token to only K of them --
cutting compute per token while multiplying total model capacity.

---

## Prerequisites

Before this module, you should have completed:

- **M05 - Building an LLM**: You built a GPT-style model in PyTorch.
- **M18 - Qwen3.5 from Scratch**: You built a production-style LLM with
  grouped-query attention, RoPE embeddings, and SwiGLU activation.
- **PyTorch knowledge**: nn.Module, nn.Linear, attention, FFN layers,
  forward pass, loss, optimizer.

You do NOT need any new libraries. This module uses only PyTorch.

---

## What You Will Build

By the end of this module you will have built:

1. A single **Expert** (FFN layer): the basic building block of MoE.
2. A **Router** network: learns to route each token to the right experts.
3. A **MoELayer**: combines N experts + router into one drop-in FFN replacement.
4. Load balancing with **auxiliary loss** (Switch Transformer style).
5. A complete **Mini MoE GPT** (4 experts, top-2 routing, 2 transformer blocks)
   trained on toy text, running on CPU in under 60 seconds.

---

## Real-World Models Using MoE

| Model         | Total Params | Active Params | Experts | Top-K |
|---------------|-------------|---------------|---------|-------|
| Mixtral 8x7B  | ~47B        | ~13B          | 8       | 2     |
| DeepSeek-V3   | 671B        | 37B           | 256     | 8     |
| Llama 4 Scout | ~109B       | ~17B          | 16      | 1     |
| Qwen3 MoE     | 57B         | 14.3B         | 128     | 8     |

**Key insight**: DeepSeek-V3 has 671B total params but only runs 37B per token.
That is 18x less compute than a dense 671B model would require.

---

## Key Numbers to Remember

- **Dense model**: double params = double compute per token.
- **MoE model**: 8x more experts = 8x more total params, but SAME compute per token.
- **Top-K**: most models use K=2 (run 2 of 8 experts per token).
- **Load balance**: without auxiliary loss, router collapses to 1 expert.
- **Expert capacity**: each expert can process at most C tokens per batch.

---

## File List

```
19_mixture_of_experts/
|-- README.md                    <- This file: module overview
|-- 01_moe_concept.md            <- What is MoE? The core idea.
|-- 02_router_network.md         <- How the router selects experts.
|-- 03_load_balancing.md         <- Auxiliary loss and expert capacity.
|-- 04_moe_vs_dense_tradeoffs.md <- When to use MoE vs dense models.
|-- 05_build_mini_moe_gpt.md     <- Guide for building the Mini MoE GPT.
|
|-- examples/
|   |-- example_01_moe_concept.py    <- Build experts, manual routing
|   |-- example_02_router_network.py <- Build the router, top-K gating
|   |-- example_03_load_balancing.py <- Aux loss, expert collapse demo
|   |-- example_04_moe_vs_dense.py   <- Parameter and compute comparison
|   |-- example_05_mini_moe_gpt.py   <- Full Mini MoE GPT training
|
|-- exercises/
    |-- exercise_01_moe_concept.py    <- Build Expert class
    |-- exercise_02_router_network.py <- Build Router class
    |-- exercise_03_load_balancing.py <- Implement aux_loss
    |-- exercise_04_moe_vs_dense.py   <- Count params, compare
    |-- exercise_05_mini_moe_gpt.py   <- Complete MoE GPT
```

---

## How to Run

```bash
# Activate your virtual environment first
venv\Scripts\activate

# Run the main Mini MoE GPT example (runs in under 60 seconds on CPU)
python examples/example_05_mini_moe_gpt.py

# Run all examples in order
python examples/example_01_moe_concept.py
python examples/example_02_router_network.py
python examples/example_03_load_balancing.py
python examples/example_04_moe_vs_dense.py
python examples/example_05_mini_moe_gpt.py

# Try the exercises
python exercises/exercise_01_moe_concept.py
```

---

## Learning Objectives

After completing this module, you will be able to:

1. Explain why MoE models can be larger than dense models without more compute.
2. Build a router network that learns to assign tokens to experts.
3. Implement auxiliary loss to prevent router collapse.
4. Compare MoE and dense models on total params vs active params.
5. Build a working Mini MoE GPT from scratch in PyTorch.

---

## Suggested Study Order

1. Read **01_moe_concept.md** -- understand the big idea first.
2. Run **example_01_moe_concept.py** -- see experts in action.
3. Read **02_router_network.md** -- learn how routing works.
4. Run **example_02_router_network.py** -- see the router pick experts.
5. Read **03_load_balancing.md** -- understand why balance matters.
6. Run **example_03_load_balancing.py** -- watch router collapse, then fix it.
7. Read **04_moe_vs_dense_tradeoffs.md** -- understand the engineering trade-offs.
8. Run **example_04_moe_vs_dense.py** -- compare params and compute.
9. Read **05_build_mini_moe_gpt.md** -- study the full architecture.
10. Run **example_05_mini_moe_gpt.py** -- train your Mini MoE GPT!
11. Complete all exercises to reinforce your understanding.
