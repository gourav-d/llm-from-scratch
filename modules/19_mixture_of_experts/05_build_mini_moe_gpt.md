# Lesson 05: Building the Mini MoE GPT

## Overview

In this lesson you will build a complete GPT-style language model where
every transformer block's FFN layer is replaced by a Mixture of Experts layer.

The architecture has these components (from bottom to top):

```
MoEGPT
  |
  +-- token_embedding   : nn.Embedding(vocab_size, d_model)
  |
  +-- MoETransformerBlock x num_layers
  |     |
  |     +-- LayerNorm (pre-attention norm)
  |     +-- MultiHeadAttention (standard, from M05/M18)
  |     +-- LayerNorm (pre-FFN norm)
  |     +-- MoELayer  (replaces standard FFN)
  |           |
  |           +-- Router: Linear(d_model, num_experts) -> softmax -> top-K
  |           +-- Expert 0: Linear -> GELU -> Linear
  |           +-- Expert 1: Linear -> GELU -> Linear
  |           +-- Expert 2: Linear -> GELU -> Linear
  |           +-- Expert 3: Linear -> GELU -> Linear
  |
  +-- final_norm  : LayerNorm
  +-- output_head : Linear(d_model, vocab_size)
```

---

## Component 1: Expert (Standard FFN)

The Expert is IDENTICAL to the FFN you built in M05 and M18.
No changes. Just packaged as an nn.Module class.

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # expand: d_model -> d_ff
        self.fc2 = nn.Linear(d_ff, d_model)  # contract: d_ff -> d_model

    def forward(self, x):
        x = F.gelu(self.fc1(x))  # Linear + GELU activation
        x = self.fc2(x)          # Linear projection back
        return x
```

Shape flow: [batch, seq, d_model] -> [batch, seq, d_ff] -> [batch, seq, d_model]

---

## Component 2: Router

The Router is one Linear layer that outputs expert probabilities.

```python
class Router(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        # one tiny linear layer: d_model -> num_experts logits
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x, k):
        # x shape: [batch, seq_len, d_model]
        logits = self.gate(x)               # [batch, seq_len, num_experts]
        probs = F.softmax(logits, dim=-1)   # probabilities sum to 1
        topk_probs, topk_idx = probs.topk(k, dim=-1)  # top-K selection
        # renormalize: the K selected probs must sum to 1
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        return topk_probs, topk_idx, probs  # probs needed for aux_loss
```

---

## Component 3: MoELayer

The MoELayer combines the Router with N Experts.

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # create N independent expert FFNs (stored as a list)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router = Router(d_model, num_experts)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        # get routing decisions
        topk_probs, topk_idx, all_probs = self.router(x, self.top_k)

        # flatten batch and seq for easier expert dispatch
        x_flat = x.view(batch * seq_len, d_model)   # [B*S, d_model]
        topk_probs_flat = topk_probs.view(batch * seq_len, self.top_k)
        topk_idx_flat = topk_idx.view(batch * seq_len, self.top_k)

        # accumulate weighted expert outputs
        output = torch.zeros_like(x_flat)            # [B*S, d_model]
        for k_idx in range(self.top_k):
            for expert_id in range(self.num_experts):
                # find tokens assigned to this expert at this k position
                mask = (topk_idx_flat[:, k_idx] == expert_id)
                if mask.any():
                    token_input = x_flat[mask]           # tokens for this expert
                    expert_out = self.experts[expert_id](token_input)
                    weight = topk_probs_flat[mask, k_idx].unsqueeze(-1)
                    output[mask] += weight * expert_out  # weighted add

        output = output.view(batch, seq_len, d_model)
        return output, all_probs  # return all_probs for aux_loss
```

---

## Component 4: MoETransformerBlock

Identical to a standard TransformerBlock except FFN is replaced by MoELayer.

```python
class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts, top_k):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)  # same as M05
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)  # MoE replaces FFN

    def forward(self, x, mask=None):
        # pre-norm attention (with residual)
        x = x + self.attention(self.norm1(x), mask)
        # pre-norm MoE (with residual), collect all_probs for aux_loss
        moe_out, all_probs = self.moe(self.norm2(x))
        x = x + moe_out
        return x, all_probs
```

---

## Component 5: MoEGPT (Full Model)

```python
class MoEGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, num_experts, top_k, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            MoETransformerBlock(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        batch, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(positions)

        all_router_probs = []  # collect router probs from all blocks
        for block in self.blocks:
            x, router_probs = block(x)
            all_router_probs.append(router_probs)

        x = self.final_norm(x)
        logits = self.output_head(x)   # [batch, seq_len, vocab_size]
        return logits, all_router_probs
```

---

## Training Loop: Adding the Auxiliary Loss

The key difference from a standard GPT training loop is adding aux_loss:

```python
def compute_aux_loss(all_router_probs, num_experts):
    """Switch Transformer auxiliary loss for load balancing."""
    total_aux = 0.0
    for probs in all_router_probs:
        # probs shape: [batch, seq_len, num_experts]
        batch, seq, ne = probs.shape
        num_tokens = batch * seq

        # P_i: average router probability per expert (differentiable)
        # shape: [num_experts]
        P = probs.mean(dim=[0, 1])

        # f_i: fraction of tokens routed to each expert
        # We approximate using the TOP-1 routing decisions
        topk_idx = probs.argmax(dim=-1)  # [batch, seq] -- primary expert
        topk_idx_flat = topk_idx.view(-1)
        f = torch.zeros(num_experts, device=probs.device)
        for i in range(num_experts):
            f[i] = (topk_idx_flat == i).float().mean()

        # aux_loss = N * sum(f_i * P_i)
        aux = num_experts * (f * P).sum()
        total_aux = total_aux + aux

    return total_aux / len(all_router_probs)  # average across layers


# Training loop
alpha = 0.01  # load balance loss weight
for step in range(num_steps):
    logits, all_router_probs = model(input_ids)
    # main language modeling loss
    main_loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                                targets[:, 1:].reshape(-1))
    # auxiliary load balancing loss
    aux = compute_aux_loss(all_router_probs, num_experts)
    # total loss
    loss = main_loss + alpha * aux
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## Mini Config (Runs on CPU in Under 60 Seconds)

```python
config = {
    "vocab_size"   : 27,    # a-z + space = 27 characters
    "d_model"      : 64,    # embedding dimension
    "num_heads"    : 4,     # attention heads
    "d_ff"         : 128,   # expert hidden dimension (2x d_model)
    "num_layers"   : 2,     # transformer blocks
    "num_experts"  : 4,     # 4 experts per block
    "top_k"        : 2,     # route each token to 2 experts
    "max_seq_len"  : 32,    # sequence length
    "batch_size"   : 8,     # training batch size
    "num_steps"    : 100,   # training steps
    "lr"           : 3e-3,  # learning rate
    "alpha"        : 0.01,  # aux loss coefficient
}
```

Total parameters with this config:
- Token embedding: 27 * 64 = 1,728
- Pos embedding: 32 * 64 = 2,048
- Per block: attention (~16K) + MoE (4 experts * ~16K = 64K) + norms
- 2 blocks: ~162K total
- Output head: 64 * 27 = 1,728

Very small! Trains in seconds on CPU.

---

## Verification: Checking Expert Specialization

After training, you can inspect which experts each token uses:

```python
with torch.no_grad():
    logits, all_probs = model(sample_input)
    block0_probs = all_probs[0]  # router probs from block 0
    # shape: [batch, seq_len, num_experts]
    topk_idx = block0_probs.argmax(dim=-1)  # primary expert per token
    # print which expert each character token uses most
```

---

## Expected Training Output

When you run example_05_mini_moe_gpt.py, you should see:

```
Step   0: loss = 3.28, aux = 1.012
Step  10: loss = 2.94, aux = 1.003
Step  20: loss = 2.51, aux = 1.001
Step  50: loss = 1.87, aux = 1.000
Step 100: loss = 1.23, aux = 1.000
```

- main_loss decreasing: the model is learning the toy text.
- aux_loss near 1.000: experts are balanced (remember: min value = 1.0 when balanced).

---

## Quiz Questions

**Question 1**: In the MoEGPT architecture, what is the ONLY structural
difference between a standard TransformerBlock (from M05) and MoETransformerBlock?

A) The attention mechanism is replaced by a router.
B) The FFN (feed-forward network) is replaced by a MoELayer.  <-- CORRECT
C) The layer norms are removed.
D) Positional embeddings are inside each block.

**Explanation**: MoE is a DIRECT DROP-IN REPLACEMENT for the FFN. Everything
else -- attention, layer norms, residual connections, embeddings -- stays the same.

---

**Question 2**: After training the Mini MoE GPT, you print the auxiliary loss and
see it equals 1.000. What does this mean?

A) The auxiliary loss is not working -- it should be 0.
B) The load balance is PERFECT -- all experts receive equal fractions of tokens,
   which is the theoretical minimum of the Switch Transformer aux_loss formula.  <-- CORRECT
C) Something is wrong -- aux_loss should be greater than 1.5 for a good model.
D) Only 1 expert is being used (aux_loss = 1 means collapse).

**Explanation**: Recall the formula: aux_loss = N * sum(f_i * P_i).
When f_i = P_i = 1/N (perfect balance): sum = N * (1/N * 1/N) = 1/N.
aux_loss = N * (1/N) = 1.0. So 1.0 IS the balanced minimum. Lower = impossible.

---

**Question 3**: The training loop computes:
total_loss = main_loss + 0.01 * aux_loss
Why is alpha (0.01) kept very small?

A) Because larger alpha would make training faster.
B) Because the aux_loss formula is on a different scale than main_loss.
C) Because too large an alpha would cause the model to focus on balancing experts
   at the expense of actually learning the language task (main_loss).  <-- CORRECT
D) alpha = 0.01 is required by the PyTorch optimizer.

**Explanation**: The model has two objectives: learn the language (main_loss)
and keep experts balanced (aux_loss). If alpha is too large, the model sacrifices
quality to achieve perfect balance. Typical alpha = 0.001 to 0.01 in practice.
