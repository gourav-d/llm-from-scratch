"""
Module 19 - Mixture of Experts
Exercise 05: Build the Mini MoE GPT (Capstone Exercise)

This is the capstone exercise of Module 19.
You will complete a Mini MoE GPT by filling in the missing parts.
The full architecture is already structured -- you complete each piece.

GLOSSARY
--------
MoETransformerBlock: One transformer block where the FFN is replaced by MoELayer.
               Contains: LayerNorm -> Attention (residual) -> LayerNorm -> MoELayer (residual).
               Output: (x, all_probs) where all_probs is for aux_loss.

MoEGPT       : Complete GPT-style model using MoETransformerBlocks.
               token_embedding + positional_embedding -> blocks -> final_norm -> output_head.

total_loss   : main_cross_entropy_loss + alpha * aux_load_balance_loss
               alpha is small (0.01) so aux_loss guides but does not dominate.

Cross entropy: Language modeling loss: predict the next character/token correctly.
               F.cross_entropy(logits.view(-1, V), targets.view(-1))
               Measures how surprised the model is by the true next token.

Router probs : Shape [batch, seq_len, num_experts]. One set per transformer block.
               Needed for computing aux_loss (load balancing).

Autoregressive: Language model generates one token at a time, feeding each output
               back as input for the next step. "Auto" = self, "regressive" = predicts from past.

Routing stats: After training, inspect which experts handle which tokens.
               probs.argmax(dim=-1) gives the primary expert for each token.
               Good training -> experts specialize in different token types.
"""

import torch                           # PyTorch: deep learning framework
import torch.nn as nn                  # nn: neural network building blocks
import torch.nn.functional as F        # F: cross_entropy, softmax, gelu
import time                            # time: measure training speed

torch.manual_seed(42)                  # fixed seed for reproducibility

print("=" * 65)
print("Exercise 05: Mini MoE GPT -- Capstone Exercise")
print("=" * 65)
print()

# ============================================================
# CONFIGURATION (DO NOT MODIFY -- these make training fast on CPU)
# ============================================================
VOCAB_SIZE   = 27     # a-z + space
D_MODEL      = 64     # embedding dimension
NUM_HEADS    = 4      # attention heads
D_FF         = 128    # expert hidden dim
NUM_LAYERS   = 2      # transformer blocks
NUM_EXPERTS  = 4      # experts per layer
TOP_K        = 2      # active experts per token
MAX_SEQ_LEN  = 32     # max context length
BATCH_SIZE   = 8      # training batch size
NUM_STEPS    = 150    # training steps
LR           = 3e-3   # learning rate
ALPHA        = 0.01   # aux_loss coefficient

# ============================================================
# PROVIDED: Dataset preparation (DO NOT MODIFY)
# ============================================================
CHARS = "abcdefghijklmnopqrstuvwxyz "  # 27 characters (ASCII only)
char_to_idx = {c: i for i, c in enumerate(CHARS)}   # char -> int
idx_to_char = {i: c for i, c in enumerate(CHARS)}   # int -> char

raw_text = (
    "hello world this is a test of mixture of experts "
    "the quick brown fox jumps over the lazy dog "
    "learning to build llms from scratch is rewarding "
    "experts specialize in different types of knowledge "
) * 25   # repeat to have enough data

clean_text = "".join(c for c in raw_text.lower() if c in char_to_idx)
text_ids   = [char_to_idx[c] for c in clean_text]
text_tensor = torch.tensor(text_ids, dtype=torch.long)

def get_batch(text_tensor, batch_size, seq_len):
    """Sample a random batch from the text tensor."""
    max_start = len(text_tensor) - seq_len - 1
    starts    = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([text_tensor[s:s+seq_len]   for s in starts])   # input
    y = torch.stack([text_tensor[s+1:s+seq_len+1] for s in starts]) # target (shifted)
    return x, y

# ============================================================
# PROVIDED: Expert class (DO NOT MODIFY)
# ============================================================
class Expert(nn.Module):
    """One expert FFN: Linear -> GELU -> Linear."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)    # expand
        self.fc2 = nn.Linear(d_ff, d_model)    # contract

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

# ============================================================
# PROVIDED: Router class (DO NOT MODIFY)
# ============================================================
class Router(nn.Module):
    """Router: Linear -> softmax -> top-K -> renorm."""
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x, top_k):
        logits    = self.gate(x)                                         # [B, S, N]
        probs     = F.softmax(logits, dim=-1)                            # [B, S, N]
        topk_w, topk_idx = torch.topk(probs, k=top_k, dim=-1)           # [B, S, K]
        topk_w    = topk_w / topk_w.sum(dim=-1, keepdim=True)            # renorm
        return topk_w, topk_idx, probs

# ============================================================
# PROVIDED: MoELayer class (DO NOT MODIFY)
# ============================================================
class MoELayer(nn.Module):
    """MoE Layer: router + N experts."""
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.experts     = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router      = Router(d_model, num_experts)

    def forward(self, x):
        B, S, D         = x.shape
        topk_w, topk_idx, all_probs = self.router(x, self.top_k)
        x_flat          = x.view(B * S, D)
        w_flat          = topk_w.view(B * S, self.top_k)
        idx_flat        = topk_idx.view(B * S, self.top_k)
        output          = torch.zeros_like(x_flat)
        for k_pos in range(self.top_k):
            for eid in range(self.num_experts):
                mask = (idx_flat[:, k_pos] == eid)
                if not mask.any(): continue
                expert_out = self.experts[eid](x_flat[mask])
                weight     = w_flat[mask, k_pos].unsqueeze(-1)
                output[mask] += weight * expert_out
        return output.view(B, S, D), all_probs

# ============================================================
# PROVIDED: CausalSelfAttention (DO NOT MODIFY)
# ============================================================
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (same as M05/M18)."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, D  = x.shape
        qkv      = self.qkv(x)
        q, k, v  = qkv.split(D, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scale   = self.head_dim ** -0.5
        scores  = (q @ k.transpose(-2, -1)) * scale
        mask    = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores  = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out     = (weights @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(out)

# ============================================================
#  EXERCISE 1
#  Topic: Complete MoETransformerBlock.forward
#
#  Background:
#    A MoE Transformer Block has:
#      1. Pre-norm + Attention + residual: x = x + attn(norm1(x))
#      2. Pre-norm + MoE + residual: moe_out, probs = moe(norm2(x)); x = x + moe_out
#    The block RETURNS both x and probs (needed for aux_loss).
#
#  Your Task:
#    Complete the forward method of MoETransformerBlock:
#      - Apply norm1 -> attn -> add residual
#      - Apply norm2 -> moe -> add residual (save probs)
#      - Return (x, probs)
#
#  C# Analogy:
#    (Tensor x, Tensor probs) Forward(Tensor x) {
#        x = x + attn.Forward(norm1.Forward(x));     // attention residual
#        var (moeOut, probs) = moe.Forward(norm2.Forward(x));  // MoE
#        x = x + moeOut;                              // MoE residual
#        return (x, probs);
#    }
# ============================================================

class MoETransformerBlock(nn.Module):
    """One transformer block with MoE FFN instead of standard FFN."""
    def __init__(self, d_model, num_heads, d_ff, num_experts, top_k):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)                           # pre-attention norm
        self.attn  = CausalSelfAttention(d_model, num_heads)         # multi-head attention
        self.norm2 = nn.LayerNorm(d_model)                           # pre-MoE norm
        self.moe   = MoELayer(d_model, d_ff, num_experts, top_k)    # MoE replaces FFN

    def forward(self, x):
        """
        Forward pass for one MoE Transformer Block.

        Args:
            x : [batch, seq_len, d_model] -- input token representations

        Returns:
            x     : [batch, seq_len, d_model] -- updated representations
            probs : [batch, seq_len, num_experts] -- router probs (for aux_loss)
        """
        # TODO: Apply norm1 -> attn, add result to x (residual connection)
        # x = x + self.attn(self.norm1(x))

        # TODO: Apply norm2 -> moe (which returns (moe_out, probs)), add moe_out to x
        # moe_out, probs = self.moe(self.norm2(x))
        # x = x + moe_out

        # TODO: return x, probs
        pass


# --- Test Exercise 1 ---
print("--- Exercise 1 Test ---")
test_block = MoETransformerBlock(D_MODEL, NUM_HEADS, D_FF, NUM_EXPERTS, TOP_K)
test_input = torch.randn(2, 8, D_MODEL)    # [batch=2, seq=8, d_model=64]
block_result = test_block(test_input)

if block_result is not None:
    x_out, probs_out = block_result
    print(f"Block input  shape: {test_input.shape}")
    print(f"Block output shape: {x_out.shape}   (should be same as input)")
    print(f"Router probs shape: {probs_out.shape} (should be [2, 8, 4])")
    assert x_out.shape == test_input.shape, "Block output shape must match input!"
    assert probs_out.shape == (2, 8, NUM_EXPERTS), "Router probs shape wrong!"
    print("PASS: Block output shapes are correct.")
else:
    print("Hint: Return (x_after_residuals, probs_from_moe)")
print()

# ============================================================
#  EXERCISE 2
#  Topic: Complete MoEGPT.forward
#
#  Background:
#    The full MoE GPT forward pass:
#      1. x = token_emb(token_ids) + pos_emb(positions)
#      2. For each block: x, probs = block(x); save probs
#      3. x = final_norm(x)
#      4. logits = output_head(x)
#      5. Return (logits, all_router_probs)
#
#  Your Task:
#    Complete MoEGPT.forward:
#      - Create position indices using torch.arange(seq_len)
#      - Embed tokens and add positional embeddings
#      - Run through all blocks, collecting router probs
#      - Apply final norm and output head
#      - Return (logits, all_router_probs)
# ============================================================

class MoEGPT(nn.Module):
    """Complete MoE GPT language model."""
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, num_experts, top_k, max_seq_len):
        super().__init__()
        self.token_emb  = nn.Embedding(vocab_size, d_model)           # token lookup table
        self.pos_emb    = nn.Embedding(max_seq_len, d_model)          # position lookup table
        self.blocks     = nn.ModuleList([
            MoETransformerBlock(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)                       # normalize before output
        self.output_head = nn.Linear(d_model, vocab_size, bias=False) # predict next token

    def forward(self, token_ids):
        """
        Forward pass for the full MoE GPT.

        Args:
            token_ids : [batch, seq_len] -- integer token indices

        Returns:
            logits          : [batch, seq_len, vocab_size] -- next-token predictions
            all_router_probs: list of [batch, seq_len, num_experts], one per block
        """
        B, S = token_ids.shape                 # batch size and sequence length

        # TODO: create position indices: [0, 1, 2, ..., S-1]
        # positions = torch.arange(S, device=token_ids.device)

        # TODO: embed tokens and add positional embeddings
        # x = self.token_emb(token_ids) + self.pos_emb(positions)

        # TODO: run through each block, collect all router probs
        # all_router_probs = []
        # for block in self.blocks:
        #     x, probs = block(x)
        #     all_router_probs.append(probs)

        # TODO: apply final normalization
        # x = self.final_norm(x)

        # TODO: project to vocabulary
        # logits = self.output_head(x)

        # TODO: return (logits, all_router_probs)
        pass


# --- Test Exercise 2 ---
print("--- Exercise 2 Test ---")
model = MoEGPT(
    vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS,
    d_ff=D_FF, num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS,
    top_k=TOP_K, max_seq_len=MAX_SEQ_LEN
)
test_ids = torch.randint(0, VOCAB_SIZE, (2, 8))   # [batch=2, seq=8]
gpt_result = model(test_ids)

if gpt_result is not None:
    logits, all_probs = gpt_result
    print(f"Input token_ids shape:  {test_ids.shape}")
    print(f"Output logits shape:    {logits.shape}  (should be [2, 8, {VOCAB_SIZE}])")
    print(f"Router probs: {len(all_probs)} layers, each {all_probs[0].shape}")
    assert logits.shape == (2, 8, VOCAB_SIZE), "Logits shape wrong!"
    assert len(all_probs) == NUM_LAYERS, "Should have one probs set per layer!"
    print("PASS: MoEGPT forward pass shapes are correct.")
else:
    print("Hint: embed -> blocks (collect probs) -> final_norm -> output_head -> return")
print()

# ============================================================
#  EXERCISE 3
#  Topic: Compute total_loss = cross_entropy + alpha * aux_loss
#
#  Background:
#    The training loss has two parts:
#      1. main_loss = cross_entropy(logits, targets) -- learn the language
#      2. aux_loss  = Switch Transformer load balancing loss
#      3. total_loss = main_loss + alpha * aux_loss
#    aux_loss formula: N * sum(f_i * P_i) averaged across layers.
#
#  Your Task:
#    Complete compute_total_loss(logits, targets, all_router_probs, alpha, num_experts) to:
#      - Compute main_loss using F.cross_entropy
#      - Compute aux_loss using the Switch formula
#      - Return total_loss = main_loss + alpha * aux_loss
# ============================================================

def compute_total_loss(logits, targets, all_router_probs, alpha, num_experts):
    """
    Compute total training loss = cross_entropy + alpha * aux_loss.

    Args:
        logits          : [B, S, vocab_size]
        targets         : [B, S] -- true next-token indices
        all_router_probs: list of [B, S, N] tensors, one per block
        alpha           : load balancing coefficient (e.g. 0.01)
        num_experts     : N

    Returns:
        total_loss : scalar tensor (differentiable, for backprop)
        main_loss  : scalar tensor (for logging)
        aux_loss   : scalar tensor (for logging)
    """
    B, S, V = logits.shape

    # TODO: compute main cross-entropy loss
    # Reshape: logits from [B, S, V] to [B*S, V], targets from [B, S] to [B*S]
    # main_loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))

    # TODO: compute auxiliary loss (Switch Transformer formula)
    # Loop over all_router_probs:
    #   for each probs [B, S, N]:
    #     P_i = probs.mean(dim=[0, 1])
    #     primary = probs.argmax(dim=-1).view(-1)
    #     f_i: fraction assigned to each expert
    #     layer_aux = num_experts * (f * P).sum()
    # aux_loss = average of layer_aux across all layers

    # TODO: total_loss = main_loss + alpha * aux_loss
    # TODO: return (total_loss, main_loss, aux_loss)
    pass


# --- Test Exercise 3 ---
print("--- Exercise 3 Test ---")
if gpt_result is not None:
    test_x, test_y = get_batch(text_tensor, 2, 8)   # small batch
    test_logits, test_probs = model(test_x)
    loss_result = compute_total_loss(test_logits, test_y, test_probs, ALPHA, NUM_EXPERTS)

    if loss_result is not None:
        total_l, main_l, aux_l = loss_result
        print(f"main_loss:  {main_l.item():.4f}")
        print(f"aux_loss:   {aux_l.item():.4f}  (should be >= 1.0, decreases toward 1.0)")
        print(f"total_loss: {total_l.item():.4f}")
        print(f"  = {main_l.item():.4f} + {ALPHA} * {aux_l.item():.4f}")
        # Total loss must be differentiable (has grad_fn for backprop)
        assert total_l.requires_grad or total_l.grad_fn is not None, \
            "total_loss must be differentiable for backprop!"
        print("PASS: total_loss is differentiable (can call .backward())")
    else:
        print("Hint: main = cross_entropy, aux = N*sum(f_i*P_i), total = main + alpha*aux")
else:
    print("Skipped (depends on Exercise 2)")
print()

# ============================================================
#  EXERCISE 4
#  Topic: Run 50 training steps, verify loss decreases
#
#  Background:
#    A working language model should show DECREASING loss over training.
#    Steps: sample batch -> forward pass -> compute loss -> backward -> update.
#    Standard PyTorch training loop:
#      optimizer.zero_grad()  -- clear old gradients
#      loss.backward()        -- compute new gradients
#      optimizer.step()       -- update parameters
#
#  Your Task:
#    Complete training_step(model, optimizer, text_tensor) to:
#      - Sample a batch using get_batch
#      - Forward pass: logits, all_router_probs = model(x)
#      - Compute total_loss using compute_total_loss
#      - Backward pass and optimizer step
#      - Return (total_loss, main_loss, aux_loss) as scalar values (use .item())
# ============================================================

def training_step(model, optimizer, text_tensor):
    """
    Run one training step and return loss values.

    Returns:
        (total_loss_val, main_loss_val, aux_loss_val) -- scalar floats (not tensors)
    """
    model.train()                              # enable training mode

    # TODO: sample a batch: x, y = get_batch(text_tensor, BATCH_SIZE, MAX_SEQ_LEN)

    # TODO: forward pass
    # logits, all_probs = model(x)

    # TODO: compute total loss
    # loss_result = compute_total_loss(logits, y, all_probs, ALPHA, NUM_EXPERTS)
    # if loss_result is None: return None

    # TODO: zero_grad -> backward -> clip_grad_norm -> step
    # optimizer.zero_grad()
    # total_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # optimizer.step()

    # TODO: return (total_loss.item(), main_loss.item(), aux_loss.item())
    pass


# --- Test Exercise 4 ---
print("--- Exercise 4 Test ---")
if gpt_result is not None and loss_result is not None:
    optimizer_test = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)
    print(f"Running 50 training steps...")
    print()
    print(f"{'Step':>5} | {'main_loss':>10} | {'aux_loss':>10}")
    print("-" * 35)

    first_loss = None
    last_loss  = None

    for step in range(50):
        step_result = training_step(model, optimizer_test, text_tensor)
        if step_result is None:
            print("Hint: Implement training_step and run again.")
            break
        total_v, main_v, aux_v = step_result
        if first_loss is None: first_loss = main_v   # save first loss
        last_loss = main_v                           # always update last

        if step % 10 == 0 or step == 49:
            print(f"{step:>5} | {main_v:>10.4f} | {aux_v:>10.4f}")

    if first_loss is not None and last_loss is not None:
        print()
        print(f"First step main_loss: {first_loss:.4f}")
        print(f"Final step main_loss: {last_loss:.4f}")
        if last_loss < first_loss:
            print("PASS: Loss decreased over 50 steps -- model is learning!")
        else:
            print("Loss did not decrease. Try increasing NUM_STEPS or adjusting LR.")
else:
    print("Skipped (depends on Exercises 2 and 3)")
print()

# ============================================================
#  EXERCISE 5
#  Topic: Print routing statistics -- which experts are used most?
#
#  Background:
#    After training, inspect the routing to see if experts are balanced.
#    Run the model on a sample, look at router probs from each block.
#    probs.argmax(dim=-1) gives the primary expert for each token.
#    Count fraction per expert: want ~25% each (for 4 experts, top-2 routing).
#
#  Your Task:
#    Complete print_routing_stats(model, text_tensor, num_experts) to:
#      - Run model on a small sample batch (no gradient needed)
#      - For each transformer block, compute primary expert per token
#      - Print fraction of tokens going to each expert (as percentage + ASCII bar)
# ============================================================

def print_routing_stats(model, text_tensor, num_experts):
    """
    Print expert routing statistics for each transformer block.

    Args:
        model       : trained MoEGPT
        text_tensor : training data tensor
        num_experts : N
    """
    model.eval()                               # evaluation mode (no dropout etc.)
    with torch.no_grad():                      # no gradient tracking needed
        # TODO: sample a small batch (batch_size=4, seq_len=16)
        # x, _ = get_batch(text_tensor, 4, 16)

        # TODO: forward pass to get all_router_probs
        # _, all_probs = model(x)

        # TODO: for each block (enumerate all_probs):
        #   probs shape: [B, S, N]
        #   primary_expert = probs.argmax(dim=-1)         # [B, S]
        #   primary_flat   = primary_expert.view(-1)      # [B*S]
        #   for each expert i: fraction = (primary_flat == i).float().mean()
        #   print formatted bar chart

        # Example output format (fill in actual values):
        # Block 0 routing:
        #   Expert 0: |#########      | 30%
        #   Expert 1: |######         | 22%
        #   Expert 2: |########       | 26%
        #   Expert 3: |#######        | 22%
        pass


# --- Test Exercise 5 ---
print("--- Exercise 5 Test ---")
print("Expert routing statistics (after partial training):")
print()
print_routing_stats(model, text_tensor, NUM_EXPERTS)
print()
print("If all experts show roughly 25% usage, load balancing is working!")
print("If one expert shows 90%+, the router has collapsed (needs more training or larger alpha).")

print()
print("=" * 65)
print("Exercise 05 complete! (Capstone Exercise)")
print()
print("You built the full Mini MoE GPT by completing:")
print("  1. MoETransformerBlock.forward  (attention + MoE residuals)")
print("  2. MoEGPT.forward               (embed -> blocks -> head)")
print("  3. compute_total_loss           (cross_entropy + aux_loss)")
print("  4. training_step                (forward -> loss -> backward -> step)")
print("  5. print_routing_stats          (inspect which experts handle what)")
print()
print("This is the same architecture used in Mixtral 8x7B and Qwen3 MoE,")
print("just scaled down to run on your CPU in minutes!")
print("=" * 65)
