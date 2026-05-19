"""
=============================================================================
PROJECT: Personal Knowledge Base Q&A  --  Phase 4 of 5
FILE   : generator.py
=============================================================================

WHAT THIS FILE DOES
--------------------
Trains a small GPT model on your .md notes so it learns the content.
Then generates text completions given a context prompt.

Used by app.py to generate answers from retrieved chunks.

SKILLS REUSED FROM MODULE 05
------------------------------
  example_04_gpt_pytorch.py  -->  TinyGPT (copy of the same class)
                                  training loop
                                  generate() method

IMPORTANT -- HONEST EXPECTATION
---------------------------------
This is a TINY GPT trained on a small amount of text.
It will NOT give clean "Answer: X" responses like ChatGPT.
It will continue text in the style of your notes.

This is the educational Path A. The goal is to understand HOW it works,
not to compete with GPT-4. Once you understand this, Path B (swap in an API)
gives you production-quality answers with zero architecture changes.

HOW TO RUN (standalone -- trains + tests generator)
------------------------------------------------------
  python generator.py

This trains the model and saves it. Run once (or re-run to retrain).

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

NOTES_DIR    = Path(__file__).parent.parent.parent   # = modules/05_building_llm/
SAVED_DIR    = Path(__file__).parent / "saved"
TRAIN_STEPS  = 1000   # increase for better quality (try 3000 for more training)
BATCH_SIZE   = 8
CONTEXT_SIZE = 128    # GPT looks at last 128 characters to predict next
D_MODEL      = 64     # embedding dimension for GPT
NUM_HEADS    = 2      # attention heads
NUM_LAYERS   = 2      # transformer blocks stacked
DROPOUT      = 0.1
LEARNING_RATE = 1e-3

# =============================================================================
# STEP 1: Load + prepare text data
# =============================================================================

def load_notes_text(notes_dir):
    """Read all .md files, return one combined string."""
    all_text = ""
    for md_file in sorted(Path(notes_dir).rglob("*.md")):
        try:
            all_text += md_file.read_text(encoding="utf-8", errors="ignore") + "\n"
        except Exception:
            pass
    return all_text


# =============================================================================
# TINY GPT -- direct copy from example_04_gpt_pytorch.py
# (same architecture, same code -- this is intentional so you see the connection)
# =============================================================================

class GPTConfig:
    """All GPT settings in one place. Same as example_04."""
    def __init__(self, vocab_size, d_model=64, num_heads=2,
                 num_layers=2, context_size=128, dropout=0.1):
        self.vocab_size   = vocab_size
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.num_layers   = num_layers
        self.context_size = context_size
        self.dropout      = dropout


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention. Same as example_04."""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_k       = config.d_model // config.num_heads
        self.qkv_proj  = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj  = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv     = self.qkv_proj(x)
        Q, K, V = qkv.split(C, dim=2)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scale  = self.d_k ** -0.5
        scores = (Q @ K.transpose(-2, -1)) * scale
        mask   = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out    = weights @ V
        out    = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Two-layer MLP per token. Same as example_04."""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer layer. Same as example_04."""
    def __init__(self, config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.ff   = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    Complete GPT model. Same as example_04_gpt_pytorch.py.

    The ONLY difference from example_04: this one is saved and loaded
    from disk so app.py can use it without retraining.
    """
    def __init__(self, config):
        super().__init__()
        self.config    = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.context_size, config.d_model)
        self.drop      = nn.Dropout(config.dropout)
        self.blocks    = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.ln_f      = nn.LayerNorm(config.d_model)
        self.head      = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight   # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T   = idx.shape
        tok    = self.token_emb(idx)
        pos    = self.pos_emb(torch.arange(T, device=idx.device))
        x      = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        """
        Autoregressive generation. Same as example_04.
        temperature: lower = more focused, higher = more random.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :] / temperature
            probs     = F.softmax(logits, dim=-1)
            next_idx  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, next_idx], dim=1)
        return idx


# =============================================================================
# FUNCTIONS USED BY app.py
# =============================================================================

def load_gpt_model(saved_dir):
    """
    Load the trained TinyGPT from disk.
    Returns (model, char_to_idx, idx_to_char) tuple.
    """
    gpt_vocab_path = saved_dir / "gpt_vocab.json"
    gpt_model_path = saved_dir / "gpt_model.pt"

    for p in [gpt_vocab_path, gpt_model_path]:
        if not p.exists():
            return None, None, None

    with open(gpt_vocab_path, "r", encoding="utf-8") as f:
        gpt_vocab = json.load(f)

    char_to_idx = gpt_vocab["char_to_idx"]
    idx_to_char = {int(k): v for k, v in gpt_vocab["idx_to_char"].items()}
    cfg = GPTConfig(
        vocab_size   = gpt_vocab["vocab_size"],
        d_model      = gpt_vocab["d_model"],
        num_heads    = gpt_vocab["num_heads"],
        num_layers   = gpt_vocab["num_layers"],
        context_size = gpt_vocab["context_size"],
    )
    model = TinyGPT(cfg)
    model.load_state_dict(torch.load(gpt_model_path, map_location="cpu"))
    model.eval()
    return model, char_to_idx, idx_to_char


def generate_answer(context_chunks, question, model, char_to_idx, idx_to_char,
                    max_new_tokens=200, temperature=0.7):
    """
    Build a prompt from context chunks + question, then generate continuation.

    HOW THE PROMPT WORKS:
      We build a string like:
        "Context: {retrieved text}
         Question: {user question}
         Answer:"

      Then we feed this to GPT and let it continue from "Answer:".
      The GPT was trained on notes text, so it will generate notes-style text.

      This is the RAG (Retrieval Augmented Generation) pattern:
        Retrieve relevant text -> Augment the prompt -> Generate

    LIMITATION:
      Our small GPT won't give clean Q&A answers -- it will continue
      text that LOOKS like your notes. That is expected at this stage.
      The architecture is correct; the model just needs more scale.
    """
    if model is None:
        return "[Generator not ready. Run generator.py first.]"

    # Build the prompt
    context_text = "\n\n".join(c["text"] for c in context_chunks)

    # Truncate context to fit in context window (leave room for question + generation)
    max_context_chars = CONTEXT_SIZE * 3   # rough estimate (3 chars per token avg)
    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars]

    prompt = f"Context: {context_text}\nQuestion: {question}\nAnswer:"

    # Encode prompt to token IDs
    # Use only characters that exist in our vocabulary (skip unknown ones)
    ids = [char_to_idx.get(ch, 0) for ch in prompt]

    # Crop to fit in context window
    max_prompt_len = CONTEXT_SIZE - max_new_tokens - 1
    if len(ids) > max_prompt_len:
        ids = ids[-max_prompt_len:]   # keep the LAST part (most relevant)

    # Convert to tensor: shape (1, seq_len) -- batch of 1 sequence
    idx = torch.tensor([ids], dtype=torch.long)

    # Generate continuation
    out_ids  = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)

    # Decode only the NEW tokens (everything after the prompt)
    new_ids  = out_ids[0, len(ids):].tolist()
    answer   = "".join(idx_to_char.get(i, "?") for i in new_ids)

    return answer.strip()


# =============================================================================
# TRAIN THE MODEL (runs when you execute this file directly)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4: Training the GPT Generator")
    print("=" * 60)

    # Load notes
    print(f"\nLoading notes from: {NOTES_DIR}")
    text = load_notes_text(NOTES_DIR)
    if not text:
        print("ERROR: No notes found. Check NOTES_DIR.")
        exit(1)
    print(f"Loaded {len(text):,} characters")

    # Build vocabulary
    chars       = sorted(set(text))
    vocab_size  = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    print(f"Vocabulary: {vocab_size} unique characters")

    # Encode
    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

    # Create model
    cfg = GPTConfig(
        vocab_size   = vocab_size,
        d_model      = D_MODEL,
        num_heads    = NUM_HEADS,
        num_layers   = NUM_LAYERS,
        context_size = CONTEXT_SIZE,
        dropout      = DROPOUT,
    )
    model     = TinyGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"\nTraining for {TRAIN_STEPS} steps...")
    print("(This will take a few minutes on CPU)")
    print()

    # Training loop -- same pattern as example_04_gpt_pytorch.py
    for step in range(TRAIN_STEPS):
        starts  = torch.randint(len(data) - CONTEXT_SIZE - 1, (BATCH_SIZE,))
        x_batch = torch.stack([data[s     : s + CONTEXT_SIZE    ] for s in starts])
        y_batch = torch.stack([data[s + 1 : s + CONTEXT_SIZE + 1] for s in starts])

        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == TRAIN_STEPS - 1:
            print(f"  Step {step:4d}: loss = {loss.item():.4f}")

    print(f"\nTraining complete. Final loss: {loss.item():.4f}")

    # Save model
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    gpt_model_path = SAVED_DIR / "gpt_model.pt"
    torch.save(model.state_dict(), gpt_model_path)
    print(f"\nSaved GPT model to: {gpt_model_path}")

    gpt_vocab = {
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        "vocab_size":  vocab_size,
        "d_model":     D_MODEL,
        "num_heads":   NUM_HEADS,
        "num_layers":  NUM_LAYERS,
        "context_size": CONTEXT_SIZE,
    }
    gpt_vocab_path = SAVED_DIR / "gpt_vocab.json"
    with open(gpt_vocab_path, "w", encoding="utf-8") as f:
        json.dump(gpt_vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved GPT vocab to: {gpt_vocab_path}")

    # Quick generation test
    print("\n--- Quick generation test ---")
    model.eval()
    test_prompt = "nn.Embedding is"
    seed_ids    = [char_to_idx.get(ch, 0) for ch in test_prompt]
    seed_tensor = torch.tensor([seed_ids], dtype=torch.long)
    out         = model.generate(seed_tensor, max_new_tokens=100, temperature=0.7)
    new_chars   = "".join(idx_to_char.get(i.item(), "?") for i in out[0, len(seed_ids):])
    print(f"Prompt:    '{test_prompt}'")
    print(f"Generated: '{test_prompt}{new_chars}'")

    print("\n" + "=" * 60)
    print("Phase 4 DONE.")
    print("Next step: run  python app.py  to use the full Q&A app")
    print("=" * 60)
