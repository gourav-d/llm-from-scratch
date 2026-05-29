"""
=============================================================================
EXAMPLE 03C: Bigram on Code Token Sequences
=============================================================================

GLOSSARY
---------
Code token     : A keyword, symbol, or identifier treated as a single unit.
                 "public class MyService" -> ["public", "class", "MyService"]
                 Same as word tokenization, but applied to source code.

Token sequence : A list of tokens extracted from real code patterns.
                 The bigram learns: after "if" -> likely "("
                                    after "for" -> likely "("
                                    after "class" -> likely identifier

Keyword bigram : A bigram model trained on code token sequences.
                 Learns the grammar of a programming language statistically.
                 NOT a rule-based parser -- pure pattern matching from data.

IDE autocomplete: When VS Code suggests "class" after "public", it uses
                  a much fancier version of exactly this idea.
                  The bigram is the simplest possible version of that.

=============================================================================
WHY THIS IS INTERESTING FOR A .NET DEVELOPER
=============================================================================

You already KNOW these patterns intuitively:
  After "public"  -> "class", "void", "static", "interface"
  After "if"      -> "("
  After "return"  -> a value or expression
  After "new"     -> a type name

The bigram will LEARN these patterns from data.
No rules. No grammar. Pure statistics.

This is exactly what GitHub Copilot does at a vastly larger scale:
  Bigram/N-gram:   look at 1-2 previous tokens
  Copilot (GPT):   look at the ENTIRE file (thousands of tokens)

Same idea. Different context window.

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("EXAMPLE 03C: Bigram on Code Token Sequences")
print("=" * 60)

# =============================================================================
# STEP 1: Training data -- simplified C# and Python code patterns
# =============================================================================

print("""
STEP 1: Training Data
We give the model real code patterns as token sequences.
Each sequence is one "code fragment" tokenized by whitespace.

Note: This is a toy dataset. Real tools use millions of lines of code.
""")

# Each string is a sequence of code tokens (space-separated for easy splitting)
# Mix of C# and Python patterns -- both follow similar structural rules
code_sequences = [
    # C# class and method patterns
    "public class MyService",
    "public class UserRepository",
    "public class OrderProcessor",
    "private class InternalHelper",
    "public static class Extensions",
    "public interface IRepository",
    "public interface IService",

    # Method signatures
    "public void ProcessOrder",
    "public void SendEmail",
    "public void UpdateRecord",
    "private void LogError",
    "public string GetName",
    "public int GetCount",
    "public bool IsValid",
    "public bool IsEmpty",

    # Control flow -- C#
    "if ( condition )",
    "if ( value != null )",
    "if ( count > 0 )",
    "else if ( condition )",
    "for ( int i = 0 )",
    "foreach ( var item in collection )",
    "while ( isRunning )",
    "try { action }",
    "catch ( Exception ex )",

    # Control flow -- Python
    "if condition :",
    "if value is not None :",
    "for item in collection :",
    "for i in range ( count ) :",
    "while is_running :",
    "def process_order ( self ) :",
    "def send_email ( self ) :",
    "def get_name ( self ) :",
    "class MyService :",
    "class UserRepository :",

    # Common patterns both languages share
    "return value",
    "return result",
    "return None",
    "return True",
    "return False",
    "throw new Exception",
    "raise ValueError",
]

print(f"Total code sequences: {len(code_sequences)}")
print("\nSample sequences:")
for seq in code_sequences[:5]:
    print(f"  {seq}")
print("  ...")

# =============================================================================
# STEP 2: Build vocabulary from code tokens
# =============================================================================

print("""
STEP 2: Vocabulary
Every unique token gets an integer ID.
Same process as word bigram -- just the "words" are code tokens.
""")

# Flatten all sequences into one list of tokens
all_tokens = [tok for seq in code_sequences for tok in seq.split()]

# Unique tokens sorted alphabetically for consistent IDs
vocab = sorted(set(all_tokens))
vocab_size = len(vocab)

token_to_idx = {t: i for i, t in enumerate(vocab)}
idx_to_token = {i: t for i, t in enumerate(vocab)}

print(f"Total tokens in data : {len(all_tokens)}")
print(f"Unique tokens (vocab) : {vocab_size}")
print(f"\nSample vocab entries:")
for tok, idx in list(token_to_idx.items())[:10]:
    print(f"  '{tok}' -> {idx}")
print("  ...")

# =============================================================================
# STEP 3: Build (current_token, next_token) training pairs
# =============================================================================

print("""
STEP 3: Training Pairs
For each consecutive token pair in every sequence:
  "public class MyService"
    -> ("public", "class")
    -> ("class", "MyService")
""")

x_list, y_list = [], []

for seq in code_sequences:
    tokens = seq.split()
    for i in range(len(tokens) - 1):
        x_list.append(token_to_idx[tokens[i]])
        y_list.append(token_to_idx[tokens[i + 1]])

x_train = torch.tensor(x_list, dtype=torch.long)
y_train = torch.tensor(y_list, dtype=torch.long)

print(f"Total training pairs : {len(x_train)}")
print("\nFirst 8 pairs:")
for i in range(8):
    src = idx_to_token[x_train[i].item()]
    tgt = idx_to_token[y_train[i].item()]
    print(f"  '{src}' -> '{tgt}'")

# =============================================================================
# STEP 4: Model -- identical BigramModel, just bigger vocab
# =============================================================================

class BigramModel(nn.Module):
    """
    Single embedding table that maps each token to scores for next token.
    Identical architecture to example_03_bigram_pytorch.py and 03b.
    The model doesn't "know" it's working on code -- just token IDs.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)           # shape: (N, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def predict_top_k(self, token_idx, k):
        """Return top-k most likely next tokens with probabilities."""
        idx_tensor = torch.tensor([token_idx], dtype=torch.long)
        logits, _ = self(idx_tensor)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], k)
        return [(idx_to_token[top_indices[i].item()], top_probs[i].item())
                for i in range(k)]

    @torch.no_grad()
    def generate_sequence(self, start_token, length):
        """Generate a token sequence starting from start_token."""
        if start_token not in token_to_idx:
            return f"('{start_token}' not in vocab)"
        current_id = token_to_idx[start_token]
        result = [start_token]
        for _ in range(length - 1):
            idx_tensor = torch.tensor([current_id], dtype=torch.long)
            logits, _ = self(idx_tensor)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs[0], num_samples=1).item()
            result.append(idx_to_token[next_id])
            current_id = next_id
        return " ".join(result)


model = BigramModel(vocab_size)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel created. Vocab: {vocab_size} tokens. Parameters: {n_params}.")

# =============================================================================
# STEP 5: Train
# =============================================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

print("\nTraining for 500 steps...")
for step in range(500):
    logits, loss = model(x_train, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"  Step {step:3d}: loss = {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.4f}")

# =============================================================================
# STEP 6: IDE Autocomplete Demo
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: IDE Autocomplete Simulation")
print("=" * 60)
print("""
Simulate what an IDE does: you type a token, the model suggests what comes next.

VS Code / Rider / IntelliJ do a much fancier version of exactly this.
The bigram is the minimum viable autocomplete.
""")

demo_tokens = ["public", "private", "if", "for", "return", "class", "def"]

for token in demo_tokens:
    if token not in token_to_idx:
        print(f"  '{token}': not in vocabulary")
        continue
    suggestions = model.predict_top_k(token_to_idx[token], k=4)
    parts = "  ".join(f"'{t}' ({p*100:.0f}%)" for t, p in suggestions)
    print(f"  After '{token}': {parts}")

# =============================================================================
# STEP 7: Generate code fragment sequences
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Generate Code Fragment Sequences")
print("=" * 60)
print("""
Start with a token, generate a sequence of 4-5 tokens.
The model learned code patterns -- let's see what it invents.
""")

start_tokens = ["public", "if", "for", "return", "class"]
for start in start_tokens:
    generated = model.generate_sequence(start, length=5)
    print(f"  Start '{start}': {generated}")

# =============================================================================
# STEP 8: Key Insight -- Bigram Learns Syntax, Not Semantics
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: What the Model Learned (and Did NOT Learn)")
print("=" * 60)
print("""
WHAT IT LEARNED (syntax patterns from data):
  'public'  -> usually followed by 'class', 'void', 'static', 'interface'
  'if'      -> usually followed by '('
  'return'  -> usually followed by 'value', 'result', 'True', 'None'
  'class'   -> usually followed by an identifier
  'for'     -> usually followed by '(' or 'item'

WHAT IT DID NOT LEARN:
  It does NOT know that "public void return" is invalid.
  It does NOT understand that the identifier after "class" must be unique.
  It does NOT track scope (open/close braces, indentation).
  It ONLY knows: "given this one token, what usually comes next?"

BIGRAM LIMIT = context window of 1 token.

GPT fixes this with the attention mechanism:
  Instead of looking at 1 previous token,
  GPT looks at ALL previous tokens in the context.

  Bigram:  "public" -> "class"     (1 token of context)
  GPT:     "public async Task<string> Get ... public" -> "void"
           (hundreds of tokens of context, knows "static" was used 3 lines up)

This is WHY transformers exist. The bigram is the starting point.
""")

print("=" * 60)
print("Next: example_03d_trigram.py -- extend context from 1 to 2 tokens")
print("=" * 60)
