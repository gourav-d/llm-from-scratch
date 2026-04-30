"""
Example 04: Positional Encoding - PyTorch Version

SAME example as NumPy version, using PyTorch tensors.

The positional encoding MATH is identical - only the tensor API changes.
This example also shows how to create a reusable nn.Module for positional encoding.

NumPy vs PyTorch for this example:
  ┌─────────────────────────────────┬──────────────────────────────────────┐
  │ NumPy                           │ PyTorch                              │
  ├─────────────────────────────────┼──────────────────────────────────────┤
  │ np.arange(max_seq_len)          │ torch.arange(max_seq_len)            │
  │ np.zeros((max_seq_len, d))      │ torch.zeros(max_seq_len, d)          │
  │ np.exp(...)                     │ torch.exp(...)                       │
  │ np.log(10000.0)                 │ torch.log(torch.tensor(10000.0))     │
  │ np.sin(x), np.cos(x)           │ torch.sin(x), torch.cos(x)          │
  │ PE[:, 0::2] = np.sin(...)      │ PE[:, 0::2] = torch.sin(...)        │
  │ arr + arr (element-wise add)    │ tensor + tensor (SAME!)              │
  └─────────────────────────────────┴──────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)

print("=" * 70)
print("POSITIONAL ENCODING - PyTorch Version")
print("=" * 70)

# ==============================================================================
# PART 1: The Problem (Same Concept as NumPy)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: The Problem - Attention is Order-Blind")
print("=" * 70)

print("""
SAME CONCEPT as the NumPy version - nothing changes here!

Attention is permutation-invariant (order-blind):
  "The cat sat on the mat"  →  same attention as  →  "mat the on sat cat The"

SOLUTION: Add positional encodings to word embeddings!
  Enhanced embedding = word embedding + positional encoding

The math formula is IDENTICAL between NumPy and PyTorch:
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Only the function names change: np.sin() → torch.sin()
""")

# ==============================================================================
# PART 2: Positional Encoding Function
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Implementing Positional Encoding in PyTorch")
print("=" * 70)

def positional_encoding_pytorch(max_seq_len, d_model):
    """
    Generate sinusoidal positional encodings using PyTorch tensors.

    Same math as NumPy version, different function names:
      np.arange()  → torch.arange()
      np.zeros()   → torch.zeros()
      np.exp()     → torch.exp()
      np.sin()     → torch.sin()
      np.cos()     → torch.cos()
      np.log()     → torch.log()  (or math.log for scalar)

    Args:
        max_seq_len: Maximum sequence length
        d_model: Embedding dimension

    Returns:
        PE: shape (max_seq_len, d_model)  as a PyTorch tensor
    """

    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    # NumPy:   position = np.arange(max_seq_len)[:, np.newaxis]
    # PyTorch: position = torch.arange(max_seq_len).unsqueeze(1)
    position = torch.arange(max_seq_len).unsqueeze(1).float()  # Shape: (max_seq_len, 1)
    # .float() converts int tensor to float (needed for math operations)

    # Compute the division term (frequency factor for each dimension pair)
    # NumPy:   div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    # PyTorch: div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)
    )

    # Initialize positional encoding matrix with zeros
    # NumPy:   PE = np.zeros((max_seq_len, d_model))
    # PyTorch: PE = torch.zeros(max_seq_len, d_model)
    PE = torch.zeros(max_seq_len, d_model)

    # Even indices (0, 2, 4, ...): use sine
    # NumPy:   PE[:, 0::2] = np.sin(position * div_term)
    # PyTorch: PE[:, 0::2] = torch.sin(position * div_term)    (SAME SLICING!)
    PE[:, 0::2] = torch.sin(position * div_term)

    # Odd indices (1, 3, 5, ...): use cosine
    PE[:, 1::2] = torch.cos(position * div_term)

    return PE

# Generate positional encodings
max_seq_len = 50
d_model     = 128

PE = positional_encoding_pytorch(max_seq_len, d_model)

print(f"Generated positional encoding tensor:")
print(f"  Shape: {PE.shape}")
print(f"  Type:  {PE.dtype}  (float32 by default in PyTorch)")

print(f"\nFirst 8 dimensions for position 0:")
print(PE[0, :8])

print(f"\nFirst 8 dimensions for position 10:")
print(PE[10, :8])

# ==============================================================================
# PART 3: PositionalEncoding as nn.Module (Best Practice)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: PositionalEncoding as a Reusable nn.Module")
print("=" * 70)

print("""
In PyTorch, we register positional encoding as a 'buffer' inside nn.Module.

WHAT is a buffer?
  - A tensor that belongs to the module BUT is NOT a trainable parameter
  - Positional encoding is FIXED (not learned), so it's a buffer
  - register_buffer() saves it in .state_dict() for easy save/load
  - Buffer moves to GPU automatically when you call .to('cuda')

  In C# terms:
    Parameter (nn.Parameter) = public mutable property (will be trained)
    Buffer (register_buffer)  = public readonly field (fixed, not trained)
""")

class PositionalEncoding(nn.Module):
    """
    Positional Encoding as a reusable nn.Module.

    Registers PE as a non-trainable buffer.
    This is the standard pattern used in real transformer libraries!
    """

    def __init__(self, d_model, max_seq_len=512):
        super().__init__()

        self.d_model = d_model

        # Compute positional encoding
        PE = self._generate_PE(max_seq_len, d_model)

        # Register as buffer (not a trainable parameter)
        # Buffers: saved in state_dict, move to GPU with .to(), but NOT optimized
        # register_buffer(name, tensor)
        self.register_buffer('PE', PE)
        # Now accessible as: self.PE

    def _generate_PE(self, max_seq_len, d_model):
        """Generate sinusoidal positional encodings."""
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        PE = torch.zeros(max_seq_len, d_model)
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        return PE

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Input embeddings, shape (seq_len, d_model)
               or (batch_size, seq_len, d_model)

        Returns:
            x + positional encoding, same shape as input
        """
        if x.dim() == 2:
            # Shape: (seq_len, d_model)
            seq_len = x.shape[0]
            return x + self.PE[:seq_len]          # Add PE for the first seq_len positions
        else:
            # Shape: (batch_size, seq_len, d_model)
            seq_len = x.shape[1]
            return x + self.PE[:seq_len].unsqueeze(0)   # Add batch dimension to PE

# Test the module
pe_module = PositionalEncoding(d_model=8, max_seq_len=100)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
word_embeddings_test = torch.randn(len(sentence), 8) * 0.5

enhanced = pe_module(word_embeddings_test)

print(f"\nInput embeddings shape:  {word_embeddings_test.shape}")
print(f"Enhanced embeddings shape: {enhanced.shape}")
print("\nPositional encoding buffers:")
for name, buf in pe_module.named_buffers():
    print(f"  Buffer '{name}': shape {buf.shape} (not trainable)")
print("\nTrainable parameters:")
params = list(pe_module.parameters())
print(f"  {len(params)} parameters  (0 - PE is fixed, not learned!)")

# ==============================================================================
# PART 4: Applying to Word Embeddings
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Adding Positional Encoding to Word Embeddings")
print("=" * 70)

d_model_small = 8
seq_len = len(sentence)

word_embeddings = torch.randn(seq_len, d_model_small) * 0.5

# Using our module
PE_small = positional_encoding_pytorch(seq_len, d_model_small)

# Element-wise addition (SAME as NumPy!)
# NumPy:   enhanced = word_embeddings + PE_small
# PyTorch: enhanced = word_embeddings + PE_small  (IDENTICAL!)
enhanced_embeddings = word_embeddings + PE_small

print(f"Word embeddings shape: {word_embeddings.shape}")
print(f"PE shape:              {PE_small.shape}")
print(f"Enhanced shape:        {enhanced_embeddings.shape}")

word_idx = 2   # "sat"
print(f"\nFor word '{sentence[word_idx]}' at position {word_idx}:")
print(f"  Word embedding:     {word_embeddings[word_idx].tolist()}")
print(f"  Positional encoding:{PE_small[word_idx].tolist()}")
print(f"  Combined:           {enhanced_embeddings[word_idx].tolist()}")

# ==============================================================================
# PART 5: Visualizations
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Visualizations")
print("=" * 70)

# Convert tensors to NumPy for plotting
PE_np              = PE.numpy()
word_emb_np        = word_embeddings.numpy()
PE_small_np        = PE_small.numpy()
enhanced_np        = enhanced_embeddings.detach().numpy()

fig = plt.figure(figsize=(16, 12))

# Subplot 1: Full positional encoding heatmap
ax1 = plt.subplot(3, 2, 1)
plt.imshow(PE_np, cmap='RdBu', aspect='auto')
plt.colorbar(label='Encoding Value')
plt.title('Positional Encoding Heatmap (PyTorch)\n(50 positions x 128 dims)',
          fontsize=12, fontweight='bold')
plt.xlabel('Dimension')
plt.ylabel('Position')

# Subplot 2: Encoding for different positions
ax2 = plt.subplot(3, 2, 2)
for pos in [0, 5, 10, 20, 30, 40]:
    plt.plot(PE_np[pos], label=f'Position {pos}', alpha=0.7)
plt.title('Encoding Patterns for Different Positions', fontsize=12, fontweight='bold')
plt.xlabel('Dimension')
plt.ylabel('Value')
plt.legend()
plt.grid(alpha=0.3)

# Subplot 3: How dimensions vary across positions
ax3 = plt.subplot(3, 2, 3)
for dim in [0, 1, 2, 3]:
    plt.plot(PE_np[:, dim], label=f'Dim {dim}', alpha=0.7)
plt.title('Dimensions Varying Across Positions', fontsize=12, fontweight='bold')
plt.xlabel('Position')
plt.ylabel('Value')
plt.legend()
plt.grid(alpha=0.3)

# Subplot 4: Zoomed fingerprint
ax4 = plt.subplot(3, 2, 4)
plt.imshow(PE_np[:20, :20], cmap='RdBu', aspect='auto')
plt.colorbar(label='Value')
plt.title('Position Fingerprints (Zoomed)\n(20 positions x 20 dims)', fontsize=12, fontweight='bold')
plt.xlabel('Dimension')
plt.ylabel('Position')

# Subplot 5: 2D position space
ax5 = plt.subplot(3, 2, 5)
plt.scatter(PE_np[:, 0], PE_np[:, 1], c=np.arange(max_seq_len), cmap='viridis', s=50, alpha=0.6)
plt.colorbar(label='Position')
plt.title('2D Position Space (Dims 0 and 1)', fontsize=12, fontweight='bold')
plt.xlabel('Dim 0 (Sine)')
plt.ylabel('Dim 1 (Cosine)')
plt.grid(alpha=0.3)

# Subplot 6: Before/after comparison for sentence
ax6 = plt.subplot(3, 2, 6)
plt.imshow(enhanced_np, cmap='RdBu', aspect='auto')
plt.colorbar(label='Value')
plt.title('Word Embeddings + Positional Encoding\n(For our sentence)', fontsize=12, fontweight='bold')
plt.yticks(range(seq_len), sentence)
plt.xlabel('Dimension')
plt.ylabel('Word (Position)')

plt.tight_layout()
plt.show()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - Positional Encoding: NumPy vs PyTorch")
print("=" * 70)

print("""
SAME MATH, SLIGHTLY DIFFERENT SYNTAX:

Operation              │ NumPy                         │ PyTorch
───────────────────────┼───────────────────────────────┼────────────────────────────────
Create sequence        │ np.arange(n)[:, None]         │ torch.arange(n).unsqueeze(1)
Zeros matrix           │ np.zeros((n, d))              │ torch.zeros(n, d)
Exponential            │ np.exp(x)                     │ torch.exp(x)
Log                    │ np.log(10000.0)               │ torch.log(torch.tensor(10000.))
Sine                   │ np.sin(x)                     │ torch.sin(x)
Cosine                 │ np.cos(x)                     │ torch.cos(x)
Slice assignment       │ PE[:, 0::2] = np.sin(...)     │ PE[:, 0::2] = torch.sin(...)
Add embeddings         │ emb + PE                      │ emb + PE  (IDENTICAL!)
For plotting           │ use array directly            │ tensor.numpy()

PyTorch-SPECIFIC PATTERN:
  register_buffer()  → Fixed tensor stored with model, moved to GPU, saved to file
  nn.Module.to()     → Moves all parameters AND buffers to GPU automatically

KEY INSIGHT:
  The MATH is identical. The API is slightly different.
  Once you know NumPy, PyTorch tensors are easy to learn!

Next:
  example_05: Complete transformer block using nn.TransformerEncoderLayer
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 04 - PyTorch Version")
print("=" * 70)
