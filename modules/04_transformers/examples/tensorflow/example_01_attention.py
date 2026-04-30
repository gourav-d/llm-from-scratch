"""
Example 01: Basic Attention Mechanism - TensorFlow Version

SAME example as NumPy and PyTorch versions, now using TensorFlow!

NumPy vs PyTorch vs TensorFlow at a glance:
  ┌─────────────────────────────┬────────────────────────┬───────────────────────────┐
  │ NumPy                       │ PyTorch                │ TensorFlow                │
  ├─────────────────────────────┼────────────────────────┼───────────────────────────┤
  │ np.random.seed(42)          │ torch.manual_seed(42)  │ tf.random.set_seed(42)    │
  │ np.random.randn(6, 4)       │ torch.randn(6, 4)      │ tf.random.normal([6, 4])  │
  │ Q @ K.T                     │ Q @ K.T                │ Q @ tf.transpose(K)       │
  │ custom softmax              │ F.softmax(x, dim=-1)   │ tf.nn.softmax(x, axis=-1) │
  │ arr.shape → (6, 4)          │ tensor.shape → [6, 4]  │ tensor.shape → (6, 4)     │
  │ arr (for matplotlib)        │ tensor.detach().numpy()│ tensor.numpy()            │
  │ No training                 │ requires_grad=True     │ tf.GradientTape()         │
  └─────────────────────────────┴────────────────────────┴───────────────────────────┘

TensorFlow vs PyTorch:
  Both are full deep learning frameworks.
  TensorFlow is backed by Google, PyTorch by Meta/Facebook.
  TensorFlow is popular in industry/production; PyTorch in research.

In C#/.NET terms:
  NumPy     = MathNet.Numerics  (pure math)
  PyTorch   = ML.NET            (Microsoft's ML framework)
  TensorFlow = TensorFlow.NET   (Google's ML framework)
"""

import tensorflow as tf               # TensorFlow main library
import numpy as np                    # Still use NumPy for matplotlib compatibility
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed
# NumPy:   np.random.seed(42)
# PyTorch: torch.manual_seed(42)
# TF:      tf.random.set_seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("BASIC ATTENTION MECHANISM - TensorFlow Version")
print("=" * 70)

print(f"\nTensorFlow version: {tf.__version__}")

# ==============================================================================
# PART 1: Understanding Query, Key, Value
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Query, Key, Value Concept")
print("=" * 70)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
print(f"\nInput sentence: {' '.join(sentence)}")

d_model = 4

# Create random embeddings
# NumPy:   np.random.randn(len(sentence), d_model)
# PyTorch: torch.randn(len(sentence), d_model)
# TF:      tf.random.normal([len(sentence), d_model])
word_embeddings = tf.random.normal([len(sentence), d_model])

print(f"\nWord embeddings shape: {word_embeddings.shape}")
print(f"  → TensorFlow uses 'TensorShape' - same as (6, 4)")
print(f"\nExample - embedding for '{sentence[0]}':")
print(word_embeddings[0].numpy())   # .numpy() converts TF tensor to NumPy array
print(f"  → tf.Tensor wraps values, .numpy() extracts the raw numbers")

print("""
KEY DIFFERENCE: TensorFlow Tensors vs PyTorch Tensors

  PyTorch: tensor.detach().numpy()   ← must call .detach() first
  TF:      tensor.numpy()            ← simpler! TF 2.x uses eager execution

  WHAT is eager execution?
    TF 2.x runs code IMMEDIATELY (like NumPy), not as a graph.
    This makes debugging much easier!
    TF 1.x used a "graph mode" (build graph first, then run) - confusing!

  In C# terms:
    TF 1.x = deferred execution (IQueryable that runs later)
    TF 2.x = immediate execution (IEnumerable that runs now)
""")

# ==============================================================================
# PART 2: Creating Query, Key, Value Matrices
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Creating Queries, Keys, and Values")
print("=" * 70)

Q = word_embeddings
K = word_embeddings
V = word_embeddings

print(f"Q shape: {Q.shape} | K shape: {K.shape} | V shape: {V.shape}")

# ==============================================================================
# PART 3: Computing Attention Scores
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Computing Attention Scores")
print("=" * 70)

print("""
KEY DIFFERENCE: Matrix multiplication and transpose

  NumPy:   Q @ K.T                     ← .T is transpose
  PyTorch: Q @ K.T                     ← same!
  TF:      Q @ tf.transpose(K)         ← tf.transpose() instead of .T
        or tf.matmul(Q, K, transpose_b=True)  ← explicit transpose flag

  For 2D tensors in TF 2.x, .T actually works too:
    Q @ K.T   ← this works in TF 2.x for 2D tensors!
""")

# TF way: Q @ tf.transpose(K)
# NumPy:  Q @ K.T
attention_scores = Q @ tf.transpose(K)    # Shape: (6, 4) @ (4, 6) = (6, 6)

print(f"Attention scores shape: {attention_scores.shape}")
print("\nAttention scores (before scaling):")
print(attention_scores.numpy())

# Scale by sqrt(d_k)
d_k = d_model
scaled_scores = attention_scores / (d_k ** 0.5)

print(f"\nScaling factor: sqrt({d_k}) = {d_k ** 0.5:.2f}")
print("\nScaled attention scores:")
print(scaled_scores.numpy())

# ==============================================================================
# PART 4: Applying Softmax
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Converting Scores to Weights with Softmax")
print("=" * 70)

print("""
Softmax comparison:

  NumPy:   custom softmax function (you wrote it yourself)
  PyTorch: F.softmax(x, dim=-1)     ← 'dim' parameter
  TF:      tf.nn.softmax(x, axis=-1) ← 'axis' parameter (same as NumPy!)

  Note: TF uses 'axis' (like NumPy), PyTorch uses 'dim'
  Same concept, just different naming conventions.
""")

# NumPy:   attention_weights = softmax(scaled_scores, axis=-1)
# PyTorch: attention_weights = F.softmax(scaled_scores, dim=-1)
# TF:      attention_weights = tf.nn.softmax(scaled_scores, axis=-1)
attention_weights = tf.nn.softmax(scaled_scores, axis=-1)

print("Attention weights (each row sums to 1.0):")
print(attention_weights.numpy())

print("\nVerify each row sums to 1.0:")
for i, word in enumerate(sentence):
    # TF: use .numpy() to get a Python float, then access the value
    row_sum = float(tf.reduce_sum(attention_weights[i]).numpy())
    print(f"  '{word}': {row_sum:.6f}")

print("""
TF SUM FUNCTIONS:
  NumPy:   x.sum()          or  np.sum(x)
  PyTorch: x.sum()          or  torch.sum(x)
  TF:      tf.reduce_sum(x) ← notice 'reduce_' prefix in TF!

  Other reduce operations:
    tf.reduce_mean(x)  = np.mean(x)
    tf.reduce_max(x)   = np.max(x)
    tf.reduce_min(x)   = np.min(x)

  Why 'reduce_'? It 'reduces' a tensor along an axis (makes it smaller).
""")

# ==============================================================================
# PART 5: Computing Weighted Sum of Values
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Computing Context-Aware Representations")
print("=" * 70)

# NumPy:   attention_output = attention_weights @ V
# PyTorch: attention_output = attention_weights @ V   (same)
# TF:      attention_output = attention_weights @ V   (same!)
attention_output = attention_weights @ V

print(f"Attention output shape: {attention_output.shape}")
print(f"\nExample - new representation for '{sentence[2]}' (sat):")
print(f"Original embedding: {V[2].numpy()}")
print(f"After attention:    {attention_output[2].numpy()}")

# ==============================================================================
# PART 6: Visualizing Attention Patterns
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Visualizing Attention Weights")
print("=" * 70)

print("""
Converting TensorFlow tensors to NumPy for matplotlib:

  PyTorch: tensor.detach().numpy()   ← need .detach() first
  TF:      tensor.numpy()            ← simpler! Just call .numpy()

  TF 2.x eager execution means tensors are already computed,
  so you don't need to "detach" from a computation graph.
""")

# TF → NumPy for matplotlib (simpler than PyTorch!)
attention_weights_np = attention_weights.numpy()

plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights_np,
            annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=sentence, yticklabels=sentence,
            cbar_kws={'label': 'Attention Weight'})

plt.title('Attention Weights (TensorFlow): Who Attends to Whom?', fontsize=14, fontweight='bold')
plt.xlabel('Keys (attending TO these words)', fontsize=12)
plt.ylabel('Queries (attention FROM these words)', fontsize=12)
plt.tight_layout()
plt.show()

# ==============================================================================
# PART 7: How 'cat' Attends to All Words
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Detailed Look - How 'cat' Attends to All Words")
print("=" * 70)

word_idx   = 1
focus_word = sentence[word_idx]

print(f"\nHow '{focus_word}' attends to each word:\n")
for i, word in enumerate(sentence):
    # TF: .numpy() converts tensor to Python value
    weight = float(attention_weights[word_idx, i].numpy())
    bar = '█' * int(weight * 50)
    print(f"  {word:6s}: {weight:.3f} {bar}")

# ==============================================================================
# PART 8: TensorFlow's Training Mechanism - GradientTape
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: TensorFlow's Training Mechanism - tf.GradientTape")
print("=" * 70)

print("""
GRADIENT COMPUTATION COMPARISON:

  NumPy:   NOT POSSIBLE (no gradient support)

  PyTorch: Automatic gradient (always on for tensors with requires_grad=True)
    W = torch.randn(4, 4, requires_grad=True)
    output = Q @ W.T
    loss = output.sum()
    loss.backward()     ← gradients computed automatically
    print(W.grad)

  TensorFlow: Manual gradient tape (you control when gradients are recorded)
    W = tf.Variable(tf.random.normal([4, 4]))
    with tf.GradientTape() as tape:   ← "record" computations
        output = Q @ tf.transpose(W)
        loss = tf.reduce_sum(output)
    grads = tape.gradient(loss, W)    ← compute gradients from tape
    print(grads)

  ANALOGY:
    PyTorch = Security camera always recording (always tracks gradients)
    TF Tape = A cassette tape recorder - you press record when needed

  tf.Variable vs tf.Tensor:
    tf.Tensor   = immutable (like C# readonly)
    tf.Variable = mutable, trainable (like C# public property)
    Trainable weights MUST be tf.Variable!
""")

# Demonstrate TensorFlow GradientTape
W_trainable = tf.Variable(tf.random.normal([d_model, d_model]))

# Record computations inside the GradientTape context
with tf.GradientTape() as tape:
    output_demo = word_embeddings @ tf.transpose(W_trainable)
    loss = tf.reduce_sum(output_demo)

# Compute gradients AFTER the tape context
gradients = tape.gradient(loss, W_trainable)

print(f"Trainable variable W (first 2 rows):\n{W_trainable[:2].numpy()}\n")
print(f"Gradient of W (computed by TF):")
print(gradients[:2].numpy())
print("\n  → TF computed the gradient using GradientTape - no manual math needed!")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - NumPy vs PyTorch vs TensorFlow")
print("=" * 70)

print("""
FULL THREE-WAY COMPARISON:

Operation        │ NumPy                    │ PyTorch              │ TensorFlow
─────────────────┼──────────────────────────┼──────────────────────┼─────────────────────
Seed             │ np.random.seed(42)       │ torch.manual_seed(42)│ tf.random.set_seed(42)
Create tensor    │ np.random.randn(6, 4)    │ torch.randn(6, 4)    │ tf.random.normal([6,4])
Matrix multiply  │ Q @ K.T                  │ Q @ K.T              │ Q @ tf.transpose(K)
Softmax          │ custom function          │ F.softmax(x, dim=-1) │ tf.nn.softmax(x, axis=-1)
Sum              │ x.sum()                  │ x.sum()              │ tf.reduce_sum(x)
Mean             │ x.mean()                 │ x.mean()             │ tf.reduce_mean(x)
To NumPy         │ already NumPy            │ tensor.detach().numpy│ tensor.numpy()
Gradients        │ NOT POSSIBLE             │ requires_grad=True   │ tf.GradientTape()
Trainable var    │ just np.array            │ requires_grad=True   │ tf.Variable()

WHEN TO USE WHICH:
  NumPy      → Learning the math, small experiments
  PyTorch    → AI research, academic papers, flexibility
  TensorFlow → Production deployment, mobile, Google Cloud TPUs

Next:
  example_02: Self-attention using tf.keras.layers.Dense
  example_03: Multi-head attention using tf.keras.layers.MultiHeadAttention
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 01 - TensorFlow Version")
print("=" * 70)
