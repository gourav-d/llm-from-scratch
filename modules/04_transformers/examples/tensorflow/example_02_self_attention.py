"""
Example 02: Self-Attention Layer - TensorFlow Version

SAME example as NumPy/PyTorch versions, using Keras layers.

KEY NEW CONCEPT: tf.keras.layers.Dense
  In NumPy:   W_q = np.random.randn(d_model, d_model) * 0.1
              Q   = X @ W_q
  In PyTorch: self.W_q = nn.Linear(d_model, d_model, bias=False)
              Q        = self.W_q(X)
  In TF:      self.W_q = tf.keras.layers.Dense(d_model, use_bias=False)
              Q        = self.W_q(X)

  tf.keras.layers.Dense = same as nn.Linear in PyTorch!
  'Dense' is the standard neural network term for a fully-connected layer.

  In C#/.NET terms:
    tf.keras.layers.Dense = A built-in neuron layer from TensorFlow library

Also new: tf.keras.layers.Layer
  All custom TF layers inherit from tf.keras.layers.Layer.
  Like PyTorch's nn.Module, but with slightly different method names:

  PyTorch           │ TensorFlow (Keras)
  ──────────────────┼──────────────────────────────────
  nn.Module         │ tf.keras.layers.Layer
  __init__()        │ __init__()  (same)
  forward(x)        │ call(x)     ← 'call' instead of 'forward'
  super().__init__()│ super().__init__()  (same)
  model(x)          │ layer(x)    (same - calls call() automatically)
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tf.random.set_seed(42)

print("=" * 70)
print("SELF-ATTENTION WITH LEARNED WEIGHT MATRICES - TensorFlow Version")
print("=" * 70)

# ==============================================================================
# PART 1: What Makes Self-Attention 'Self'?
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: What Makes Self-Attention 'Self'?")
print("=" * 70)

print("""
SAME CONCEPT as NumPy/PyTorch:
  Q, K, V all from same input, through LEARNED transformations:
    Q = Input @ W_q
    K = Input @ W_k
    V = Input @ W_v

TF IMPLEMENTATION: Use tf.keras.layers.Dense (same as nn.Linear in PyTorch)

  NumPy:    W_q = np.random.randn(d, d) * 0.1  | Q = X @ W_q
  PyTorch:  self.W_q = nn.Linear(d, d)          | Q = self.W_q(X)
  TF:       self.W_q = Dense(d, use_bias=False)  | Q = self.W_q(X)

  ALL THREE compute the same math! Just different library syntax.

  Dense vs Linear - same thing, different names:
    TensorFlow (Google) calls it: Dense
    PyTorch (Meta) calls it: Linear
    Math: output = input @ weights  (optional + bias)
""")

# ==============================================================================
# PART 2: Input Setup
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Input Sentence and Embeddings")
print("=" * 70)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
d_model  = 8
seq_len  = len(sentence)

print(f"Sentence: {' '.join(sentence)}")

# NumPy:   X = np.random.randn(seq_len, d_model) * 0.5
# PyTorch: X = torch.randn(seq_len, d_model) * 0.5
# TF:      X = tf.random.normal([seq_len, d_model]) * 0.5
X = tf.random.normal([seq_len, d_model]) * 0.5

print(f"Input shape: {X.shape}")

# ==============================================================================
# PART 3: Self-Attention Layer using Keras
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: SelfAttention as a Keras Layer")
print("=" * 70)

print("""
TF LAYERS INHERIT FROM tf.keras.layers.Layer

Comparison:
  PyTorch                              │ TensorFlow (Keras)
  ─────────────────────────────────────┼──────────────────────────────────────
  class MyLayer(nn.Module):           │ class MyLayer(keras.layers.Layer):
      def __init__(self, ...):        │     def __init__(self, ...):
          super().__init__()          │         super().__init__()
          self.linear = nn.Linear(d)  │         self.dense = Dense(d)
      def forward(self, x):          │     def call(self, x):    ← 'call'!
          return self.linear(x)       │         return self.dense(x)
  layer(x)  # calls forward          │ layer(x)  # calls call

KEY DIFFERENCE: 'forward' in PyTorch = 'call' in TensorFlow/Keras
""")

class SelfAttentionTF(keras.layers.Layer):
    """
    Self-Attention layer using TensorFlow Keras.

    Same math as NumPy and PyTorch versions.
    Only the API is different.

    C# equivalent:
        class SelfAttention : KerasLayerBase {
            Dense W_q, W_k, W_v;
            int d_model;

            public SelfAttention(int d_model) {
                W_q = new Dense(d_model, use_bias: false);
                W_k = new Dense(d_model, use_bias: false);
                W_v = new Dense(d_model, use_bias: false);
            }

            public (Tensor output, Tensor weights) Call(Tensor X) { ... }
        }
    """

    def __init__(self, d_model, **kwargs):
        """
        Initialize self-attention.

        **kwargs: passes extra arguments to parent Layer class
        (like name='my_layer', trainable=True, etc.)
        In C#: base class optional constructor params
        """
        super().__init__(**kwargs)

        self.d_model = d_model

        # tf.keras.layers.Dense = same as nn.Linear in PyTorch
        # Dense(units, use_bias=False):
        #   units    = output size (same as out_features in nn.Linear)
        #   use_bias = whether to add bias term (False = no bias)
        self.W_q = keras.layers.Dense(d_model, use_bias=False)   # Query projection
        self.W_k = keras.layers.Dense(d_model, use_bias=False)   # Key projection
        self.W_v = keras.layers.Dense(d_model, use_bias=False)   # Value projection

    def call(self, X, training=False):
        """
        Forward pass.

        NOTE: In Keras it's 'call()' not 'forward()' like PyTorch!

        The 'training' parameter is important in Keras:
          training=True  → training mode (dropout, batchnorm behave differently)
          training=False → inference mode

        In C#: like having an 'isTraining' flag that changes behavior.
        """
        # Project input to Q, K, V using Dense layers
        # Dense layer called as function: self.W_q(X)
        # Same as PyTorch: self.W_q(X)  or NumPy: X @ W_q
        Q = self.W_q(X)    # Shape: (6, 8) - learned query projection
        K = self.W_k(X)    # Shape: (6, 8) - learned key projection
        V = self.W_v(X)    # Shape: (6, 8) - learned value projection

        # Compute attention scores
        # TF: Q @ tf.transpose(K) instead of Q @ K.T
        scores = Q @ tf.transpose(K) / (self.d_model ** 0.5)

        # Softmax attention weights
        # TF: tf.nn.softmax(x, axis=-1)  vs  PyTorch: F.softmax(x, dim=-1)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Weighted sum of values
        output = attention_weights @ V

        return output, attention_weights

# ==============================================================================
# PART 4: Testing the Layer
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Testing SelfAttentionTF")
print("=" * 70)

# Create layer
attn = SelfAttentionTF(d_model=8)

# Call the layer (calls call() internally)
output, attention_weights = attn(X)

print(f"Input shape:  {X.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")

# List trainable weights
print(f"\nTrainable weights:")
for weight in attn.trainable_weights:
    print(f"  {weight.name}: shape {weight.shape}  ({weight.numpy().size:,} values)")

total_params = sum(w.numpy().size for w in attn.trainable_weights)
print(f"Total trainable parameters: {total_params:,}")
print("\n✓ TensorFlow SelfAttention layer works!")

# ==============================================================================
# PART 5: Training Mode vs Inference Mode
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Training Mode vs Inference Mode in Keras")
print("=" * 70)

print("""
Keras layers can behave differently in training vs inference:

  Training mode:   layer(X, training=True)
  Inference mode:  layer(X, training=False)  or  layer(X)

  WHY does this matter?
    Dropout layers: during training, randomly zero out neurons
                    during inference, use all neurons (no dropout)
    BatchNorm:      uses batch statistics during training
                    uses running averages during inference

  In C# terms:
    Like a method with a bool parameter:
    ProcessData(data, isTraining: true)   // Different behavior!

  For our SelfAttention, training=True/False has no effect (no dropout).
  But it's good practice to always pass it!
""")

# Inference call (explicitly set training=False)
output_inf, weights_inf = attn(X, training=False)
print(f"Inference output shape: {output_inf.shape}")

# ==============================================================================
# PART 6: GradientTape - Training Example
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Training with GradientTape")
print("=" * 70)

print("""
TF uses GradientTape for gradient computation (vs PyTorch's .backward()).

  PyTorch:
    loss.backward()              ← PyTorch computes gradients
    for p in model.parameters():
        print(p.grad)            ← access gradients

  TensorFlow:
    with tf.GradientTape() as tape:
        output, _ = attn(X)
        loss = tf.reduce_sum(output)
    grads = tape.gradient(loss, attn.trainable_variables)  ← compute gradients

  attn.trainable_variables = list of all trainable tf.Variable in the layer
  Same as: list(attn.parameters()) in PyTorch
""")

with tf.GradientTape() as tape:
    output_train, _ = attn(X, training=True)
    dummy_loss = tf.reduce_mean(output_train ** 2)    # Mean squared output as demo loss

# Compute gradients for all trainable variables
gradients = tape.gradient(dummy_loss, attn.trainable_variables)

print(f"Loss value: {dummy_loss.numpy():.4f}")
print(f"\nGradients computed for {len(gradients)} weight matrices:")
for i, (var, grad) in enumerate(zip(attn.trainable_variables, gradients)):
    print(f"  {var.name}: grad shape {grad.shape}, grad norm = {tf.norm(grad).numpy():.4f}")

# ==============================================================================
# PART 7: Visualization
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Visualizing Attention Patterns")
print("=" * 70)

# TF → NumPy: simpler than PyTorch (just .numpy(), no .detach() needed)
attention_weights_np = attention_weights.numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(attention_weights_np,
            annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=sentence, yticklabels=sentence,
            cbar_kws={'label': 'Attention Weight'},
            ax=axes[0])
axes[0].set_title('Self-Attention Weights (TensorFlow)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Attending TO (Keys)')
axes[0].set_ylabel('Attending FROM (Queries)')

word_idx = 2
word = sentence[word_idx]
axes[1].barh(sentence, attention_weights_np[word_idx], color='steelblue')
axes[1].set_xlabel('Attention Weight')
axes[1].set_title(f'How "{word}" Attends to Other Words', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)
for i, (w, weight) in enumerate(zip(sentence, attention_weights_np[word_idx])):
    axes[1].text(weight, i, f' {weight:.3f}', va='center')

plt.tight_layout()
plt.show()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - Self-Attention: Three-Way Comparison")
print("=" * 70)

print("""
APPROACH COMPARISON:

                    │ NumPy                  │ PyTorch              │ TensorFlow
────────────────────┼────────────────────────┼──────────────────────┼──────────────────────
Base class          │ just Python class       │ nn.Module            │ keras.layers.Layer
Linear/Dense layer  │ W = np.randn(d, d)      │ nn.Linear(d, d)      │ Dense(d)
Call convention     │ layer.forward(X)        │ layer(X)             │ layer(X)
Method name         │ forward()              │ forward()            │ call()
Transpose           │ K.T                    │ K.T                  │ tf.transpose(K)
Softmax             │ custom                 │ F.softmax(x, dim=-1) │ tf.nn.softmax(x, axis=-1)
List weights        │ no built-in            │ model.parameters()   │ layer.trainable_weights
Gradient tape       │ not available          │ loss.backward()      │ tf.GradientTape()

TF-SPECIFIC CONCEPTS:
  ✓ keras.layers.Layer  → base class for custom layers
  ✓ Dense layer         → same as nn.Linear (fully connected)
  ✓ call()              → forward pass method (called by layer(X))
  ✓ trainable_weights   → list of all learnable tf.Variables
  ✓ GradientTape        → records computations to compute gradients later
  ✓ tf.Variable         → mutable, trainable tensor
  ✓ training parameter  → switch between train/inference behavior

Next:
  example_03: Multi-head attention using tf.keras.layers.MultiHeadAttention
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 02 - TensorFlow Version")
print("=" * 70)
