"""
Example 06: Mini-GPT - TensorFlow Version

THE CAPSTONE EXAMPLE - using TensorFlow/Keras!

KEY NEW CONCEPT: tf.keras.layers.Embedding
  NumPy:   self.embeddings = np.random.randn(vocab, d) * 0.02
           result = self.embeddings[token_ids]
  PyTorch: self.token_emb = nn.Embedding(vocab, d)
           result = self.token_emb(token_ids)
  TF:      self.token_emb = tf.keras.layers.Embedding(vocab, d)
           result = self.token_emb(token_ids)

  Identical concept in all three! Just different library names.

ALSO NEW: tf.keras.Model (vs tf.keras.layers.Layer)
  Use Layer  when building a component (attention, FFN, etc.)
  Use Model  when building the FULL model (Mini-GPT)

  Benefits of Model:
    ✓ model.summary()     → print full architecture
    ✓ model.compile()     → set up optimizer and loss
    ✓ model.fit()         → training loop in ONE line!
    ✓ model.save()        → save entire model to disk
    ✓ model.predict()     → inference on new data

  PyTorch uses nn.Module for both components and full models.
  TF/Keras distinguishes between Layer and Model (both are fine to use).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Embedding, Dense, LayerNormalization, MultiHeadAttention, Dropout
)
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(42)

print("=" * 70)
print("MINI-GPT: COMPLETE TRANSFORMER ARCHITECTURE - TensorFlow Version")
print("=" * 70)

# ==============================================================================
# PART 1: Token Embeddings with Keras Embedding Layer
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Token Embeddings - tf.keras.layers.Embedding")
print("=" * 70)

print("""
Embedding layer comparison:

  NumPy:
    self.embeddings = np.random.randn(vocab_size, d_model) * 0.02
    result = self.embeddings[token_ids]   # direct array indexing

  PyTorch:
    self.token_emb = nn.Embedding(vocab_size, d_model)
    result = self.token_emb(token_ids)   # call like a function

  TensorFlow:
    self.token_emb = Embedding(vocab_size, d_model)
    result = self.token_emb(token_ids)   # SAME as PyTorch!

  ALL THREE: lookup table that maps integer token IDs → dense vectors.
  The only difference is which library provides it.

  In C# terms:
    All three = Dictionary<int, float[]> where entries are trainable
""")

vocab_size = 20
d_model    = 8

# Create embedding layer
token_emb = Embedding(input_dim=vocab_size, output_dim=d_model)
# input_dim  = vocabulary size (number of unique tokens)
# output_dim = embedding dimension

# Test
test_tokens = tf.constant([0, 5, 10, 15])    # Token IDs as TF tensor
test_embeddings = token_emb(test_tokens)

print(f"Embedding table: {vocab_size} tokens × {d_model} dims")
print(f"Input token IDs: {test_tokens.numpy()}")
print(f"Embeddings shape: {test_embeddings.shape}")
print(f"\nEmbedding for token 0:\n{test_embeddings[0].numpy()}")

# ==============================================================================
# PART 2: Positional Encoding
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Positional Encoding")
print("=" * 70)

def positional_encoding_tf(max_seq_len, d_model):
    """Sinusoidal positional encoding (same as Example 04)."""
    position = tf.cast(tf.range(max_seq_len), tf.float32)[:, tf.newaxis]
    i        = tf.cast(tf.range(0, d_model, 2), tf.float32)
    div_term = tf.exp(i * -(tf.math.log(tf.constant(10000.0)) / d_model))

    PE = tf.reshape(
        tf.stack([tf.sin(position * div_term),
                  tf.cos(position * div_term)], axis=-1),
        [max_seq_len, d_model]
    )
    return PE

print("positional_encoding_tf() defined (see Example 04 for details)")

# ==============================================================================
# PART 3: Causal Mask in TensorFlow
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Causal Masking in TensorFlow")
print("=" * 70)

print("""
CAUSAL MASK COMPARISON:

  NumPy:
    mask = np.tril(np.ones((seq_len, seq_len))).astype(bool)
    scores = np.where(mask, scores, -1e9)

  PyTorch:
    mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
    # Returns float mask: 0.0 = attend, -inf = blocked

  TensorFlow:
    # Option A: Create boolean mask
    mask = tf.linalg.band_part(tf.ones([n, n], dtype=tf.bool), -1, 0)
    # tf.linalg.band_part(x, num_lower, num_upper):
    #   num_lower=-1: keep ALL lower rows
    #   num_upper=0:  keep only the diagonal (no upper triangle)

    # MultiHeadAttention in TF uses boolean mask:
    #   True  = attend to this position
    #   False = DO NOT attend (masked/blocked)

    # For CAUSAL mask: True below and on diagonal, False above
    causal_mask = tf.linalg.band_part(tf.ones([n, n], dtype=tf.bool), -1, 0)
""")

seq_len_demo = 6

# Create causal mask for TF
# tf.linalg.band_part: keeps a band of the matrix
#   band_part(input, num_lower=-1, num_upper=0) = lower triangular
causal_mask = tf.linalg.band_part(
    tf.ones([seq_len_demo, seq_len_demo], dtype=tf.bool),
    -1,    # -1 = keep all lower rows (no limit)
    0      # 0  = only keep diagonal in upper half (= lower triangular)
)

print(f"Causal mask shape: {causal_mask.shape}")
print(f"\nCausal mask (True = can attend):")
print(causal_mask.numpy().astype(int))

plt.figure(figsize=(8, 6))
plt.imshow(causal_mask.numpy().astype(int), cmap='RdYlGn', interpolation='nearest')
plt.title('Causal Attention Mask (TensorFlow)\n(Green = Attend, Red = Blocked)',
          fontsize=14, fontweight='bold')
words = ["The", "cat", "sat", "on", "the", "mat"]
plt.xticks(range(6), words)
plt.yticks(range(6), words)
plt.xlabel('Attending TO (keys)')
plt.ylabel('Attending FROM (queries)')
plt.tight_layout()
plt.show()

# ==============================================================================
# PART 4: Transformer Block Component (reused from example_05)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Transformer Block Component")
print("=" * 70)

class TransformerBlock(keras.layers.Layer):
    """Transformer block with causal attention support."""

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        d_k = d_model // num_heads

        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_k)
        self.ffn_dense1 = Dense(d_ff, activation='relu')
        self.ffn_dense2 = Dense(d_model)
        self.norm1      = LayerNormalization(axis=-1)
        self.norm2      = LayerNormalization(axis=-1)
        self.dropout    = Dropout(dropout_rate) if dropout_rate > 0 else None

    def call(self, x, training=False, attention_mask=None):
        # Self-attention with optional mask
        attn_out = self.attention(
            query=x, key=x, value=x,
            attention_mask=attention_mask,
            training=training
        )
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn_dense2(self.ffn_dense1(x, training=training), training=training)
        x = self.norm2(x + ffn_out)

        return x

print("TransformerBlock defined (same as Example 05)")

# ==============================================================================
# PART 5: Complete Mini-GPT as a Keras Model
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Mini-GPT as a tf.keras.Model")
print("=" * 70)

print("""
Using tf.keras.Model instead of tf.keras.layers.Layer:

  keras.layers.Layer → for building COMPONENTS (attention, FFN, etc.)
  keras.Model        → for building the FULL MODEL (Mini-GPT)

  KEY EXTRA FEATURES from keras.Model:
    ✓ model.summary()     → print architecture table
    ✓ model.compile()     → set optimizer and loss
    ✓ model.fit(X, y)     → train model in one line!
    ✓ model.evaluate()    → compute metrics
    ✓ model.predict()     → generate predictions
    ✓ model.save()        → save to .h5 or SavedModel format
    ✓ model.load_weights()→ restore from checkpoint

  In C#/.NET terms:
    keras.Layer  = abstract class (building block)
    keras.Model  = concrete runnable class with train/predict/save methods
""")

class MiniGPT(keras.Model):
    """
    Complete Mini-GPT language model using TensorFlow/Keras.

    Architecture:
      Token IDs → Embedding → +PositionalEncoding → N×TransformerBlock → LayerNorm → Dense → Logits
    """

    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_seq_len=512, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # 1. Token embeddings (lookup table: token_id → vector)
        self.token_embedding = Embedding(input_dim=vocab_size, output_dim=d_model)

        # 2. Positional encoding (precomputed, not learned)
        # Store as regular Python attribute - NOT a tf.Variable (not trainable)
        self.pos_encoding = positional_encoding_tf(max_seq_len, d_model)

        # 3. Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]

        # 4. Final layer normalization
        self.final_norm = LayerNormalization(axis=-1)

        # 5. Language modeling head (maps d_model → vocab logits)
        self.lm_head = Dense(vocab_size, use_bias=False)

    def call(self, token_ids, training=False):
        """
        Forward pass.

        token_ids: shape (seq_len,) or (batch_size, seq_len)
        Returns: logits of shape (seq_len, vocab_size) or (batch, seq_len, vocab_size)
        """
        squeeze = (len(token_ids.shape) == 1)
        if squeeze:
            token_ids = token_ids[tf.newaxis, :]   # → (1, seq_len)

        batch_size = tf.shape(token_ids)[0]
        seq_len    = tf.shape(token_ids)[1]

        # 1. Token embeddings
        x = self.token_embedding(token_ids)          # (batch, seq_len, d_model)

        # 2. Add positional encoding
        x = x + self.pos_encoding[:seq_len]          # Broadcasting along batch dim

        # 3. Create causal mask
        causal_mask = tf.linalg.band_part(
            tf.ones([seq_len, seq_len], dtype=tf.bool), -1, 0
        )

        # 4. Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training, attention_mask=causal_mask)

        # 5. Final layer norm
        x = self.final_norm(x)                       # (batch, seq_len, d_model)

        # 6. Language model head
        logits = self.lm_head(x)                     # (batch, seq_len, vocab_size)

        if squeeze:
            logits = logits[0]                        # → (seq_len, vocab_size)

        return logits

    def generate(self, start_tokens, max_new_tokens, strategy='greedy',
                 temperature=1.0, top_k=None):
        """
        Generate text auto-regressively.

        @tf.function can be added here to compile for speed:
          @tf.function
          def generate(self, ...):  ← compiled to graph, much faster!
        """
        tokens = list(start_tokens)

        for _ in range(max_new_tokens):
            token_tensor = tf.constant(tokens)
            logits = self(token_tensor, training=False)   # No gradient tracking

            next_logits = logits[-1] / temperature        # Last position logits

            if strategy == 'greedy':
                next_token = int(tf.argmax(next_logits).numpy())

            elif strategy == 'sample':
                probs = tf.nn.softmax(next_logits)
                # tf.random.categorical: sample from categorical distribution
                # shape [1, vocab_size] → [1] sample → scalar
                next_token = int(tf.random.categorical(
                    tf.expand_dims(tf.math.log(probs), 0),   # log_probs shape: [1, vocab]
                    num_samples=1
                )[0, 0].numpy())

            elif strategy == 'top_k':
                # Get top-k values and indices
                top_k_vals, top_k_idx = tf.math.top_k(next_logits, k=top_k)
                top_k_probs = tf.nn.softmax(top_k_vals)
                # Sample from top-k
                chosen = int(tf.random.categorical(
                    tf.expand_dims(tf.math.log(top_k_probs), 0),
                    num_samples=1
                )[0, 0].numpy())
                next_token = int(top_k_idx[chosen].numpy())

            tokens.append(next_token)

        return tokens

# ==============================================================================
# PART 6: Create and Test Mini-GPT
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Create and Test Mini-GPT")
print("=" * 70)

mini_gpt = MiniGPT(
    vocab_size=20,
    d_model=8,
    num_layers=2,
    num_heads=2,
    d_ff=32,
    max_seq_len=16
)

# Forward pass (this also builds the model - TF builds lazily)
test_input = tf.constant([0, 5, 10, 15, 7])
logits = mini_gpt(test_input, training=False)

print(f"Forward pass:")
print(f"  Input token IDs: {test_input.numpy()}")
print(f"  Output logits shape: {logits.shape}")

predicted = int(tf.argmax(logits[-1]).numpy())
print(f"  Predicted next token: {predicted}")

# model.summary() - only works after building (after first call)
print(f"\nModel summary:")
mini_gpt.summary()

# ==============================================================================
# PART 7: Text Generation
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Text Generation")
print("=" * 70)

start = [0, 5]
print(f"Starting tokens: {start}")

print("\n1. GREEDY Generation:")
generated = mini_gpt.generate(start, max_new_tokens=8, strategy='greedy')
print(f"   {generated}")

print("\n2. SAMPLING Generation:")
generated = mini_gpt.generate(start, max_new_tokens=8, strategy='sample')
print(f"   {generated}")

print("\n3. TOP-K Generation (k=5):")
generated = mini_gpt.generate(start, max_new_tokens=8, strategy='top_k', top_k=5)
print(f"   {generated}")

# ==============================================================================
# PART 8: Keras Training (The Great Advantage Over NumPy!)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Training with Keras - model.compile() + model.fit()")
print("=" * 70)

print("""
THE BIGGEST ADVANTAGE OF KERAS: model.fit() trains the model automatically!

  NumPy: Cannot train at all
  PyTorch: Manual training loop (you write forward, backward, optimizer step)
  TF/Keras: model.compile() + model.fit() = automatic training!

  Compare:
  ─────────────────────────────────────────────────────────────
  PyTorch training loop:              TF/Keras training:
  ─────────────────────────────────   ────────────────────────
  for epoch in range(epochs):         model.compile(
      for batch in dataloader:            optimizer='adam',
          logits = model(batch)           loss='sparse_categorical_crossentropy'
          loss = criterion(logits, y)  )
          optimizer.zero_grad()        model.fit(
          loss.backward()                  x=X_train,
          optimizer.step()                 y=y_train,
                                           epochs=10,
                                           batch_size=32
                                       )
  ─────────────────────────────────   ────────────────────────
  ~10 lines per training loop         ~5 lines total!

  In C# terms:
    PyTorch = writing your own training controller
    Keras   = using ASP.NET's built-in training pipeline
""")

# Create dummy training data
dummy_input   = tf.constant([[0, 5, 10, 15, 7], [3, 8, 12, 2, 6]], dtype=tf.int32)
dummy_targets = tf.constant([[5, 10, 15, 7, 3], [8, 12, 2, 6, 11]], dtype=tf.int32)

print("Compiling model with Adam optimizer and cross-entropy loss...")
mini_gpt.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

print("\nRunning 3 training steps to demonstrate:")

# Create a simple dataset
dataset = tf.data.Dataset.from_tensors((dummy_input, dummy_targets)).repeat(3)

# NOTE: For language modeling, input = tokens[:-1], target = tokens[1:]
# We're using dummy data here just to show the API works
history = mini_gpt.fit(
    dummy_input,
    dummy_targets,
    epochs=3,
    batch_size=2,
    verbose=1
)

print(f"\nTraining loss values: {history.history['loss']}")

# ==============================================================================
# PART 9: Model Save and Load (TF/Keras Exclusive!)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 9: Saving and Loading Models - Keras Advantage")
print("=" * 70)

print("""
TF/Keras makes model saving incredibly easy!

  Save:
    model.save('mini_gpt_model')     ← saves entire model (weights + architecture)
    model.save_weights('weights.h5') ← saves weights only

  Load:
    model = keras.models.load_model('mini_gpt_model')
    model.load_weights('weights.h5')

  PyTorch equivalent (slightly more complex):
    torch.save(model.state_dict(), 'model.pth')     ← PyTorch save weights
    model.load_state_dict(torch.load('model.pth'))   ← PyTorch load

  In C# terms:
    Keras  = BinaryFormatter.Serialize() / Deserialize()  (1 line each)
    PyTorch = implementing ISerializable yourself          (more code)
""")

print("(Skipping actual file save to avoid creating files during demo)")
print("To save: mini_gpt.save('mini_gpt_saved_model')")
print("To load: loaded = keras.models.load_model('mini_gpt_saved_model')")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY - Three-Way Comparison")
print("=" * 70)

print("""
COMPLETE COMPARISON - NumPy vs PyTorch vs TensorFlow:

Component          │ NumPy                   │ PyTorch                  │ TensorFlow
───────────────────┼─────────────────────────┼──────────────────────────┼──────────────────────────
Token embeddings   │ np.array lookup table   │ nn.Embedding(vocab, d)   │ Embedding(vocab, d)
Positional enc.    │ custom function         │ register_buffer          │ plain attribute
Attention          │ custom code             │ nn.MultiheadAttention    │ MultiHeadAttention
FFN                │ manual matrix math      │ nn.Sequential(...)       │ Dense(d, 'relu') + Dense
LayerNorm          │ class LayerNorm (25 ln) │ nn.LayerNorm(d)          │ LayerNormalization(axis=-1)
Full model         │ Python class            │ nn.Module subclass       │ keras.Model subclass
Training loop      │ NOT POSSIBLE            │ manual (10+ lines)       │ model.compile() + fit()
Gradient           │ NOT POSSIBLE            │ loss.backward()          │ tf.GradientTape()
Save model         │ NOT POSSIBLE            │ torch.save(state_dict)   │ model.save()
GPU support        │ NOT SUPPORTED           │ .to('cuda')              │ tf.device('/GPU:0')
Mobile deployment  │ NOT SUPPORTED           │ TorchScript / CoreML     │ TensorFlow Lite (best!)

CHOOSE BASED ON YOUR GOAL:
  Learning math   → NumPy    (simplest, see every calculation)
  AI research     → PyTorch  (most flexible, standard in academia)
  Production/app  → TensorFlow/Keras  (best tooling, mobile-ready, Google Cloud)

YOU NOW UNDERSTAND:
  ✅ How the same transformer math is expressed in 3 different frameworks
  ✅ When to use each framework
  ✅ How attention, FFN, and transformer blocks work in all three
  ✅ How a full GPT model is assembled in all three
  ✅ How to train in PyTorch and TensorFlow
  ✅ This IS how ChatGPT, Gemini, and all modern LLMs work!
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 06 - TensorFlow Mini-GPT COMPLETE!")
print("=" * 70)
print("\nModule 04: Transformers - ALL THREE VERSIONS COMPLETE!")
print("  NumPy    → modules/04_transformers/examples/")
print("  PyTorch  → modules/04_transformers/examples/pytorch/")
print("  TensorFlow → modules/04_transformers/examples/tensorflow/")
