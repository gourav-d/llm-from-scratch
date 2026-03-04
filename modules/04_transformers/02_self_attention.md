# Lesson 2: Self-Attention

**Where Q = K = V: The secret sauce of transformers!**

---

## 🎯 What You'll Learn

- ✅ What makes self-attention "self"
- ✅ How words attend to each other
- ✅ Building context-aware representations
- ✅ Implementation from scratch
- ✅ Visualizing attention patterns

**Time:** 2-3 hours
**Difficulty:** ⭐⭐⭐⭐☆

**Prerequisites:** Lesson 1 (Attention Mechanism)

---

## 🤔 What Is Self-Attention?

### The Key Difference

**Regular Attention (Lesson 1):**
```python
Q = query_input     # What we're looking for
K = search_database # Where we search
V = search_database # What we retrieve

# Different sources!
```

**Self-Attention (This Lesson):**
```python
Q = sentence  # From the SAME input
K = sentence  # From the SAME input
V = sentence  # From the SAME input

# Everything from the SAME source!
```

**Self-Attention = The input attends to itself!**

---

### Why Is This Powerful?

**Analogy: Reading Comprehension**

**Regular attention (separate query):**
```
Question: "What color is the cat?"
Text: "The brown dog and the black cat sat together."
      → Find answer in text
```

**Self-attention (understanding the text itself):**
```
Text: "The brown dog and the black cat sat together."

Understanding relationships:
- "brown" describes "dog"
- "black" describes "cat"
- "dog" and "cat" both relate to "sat"

→ Each word understands its role!
```

---

## 🔑 The Core Concept

### How Words Relate to Each Other

**Example Sentence:** "The cat sat on the mat"

**Without self-attention:**
```
Each word is independent:
["The", "cat", "sat", "on", "the", "mat"]

No relationships captured!
```

**With self-attention:**
```
"The" learns: → relates to "cat" (article-noun)
"cat" learns: → relates to "sat" (subject-verb)
"sat" learns: → relates to "cat" (verb-subject) and "mat" (verb-object)
"on" learns: → relates to "sat" and "mat" (preposition)
"mat" learns: → relates to "sat" (object-verb)

Every word understands its context!
```

---

## 🧮 The Math: Same Formula, Different Meaning

### Recap: Attention Formula

```python
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

### Self-Attention Formula

```python
# Given input: X (sentence embeddings)
Q = X @ W_Q  # Transform input to queries
K = X @ W_K  # Transform input to keys
V = X @ W_V  # Transform input to values

# Then apply attention (same as before)
SelfAttention(X) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

**Key insight:** Q, K, V all come from the same input X, but transformed differently!

---

## 📊 Step-by-Step Example

### Setup

**Input sentence:** "The cat sat"

**Word embeddings (simplified to 3D):**
```python
import numpy as np

# Each word is a 3D vector
X = np.array([
    [1.0, 0.5, 0.2],  # "The"
    [0.8, 0.9, 0.1],  # "cat"
    [0.3, 0.4, 0.7]   # "sat"
])

# Shape: (3, 3) = (num_words, embedding_dim)
```

---

### Step 1: Create Q, K, V Matrices

**In real transformers, we use learned weight matrices:**

```python
# Weight matrices (learned during training)
d_model = 3  # embedding dimension

# Initialize random weights (normally learned)
W_Q = np.random.randn(d_model, d_model) * 0.1
W_K = np.random.randn(d_model, d_model) * 0.1
W_V = np.random.randn(d_model, d_model) * 0.1

# For this example, let's use simple identity matrices
W_Q = np.eye(3)
W_K = np.eye(3)
W_V = np.eye(3)
```

---

### Step 2: Transform Input

```python
# Transform X into Q, K, V
Q = X @ W_Q  # (3, 3) @ (3, 3) = (3, 3)
K = X @ W_K  # (3, 3) @ (3, 3) = (3, 3)
V = X @ W_V  # (3, 3) @ (3, 3) = (3, 3)

# With identity matrices: Q = K = V = X
print("Q (Queries):")
print(Q)
# [[1.0, 0.5, 0.2],   "The"
#  [0.8, 0.9, 0.1],   "cat"
#  [0.3, 0.4, 0.7]]   "sat"

print("\nK (Keys):")
print(K)
# Same as Q (with identity matrices)

print("\nV (Values):")
print(V)
# Same as Q (with identity matrices)
```

---

### Step 3: Compute Attention Scores

```python
# Compute similarity between all word pairs
scores = Q @ K.T  # (3, 3) @ (3, 3) = (3, 3)

print("Attention scores:")
print(scores)

#         The    cat    sat
# The  [[1.29   1.20   0.55]
# cat   [1.20   1.46   0.55]
# sat   [0.55   0.55   0.74]]

# Interpretation:
# - "cat" and "cat" = 1.46 (high similarity with itself)
# - "The" and "cat" = 1.20 (related: article-noun)
# - "sat" and other words = lower scores
```

---

### Step 4: Scale Scores

```python
d_k = K.shape[-1]  # = 3
scores_scaled = scores / np.sqrt(d_k)

print("Scaled scores:")
print(scores_scaled)

#         The    cat    sat
# The  [[0.74   0.69   0.32]
# cat   [0.69   0.84   0.32]
# sat   [0.32   0.32   0.43]]
```

**Why scale?** Prevents gradients from vanishing/exploding during training!

---

### Step 5: Apply Softmax (Row-wise)

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores_scaled)

print("Attention weights:")
print(attention_weights)

#         The    cat    sat
# The  [[0.38   0.37   0.25]
# cat   [0.32   0.40   0.28]
# sat   [0.31   0.31   0.38]]
```

**Interpretation:**
- Each row sums to 1.0 (probabilities!)
- Row 1: "The" pays 38% attention to itself, 37% to "cat", 25% to "sat"
- Row 2: "cat" pays 40% to itself (most important for "cat")
- Row 3: "sat" pays equal attention to all words

---

### Step 6: Weighted Sum of Values

```python
output = attention_weights @ V  # (3, 3) @ (3, 3) = (3, 3)

print("Output (context-aware representations):")
print(output)

#         dim1   dim2   dim3
# The  [[0.70   0.60   0.33]
# cat   [0.70   0.62   0.32]
# sat   [0.70   0.60   0.33]]

print("\nOriginal input:")
print(X)
# [[1.0, 0.5, 0.2],   "The"
#  [0.8, 0.9, 0.1],   "cat"
#  [0.3, 0.4, 0.7]]   "sat"
```

**Notice:** Each word's representation now includes context from other words!

---

## 🎨 Visualizing Self-Attention

### Attention Heatmap

```
Sentence: "The cat sat on the mat"

Attention weights (darker = more attention):

         Attending to →
         The  cat  sat  on   the  mat
Query ↓
The     ███  ██   █    █    ██   █
cat     ██   ███  ██   █    █    █
sat     █    ██   ███  ██   █    ██
on      █    █    ██   ███  █    ██
the     ██   █    █    █    ███  ██
mat     █    █    ██   ██   ██   ███

Patterns:
- Diagonal is dark (words attend to themselves)
- "cat" → "sat" strong (subject-verb)
- "sat" → "mat" strong (verb-object)
- "on" → "sat", "mat" (prepositional phrase)
```

---

## 💻 Complete Implementation

### Self-Attention Layer (NumPy)

```python
import numpy as np

class SelfAttention:
    """
    Self-Attention layer from scratch.

    This is the building block of transformers!
    """

    def __init__(self, d_model):
        """
        Initialize self-attention layer.

        Args:
            d_model: Embedding dimension
        """
        self.d_model = d_model

        # Initialize weight matrices (normally learned)
        # Using small random values (Xavier initialization)
        scale = 1.0 / np.sqrt(d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale

    def softmax(self, x):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def forward(self, X):
        """
        Forward pass of self-attention.

        Args:
            X: Input embeddings (seq_len, d_model)

        Returns:
            output: Context-aware representations (seq_len, d_model)
            attention_weights: Attention matrix (seq_len, seq_len)
        """
        # Step 1: Linear transformations
        Q = X @ self.W_Q  # (seq_len, d_model)
        K = X @ self.W_K  # (seq_len, d_model)
        V = X @ self.W_V  # (seq_len, d_model)

        # Step 2: Compute attention scores
        scores = Q @ K.T  # (seq_len, seq_len)

        # Step 3: Scale by sqrt(d_k)
        d_k = self.d_model
        scores_scaled = scores / np.sqrt(d_k)

        # Step 4: Apply softmax
        attention_weights = self.softmax(scores_scaled)

        # Step 5: Weighted sum of values
        output = attention_weights @ V  # (seq_len, d_model)

        return output, attention_weights
```

---

### Example Usage

```python
# Example: Process a sentence
sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(sentence)
d_model = 8  # embedding dimension

# Create random embeddings (normally from embedding layer)
np.random.seed(42)
X = np.random.randn(seq_len, d_model)

print("Input shape:", X.shape)  # (6, 8)

# Create self-attention layer
self_attn = SelfAttention(d_model)

# Forward pass
output, attn_weights = self_attn.forward(X)

print("\nOutput shape:", output.shape)  # (6, 8)
print("Attention weights shape:", attn_weights.shape)  # (6, 6)

# Visualize attention weights
print("\nAttention weights:")
print("         ", "  ".join(sentence))
for i, word in enumerate(sentence):
    print(f"{word:8s}", " ".join(f"{w:.2f}" for w in attn_weights[i]))
```

**Output example:**
```
         The  cat  sat  on   the  mat
The      0.22 0.18 0.15 0.14 0.16 0.15
cat      0.18 0.24 0.17 0.13 0.14 0.14
sat      0.16 0.19 0.21 0.18 0.13 0.13
on       0.15 0.14 0.19 0.22 0.15 0.15
the      0.17 0.15 0.14 0.15 0.23 0.16
mat      0.16 0.14 0.13 0.14 0.16 0.27
```

---

## 🔍 Understanding the Output

### What Changed?

**Input (original embeddings):**
```python
X[0] = [0.49, -0.14, ...]  # "The" - generic embedding
X[1] = [1.45, -0.32, ...]  # "cat" - generic embedding
```

**Output (context-aware):**
```python
output[0] = [0.82, -0.21, ...]  # "The" - knows it precedes "cat"
output[1] = [1.03, -0.25, ...]  # "cat" - knows it's subject of "sat"
```

**Key insight:** Same words in different contexts get different representations!

**Example:**
```python
Sentence 1: "The cat sat on the mat"
            "cat" representation = influenced by "sat" (subject)

Sentence 2: "I saw the cat yesterday"
            "cat" representation = influenced by "saw" (object)

Same word, different context, different representation!
```

---

## 🎯 Self-Attention vs. Regular Attention

### Comparison Table

| Feature | Regular Attention | Self-Attention |
|---------|------------------|----------------|
| **Q source** | External query | Same input |
| **K source** | Database | Same input |
| **V source** | Database | Same input |
| **Use case** | Question answering | Understanding text |
| **Example** | "Find red items" in product catalog | "Understand relationships" in sentence |
| **Output** | Answer to query | Context-aware representation |

---

### When to Use Each?

**Self-Attention (this lesson):**
- ✅ Understanding language (GPT, BERT)
- ✅ Processing sequences (text, audio, video)
- ✅ Finding relationships within data
- ✅ Encoder models

**Cross-Attention (regular attention):**
- ✅ Machine translation (English → French)
- ✅ Question answering (query → document)
- ✅ Image captioning (image → text)
- ✅ Decoder models (attending to encoder)

---

## 🧪 Practice Problems

### Problem 1: Manual Calculation

Given a 2-word sentence with embeddings:
```python
X = np.array([
    [1.0, 0.0],  # Word 1
    [0.0, 1.0]   # Word 2
])

# Identity weight matrices
W_Q = W_K = W_V = np.eye(2)
```

**Tasks:**
1. Compute Q, K, V
2. Calculate attention scores (Q @ K.T)
3. Apply softmax
4. Compute output

<details>
<summary>Solution</summary>

```python
# Step 1: Q, K, V (with identity matrices)
Q = X @ W_Q = X = [[1, 0], [0, 1]]
K = X @ W_K = X = [[1, 0], [0, 1]]
V = X @ W_V = X = [[1, 0], [0, 1]]

# Step 2: Attention scores
scores = Q @ K.T = [[1, 0],
                     [0, 1]]

# Step 3: Scaled scores (d_k = 2)
scores_scaled = scores / sqrt(2) = [[0.71, 0],
                                      [0, 0.71]]

# Step 4: Softmax
weights = softmax(scores_scaled)
        ≈ [[0.68, 0.32],
           [0.32, 0.68]]

# Step 5: Output
output = weights @ V
       ≈ [[0.68, 0.32],
          [0.32, 0.68]]

# Each word now has information from the other!
```
</details>

---

### Problem 2: Implement Masking

In language models, we can't look at future words! Implement **masked self-attention**:

```python
def masked_self_attention(X, W_Q, W_K, W_V):
    """
    Self-attention with future masking.

    For position i, can only attend to positions 0...i
    (cannot see future words)
    """
    # Your code here
    pass

# Test
X = np.random.randn(4, 8)  # 4 words, 8-dim embeddings
# ... implement and test
```

<details>
<summary>Solution</summary>

```python
def masked_self_attention(X, W_Q, W_K, W_V):
    seq_len = X.shape[0]

    # Linear transformations
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # Compute scores
    scores = Q @ K.T / np.sqrt(W_Q.shape[0])

    # Create mask (lower triangular matrix)
    mask = np.tril(np.ones((seq_len, seq_len)))

    # Apply mask: set future positions to -infinity
    scores = np.where(mask == 1, scores, -1e9)

    # Softmax (future positions will have ~0 probability)
    weights = softmax(scores)

    # Output
    output = weights @ V

    return output, weights

# The attention matrix will be lower triangular:
#     w1  w2  w3  w4
# w1 [x   0    0   0 ]  (only attends to self)
# w2 [x   x   0   0 ]  (attends to w1, w2)
# w3 [x   x   x   0 ]  (attends to w1, w2, w3)
# w4 [x   x   x   x ]  (attends to all)
```
</details>

---

## 🌟 Real-World Example: GPT Text Generation

### How GPT Uses Self-Attention

**Prompt:** "The cat sat on"

**GPT processing:**
```python
Step 1: Embed words
X = embed(["The", "cat", "sat", "on"])

Step 2: Self-attention (masked!)
# "The" sees: ["The"]
# "cat" sees: ["The", "cat"]
# "sat" sees: ["The", "cat", "sat"]
# "on" sees: ["The", "cat", "sat", "on"]

Step 3: Each word understands its context
# "on" knows:
#   - It follows "sat" (verb)
#   - "cat" is the subject
#   - Next word should complete the prepositional phrase

Step 4: Predict next word
# Likely predictions: "the", "a", "top", "it"
# "the" (mat/table/floor)

Output: "The cat sat on the mat"
```

**Self-attention helped GPT understand that "on" needs a noun to complete the phrase!**

---

## 🔑 Key Takeaways

### Remember These Points

1. **Self-Attention = Input attending to itself**
   - Q, K, V all from same source
   - But transformed differently (W_Q, W_K, W_V)

2. **Captures relationships within sequence**
   - Subject-verb relationships
   - Modifier-noun relationships
   - Long-range dependencies

3. **Each word gets context from all others**
   - Not just neighboring words
   - Global context in one step!

4. **Output = Context-aware representations**
   - Same word, different contexts → different representations
   - Critical for understanding language!

5. **Building block of transformers**
   - GPT uses masked self-attention
   - BERT uses bidirectional self-attention
   - Foundation of all modern LLMs!

---

## ✅ Self-Check

Before moving to Lesson 3 (Multi-Head Attention), ensure you can:

- [ ] Explain difference between attention and self-attention
- [ ] Describe how Q, K, V are created from input
- [ ] Implement self-attention in NumPy
- [ ] Understand why self-attention is powerful
- [ ] Explain masked self-attention (for GPT)
- [ ] Visualize attention patterns

**If you checked all boxes:** Ready for Multi-Head Attention! 🎉

**If not:** Review this lesson, run the code examples, experiment!

---

## 💬 Common Questions

**Q: Why do we need W_Q, W_K, W_V if they all come from X?**
A: These learned transformations let the model create different "views" of the input. Q might focus on "what to look for", K on "what's available", V on "what to return". This flexibility is crucial!

**Q: Isn't self-attention just averaging all words?**
A: No! The attention weights are learned and context-dependent. Different words get different weights based on relevance.

**Q: What's the difference from averaging embeddings?**
A: Averaging treats all words equally. Self-attention learns which words are relevant for each position.

**Q: Why is this better than RNNs?**
A:
- ✅ Parallel processing (RNNs are sequential)
- ✅ Direct connections (RNNs pass through all positions)
- ✅ Better for long sequences (RNNs forget distant info)

---

## 📖 Further Reading

### Next Steps

1. **Run example code:**
   - `examples/example_02_self_attention.py` (when available)
   - Visualize attention patterns!

2. **Next lesson:**
   - `03_multi_head_attention.md`
   - Multiple attention patterns in parallel!

3. **Deep dive:**
   - "Attention Is All You Need" (Section 3.2.1)
   - "The Illustrated Transformer" (Jay Alammar)

---

## 🎊 Congratulations!

**You've learned self-attention - the core of GPT and BERT!**

**Key achievement:**
> You understand how transformers learn relationships within sequences!

**What you can build:**
- Basic language model (with masking)
- Text encoder (bidirectional)
- Foundation for GPT architecture!

---

**Next:** Lesson 3 - Multi-Head Attention (Learn multiple patterns at once!)

**Time to next lesson:** When comfortable with self-attention

**Practice first!** Try the problems, visualize attention, experiment! 🚀
