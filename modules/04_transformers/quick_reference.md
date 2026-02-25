# Module 4: Transformers - Quick Reference

**Cheat sheet for transformer concepts**

---

## 🔑 Core Formulas

### Attention Mechanism
```python
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

**Where:**
- Q = Query (what we're looking for)
- K = Keys (what we're comparing against)
- V = Values (what we return)
- d_k = dimension of keys

---

### Self-Attention
```python
# Q, K, V all from same input
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

output = Attention(Q, K, V)
```

---

### Multi-Head Attention
```python
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
```

---

### Positional Encoding
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

## 📐 Shapes Reference

**For sequence length = n, embedding dim = d:**

| Component | Input Shape | Output Shape |
|-----------|-------------|--------------|
| **Attention** | Q:(n,d), K:(n,d), V:(n,d) | (n, d) |
| **Self-Attention** | (n, d) | (n, d) |
| **Multi-Head (h heads)** | (n, d) | (n, d) |
| **Positional Encoding** | (n, d) | (n, d) |
| **FFN** | (n, d) | (n, d) |
| **Transformer Block** | (n, d) | (n, d) |

---

## 🎯 Key Concepts

### Q, K, V Analogy
```
Search Engine:
- Query = your search term
- Keys = document indices
- Values = actual documents

Attention:
- Query = what word is looking for
- Keys = what to compare with
- Values = information to retrieve
```

---

### Attention Steps
1. **Scores:** `Q @ K.T` (similarity)
2. **Scale:** `/ sqrt(d_k)` (stability)
3. **Softmax:** probabilities
4. **Weighted sum:** `weights @ V`

---

### Why Transformers Work
- ✅ **Parallel processing** (all words at once)
- ✅ **Long-range dependencies** (attention to any word)
- ✅ **Position-aware** (positional encoding)
- ✅ **Scalable** (no sequential bottleneck)

---

## 🔧 Common Operations

### Softmax
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)
```

### Layer Norm
```python
def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)
```

### Residual Connection
```python
output = x + sublayer(x)
```

---

## 🏗️ Architecture Patterns

### Transformer Block
```python
# Self-attention + residual + norm
x = layer_norm(x + multi_head_attention(x))

# Feed-forward + residual + norm
x = layer_norm(x + feed_forward(x))
```

### GPT Architecture (Decoder-Only)
```python
x = embeddings + positional_encoding

for block in transformer_blocks:
    x = block(x)  # Masked self-attention

logits = x @ output_projection
```

---

## 💻 Code Snippets

### Basic Attention
```python
scores = Q @ K.T / np.sqrt(d_k)
weights = softmax(scores)
output = weights @ V
```

### Self-Attention Layer
```python
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
return attention(Q, K, V)
```

### Multi-Head Split
```python
# Split d_model into h heads
d_head = d_model // h
Q_heads = Q.reshape(batch, n, h, d_head)
```

---

## 🎓 Transformer vs RNN

| Aspect | RNN/LSTM | Transformer |
|--------|----------|-------------|
| **Processing** | Sequential | Parallel |
| **Speed** | Slow | Fast |
| **Long context** | Forgets | Remembers all |
| **Parallelization** | No | Yes |
| **Training** | Hard | Easier |

---

## 📊 Hyperparameters

**GPT-3 Small:**
- Layers: 12
- Heads: 12
- d_model: 768
- d_ff: 3072
- Context: 2048 tokens

**GPT-3:**
- Layers: 96
- Heads: 96
- d_model: 12288
- d_ff: 49152
- Context: 2048 tokens

---

## ⚡ Quick Lookup

**Need to understand:**
- Attention? → Lesson 1
- Self-attention? → Lesson 2
- Multi-head? → Lesson 3
- Position? → Lesson 4
- Full architecture? → Lesson 6

**Need to implement:**
- Basic attention → example_01
- Self-attention → example_02
- Transformer block → example_05
- Mini-GPT → example_06

---

## 🐛 Common Issues

**Problem:** Attention weights all equal
**Solution:** Check if Q and K are too similar

**Problem:** NaN in softmax
**Solution:** Add numerical stability (subtract max)

**Problem:** Wrong shapes
**Solution:** Check matrix dimensions carefully

**Problem:** Slow training
**Solution:** Use batching, optimize matrix operations

---

## 📚 Papers to Read

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Original transformer paper

2. **"BERT"** (Devlin et al., 2018)
   - Encoder-only transformer

3. **"GPT-2"** (Radford et al., 2019)
   - Decoder-only, language modeling

4. **"GPT-3"** (Brown et al., 2020)
   - Scaling laws, few-shot learning

---

## ✅ Checklist

Before moving to Module 5:

- [ ] Understand attention mechanism
- [ ] Implement self-attention
- [ ] Understand multi-head attention
- [ ] Know positional encoding
- [ ] Can explain transformer architecture
- [ ] Built mini-GPT
- [ ] Read "Attention Is All You Need"

---

**Keep this handy while learning!** 📖
