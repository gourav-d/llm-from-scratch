# Lesson 6: Optimizers - Advanced Learning Algorithms

## 🎯 Learning Objectives

After this lesson, you will:
- Understand why vanilla gradient descent is slow
- Learn how Momentum speeds up training
- Understand RMSProp and adaptive learning rates
- Master Adam optimizer (used to train GPT-3!)
- Know when to use which optimizer
- Understand production best practices

---

## 📖 Table of Contents

1. [The Problem with Vanilla Gradient Descent](#1-the-problem-with-vanilla-gradient-descent)
2. [Momentum Optimizer](#2-momentum-optimizer)
3. [RMSProp Optimizer](#3-rmsprop-optimizer)
4. [Adam Optimizer](#4-adam-optimizer)
5. [Comparison and When to Use Which](#5-comparison-and-when-to-use-which)
6. [Connection to GPT and Modern LLMs](#6-connection-to-gpt-and-modern-llms)
7. [Summary](#7-summary)

---

## 1. The Problem with Vanilla Gradient Descent

### What We've Been Using

In previous lessons, we updated weights like this:

```python
# Vanilla Gradient Descent (SGD)
W = W - learning_rate * gradient
```

**This is called Stochastic Gradient Descent (SGD).**

### The Problems

#### Problem 1: Slow Progress in Valleys

Imagine you're rolling a ball down a long, narrow valley:

```
    |     |
    |     |
    |     |  ← Ball bounces side-to-side
    |  o  |     (wastes energy)
    | / \ |
    |/   \|
    ------● Goal
```

**What happens:**
- Steep sides → Large gradients → Ball bounces
- Gentle slope toward goal → Small gradients → Slow progress
- Wastes time going back and forth instead of down!

**In neural networks:**
- Some dimensions have steep gradients (bounce)
- Other dimensions have gentle gradients (slow)
- Training oscillates and takes forever!

#### Problem 2: Different Features Need Different Learning Rates

```python
# Feature 1: Changes a lot (large gradients)
# Feature 2: Changes a little (small gradients)

# Problem: Same learning_rate for both!
W1 = W1 - 0.01 * gradient1  # Too fast? (oscillates)
W2 = W2 - 0.01 * gradient2  # Too slow? (stuck)
```

**What we need:**
- Adaptive learning rates per parameter
- Memory of past gradients
- Smoother, faster convergence

---

## 2. Momentum Optimizer

### The Big Idea

**Add momentum like a rolling ball!**

Instead of moving in the direction of the current gradient alone, remember previous gradients and build up velocity.

```
Without Momentum:        With Momentum:
    ↓                       ↓
   →↓                      ↘↓
  ←↓→                     ↘↓
 →↓←                     ↘↓
↓                       ↘↓
(zigzag)               ↘↓
                      ↘● (smooth!)
```

### The Math

#### Vanilla SGD:
```python
W = W - learning_rate * gradient
```

#### Momentum SGD:
```python
# Step 1: Update velocity (exponential moving average of gradients)
velocity = beta * velocity + gradient

# Step 2: Update weights using velocity
W = W - learning_rate * velocity
```

**Key parameter:** `beta` (typically 0.9)
- Higher beta = more momentum = smoother path
- Lower beta = less momentum = more responsive to current gradient

### How It Works

**Think of velocity as "accumulated gradients":**

```python
# Iteration 1:
velocity = 0 + gradient₁ = gradient₁
W = W - lr * gradient₁

# Iteration 2:
velocity = 0.9 * gradient₁ + gradient₂
          = 0.9 * gradient₁ + 1.0 * gradient₂
W = W - lr * (0.9 * gradient₁ + gradient₂)

# Iteration 3:
velocity = 0.9 * (0.9 * gradient₁ + gradient₂) + gradient₃
          = 0.81 * gradient₁ + 0.9 * gradient₂ + 1.0 * gradient₃
```

**Notice:**
- Recent gradients have more weight (1.0, 0.9, 0.81...)
- Old gradients fade out exponentially
- If gradients point in the same direction → velocity builds up!
- If gradients cancel out → velocity dampens (good!)

### Benefits

1. **Faster in valleys:** Builds up speed in consistent directions
2. **Dampens oscillations:** Cancels out back-and-forth movements
3. **Escapes plateaus:** Can push through flat regions with accumulated velocity

### Code Example

```python
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = None  # Initialize on first use

    def update(self, weights, gradients):
        # Initialize velocity on first call
        if self.velocity is None:
            self.velocity = np.zeros_like(gradients)

        # Update velocity: v = β*v + g
        self.velocity = self.beta * self.velocity + gradients

        # Update weights: W = W - lr * v
        weights -= self.learning_rate * self.velocity

        return weights
```

### C# Comparison

```csharp
// Similar to exponential smoothing in time series
// Or like a low-pass filter in signal processing

double velocity = 0.0;
const double beta = 0.9;

foreach (var gradient in gradients) {
    velocity = beta * velocity + gradient;  // Exponential moving average
    weight -= learningRate * velocity;
}
```

---

## 3. RMSProp Optimizer

### The Big Idea

**Adapt learning rate per parameter based on gradient magnitude history.**

If a parameter has consistently large gradients → reduce its learning rate
If a parameter has consistently small gradients → increase its learning rate

### The Problem Momentum Doesn't Solve

Momentum helps with consistent directions, but doesn't solve:
- Different parameters needing different learning rates
- Parameters with large vs. small gradient magnitudes

### The Math

```python
# Step 1: Accumulate squared gradients (moving average)
cache = beta * cache + (1 - beta) * (gradient ** 2)

# Step 2: Adapt learning rate per parameter
W = W - learning_rate * gradient / (sqrt(cache) + epsilon)
```

**Key components:**
- `cache`: Running average of **squared gradients** (gradient²)
- `beta`: Decay rate (typically 0.9 or 0.99)
- `epsilon`: Small constant (1e-8) to prevent division by zero
- Division by `sqrt(cache)`: Normalizes gradient by its typical magnitude

### How It Works

**Intuition:**

```python
# Parameter with LARGE gradients:
gradient = [100, 95, 105, 98, ...]
cache ≈ 100² = 10,000
effective_gradient = 100 / sqrt(10,000) = 100 / 100 = 1.0
# → Learning rate effectively reduced!

# Parameter with SMALL gradients:
gradient = [0.01, 0.02, 0.01, ...]
cache ≈ 0.01² = 0.0001
effective_gradient = 0.01 / sqrt(0.0001) = 0.01 / 0.01 = 1.0
# → Learning rate effectively increased!
```

**Result:** All parameters take similar-sized steps, regardless of their natural gradient scale!

### Why Square the Gradients?

```python
cache = beta * cache + (1 - beta) * gradient²
```

**Reasons:**
1. **Sign doesn't matter:** We care about magnitude, not direction
   - `gradient² ≥ 0` always (no cancellation)
   - Positive and negative gradients both contribute

2. **Large gradients dominate:** Penalize parameters with large, noisy gradients
   - Gradient of 10 → contributes 100 to cache
   - Gradient of 1 → contributes 1 to cache

3. **Statistics:** We're tracking the variance/magnitude of gradients

### Code Example

```python
class RMSPropOptimizer:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.cache = None

    def update(self, weights, gradients):
        # Initialize cache on first call
        if self.cache is None:
            self.cache = np.zeros_like(gradients)

        # Accumulate squared gradients
        self.cache = self.beta * self.cache + (1 - self.beta) * (gradients ** 2)

        # Adaptive update
        weights -= self.learning_rate * gradients / (np.sqrt(self.cache) + self.epsilon)

        return weights
```

### Benefits

1. **Adaptive learning rates:** Each parameter gets its own effective learning rate
2. **Handles different scales:** Works well with features of different magnitudes
3. **Robust to hyperparameters:** Less sensitive to global learning rate choice

### Limitation

RMSProp doesn't have momentum → Can still oscillate in valleys!

---

## 4. Adam Optimizer

### The Big Idea

**Combine the best of both worlds: Momentum + RMSProp!**

Adam = **Ada**ptive **M**oment estimation

### The Math

Adam maintains **two** moving averages:

```python
# First moment: Mean of gradients (like Momentum)
m = beta1 * m + (1 - beta1) * gradient

# Second moment: Mean of squared gradients (like RMSProp)
v = beta2 * v + (1 - beta2) * (gradient ** 2)

# Bias correction (important for early iterations!)
m_corrected = m / (1 - beta1^t)
v_corrected = v / (1 - beta2^t)

# Update weights
W = W - learning_rate * m_corrected / (sqrt(v_corrected) + epsilon)
```

**Key parameters:**
- `beta1`: Decay rate for first moment (typically 0.9)
- `beta2`: Decay rate for second moment (typically 0.999)
- `learning_rate`: Step size (typically 0.001)
- `epsilon`: Numerical stability (1e-8)
- `t`: Iteration number (for bias correction)

### Breaking It Down

#### 1. First Moment (m): Direction with Momentum

```python
m = beta1 * m + (1 - beta1) * gradient
```

**This is like Momentum's velocity:**
- Accumulates gradients over time
- Points in average gradient direction
- Dampens oscillations

**With beta1=0.9:**
```python
m = 0.9 * m + 0.1 * gradient
```
- 90% old direction
- 10% new gradient

#### 2. Second Moment (v): Adaptive Learning Rate

```python
v = beta2 * v + (1 - beta2) * (gradient ** 2)
```

**This is like RMSProp's cache:**
- Tracks gradient magnitude squared
- Used to normalize step sizes
- Adapts learning rate per parameter

**With beta2=0.999:**
```python
v = 0.999 * v + 0.001 * gradient²
```
- 99.9% old magnitude
- 0.1% new magnitude
- Very smooth estimate!

#### 3. Bias Correction

**The problem:** `m` and `v` are initialized to zero.

In early iterations:
```python
# Iteration 1:
m = 0.9 * 0 + 0.1 * gradient = 0.1 * gradient  # Too small!
v = 0.999 * 0 + 0.001 * gradient² = 0.001 * gradient²  # Too small!
```

**The solution:** Divide by `(1 - beta^t)` to correct the bias.

```python
# Iteration 1 (t=1):
m_corrected = m / (1 - 0.9^1) = m / 0.1 = 10 * m  # ✓ Corrected!
v_corrected = v / (1 - 0.999^1) = v / 0.001 = 1000 * v  # ✓ Corrected!

# Iteration 100 (t=100):
m_corrected = m / (1 - 0.9^100) ≈ m / 1.0 ≈ m  # Negligible correction
v_corrected = v / (1 - 0.999^100) ≈ v / 1.0 ≈ v  # Negligible correction
```

**As t → ∞, (1 - beta^t) → 1, so bias correction becomes unnecessary.**

### Complete Code Example

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Iteration counter

    def update(self, weights, gradients):
        # Initialize moments on first call
        if self.m is None:
            self.m = np.zeros_like(gradients)
            self.v = np.zeros_like(gradients)

        # Increment iteration counter
        self.t += 1

        # Update first moment (momentum)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients

        # Update second moment (RMSProp)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        # Bias correction
        m_corrected = self.m / (1 - self.beta1 ** self.t)
        v_corrected = self.v / (1 - self.beta2 ** self.t)

        # Update weights
        weights -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

        return weights
```

### Why Adam is Popular

1. **Best of both:** Combines momentum and adaptive learning rates
2. **Robust:** Works well with default hyperparameters (beta1=0.9, beta2=0.999, lr=0.001)
3. **Fast convergence:** Often reaches good solutions quickly
4. **Handles sparse gradients:** Good for NLP and embeddings
5. **Industry standard:** Used to train GPT-2, GPT-3, BERT, and most modern models

### C# Comparison

```csharp
// Similar to Kalman filter (state estimation)
// Combines multiple sources of information

class AdamOptimizer {
    double m = 0.0;  // First moment (like velocity)
    double v = 0.0;  // Second moment (like variance)
    int t = 0;       // Time step

    void Update(ref double weight, double gradient) {
        t++;

        // Exponential moving averages
        m = 0.9 * m + 0.1 * gradient;
        v = 0.999 * v + 0.001 * gradient * gradient;

        // Bias correction
        double m_hat = m / (1 - Math.Pow(0.9, t));
        double v_hat = v / (1 - Math.Pow(0.999, t));

        // Adaptive update
        weight -= 0.001 * m_hat / (Math.Sqrt(v_hat) + 1e-8);
    }
}
```

---

## 5. Comparison and When to Use Which

### Quick Comparison Table

| Optimizer | Speed | Stability | Memory | When to Use |
|-----------|-------|-----------|--------|-------------|
| **SGD** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Simple problems, careful tuning |
| **SGD + Momentum** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | CNNs, vision tasks, final fine-tuning |
| **RMSProp** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | RNNs, online learning |
| **Adam** | ⭐⭐⭐ | ⭐⭐ | ⭐ | Default choice, NLP, transformers |

### Detailed Comparison

#### SGD (Vanilla Gradient Descent)

**Formula:** `W = W - lr * gradient`

**Pros:**
- Simple, well-understood
- Can find better minima (sharper, more generalizable)
- Low memory usage

**Cons:**
- Slow convergence
- Sensitive to learning rate
- Struggles with different scales

**Use when:**
- Final fine-tuning (after pre-training with Adam)
- Small datasets
- Need best possible generalization

**Typical hyperparameters:**
```python
learning_rate = 0.01
```

#### SGD + Momentum

**Formula:**
```python
v = beta * v + gradient
W = W - lr * v
```

**Pros:**
- Faster than vanilla SGD
- Dampens oscillations
- Simple to implement

**Cons:**
- Still sensitive to learning rate
- One global learning rate for all parameters

**Use when:**
- Training CNNs for image classification
- Want better generalization than Adam
- Have time to tune learning rate

**Typical hyperparameters:**
```python
learning_rate = 0.01
beta = 0.9
```

#### RMSProp

**Formula:**
```python
cache = beta * cache + (1 - beta) * gradient²
W = W - lr * gradient / (sqrt(cache) + eps)
```

**Pros:**
- Adaptive learning rates
- Good for RNNs
- Handles different scales well

**Cons:**
- No momentum (can still oscillate)
- Less popular than Adam
- Can be unstable with large learning rates

**Use when:**
- Training RNNs (recurrent neural networks)
- Online learning (streaming data)
- Parameters have very different scales

**Typical hyperparameters:**
```python
learning_rate = 0.001
beta = 0.9
epsilon = 1e-8
```

#### Adam

**Formula:**
```python
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient²
W = W - lr * m / (sqrt(v) + eps)
```
(Plus bias correction)

**Pros:**
- Fast convergence
- Robust to hyperparameters
- Works "out of the box"
- Handles sparse gradients well
- Industry standard

**Cons:**
- Can generalize worse than SGD+Momentum
- Uses more memory (stores m and v)
- Sometimes doesn't converge to best solution

**Use when:**
- Training transformers (GPT, BERT)
- NLP tasks with word embeddings
- Default choice for new projects
- Need fast initial training

**Typical hyperparameters:**
```python
learning_rate = 0.001  # or 3e-4
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
```

### Decision Tree

```
START
  |
  ├─ Training Transformer/GPT?
  |    └─→ YES → Use Adam (lr=3e-4)
  |
  ├─ Training CNN (ResNet/VGG)?
  |    └─→ YES → Use SGD+Momentum (lr=0.1, decay schedule)
  |
  ├─ Training RNN/LSTM?
  |    └─→ YES → Use Adam or RMSProp (lr=0.001)
  |
  ├─ Small dataset, need best accuracy?
  |    └─→ YES → Use SGD+Momentum (lr=0.01)
  |
  └─ Not sure / Experimenting?
       └─→ Use Adam (lr=0.001) as default
```

### Real-World Example: Training GPT-3

**OpenAI used Adam for GPT-3 training:**

```python
optimizer = Adam(
    learning_rate = 0.0006,  # 6e-4
    beta1 = 0.9,
    beta2 = 0.95,  # Note: Lower than default!
    epsilon = 1e-8
)
```

**Why Adam for GPT?**
1. 175 billion parameters → Need adaptive learning rates
2. Sparse gradients from embeddings → Adam handles this well
3. Mixed precision training → Adam is numerically stable
4. Fast iteration → Adam converges quickly

**Fun fact:** Training GPT-3 took 1,000+ GPUs and cost ~$5 million!

---

## 6. Connection to GPT and Modern LLMs

### How GPT is Trained

**GPT-3 Training Process:**

```python
# 1. Initialize GPT-3 model (175 billion parameters!)
model = GPT3(
    layers=96,
    hidden_size=12288,
    attention_heads=96,
    vocab_size=50257
)

# 2. Create Adam optimizer
optimizer = Adam(
    learning_rate=6e-4,
    beta1=0.9,
    beta2=0.95
)

# 3. Training loop
for batch in training_data:
    # Forward pass
    predictions = model(batch.input_ids)
    loss = cross_entropy(predictions, batch.target_ids)

    # Backward pass
    gradients = compute_gradients(loss)

    # Update with Adam (THIS is what we learned!)
    for param in model.parameters():
        optimizer.update(param, gradients[param])
```

**Key insights:**

1. **Adam is essential:** With 175B parameters, you NEED adaptive learning rates
2. **Same algorithm:** The Adam you learned is the EXACT same algorithm used in GPT
3. **Scale difference:** GPT has billions of parameters, our examples have hundreds
4. **Same principles:** Momentum + adaptive learning rates = fast, stable training

### Why Modern LLMs Use Adam

**Reasons:**

1. **Sparse embeddings:**
   - Word embeddings have sparse gradients
   - Most words not in each batch → gradients are zero
   - Adam handles sparse updates well

2. **Mixed scales:**
   - Embedding layer: Large, sparse updates
   - Attention weights: Dense, small updates
   - Feed-forward layers: Medium updates
   - Adam adapts to each automatically!

3. **Fast iteration:**
   - Training LLMs costs millions of dollars
   - Need to iterate quickly on architectures
   - Adam works "out of the box" without tuning

4. **Stability:**
   - Training for weeks/months
   - Can't afford divergence or NaN values
   - Adam is numerically stable

### Learning Rate Schedules

Modern LLMs don't use constant learning rates:

```python
def get_learning_rate(step, warmup_steps=4000, max_lr=0.001):
    """Learning rate schedule used in transformers."""
    if step < warmup_steps:
        # Warmup: Linear increase
        return max_lr * (step / warmup_steps)
    else:
        # Decay: Inverse square root
        return max_lr * (warmup_steps / step) ** 0.5

# Usage with Adam
for step in range(total_steps):
    lr = get_learning_rate(step)
    optimizer.learning_rate = lr
    optimizer.update(weights, gradients)
```

**Why warmup?**
- Early in training, parameters are random
- Large learning rate → unstable updates
- Warmup gradually increases lr → stable start

**Why decay?**
- Later in training, close to solution
- Large learning rate → overshoot
- Decay reduces lr → fine-tune solution

### The Complete Picture

**What you've learned in Module 3:**

```
Lesson 1: Perceptron
  └─→ How a single neuron works

Lesson 2: Activation Functions
  └─→ Non-linearity (GELU used in GPT!)

Lesson 3: Multi-Layer Networks
  └─→ Stacking layers (GPT has 96 layers!)

Lesson 4: Backpropagation
  └─→ Computing gradients for ALL parameters

Lesson 5: Training Loop
  └─→ Batching, epochs, monitoring

Lesson 6: Optimizers (YOU ARE HERE!)
  └─→ Adam: How GPT learns efficiently
```

**You now understand the COMPLETE training process for GPT!** 🎉

The only thing left is **attention mechanism** (Module 4), which makes GPT powerful at understanding context.

---

## 7. Summary

### Key Takeaways

#### Vanilla SGD
- **Formula:** `W = W - lr * gradient`
- **Problem:** Slow, sensitive to learning rate, oscillates
- **Use:** Simple problems, final fine-tuning

#### Momentum
- **Formula:** `v = β*v + g; W = W - lr*v`
- **Benefit:** Builds up speed, dampens oscillations
- **Use:** CNNs, vision tasks

#### RMSProp
- **Formula:** `cache = β*cache + (1-β)*g²; W = W - lr*g/√cache`
- **Benefit:** Adaptive learning rates per parameter
- **Use:** RNNs, different scales

#### Adam (Most Popular!)
- **Formula:** Combines momentum + RMSProp + bias correction
- **Benefit:** Fast, robust, works "out of the box"
- **Use:** Transformers, NLP, default choice
- **Used in:** GPT-2, GPT-3, BERT, and most modern LLMs

### Visual Summary

```
Training Speed:
SGD          ════════════════════════ (slow)
Momentum     ════════════════ (faster)
RMSProp      ════════════ (fast)
Adam         ═══════ (fastest!)

Generalization:
SGD+Momentum ★★★★★ (best)
Adam         ★★★☆☆ (good)
RMSProp      ★★★☆☆ (good)
SGD          ★★★★☆ (very good)

Ease of Use:
Adam         ★★★★★ (easiest - default hyperparameters work)
RMSProp      ★★★☆☆ (medium)
Momentum     ★★☆☆☆ (requires tuning)
SGD          ★☆☆☆☆ (hardest - very sensitive)
```

### Production Tips

1. **Start with Adam:**
   - `lr=0.001, beta1=0.9, beta2=0.999`
   - Works well 90% of the time

2. **If Adam doesn't generalize well:**
   - Try SGD+Momentum with learning rate schedule
   - `lr=0.01, momentum=0.9, decay=0.1 every 30 epochs`

3. **Monitor training:**
   - If loss oscillates → reduce learning rate
   - If training is slow → increase learning rate or switch to Adam
   - If validation loss increases → add regularization or reduce learning rate

4. **Use learning rate warmup:**
   - Especially important for large models (transformers)
   - Linear warmup for 1-10% of total steps

5. **GPU memory constraints:**
   - Adam uses 2x memory (stores m and v)
   - If OOM, try SGD+Momentum

### What You've Learned

You can now:
- ✅ Explain why vanilla SGD is slow
- ✅ Implement Momentum, RMSProp, and Adam from scratch
- ✅ Choose the right optimizer for your task
- ✅ Understand how GPT-3 was trained
- ✅ Use production-level optimization techniques

### What's Next

**Module 3 is complete!** You understand:
1. Perceptrons
2. Activation functions
3. Multi-layer networks
4. Backpropagation
5. Training loops
6. Optimizers

**Module 4: Transformers** will teach:
- Attention mechanism
- Self-attention
- Multi-head attention
- Positional encoding
- Complete GPT architecture

**You're ready to build real neural networks!** 🚀

---

## 📚 Additional Resources

### Papers
- Adam Optimizer: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (2014)
- RMSProp: Mentioned in [Coursera Neural Networks course](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- Momentum: [On the importance of initialization and momentum in deep learning](http://proceedings.mlr.press/v28/sutskever13.pdf) (2013)

### Visualizations
- [Optimizer Comparison Visualizations](https://cs231n.github.io/neural-networks-3/)
- [Distill: Why Momentum Really Works](https://distill.pub/2017/momentum/)

### Practical Guides
- [PyTorch Optimizers Documentation](https://pytorch.org/docs/stable/optim.html)
- [TensorFlow Optimizers Guide](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)

---

**Next step:** Run `example_06_optimizers.py` to see these optimizers in action! 🎯
