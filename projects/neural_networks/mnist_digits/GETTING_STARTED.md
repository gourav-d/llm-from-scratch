# Getting Started: MNIST Handwritten Digit Classifier

**Step-by-step guide to building your first image classifier**

---

## 📋 Prerequisites

Before starting, ensure you've completed:

✅ **Module 2: NumPy** (array operations, linear algebra)
✅ **Module 3, Lessons 1-5** (perceptrons through training loops)
✅ **Project 1: Email Spam Classifier** (recommended but not required)

**Time Required:** 3-4 hours

---

## 🎯 Learning Goals

By the end of this project, you'll be able to:

1. ✅ Work with image data (pixels as features)
2. ✅ Build deeper neural networks (3 layers!)
3. ✅ Perform multi-class classification (10 digits: 0-9)
4. ✅ Achieve 95%+ accuracy on real benchmark dataset
5. ✅ Visualize predictions and mistakes
6. ✅ Understand how computers "see" images

---

## 🖼️ What is MNIST?

**MNIST** = Modified National Institute of Standards and Technology database

**The Dataset:**
- **70,000 images** of handwritten digits (0-9)
- **28x28 pixels** = 784 pixels per image
- **Grayscale:** Each pixel is 0-255 (0=black, 255=white)
- **Train set:** 60,000 images
- **Test set:** 10,000 images

**Sample digits:**
```
[0] [1] [2] [3] [4] [5] [6] [7] [8] [9]
 █   █  ███ ███  █ █████ ███ ███ ███ ███
███  █    █   █ ██    █ █     █ █ █ █ █
█ █  █  ███ ███ ███ ███ ███ ███ ███ ███
█ █  █  █     █   █   █ █ █   █ █ █   █
███  █  ███ ███   █ ███ ███   █ ███ ███
```

**Why it's famous:**
- ✅ The "Hello World" of deep learning
- ✅ Simple enough to train quickly
- ✅ Complex enough to be interesting
- ✅ Real-world benchmark for comparison

---

## 🚦 Step-by-Step Guide

### Step 1: Understand the Problem (20 minutes)

**Read first:**
- `README.md` (project overview)

**Think about:**
- How does a computer "see" an image?
- What's different from text classification?
- Why 10 classes instead of 2?

**Key insight:**
Each image is just a grid of numbers!

```
Image of digit "5":

  0   0   0  255 255 255   0   0
  0   0 255   0   0   0   0   0
  0   0 255 255 255   0   0   0
  0   0   0   0 255   0   0   0
  0 255 255 255   0   0   0   0

→ Flatten to vector: [0,0,0,255,255,255,0,0,0,0,255,...]
→ 784 numbers (28×28 = 784 pixels)
→ Feed to neural network!
```

---

### Step 2: Run the Simple Version (45 minutes)

**Start here!**

```bash
# Navigate to project
cd projects/neural_networks/mnist_digits

# Run simple version
python project_simple.py
```

**What happens:**

```
Step 1: Loading MNIST data...
✓ Training set: 60,000 images (28x28)
✓ Test set: 10,000 images
✓ Classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

Step 2: Preprocessing data...
✓ Normalized pixels to [0, 1] range
✓ Flattened images: 28x28 → 784 features
✓ One-hot encoded labels

Step 3: Building neural network...
Architecture: 784 → 128 → 64 → 10
✓ Layer 1: 784 → 128 (ReLU)
✓ Layer 2: 128 → 64 (ReLU)
✓ Layer 3: 64 → 10 (Softmax)
✓ Total parameters: 109,386

Step 4: Training...
Epoch 5/20:  Loss=0.285, Acc=91.5%, Test Acc=90.8%
Epoch 10/20: Loss=0.142, Acc=95.8%, Test Acc=94.2%
Epoch 15/20: Loss=0.098, Acc=97.2%, Test Acc=95.5%
Epoch 20/20: Loss=0.075, Acc=97.9%, Test Acc=96.1%

Step 5: Evaluating...
✓ Final Test Accuracy: 96.1%
✓ Top-3 accuracy: 99.2%

Step 6: Visualizing...
✓ Saved 20 sample predictions
✓ Saved confusion matrix
✓ Saved training curves

Done! Check results/ folder.
```

**Expected training time:** 2-3 minutes on CPU

**View results:**
```bash
ls results/
# You'll see:
# - training_curve.png (loss and accuracy)
# - confusion_matrix.png (10x10 grid)
# - sample_predictions.png (20 random predictions)
# - mistakes.png (where model fails)
```

---

### Step 3: Understand the Architecture (30 minutes)

**Open:** `project_simple.py`

**The network has 3 layers (deep learning!):**

```
Input Layer (784)  →  Hidden Layer 1 (128)  →  Hidden Layer 2 (64)  →  Output (10)
   [pixels]              [ReLU]                    [ReLU]              [Softmax]

Example forward pass:
[0,0,0,255,...,0]  →  [0.2, 0.8, ...]  →  [0.5, 0.1, ...]  →  [0.01, 0.05, 0.87, ...]
  784 pixels           128 features         64 features         10 probabilities
                                                                    ↑
                                                            Digit "2" (87% confident!)
```

**Key differences from Project 1:**

| Aspect | Email Spam | MNIST Digits |
|--------|-----------|--------------|
| **Input size** | 1000 words | 784 pixels |
| **Network depth** | 2 layers | 3 layers |
| **Output** | 1 neuron (binary) | 10 neurons (multi-class) |
| **Output activation** | Sigmoid | Softmax |
| **Loss function** | Binary cross-entropy | Categorical cross-entropy |

---

### Step 4: Understand Key Concepts (45 minutes)

**Open:** `CONCEPTS.md`

**Focus on:**

#### 1. Image as Numbers
```python
# Image is just a matrix!
image = np.array([
    [0, 0, 255, 255, 0],
    [0, 255, 0, 0, 255],
    [255, 0, 0, 0, 255],
    # ... 28 rows total
])

# Flatten to vector
pixels = image.flatten()  # [0, 0, 255, 255, 0, 0, 255, ...]

# Normalize to [0, 1]
pixels = pixels / 255.0   # [0.0, 0.0, 1.0, 1.0, 0.0, ...]
```

#### 2. Softmax for Multi-Class

```python
# Network outputs 10 scores (logits)
logits = [2.1, -0.5, 3.8, 1.2, 0.1, -1.0, 0.5, 1.8, 0.3, -0.8]
#         [0]   [1]   [2]  [3]  [4]   [5]  [6]  [7]  [8]   [9]

# Softmax converts to probabilities
probabilities = softmax(logits)
# [0.05, 0.01, 0.87, 0.02, 0.02, 0.00, 0.01, 0.01, 0.01, 0.00]
#   0%    1%    87%   2%   2%    0%    1%    1%    1%    0%

# Prediction: argmax = 2 (highest probability)
prediction = 2  ✓
```

**Softmax formula:**
```
softmax(x_i) = e^(x_i) / Σ(e^(x_j))

Properties:
- All outputs sum to 1 (valid probability distribution)
- Exponentiation amplifies differences
- Differentiable (can backpropagate!)
```

#### 3. One-Hot Encoding

```python
# True label: digit "7"
label = 7

# One-hot encoding: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
#                    0  1  2  3  4  5  6  7  8  9
one_hot = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
          #                      ↑
          #                 only position 7 is 1

# For .NET devs: Like enum to bool array
```

**Why?**
- Loss function needs target probabilities
- Can't use label=7 directly (not comparable to softmax output)

#### 4. Categorical Cross-Entropy

```python
# True label (one-hot): [0, 0, 1, 0, ...]  (digit "2")
# Predicted:            [0.05, 0.01, 0.87, 0.02, ...]

# Loss = -sum(true * log(pred))
#      = -(0*log(0.05) + 0*log(0.01) + 1*log(0.87) + ...)
#      = -log(0.87)
#      = 0.14  (low loss, good prediction!)

# If predicted [0.05, 0.01, 0.01, 0.02, ...] (wrong!)
# Loss = -log(0.01) = 4.61  (high loss, bad prediction!)
```

---

### Step 5: Explore the Code (60 minutes)

**Open:** `project_simple.py`

**Code structure (5 parts):**

#### Part 1: Data Loading (Lines 1-80)
```python
def load_mnist():
    # Loads MNIST from scikit-learn
    # Returns: X_train (60000, 784), y_train (60000,)
```

#### Part 2: Preprocessing (Lines 81-120)
```python
# Normalize pixels
X = X / 255.0  # [0, 255] → [0, 1]

# One-hot encode labels
y_onehot = to_categorical(y)  # 7 → [0,0,0,0,0,0,0,1,0,0]
```

#### Part 3: Neural Network (Lines 121-300)
```python
class DigitClassifier:
    def __init__(self):
        # Initialize 3 layers
        self.W1 = ...  # (784, 128)
        self.W2 = ...  # (128, 64)
        self.W3 = ...  # (64, 10)

    def softmax(self, z):
        # Softmax activation
        exp_z = np.exp(z - np.max(z))  # Numerical stability!
        return exp_z / exp_z.sum()

    def forward(self, X):
        # Layer 1
        z1 = X @ W1 + b1
        a1 = relu(z1)

        # Layer 2
        z2 = a1 @ W2 + b2
        a2 = relu(z2)

        # Layer 3 (output)
        z3 = a2 @ W3 + b3
        y_pred = softmax(z3)

        return y_pred
```

#### Part 4: Training (Lines 301-400)
```python
# Same as Project 1, but:
# - Deeper network (3 layers)
# - Softmax output
# - Categorical cross-entropy
```

#### Part 5: Evaluation (Lines 401-500)
```python
# Confusion matrix is 10x10 (not 2x2!)
# Shows which digits are confused with which
```

---

### Step 6: Run Experiments (45 minutes)

**Try these modifications:**

#### Experiment 1: Change Network Size
```python
# Original
hidden1_size = 128
hidden2_size = 64

# Try smaller
hidden1_size = 64
hidden2_size = 32

# Try larger
hidden1_size = 256
hidden2_size = 128
```

**Question:** Does larger always mean better?

#### Experiment 2: Change Depth
```python
# Try 2-layer network (remove middle layer)
# Try 4-layer network (add another layer)
```

**Question:** How does depth affect accuracy?

#### Experiment 3: Learning Rate
```python
# Original
learning_rate = 0.001

# Try faster
learning_rate = 0.01

# Try slower
learning_rate = 0.0001
```

**Question:** What's the optimal learning rate?

#### Experiment 4: Training Duration
```python
# Original
epochs = 20

# Try shorter
epochs = 10

# Try longer
epochs = 40
```

**Question:** Does it overfit with more epochs?

---

### Step 7: Analyze Mistakes (30 minutes)

**The model will be ~96% accurate. Where does the 4% fail?**

```bash
# Run to see mistakes
python project_simple.py

# Check results/mistakes.png
```

**Common mistakes:**

```
True: 4, Predicted: 9
  ██       ███       ← Top looks similar!
  ██         █
  ███      ███
    █      █ █
    █      ███

True: 7, Predicted: 1
  ███       █        ← Handwriting variation
    █       █
    █       █
    █       █
    █       █
```

**Why mistakes happen:**
- Handwriting variations
- Similar-looking digits (4 vs 9, 3 vs 8, 5 vs 6)
- Poorly written digits
- Network capacity limits

---

## 🎯 Checkpoints

### ✅ Checkpoint 1: Understanding
Can you answer these?

1. Why do we flatten 28x28 images to 784 pixels?
2. What does softmax do?
3. Why one-hot encode labels?
4. How is this different from binary classification?
5. Why 3 layers instead of 2?

**If yes:** Continue!
**If no:** Re-read README.md and CONCEPTS.md

### ✅ Checkpoint 2: Running Code
Have you:

1. ✅ Run `project_simple.py` successfully?
2. ✅ Achieved >94% test accuracy?
3. ✅ Seen training curves?
4. ✅ Viewed confusion matrix?
5. ✅ Understood the network architecture?

**If yes:** Continue!
**If no:** Debug errors, review output

### ✅ Checkpoint 3: Experimentation
Have you tried:

1. ✅ Changing network size?
2. ✅ Different learning rates?
3. ✅ Analyzing mistakes?
4. ✅ Understanding why it fails?

**If yes:** You're done! 🎉
**If no:** Go back to Step 6

---

## 🚧 Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'sklearn'"
**Solution:**
```bash
pip install scikit-learn
```

### Issue 2: "Memory Error"
**Cause:** Network too large or batch size too big
**Solution:**
```python
# Reduce batch size
batch_size = 32  # instead of 128

# Or reduce network size
hidden_size = 64  # instead of 256
```

### Issue 3: "Accuracy stuck at 10%"
**Cause:** Model is just guessing (10 classes = 10% random)
**Solutions:**
- Check learning rate (try 0.001)
- Check data normalization (must divide by 255!)
- Check loss function (use categorical cross-entropy)
- Train longer (20+ epochs)

### Issue 4: "Loss is NaN"
**Cause:** Numerical instability in softmax
**Solution:**
```python
def softmax(z):
    # Subtract max for numerical stability
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
```

### Issue 5: "Training very slow"
**Cause:** Large dataset (60,000 images)
**Solutions:**
- Use smaller subset for testing (10,000 images)
- Increase batch size (128 instead of 32)
- Use fewer epochs (10 instead of 20)

---

## 📊 Expected Results

### Beginner Level
- ✅ Run project successfully
- ✅ Achieve 90%+ accuracy
- ✅ Understand input/output shapes
- ✅ View visualizations

**Time:** 2-3 hours

### Intermediate Level
- ✅ Understand all code
- ✅ Achieve 95%+ accuracy
- ✅ Run experiments
- ✅ Analyze mistakes

**Time:** 3-4 hours

### Advanced Level
- ✅ Modify architecture
- ✅ Tune hyperparameters
- ✅ Achieve 97%+ accuracy
- ✅ Understand failure modes
- ✅ Compare to academic papers

**Time:** 4-5 hours

---

## ⏭️ Next Steps

### After This Project

**Option 1: Move to Project 3 (Recommended)**
→ `projects/neural_networks/sentiment_analysis/`
- More advanced NLP
- Word embeddings
- Bridge to transformers

**Option 2: Improve This Project**
- Try data augmentation (rotate, shift images)
- Implement dropout (regularization)
- Try different architectures
- Compare with CNNs (future topic!)

**Option 3: Build Custom Application**
- Collect your own handwritten digits
- Build digit recognition app
- Deploy as web service

---

## 🎓 Learning Reflection

After completing, ask yourself:

1. **What surprised you?**
   - How well does the network learn?
   - What mistakes does it make?

2. **How does this compare to Project 1?**
   - Easier or harder?
   - Different challenges?

3. **What would you do differently?**
   - Different architecture?
   - More/fewer layers?

4. **Real-world applications?**
   - Where would you use this?
   - What modifications needed?

---

## ✅ Final Checklist

Before moving to Project 3:

- ✅ Successfully ran both simple and main versions
- ✅ Achieved 95%+ test accuracy
- ✅ Understood image → pixels → features pipeline
- ✅ Understood softmax and multi-class classification
- ✅ Understood confusion matrix (10x10)
- ✅ Ran at least 2 experiments
- ✅ Read all documentation
- ✅ Can explain how it works

---

**Congratulations! You built a real image classifier!** 🎉

**You now understand:**
- ✅ How computers "see" images
- ✅ Deep neural networks (3+ layers)
- ✅ Multi-class classification
- ✅ Softmax activation
- ✅ Real-world benchmark performance

👉 **Next:** Move to Project 3 (Sentiment Analysis) or dive deeper!

🚀 **Two projects down, one to go before Module 4!**
