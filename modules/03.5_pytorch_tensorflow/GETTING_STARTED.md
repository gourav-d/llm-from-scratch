# Getting Started with Module 3.5: PyTorch & TensorFlow

**Welcome to the bridge between theory and production!**

---

## Quick Start

### Prerequisites Check

Before starting, ensure you've completed:

- [x] Module 1: Python Basics
- [x] Module 2: NumPy & Math
- [x] Module 3: Neural Networks from Scratch (including Lesson 7: AutoGrad)

**Why?** You need to understand what these frameworks automate!

---

## Installation

### Step 1: Install PyTorch

```bash
# For CPU-only (good for learning)
pip install torch torchvision torchaudio

# For GPU (if you have NVIDIA GPU with CUDA)
# Check your CUDA version first: nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install TensorFlow

```bash
# TensorFlow 2.x (works for both CPU and GPU)
pip install tensorflow

# Optional: TensorFlow datasets
pip install tensorflow-datasets
```

### Step 3: Verify Installation

```python
# test_installation.py
import torch
import tensorflow as tf
import numpy as np

print("=" * 50)
print("INSTALLATION CHECK")
print("=" * 50)

# PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# TensorFlow
print(f"\nTensorFlow version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU devices: {len(gpu_devices)}")
if gpu_devices:
    print(f"GPU: {gpu_devices[0].name}")

# NumPy
print(f"\nNumPy version: {np.__version__}")

print("\n" + "=" * 50)
print("✅ All frameworks installed successfully!")
print("=" * 50)
```

Run it:
```bash
python test_installation.py
```

---

## Choose Your Learning Path

### Path 1: PyTorch Focus (For LLM Development)
**Best for:** You want to build LLMs, work in research, or need maximum flexibility

**Timeline:** 2-3 weeks

```
Week 1: PyTorch Deep Dive
├── Day 1-2: Lesson 1 (PyTorch Fundamentals)
├── Day 3-5: Lesson 2 (Neural Networks)
└── Day 6-7: Example projects

Week 2: Practical Application
├── Day 1-3: Lesson 3 (NumPy → PyTorch)
├── Day 4-5: Project 1 (MNIST comparison)
└── Day 6-7: Project 2 (Convert your models)

Week 3: TensorFlow Awareness
├── Day 1-3: Lesson 4 (TensorFlow basics)
└── Day 4-5: Lesson 5 (Framework comparison)
```

**After this path:**
- Expert in PyTorch
- Can read/write PyTorch code fluently
- Aware of TensorFlow
- Ready for transformer implementations

---

### Path 2: Balanced (For Maximum Versatility)
**Best for:** You want job flexibility or plan to work in production environments

**Timeline:** 3-4 weeks

```
Week 1: PyTorch Foundation
└── Lessons 1-2

Week 2: PyTorch Mastery
└── Lesson 3 + Projects

Week 3: TensorFlow Foundation
└── Lesson 4 + Examples

Week 4: Integration
└── Lesson 5 + Final comparison project
```

**After this path:**
- Proficient in both frameworks
- Can choose the right tool for the job
- Competitive for any ML job
- Ready for production deployment

---

### Path 3: Fast Track (Minimum Viable Knowledge)
**Best for:** You're in a hurry or just need awareness

**Timeline:** 1 week intensive

```
Days 1-3: PyTorch essentials (Lessons 1-2)
Days 4-5: NumPy to PyTorch (Lesson 3)
Days 6-7: TensorFlow basics (Lesson 4) + Comparison (Lesson 5)
```

**After this path:**
- Functional knowledge of both
- Can read framework code
- Needs more practice for production
- Ready to move forward (can deepen later)

---

## Daily Study Plan (Path 1 - PyTorch Focus)

### Week 1

**Day 1: PyTorch Tensors**
- [ ] Read Lesson 1 (first half)
- [ ] Run `example_01_pytorch_tensors.py`
- [ ] Practice: Create various tensor types
- [ ] Exercise: Convert NumPy arrays to tensors

**Day 2: Autograd in Practice**
- [ ] Read Lesson 1 (second half)
- [ ] Understand computational graphs
- [ ] Compare to your Module 3 Lesson 7 autograd
- [ ] Exercise: Simple gradient calculations

**Day 3: Building Blocks**
- [ ] Read Lesson 2 (first half)
- [ ] Learn nn.Module
- [ ] Understand nn.Linear
- [ ] Exercise: Build single-layer network

**Day 4: Complete Networks**
- [ ] Read Lesson 2 (second half)
- [ ] Build multi-layer networks
- [ ] Use optimizers
- [ ] Run `example_02_mnist_pytorch.py`

**Day 5: Training Patterns**
- [ ] Study training loop structure
- [ ] Understand loss functions
- [ ] Practice with different optimizers
- [ ] Exercise: Train your own model

**Day 6-7: Practice Projects**
- [ ] Implement 3+ small networks
- [ ] Experiment with hyperparameters
- [ ] Debug common issues
- [ ] Prepare for week 2

### Week 2

**Day 1-2: Code Conversion**
- [ ] Read Lesson 3
- [ ] Convert Module 3 perceptron to PyTorch
- [ ] Convert Module 3 MLP to PyTorch
- [ ] Run `example_05_numpy_vs_pytorch.py`

**Day 3: Performance Comparison**
- [ ] Benchmark NumPy vs PyTorch
- [ ] Understand when each is faster
- [ ] Profile your code
- [ ] Exercise: Optimize a slow model

**Day 4-5: Project 1 - MNIST Three Ways**
- [ ] Implement MNIST in NumPy (already done!)
- [ ] Implement MNIST in PyTorch
- [ ] Compare code complexity
- [ ] Compare training speed
- [ ] Document findings

**Day 6-7: Project 2 - Convert Custom Network**
- [ ] Choose a Module 3 project
- [ ] Convert to PyTorch
- [ ] Add improvements (e.g., GPU support)
- [ ] Write comparison document

### Week 3 (Optional but Recommended)

**Day 1-2: TensorFlow Basics**
- [ ] Read Lesson 4 (first half)
- [ ] Install TensorFlow
- [ ] Run `example_04_tensorflow_mnist.py`
- [ ] Compare to PyTorch code

**Day 3: Keras APIs**
- [ ] Sequential API
- [ ] Functional API
- [ ] When to use each
- [ ] Exercise: Build model both ways

**Day 4-5: Framework Comparison**
- [ ] Read Lesson 5
- [ ] Create decision matrix
- [ ] Understand trade-offs
- [ ] Exercise: Choose framework for scenarios

**Day 6-7: Final Integration**
- [ ] Review all lessons
- [ ] Complete any missing exercises
- [ ] Build one project in each framework
- [ ] Prepare for Module 4

---

## Learning Resources

### Official Documentation
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **TensorFlow Docs**: https://www.tensorflow.org/api_docs
- **Keras Guide**: https://keras.io/guides/

### Video Tutorials
- **PyTorch Official Tutorials**: YouTube channel
- **sentdex PyTorch**: Practical tutorials
- **TensorFlow Official**: Getting started series
- **3Blue1Brown**: Neural network visualization

### Practice Platforms
- **PyTorch Examples**: https://github.com/pytorch/examples
- **Kaggle**: Real datasets and competitions
- **Google Colab**: Free GPU access
- **Papers with Code**: Implementation examples

### Communities
- **PyTorch Forums**: discuss.pytorch.org
- **r/MachineLearning**: Reddit community
- **Stack Overflow**: Tag [pytorch] or [tensorflow]
- **Discord/Slack**: ML communities

---

## Tips for Success

### 1. Code Everything Yourself
Don't just read the examples - type them out! Muscle memory matters.

### 2. Compare to Module 3
For each PyTorch concept, ask: "How did I do this in NumPy?"

Example:
```python
# NumPy (Module 3)
W = np.random.randn(10, 5)
b = np.zeros(5)
z = x @ W + b

# PyTorch (Module 3.5)
layer = nn.Linear(10, 5)  # Creates W and b automatically
z = layer(x)
```

### 3. Use Jupyter Notebooks
Interactive exploration helps learning:
```bash
jupyter notebook
```

### 4. Start Simple, Then Scale
```
Day 1: Single neuron
Day 2: Single layer
Day 3: Two layers
Day 4: Full network
Day 5: Custom architecture
```

### 5. Debug Systematically
```python
# Always check shapes!
print(f"Input shape: {x.shape}")
print(f"Weight shape: {model.layer1.weight.shape}")
print(f"Output shape: {output.shape}")

# Always check gradients!
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean = {param.grad.mean()}")
```

### 6. Use GPU (If Available)
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)

# Everything runs on GPU now!
```

### 7. Save Your Work
```python
# PyTorch
torch.save(model.state_dict(), 'model.pth')

# TensorFlow
model.save('model.h5')
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Forgetting to Zero Gradients
```python
# WRONG
for epoch in range(10):
    output = model(x)
    loss = criterion(output, y)
    loss.backward()  # Gradients accumulate!
    optimizer.step()

# CORRECT
for epoch in range(10):
    optimizer.zero_grad()  # Clear old gradients!
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
```

### Pitfall 2: Wrong Tensor Shapes
```python
# Always check shapes
print(f"Expected: {expected_shape}")
print(f"Got: {actual_tensor.shape}")

# Use .view() or .reshape() to fix
x = x.view(batch_size, -1)  # Flatten
```

### Pitfall 3: CPU/GPU Mismatch
```python
# WRONG
model = model.to('cuda')
x = torch.tensor([1, 2, 3])  # Still on CPU!
output = model(x)  # ERROR!

# CORRECT
model = model.to('cuda')
x = torch.tensor([1, 2, 3]).to('cuda')  # Move to GPU
output = model(x)  # Works!
```

### Pitfall 4: Training/Eval Mode
```python
# WRONG
model.eval()  # Set to eval mode
# ... but then training!
loss.backward()  # BatchNorm, Dropout won't work correctly

# CORRECT
model.train()  # Set to training mode
# ... training code
model.eval()  # Set to eval mode for inference
```

---

## Project Ideas for Practice

### Beginner
1. Digit classifier (MNIST)
2. Fashion item classifier (Fashion-MNIST)
3. Binary classifier (simple dataset)

### Intermediate
4. Multi-class image classifier (CIFAR-10)
5. Sentiment analysis (text classification)
6. Custom autograd function

### Advanced
7. Image segmentation
8. Transfer learning (pre-trained models)
9. Custom layer implementation
10. Distributed training

---

## Troubleshooting

### Installation Issues

**Problem**: PyTorch install fails
```bash
# Solution: Use pip instead of conda
pip install torch torchvision torchaudio

# Or specify version explicitly
pip install torch==2.0.0
```

**Problem**: TensorFlow GPU not detected
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA-compatible TensorFlow
pip install tensorflow[and-cuda]
```

### Runtime Issues

**Problem**: Out of memory (GPU)
```python
# Solution: Reduce batch size
batch_size = 16  # Instead of 128

# Or use gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Problem**: Slow training
```python
# Solution: Use DataLoader with multiple workers
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster CPU→GPU transfer
)
```

---

## Assessment

### Self-Check After Week 1
Can you:
- [ ] Create tensors in PyTorch?
- [ ] Build a simple neural network?
- [ ] Train a model on MNIST?
- [ ] Explain requires_grad?
- [ ] Use optimizer.step()?

If yes → Continue to Week 2
If no → Review Lessons 1-2

### Self-Check After Week 2
Can you:
- [ ] Convert NumPy code to PyTorch?
- [ ] Explain performance differences?
- [ ] Use GPU acceleration?
- [ ] Save and load models?
- [ ] Debug training issues?

If yes → Continue to Week 3 (or Module 4)
If no → More practice needed

### Self-Check After Week 3
Can you:
- [ ] Build models in both PyTorch and TensorFlow?
- [ ] Choose appropriate framework for a project?
- [ ] Explain trade-offs between frameworks?
- [ ] Deploy a simple model?

If yes → Ready for Module 4 (Transformers)!

---

## What's Next

### After Module 3.5

You're ready for:

**Module 4: Transformers & Attention**
- Build transformers in PyTorch
- Understand self-attention
- Implement multi-head attention

**Module 5: Building Your LLM**
- GPT from scratch (PyTorch)
- nanoGPT implementation
- Train on Shakespeare

**Production Skills**
- Model serving
- Optimization
- Deployment

---

## Get Help

### When You're Stuck

1. **Check documentation** - Most answers are there
2. **Print shapes** - 90% of bugs are shape mismatches
3. **Simplify** - Test with smaller examples first
4. **Search** - Someone has had your error before
5. **Ask** - Forums, Stack Overflow, Discord

### Good Questions Format

```
Title: [PyTorch] Shape mismatch in Linear layer

I'm trying to build a network but getting:
RuntimeError: size mismatch, m1: [32 x 784], m2: [128 x 64]

Code:
[paste minimal reproducible example]

Expected: [32 x 64]
Got: [error]

What I tried:
- Checked input shape: [32, 784]
- Layer definition: nn.Linear(128, 64)

Question: Why is m2 showing [128 x 64] instead of [784 x 64]?
```

---

## Final Checklist

Before starting Module 4, ensure you can:

### PyTorch
- [x] Create and manipulate tensors
- [x] Build custom nn.Module classes
- [x] Implement forward pass
- [x] Use automatic differentiation
- [x] Train with optimizers
- [x] Save and load models
- [x] Use GPU acceleration

### TensorFlow (Basic Awareness)
- [x] Build Sequential models
- [x] Understand Keras API
- [x] Run basic training
- [x] Export models

### Practical Skills
- [x] Convert NumPy code to PyTorch
- [x] Debug common errors
- [x] Choose appropriate framework
- [x] Benchmark performance

---

**Ready to start? Open Lesson 1: PyTorch Fundamentals!**

**Remember: You've built neural networks from scratch. Now you're learning to build them faster!**

**Let's go!** 🚀
