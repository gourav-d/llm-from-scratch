# Module 3.5: Deep Learning Frameworks - PyTorch & TensorFlow

**From NumPy to Production: Master Modern Deep Learning Frameworks**

---

## Module Overview

Now that you've built neural networks from scratch (Module 3) and understand automatic differentiation (Lesson 3.7), it's time to learn the industry-standard frameworks: **PyTorch** and **TensorFlow**.

### What You'll Learn

- PyTorch fundamentals and tensor operations
- Building neural networks with PyTorch
- TensorFlow/Keras basics
- Framework comparison and when to use each
- Converting NumPy implementations to PyTorch/TensorFlow
- GPU acceleration basics
- Model saving and loading
- Production deployment considerations

**Time:** 3-4 weeks

---

## Why This Module?

### The Transition

```
Module 3 (NumPy):           Module 3.5:              Production:
Build from scratch     →    Use frameworks      →    Deploy at scale
Understand deeply           Code efficiently         Serve billions

You've learned WHY.         Now learn HOW.          Then ship it!
```

### What You've Built vs What You'll Use

**Module 3 - You built:**
```python
# Forward pass (manual)
z1 = x @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
output = softmax(z2)

# Backward pass (manual)
dW2 = ...  # 20 lines of gradient calculations
dW1 = ...
```

**Module 3.5 - You'll use:**
```python
# PyTorch (automatic!)
class Network(nn.Module):
    def __init__(self):
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.softmax(self.layer2(x))
        return x

# Gradients computed automatically!
loss.backward()
```

---

## Module Structure

```
modules/03.5_pytorch_tensorflow/
├── README.md                          ← You are here
├── GETTING_STARTED.md                 ← Start here next
│
├── 01_pytorch_fundamentals.md         ← Lesson 1: PyTorch basics
├── 02_pytorch_neural_networks.md      ← Lesson 2: Build models in PyTorch
├── 03_numpy_to_pytorch.md             ← Lesson 3: Convert your code
├── 04_tensorflow_basics.md            ← Lesson 4: TensorFlow/Keras intro
├── 05_framework_comparison.md         ← Lesson 5: When to use what
│
├── examples/
│   ├── example_01_pytorch_tensors.py          ← Tensor operations
│   ├── example_02_mnist_pytorch.py            ← MNIST in PyTorch
│   ├── example_03_custom_autograd.py          ← Extend autograd
│   ├── example_04_tensorflow_mnist.py         ← MNIST in TensorFlow
│   ├── example_05_numpy_vs_pytorch.py         ← Side-by-side comparison
│   └── example_06_gpu_acceleration.py         ← Using GPUs
│
├── exercises/
│   ├── exercise_01_convert_perceptron.py      ← NumPy → PyTorch
│   ├── exercise_02_build_mlp.py               ← Build MLP in PyTorch
│   ├── exercise_03_custom_layer.py            ← Custom layers
│   └── exercise_04_framework_choice.py        ← Choose right framework
│
└── projects/
    ├── project_01_mnist_comparison.py         ← Same model, 3 ways
    └── project_02_production_model.py         ← Deploy-ready model
```

---

## Learning Objectives

By the end of this module, you will:

### Conceptual Understanding
- [ ] Understand the relationship between NumPy and PyTorch/TensorFlow
- [ ] Know when to use PyTorch vs TensorFlow
- [ ] Understand automatic differentiation in frameworks
- [ ] Know how to leverage GPUs for acceleration

### PyTorch Skills
- [ ] Create and manipulate tensors
- [ ] Build neural networks with nn.Module
- [ ] Use built-in loss functions and optimizers
- [ ] Convert NumPy code to PyTorch
- [ ] Train models with GPU acceleration
- [ ] Save and load trained models

### TensorFlow/Keras Skills
- [ ] Understand TensorFlow's execution model
- [ ] Build models with Keras Sequential and Functional APIs
- [ ] Use tf.data for efficient data loading
- [ ] Deploy models with TensorFlow Serving

### Production Skills
- [ ] Choose the right framework for your project
- [ ] Optimize training performance
- [ ] Export models for deployment
- [ ] Debug common framework issues

---

## Prerequisites

Before starting this module, you must complete:

✅ **Module 1**: Python Basics (all lessons)
✅ **Module 2**: NumPy & Math (matrix operations)
✅ **Module 3**: Neural Networks from Scratch (all 7 lessons, including AutoGrad)

**Why these prerequisites?**
- You need to understand what the frameworks are automating
- Building from scratch teaches you the fundamentals
- Frameworks make more sense when you know what's under the hood

---

## Lesson Overview

### Lesson 1: PyTorch Fundamentals
**File:** `01_pytorch_fundamentals.md`
**Time:** 4-6 hours

**What you'll learn:**
- Installing PyTorch
- Tensor creation and operations
- Automatic differentiation with autograd
- Moving between CPU and GPU
- PyTorch vs NumPy comparison

**Key Concepts:**
- torch.Tensor vs np.ndarray
- requires_grad flag
- Computational graphs
- .backward() and gradients
- Device management (CPU/GPU)

---

### Lesson 2: Building Neural Networks in PyTorch
**File:** `02_pytorch_neural_networks.md`
**Time:** 5-7 hours

**What you'll learn:**
- nn.Module: The base class for all models
- nn.Linear, nn.Conv2d, etc.
- Activation functions (nn.ReLU, nn.Sigmoid)
- Loss functions (nn.CrossEntropyLoss)
- Optimizers (optim.SGD, optim.Adam)
- Training loop pattern

**Key Concepts:**
- Subclassing nn.Module
- forward() method
- Model parameters
- optimizer.zero_grad()
- Training vs evaluation mode

---

### Lesson 3: Converting NumPy to PyTorch
**File:** `03_numpy_to_pytorch.md`
**Time:** 3-4 hours

**What you'll learn:**
- Side-by-side NumPy and PyTorch code
- Converting your Module 3 projects
- Performance comparison
- When NumPy is still better

**Projects:**
- Convert perceptron to PyTorch
- Convert MNIST classifier to PyTorch
- Benchmark performance improvements

---

### Lesson 4: TensorFlow & Keras Basics
**File:** `04_tensorflow_basics.md`
**Time:** 4-6 hours

**What you'll learn:**
- TensorFlow 2.x fundamentals
- Keras Sequential API
- Keras Functional API
- tf.data pipelines
- Callbacks and monitoring

**Key Concepts:**
- Eager execution vs graph mode
- Model.fit() vs custom training loops
- TensorBoard for visualization
- Model checkpointing

---

### Lesson 5: Framework Comparison
**File:** `05_framework_comparison.md`
**Time:** 2-3 hours

**What you'll learn:**
- PyTorch vs TensorFlow pros/cons
- Research vs production considerations
- Ecosystem comparison
- When to use each framework
- Hybrid approaches

**Decision Matrix:**
- Research → PyTorch
- Production → TensorFlow (historically)
- Mobile → TensorFlow Lite
- Web → TensorFlow.js or ONNX
- Flexibility → PyTorch
- Deployment → Both now equal

---

## Key Differences: NumPy vs PyTorch vs TensorFlow

### Creating Arrays/Tensors

```python
# NumPy
import numpy as np
x = np.array([[1, 2], [3, 4]], dtype=np.float32)

# PyTorch
import torch
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# TensorFlow
import tensorflow as tf
x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
```

### Matrix Operations

```python
# NumPy
result = np.matmul(a, b)  # or a @ b

# PyTorch
result = torch.matmul(a, b)  # or a @ b

# TensorFlow
result = tf.matmul(a, b)  # or a @ b

# Almost identical syntax!
```

### Automatic Differentiation

```python
# NumPy (MANUAL - what you did in Module 3)
def forward(x, w):
    return x @ w

def backward(x, w, grad_output):
    grad_w = x.T @ grad_output  # Manual!
    grad_x = grad_output @ w.T  # Manual!
    return grad_x, grad_w

# PyTorch (AUTOMATIC)
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
w = torch.tensor([[3.0], [4.0]], requires_grad=True)
y = x @ w
y.backward()
print(w.grad)  # Gradients computed automatically!

# TensorFlow (AUTOMATIC)
x = tf.Variable([[1.0, 2.0]])
w = tf.Variable([[3.0], [4.0]])
with tf.GradientTape() as tape:
    y = x @ w
grad_w = tape.gradient(y, w)  # Automatic!
```

---

## Building a Neural Network: 3 Ways

### NumPy (Module 3 - What You Built)

```python
class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(784, 128) * 0.01
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, 10) * 0.01
        self.b2 = np.zeros(10)

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return softmax(self.z2)

    def backward(self, x, y):
        # 30+ lines of manual gradient calculations
        # ...
```

### PyTorch (What You'll Learn)

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Training
model = NeuralNetwork()
optimizer = optim.Adam(model.parameters())

for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch.x)
    loss = F.cross_entropy(output, batch.y)
    loss.backward()  # Automatic gradients!
    optimizer.step()
```

### TensorFlow/Keras (What You'll Learn)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# One-line training!
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

**Same model, three approaches:**
- NumPy: Understand deeply
- PyTorch: Research and flexibility
- TensorFlow: Production and deployment

---

## Why Both PyTorch AND TensorFlow?

### Industry Reality

```
Job Postings Analysis (2026):

PyTorch: 45% of ML jobs
TensorFlow: 40% of ML jobs
Both: 15% require both!

You need BOTH to be competitive!
```

### Best Uses

**PyTorch:**
- ✅ Research and experimentation
- ✅ Custom architectures
- ✅ NLP (transformers, LLMs)
- ✅ Academic papers
- ✅ Debugging (more Pythonic)

**TensorFlow:**
- ✅ Production deployment
- ✅ Mobile (TensorFlow Lite)
- ✅ Web (TensorFlow.js)
- ✅ Enterprise (Google ecosystem)
- ✅ Mature deployment tools

**Reality in 2026:**
- Both have converged in features
- Both can do research AND production
- Learn both, specialize in one

---

## GPU Acceleration

### The Power of GPUs

```
Training MNIST on CPU:
NumPy: 5 minutes
PyTorch CPU: 4 minutes

Training MNIST on GPU:
PyTorch GPU: 30 seconds  ← 10x faster!
TensorFlow GPU: 25 seconds

For GPT-scale models:
CPU: Months (impossible!)
GPU: Weeks
Multi-GPU: Days
```

### Using GPUs in PyTorch

```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model and data to GPU
model = model.to(device)
x = x.to(device)

# Everything else is the same!
output = model(x)  # Runs on GPU automatically
```

### Using GPUs in TensorFlow

```python
# TensorFlow automatically uses GPU if available
print(tf.config.list_physical_devices('GPU'))

# Manual placement (usually not needed)
with tf.device('/GPU:0'):
    model = build_model()
    model.fit(x_train, y_train)
```

**You'll learn GPU programming without learning CUDA!**

---

## Projects

### Project 1: MNIST Three Ways

Build the MNIST classifier using:
1. NumPy (your Module 3 code)
2. PyTorch
3. TensorFlow/Keras

**Compare:**
- Lines of code
- Training time
- Ease of debugging
- Final accuracy

**Goal:** See the trade-offs firsthand

---

### Project 2: Convert Your Custom Network

Take your Module 3 neural network project and convert it to:
1. PyTorch implementation
2. TensorFlow implementation

**Deliverables:**
- Working PyTorch model
- Working TensorFlow model
- Performance benchmark
- Decision document: which framework for which use case

---

## Connection to Future Modules

### How This Enables Future Learning

```
Module 3.5 → Module 4 (Transformers)
PyTorch skills needed for transformer implementations

Module 3.5 → Module 5 (Building LLM)
All LLMs are built with PyTorch or TensorFlow

Module 3.5 → Module 6 (Training/Fine-tuning)
Framework knowledge essential for training

Module 3.5 → Module 7+ (Production)
Deployment requires framework expertise
```

**This module is the bridge from learning to doing!**

---

## Real-World Applications

After this module, you can:

### Research
- Implement papers from arXiv
- Experiment with novel architectures
- Reproduce state-of-the-art results

### Production
- Deploy models to cloud (AWS, GCP, Azure)
- Serve predictions at scale
- Optimize inference performance

### Existing Ecosystems
- **Hugging Face**: Transformers library (PyTorch/TF)
- **OpenAI**: GPT models (originally PyTorch)
- **Google**: BERT, T5 (TensorFlow)
- **Facebook**: LLaMA, OPT (PyTorch)

**You'll be able to use ALL of these!**

---

## Expected Time Investment

| Component | Time |
|-----------|------|
| **5 Lessons** | 18-26 hours |
| **6 Examples** | 6-8 hours |
| **4 Exercises** | 4-6 hours |
| **2 Projects** | 8-12 hours |
| **Total** | 36-52 hours |

**Pace Options:**
- **Intensive**: 1 week (full-time learning)
- **Moderate**: 2-3 weeks (4-5 hours/day)
- **Casual**: 4-6 weeks (2 hours/day)

---

## Success Criteria

You've mastered Module 3.5 when you can:

### PyTorch
- [ ] Create and manipulate tensors
- [ ] Build custom nn.Module classes
- [ ] Train models with automatic differentiation
- [ ] Use GPU acceleration
- [ ] Debug training issues
- [ ] Save and load models

### TensorFlow
- [ ] Build models with Sequential and Functional APIs
- [ ] Use tf.data for efficient data loading
- [ ] Implement custom training loops
- [ ] Use TensorBoard for monitoring
- [ ] Export models for serving

### Framework Choice
- [ ] Choose appropriate framework for projects
- [ ] Justify framework selection
- [ ] Convert between frameworks
- [ ] Understand trade-offs

---

## Installation

### PyTorch

```bash
# CPU-only (for learning)
pip install torch torchvision torchaudio

# GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### TensorFlow

```bash
# CPU and GPU (TensorFlow 2.x auto-detects)
pip install tensorflow

# Verify installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Verify Installation

```python
# PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# TensorFlow
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
```

---

## Resources

### Documentation
- **PyTorch**: https://pytorch.org/docs/
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **TensorFlow**: https://www.tensorflow.org/
- **Keras**: https://keras.io/

### Courses
- **PyTorch for Deep Learning** (Zero to Mastery)
- **TensorFlow in Practice** (deeplearning.ai)
- **Fast.ai** (PyTorch-based)

### Books
- "Deep Learning with PyTorch" (Stevens et al.)
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Géron)

---

## Learning Paths

### Path A: PyTorch Focus (Recommended for LLMs)
**Time:** 2-3 weeks

**Week 1:**
- Lesson 1: PyTorch Fundamentals
- Lesson 2: Neural Networks in PyTorch
- Example projects

**Week 2:**
- Lesson 3: NumPy to PyTorch conversion
- Project 1: MNIST comparison
- Project 2: Convert custom network

**Week 3 (Optional):**
- Lesson 4: TensorFlow basics (awareness)
- Lesson 5: Framework comparison

**Outcome:** PyTorch expert, TensorFlow aware

---

### Path B: Balanced (Recommended for Versatility)
**Time:** 3-4 weeks

**Week 1:** PyTorch Lessons 1-2
**Week 2:** PyTorch Lesson 3 + Projects
**Week 3:** TensorFlow Lesson 4
**Week 4:** Comparison + Final projects

**Outcome:** Proficient in both frameworks

---

### Path C: Fast Track (Minimum)
**Time:** 1 week (intensive)

**Days 1-3:** PyTorch Fundamentals + Networks
**Days 4-5:** Convert NumPy to PyTorch
**Days 6-7:** TensorFlow basics + Comparison

**Outcome:** Functional knowledge of both

---

## Next Steps

### After Module 3.5

You'll be ready for:

**Module 4: Transformers & Attention**
- Build transformers in PyTorch
- Understand BERT, GPT architectures
- Use Hugging Face Transformers library

**Module 5: Building Your LLM**
- Implement GPT from scratch (PyTorch)
- Train on real text data
- Generate coherent text

**Module 6+: Advanced Topics**
- Fine-tuning with PyTorch
- Production deployment with TensorFlow Serving
- Distributed training
- Model optimization

---

## The Big Picture

### Your Journey

```
Module 1-2: Python & Math Foundations
    ↓
Module 3: Build Neural Networks (NumPy)
    ↓
Module 3 Lesson 7: Understand AutoGrad
    ↓
Module 3.5: Use Frameworks (PyTorch/TF)  ← YOU ARE HERE
    ↓
Module 4+: Build Real AI Systems
```

**You've learned to build from scratch.**
**Now learn to build at scale!**

---

## Start Learning!

### Recommended Order

1. **Read** `GETTING_STARTED.md` - Choose your path
2. **Complete** Lesson 1: PyTorch Fundamentals
3. **Practice** with examples and exercises
4. **Build** the projects
5. **Move forward** to transformers!

---

## Remember

**Why you built from scratch first:**
- You understand what PyTorch automates
- You can debug when things go wrong
- You appreciate the power of frameworks
- You make informed architectural decisions

**Now with frameworks:**
- 10x faster development
- GPU acceleration
- Production-ready code
- Join the AI community

**Best of both worlds: Deep understanding + Modern tools!**

---

**Ready to level up from NumPy to PyTorch/TensorFlow?**

**Let's build modern AI systems!**

**Next:** Open `GETTING_STARTED.md` to begin!
