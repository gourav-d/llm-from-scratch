# Getting Started with Module 2: NumPy Fundamentals

Welcome! This guide will help you navigate the expanded Module 2 content.

---

## 📂 What's Inside This Module

```
02_numpy_math/
├── README.md                           ← Start here! Overview and motivation
├── GETTING_STARTED.md                  ← You are here
├── quick_reference.md                  ← Bookmark this for quick lookups
├── concepts.md                         ← Visual explanations with diagrams
├── python_guide_for_dotnet.md          ← NumPy for C#/.NET developers
│
├── 01_numpy_basics.md                  ← Lesson 1: Arrays, indexing, slicing
├── 02_array_operations.md              ← Lesson 2: Broadcasting, operations
├── 03_linear_algebra.md                ← Lesson 3: Matrix math for neural nets
│
├── examples/                           ← Working code to run and modify
│   ├── example_01_basics.py
│   ├── example_02_operations.py
│   ├── example_03_linear_algebra.py
│   └── example_04_neural_network_demo.py  ← Full neural network demo!
│
├── exercises/                          ← Practice problems with solutions
│   ├── exercise_01_basics.py
│   ├── exercise_02_operations.py
│   └── exercise_03_linear_algebra.py
│
└── quiz.md                             ← Test your knowledge (35 questions)
```

---

## 🚀 How to Use This Module

### Option 1: Structured Learning (Recommended for Beginners)

**Week 1 Plan:**
```
Day 1-2: Foundations
├── Read: README.md (understand WHY NumPy matters)
├── Read: concepts.md (visual mental models)
├── Read: 01_numpy_basics.md
└── Run: examples/example_01_basics.py

Day 3: Practice
├── Do: exercises/exercise_01_basics.py
└── Compare your solutions with provided answers

Day 4-5: Operations
├── Read: 02_array_operations.md
├── Run: examples/example_02_operations.py
└── Do: exercises/exercise_02_operations.py

Day 6: Linear Algebra
├── Read: 03_linear_algebra.md
├── Run: examples/example_03_linear_algebra.py
└── Do: exercises/exercise_03_linear_algebra.py

Day 7: Integration & Assessment
├── Run: examples/example_04_neural_network_demo.py
├── Take: quiz.md
└── Score 80%+ to move to Module 3
```

### Option 2: Quick Reference (For Experienced Developers)

If you have programming experience and want to move quickly:
1. Read **README.md** for context
2. Skim **quick_reference.md** for syntax
3. Run all **examples/** to see concepts in action
4. Take **quiz.md** to identify gaps
5. Review specific lessons as needed

### Option 3: .NET Developer Path

If you're coming from C#/.NET:
1. Read **README.md** for motivation
2. Read **python_guide_for_dotnet.md** (relates NumPy to C#)
3. Read **concepts.md** for visual models
4. Work through lessons 01-03
5. Run examples and compare to .NET equivalents

---

## 🎯 Learning Objectives

By the end of this module, you should be able to:

- [ ] Explain WHY NumPy is essential for building LLMs
- [ ] Create and manipulate NumPy arrays confidently
- [ ] Understand and apply broadcasting
- [ ] Perform matrix operations for neural networks
- [ ] Debug shape mismatches independently
- [ ] Read and write vectorized code (no loops!)
- [ ] Implement a simple neural network layer

---

## 💻 Setup and Installation

### 1. Ensure Your Environment is Ready

```bash
# Navigate to project root
cd "/c/Users/gourav.dwivedi/OneDrive - BLACKLINE/Documents/GD/Learning/LLM/2026/1"

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (Git Bash/WSL):
source venv/Scripts/activate

# Verify activation (you should see (venv) in prompt)
```

### 2. Install Required Packages

```bash
# Install NumPy and supporting libraries
pip install numpy matplotlib jupyter

# Verify installation
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"
```

Expected output: `NumPy version: 1.24.x` or newer

### 3. Choose Your Environment

**Option A: Jupyter Notebook (Recommended for Learning)**
```bash
# Start Jupyter
jupyter notebook

# This will open in your browser
# Navigate to: modules/02_numpy_math/examples/
# Open any .py file as a notebook
```

**Option B: Python Scripts**
```bash
# Run examples directly
python modules/02_numpy_math/examples/example_01_basics.py

# Run exercises
python modules/02_numpy_math/exercises/exercise_01_basics.py
```

**Option C: VS Code**
```bash
# Open folder in VS Code
code .

# Install Python extension if not already installed
# Open any .py file and run with Ctrl+Shift+P → "Run Python File"
```

---

## 📖 Reading Guide

### Understanding Code Examples

All code examples follow this format:

```python
# What this code does (comment)
code_line = some_operation()
print(result)  # Expected output
```

**Tips:**
- Type code yourself (don't copy-paste!)
- Run each cell/section one at a time
- Print shapes often: `print(array.shape)`
- Experiment: change values and see what happens

### Understanding Lessons

Each lesson has:
1. **Concept explanation** - What and why
2. **Code examples** - How to use it
3. **Visual diagrams** - Mental models
4. **C# comparisons** - For .NET developers
5. **Use cases** - Where you'll use this in LLMs

---

## 🔍 How to Practice Effectively

### 1. Active Learning
```python
# Don't just read:
arr = np.array([1, 2, 3])

# Do this instead:
arr = np.array([1, 2, 3])
print(f"Array: {arr}")
print(f"Shape: {arr.shape}")
print(f"Type: {type(arr)}")
print(f"Dtype: {arr.dtype}")

# Try variations:
arr2 = np.array([[1, 2], [3, 4]])
print(f"2D Shape: {arr2.shape}")
```

### 2. Debugging Practice

When you get an error:
```python
# Error: shapes (3,4) and (5,2) not aligned

# Debug steps:
1. Print shapes: print(A.shape, B.shape)
2. Check compatibility: (3,4) @ (5,2) → 4 ≠ 5 ✗
3. Fix: either reshape or transpose
4. Verify: print((A @ B.T).shape)
```

### 3. Build Intuition

For each operation, ask:
- What is the input shape?
- What is the output shape?
- Why does this operation make sense?

Example:
```python
X = np.random.randn(32, 784)  # 32 images, 784 pixels
W = np.random.randn(784, 128) # Weights
output = X @ W                 # Shape: (32, 128)

# Why? Each of 32 images → 128 features
# This is one layer in a neural network!
```

---

## 🎓 Study Tips

### For Visual Learners
- Focus on `concepts.md` first
- Draw diagrams on paper
- Visualize arrays as grids/matrices
- Use matplotlib to plot data

### For Hands-On Learners
- Start with `examples/` immediately
- Modify code and see what breaks
- Try to break things intentionally
- Fix errors to learn boundaries

### For Reading Learners
- Read all lessons in order
- Take notes in your own words
- Create summary sheets
- Explain concepts to yourself out loud

### For .NET Developers
- Compare every NumPy operation to C# equivalent
- Think: "How would I do this in LINQ?"
- Note: NumPy is 50-100x faster!
- Use `python_guide_for_dotnet.md` extensively

---

## 🐛 Common Issues and Solutions

### Issue 1: Import Error
```python
ModuleNotFoundError: No module named 'numpy'
```
**Solution:**
```bash
pip install numpy
# Or if using conda:
conda install numpy
```

### Issue 2: Shape Mismatch
```python
ValueError: shapes (3,4) and (5,2) not aligned
```
**Solution:**
```python
print(A.shape, B.shape)  # Debug
# Fix: Transpose or reshape to match inner dimensions
```

### Issue 3: Unexpected Broadcasting
```python
# Expected error, but got weird result
```
**Solution:**
```python
# Always print shapes before operations
print(f"A: {A.shape}, B: {B.shape}")
# Understand broadcasting rules in concepts.md
```

### Issue 4: Modifying Original Array
```python
# Changed slice, but original changed too!
```
**Solution:**
```python
# Use .copy() for independent arrays
arr_copy = arr.copy()
```

---

## 📊 Progress Tracking

### Self-Assessment Checklist

After each lesson, check if you can:

**Lesson 1: NumPy Basics**
- [ ] Create arrays from lists
- [ ] Use `zeros`, `ones`, `arange`, `linspace`
- [ ] Access elements with indexing
- [ ] Slice arrays (1D and 2D)
- [ ] Reshape arrays
- [ ] Understand `.shape`, `.size`, `.ndim`

**Lesson 2: Array Operations**
- [ ] Explain broadcasting
- [ ] Perform element-wise operations
- [ ] Use `@` for matrix multiplication
- [ ] Apply aggregation functions
- [ ] Stack and split arrays
- [ ] Generate random numbers

**Lesson 3: Linear Algebra**
- [ ] Compute dot products
- [ ] Multiply matrices with correct shapes
- [ ] Transpose arrays
- [ ] Understand shape compatibility
- [ ] Implement a neural network layer
- [ ] Calculate norms and distances

### Quiz Scoring Guide

- **27-35 (Excellent)**: Ready for Module 3! 🌟
- **24-26 (Good)**: Review missed topics ✅
- **20-23 (Okay)**: Revisit lessons and practice more 📚
- **<20 (Review needed)**: Go through module again 🔄

---

## 🔗 Next Steps

After completing this module:

1. **Mark Progress**
   - Update `PROGRESS.md` in project root
   - Note any challenging topics for review

2. **Optional Deep Dive**
   - Advanced NumPy features
   - NumPy C API
   - Performance optimization

3. **Move to Module 3**
   - Build real neural networks
   - Implement backpropagation
   - Train your first model

---

## 💡 Pro Tips

1. **Print shapes constantly** - `print(arr.shape)` is your best friend
2. **Start small** - Test with 2x2 matrices before scaling up
3. **Experiment freely** - Code is cheap, run it and see!
4. **Use Jupyter** - Interactive exploration builds intuition
5. **Relate to real LLMs** - Think "how does GPT use this?"
6. **Don't memorize** - Understand concepts, look up syntax
7. **Ask why** - Every operation has a purpose

---

## 📚 Additional Resources

### Official Documentation
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy API Reference](https://numpy.org/doc/stable/reference/index.html)

### Interactive Practice
- [NumPy Exercises (GitHub)](https://github.com/rougier/numpy-100)
- [Google Colab](https://colab.research.google.com/) - Free GPU access

### When Stuck
1. Check `quick_reference.md`
2. Review `concepts.md` for visual explanations
3. Run working examples to compare
4. Print everything: shapes, types, values

---

## 🎉 Ready to Begin!

Start with **README.md** to understand why NumPy is crucial for LLMs, then dive into **01_numpy_basics.md**.

Remember: Every expert started as a beginner. Take your time, experiment, and have fun building the foundation for understanding LLMs!

**Good luck! 🚀**
