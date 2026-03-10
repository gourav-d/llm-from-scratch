# Module 06: Code Examples

This directory contains **6 complete code examples** with **line-by-line explanations in layman language**.

Each example demonstrates key concepts from Module 06 lessons.

---

## 📁 Examples Overview

### Example 1: Building Complete GPT Architecture
**File:** `example_01_complete_gpt.py`
**Lesson:** 1 - Building Complete GPT
**Topics Covered:**
- Complete GPT architecture assembly
- Token embeddings and positional encoding
- Multi-head attention mechanism
- Transformer blocks
- Forward pass implementation
- Parameter counting

**What You'll Learn:**
- How to assemble all GPT components
- Understanding every layer in the architecture
- Counting parameters (where do 124M parameters come from?)
- Comparing your GPT to GPT-2/GPT-3

**Run:**
```bash
python example_01_complete_gpt.py
```

**Output:**
- GPT model creation
- Forward pass demonstration
- Parameter statistics
- Comparison to GPT-2

---

### Example 2: Text Generation and Sampling Strategies
**File:** `example_02_text_generation.py`
**Lesson:** 2 - Text Generation
**Topics Covered:**
- Autoregressive generation
- Greedy sampling (always pick best)
- Temperature sampling (control creativity)
- Top-k sampling (limit choices)
- Top-p (nucleus) sampling (GPT-3's method)
- Comparing all strategies

**What You'll Learn:**
- How GPT generates text (one token at a time)
- When to use each sampling strategy
- How temperature affects creativity
- Why ChatGPT uses top-p sampling

**Run:**
```bash
python example_02_text_generation.py
```

**Output:**
- Comparison of all sampling strategies
- Temperature effects demonstration
- Recommendations for each use case

---

### Example 3: Training GPT from Scratch
**File:** `example_03_training_gpt.py`
**Lesson:** 3 - Training GPT
**Topics Covered:**
- Loss function (cross-entropy)
- Backpropagation (simplified)
- Gradient descent
- Training loop
- Optimizers (SGD)
- Epochs and batches

**What You'll Learn:**
- What "training" actually means
- How loss measures error
- How gradients help model improve
- The complete training process

**Run:**
```bash
python example_03_training_gpt.py
```

**Output:**
- Training process demonstration
- Loss decreasing over epochs
- Before/after comparison

---

### Example 4: Fine-Tuning Pre-trained Models
**File:** `example_04_finetuning_gpt.py`
**Lesson:** 4 - Fine-Tuning
**Topics Covered:**
- Fine-tuning vs training from scratch
- Lower learning rates
- Layer freezing
- Warmup schedules
- Catastrophic forgetting prevention

**What You'll Learn:**
- Why fine-tuning is 100x cheaper than training
- How to adapt models to specific tasks
- When to freeze layers
- Best practices for fine-tuning

**Run:**
```bash
python example_04_finetuning_gpt.py
```

**Output:**
- Comparison: scratch vs fine-tuning
- Three approaches (full, partial, frozen)
- Resource requirements

---

### Example 5: RLHF and Alignment
**File:** `example_05_rlhf_alignment.py`
**Lesson:** 5 - RLHF & Alignment
**Topics Covered:**
- Phase 1: Supervised Fine-Tuning (SFT)
- Phase 2: Reward Model training
- Phase 3: PPO optimization
- Making AI helpful, harmless, honest
- How ChatGPT was created

**What You'll Learn:**
- Complete RLHF pipeline
- How to align AI models
- Why ChatGPT is different from GPT-3
- All 3 phases explained simply

**Run:**
```bash
python example_05_rlhf_alignment.py
```

**Output:**
- Complete RLHF demonstration
- GPT-3 → ChatGPT transformation
- Before/after comparison

---

### Example 6: Deployment and Optimization
**File:** `example_06_deployment_optimization.py`
**Lesson:** 6 - Deployment & Optimization
**Topics Covered:**
- Quantization (float32 → float16 → int8)
- Response caching
- Batch processing
- Model size comparison
- Performance optimization

**What You'll Learn:**
- How to make models 4x smaller
- How to make inference 10x faster
- Caching for instant responses
- Batching for throughput

**Run:**
```bash
python example_06_deployment_optimization.py
```

**Output:**
- Quantization demonstration (2-4x smaller)
- Caching effectiveness
- Batching speedup
- Production deployment example

---

## 🎯 How to Use These Examples

### Prerequisites
```bash
# Install NumPy (only dependency)
pip install numpy
```

### Running Examples

**Option 1: Run individually**
```bash
python example_01_complete_gpt.py
python example_02_text_generation.py
# etc.
```

**Option 2: Run in sequence**
```bash
# Recommended learning path
python example_01_complete_gpt.py      # Understand architecture
python example_02_text_generation.py    # Learn generation
python example_03_training_gpt.py       # See training process
python example_04_finetuning_gpt.py     # Compare fine-tuning
python example_05_rlhf_alignment.py     # Learn alignment
python example_06_deployment_optimization.py  # Deploy optimized model
```

---

## 📊 What Each Example Outputs

| Example | Output | Time |
|---------|--------|------|
| **Example 1** | GPT architecture, parameter count, forward pass | ~30 sec |
| **Example 2** | Sampling strategy comparisons, text generation | ~45 sec |
| **Example 3** | Training loop, decreasing loss, improvement | ~60 sec |
| **Example 4** | Fine-tuning vs scratch, 3 approaches | ~60 sec |
| **Example 5** | Complete RLHF pipeline, all 3 phases | ~60 sec |
| **Example 6** | Optimization techniques, speedup stats | ~45 sec |

---

## 🔑 Key Concepts by Example

### Example 1: Architecture
- **GPT = Stacking components you already know**
- Token embeddings + Positional encoding
- Transformer blocks (attention + feedforward)
- Output projection layer

### Example 2: Generation
- **Autoregressive = One token at a time**
- Temperature controls creativity
- Top-p is best overall strategy
- ChatGPT uses top-p + temperature=0.8

### Example 3: Training
- **Training = Showing examples & learning from mistakes**
- Loss = Report card (lower is better)
- Gradients = Directions to improve
- Optimizer = Study strategy

### Example 4: Fine-Tuning
- **Fine-tuning = 100-1000x cheaper than training**
- Lower learning rate preserves knowledge
- Fewer epochs needed
- Can freeze layers for small datasets

### Example 5: RLHF
- **RLHF = How ChatGPT was created from GPT-3**
- Phase 1: Learn from experts (SFT)
- Phase 2: Learn to judge quality (Reward Model)
- Phase 3: Optimize for quality (PPO)

### Example 6: Deployment
- **Quantization = 2-4x smaller, faster**
- Caching = Instant for common queries
- Batching = 5-10x throughput
- Combined = 10-100x cost reduction

---

## 💡 Learning Tips

1. **Read the code comments** - Every line is explained
2. **Run each example** - See concepts in action
3. **Experiment** - Change parameters, see what happens
4. **Compare outputs** - Notice patterns and differences
5. **Take notes** - Write down key insights

---

## 🚀 Next Steps

After completing all examples:

1. **Run the projects** (in `/projects` directory)
   - Shakespeare generator
   - Customer support chatbot
   - Code completion assistant

2. **Build your own project**
   - Choose a use case
   - Apply techniques learned
   - Combine multiple concepts

3. **Explore real implementations**
   - Hugging Face Transformers
   - OpenAI API
   - LangChain

---

## ⚠️ Note

These examples use **simplified implementations** for learning:
- Focus on **concepts**, not production code
- Use **NumPy** instead of PyTorch/TensorFlow
- **Explain every line** in layman terms

For production:
- Use PyTorch or TensorFlow
- Use automatic differentiation
- Use GPU acceleration
- Use established libraries (Transformers, etc.)

---

## 📚 Related Lessons

- `01_building_complete_gpt.md` - Architecture details
- `02_text_generation.md` - Generation strategies
- `03_training_gpt.md` - Training process
- `04_finetuning_gpt.md` - Fine-tuning guide
- `05_rlhf_alignment.md` - RLHF details
- `06_deployment_optimization.md` - Production deployment

---

**Happy Learning! 🎉**

You now have 6 complete, fully-explained examples covering the entire GPT pipeline from architecture to deployment!
