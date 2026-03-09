# Lesson 3: Training Your GPT Model from Scratch

**Build a GPT that actually learns from data!**

---

## What You'll Learn

In the previous lessons, you built a GPT architecture and learned how to generate text. But there's one critical problem: **your model doesn't know anything yet!** All those millions of parameters are just random numbers.

In this lesson, you'll learn how to **train** your GPT model - teaching it to understand and generate coherent English (or any language) by learning from examples.

**After this lesson, you'll have:**
- A GPT model that can actually write coherent sentences
- Understanding of how ChatGPT was trained
- Ability to train custom models on your own data

---

## What Is "Training"?

### Layman Definition

**Training** = Teaching the model by showing it examples and correcting its mistakes, over and over, until it gets better.

### Real-World Analogy

Think of training a GPT model like **teaching a child to write stories**:

**Day 1:** The child writes nonsense
- "Dog the sat cat on" (random words, no meaning)

**After showing 100 example sentences:** Getting better
- "The dog sat on cat" (almost right, grammar improving)

**After showing 10,000 example sentences:** Much better!
- "The dog sat on the mat" (correct grammar, makes sense)

**After showing 1,000,000 example sentences:** Expert level!
- "The golden retriever lazily sat on the colorful mat in the warm afternoon sun"

**The training process:**
1. **Show an example** ("The cat sat on the mat")
2. **Ask the child to predict the next word**
   - After "The cat sat on" → predict "the"
3. **Correct mistakes** ("You said 'a' but the answer was 'the'")
4. **Adjust understanding** (learn that "the" is more common here)
5. **Repeat 1 million times!**

That's exactly how GPT learns!

---

## Why Do We Need Training?

### The Problem

When you create a GPT model with `model = GPT(config)`:
```python
# All parameters are RANDOM numbers
weights = random.randn(512, 512)  # Completely random!
bias = random.randn(512)          # Also random!
```

**Result:** The model generates complete nonsense:
```
Input: "Once upon a time"
Output: "xkz2#blorp$@@wuv" ← Garbage!
```

**Why?** Because random weights produce random outputs!

### The Solution: Training

After training on millions of text examples:
```python
# Parameters are LEARNED from data
weights = [[0.34, -0.12, 0.89, ...], ...]  # Learned patterns!
bias = [0.05, -0.23, 0.67, ...]            # Learned biases!
```

**Result:** The model generates coherent text:
```
Input: "Once upon a time"
Output: "there was a brave knight who lived in a castle by the sea"
```

**Why?** Because the weights learned patterns from real text!

---

## How Does Training Work?

### The Big Picture

Training is a cycle of 4 steps, repeated millions of times:

```
Step 1: PREDICTION
   ↓
   Model tries to predict next word
   "The cat sat on ___" → Model guesses: "table" (60%), "mat" (30%), "dog" (10%)

Step 2: COMPARISON
   ↓
   Compare prediction to correct answer
   Correct answer: "the"
   Model's guess: "table" ← WRONG!

Step 3: MEASURE MISTAKE
   ↓
   Calculate HOW wrong (this is the "Loss")
   Loss = 2.3 (high number = very wrong)

Step 4: ADJUST WEIGHTS
   ↓
   Slightly change weights to make better predictions
   Old weight: 0.5 → New weight: 0.48

Step 5: REPEAT!
   ↓
   Go back to Step 1 with new weights
   Eventually, model learns "the" is the right word!
```

**Analogy:** Like practicing free throws in basketball:
1. Shoot the ball
2. See if it goes in
3. If you miss, note how far off you were
4. Adjust your aim slightly
5. Shoot again
6. Repeat 1,000 times → You get better!

---

## Core Concepts Explained

### 1. Training Data

**Training Data** = The examples you show the model to learn from.

**Think of it like textbooks for a student:**
- More books = smarter student
- Better books = better learning
- Bad books = confused student

**For GPT:**
```python
# Training data is just text files!
training_data = """
Once upon a time, there was a brave knight.
The knight rode a white horse.
He saved the princess from the dragon.
They lived happily ever after.
"""
```

**How much data do you need?**
| Model Size | Training Data | Example |
|------------|---------------|---------|
| **Tiny GPT** | 1 MB (small book) | Shakespeare's works |
| **Small GPT** | 100 MB | All Harry Potter books |
| **Medium GPT** | 10 GB | Wikipedia |
| **GPT-3** | 570 GB | Most of the internet! |

**C# Equivalent:**
```csharp
// C# - read training data from file
string trainingData = File.ReadAllText("training.txt");

// Python - same thing
training_data = open("training.txt").read()
```

---

### 2. Loss Function

**Loss Function** = A way to measure HOW WRONG the model's prediction is.

**Think of it like a report card:**
- **Low score (0.1)** = Almost perfect! (A+)
- **Medium score (1.5)** = Okay, but needs improvement (C)
- **High score (5.0)** = Very wrong! (F)

**Real-World Analogy - Archery:**

Imagine you're shooting arrows at a target:

```
TARGET:  🎯 (bullseye)

Shot 1:  ←5 feet away→ 🏹 (Loss = 5.0 - terrible!)
Shot 2:  ←2 feet away→ 🏹   (Loss = 2.0 - better!)
Shot 3:  ←6 inches→ 🏹       (Loss = 0.5 - close!)
Shot 4:  🎯🏹                 (Loss = 0.0 - perfect!)
```

**For GPT - Predicting the Next Word:**

```
Text: "The cat sat on the ___"
Correct answer: "mat"

Model's predictions (probabilities):
- "table" = 0.6  (60% confident)
- "mat"   = 0.3  (30% confident) ← Correct answer!
- "dog"   = 0.1  (10% confident)

Loss = How wrong this distribution is
     = 1.2 (not great - should be MORE confident in "mat")
```

**After training:**
```
Model's predictions (improved):
- "mat"   = 0.8  (80% confident) ← Correct answer!
- "table" = 0.15 (15% confident)
- "dog"   = 0.05 (5% confident)

Loss = 0.22 (much better! High confidence in correct answer)
```

**The Loss Function Formula (Cross-Entropy):**

```python
# Don't worry about the math yet - understand the CONCEPT first!
# Loss = -log(probability of correct answer)

# If model gives correct answer high probability:
loss = -log(0.8) = 0.22  # Low loss = good!

# If model gives correct answer low probability:
loss = -log(0.3) = 1.2   # High loss = bad!

# If model gives correct answer very low probability:
loss = -log(0.01) = 4.6  # Very high loss = terrible!
```

**Key Insight:**
> Lower loss = Better predictions = Smarter model!

---

### 3. Backpropagation

**Backpropagation** = The process of figuring out which weights to adjust and by how much.

**Think of it like detective work:**

Imagine your cake tastes bad. You need to figure out what went wrong:

```
Final Taste: Bad 😞
    ↑
Was it the frosting? Let me check...
    ↑
Was it the baking time? Let me check...
    ↑
Was it the flour amount? Let me check...
    ↑
AH! I added salt instead of sugar! ← Root cause!
```

**For GPT:**

```
Final Prediction: Wrong! "table" instead of "mat"
    ↑
Output layer weights: Which ones contributed to "table"?
    ↑
Last transformer block: Which neurons activated for "table"?
    ↑
Middle transformer blocks: Which attention heads focused wrong?
    ↑
First transformer block: Which features were emphasized?
    ↑
Embedding layer: Were the word vectors pointing wrong?
    ↑
ADJUST ALL OF THESE! ← Fix the whole chain!
```

**The "Back" in Backpropagation:**

```
FORWARD (prediction):
Input → Layer 1 → Layer 2 → Layer 3 → Output
 "The"    ...       ...       ...      "table" ❌

BACKWARD (correction):
Input ← Layer 1 ← Layer 2 ← Layer 3 ← Output
        adjust    adjust    adjust    Error!
```

**Why it's called "back":**
- We start from the OUTPUT (final prediction)
- Work BACKWARDS through all layers
- Calculate how each layer contributed to the mistake
- Adjust weights in each layer

**C# Analogy:**

```csharp
// Like a call stack during debugging!
Main()
  ↓ calls
CalculateTotal()
  ↓ calls
GetPrice()
  ↓ calls
ApplyDiscount() ← ERROR HERE!

// You trace BACKWARDS through the call stack:
ApplyDiscount() ← Found the bug!
  ↑ return
GetPrice()
  ↑ return
CalculateTotal()
  ↑ return
Main()
```

**Key Insight:**
> Backpropagation doesn't magically fix everything - it just tells us WHICH weights to adjust and BY HOW MUCH.

---

### 4. Gradient Descent

**Gradient Descent** = The algorithm for adjusting weights to reduce loss.

**Think of it like walking down a mountain in the fog:**

You want to reach the bottom (lowest loss) but you can't see far:

```
          🏔️ (High Loss = 5.0)
         /  \
        /    \
       /      \    ← You take small steps downward
      /   😊   \
     /          \
    /____________\  🎯 (Low Loss = 0.1)
       Valley
```

**How do you find the bottom?**
1. **Feel the slope** under your feet (this is the "gradient")
2. **Take a small step** downhill (this is "learning rate")
3. **Repeat** until you reach the bottom

**For GPT:**

```python
# Current weight value
weight = 0.5

# Current loss at this weight
loss = 2.3  (high - not good!)

# Gradient = "Which direction should I adjust this weight?"
gradient = -0.4  (negative = decrease weight)

# Learning rate = "How big should each step be?"
learning_rate = 0.01  (small steps = safe)

# Update weight
new_weight = weight - (learning_rate * gradient)
new_weight = 0.5 - (0.01 * -0.4)
new_weight = 0.5 + 0.004
new_weight = 0.504  ← Slightly adjusted!

# New loss (after adjustment)
new_loss = 2.25  (slightly better!)
```

**Repeat this for ALL 50 million parameters!**

**Real-World Analogy - Tuning a Guitar:**

```
String is too tight (sounds too high)
  ↓
Turn tuning peg slightly LEFT
  ↓
Check if sound is better
  ↓
Still too high? Turn a bit more LEFT
  ↓
Repeat until perfect pitch!
```

**Why "Descent"?**
- We're DESCENDING (going down) the loss landscape
- High loss = high mountain
- Low loss = low valley
- Goal: Reach the bottom (lowest loss)

**The Learning Rate - Goldilocks Problem:**

```
Learning Rate TOO BIG (0.5):
   🏃💨 (running down mountain)
   → Miss the valley
   → Overshoot to other side
   → Never settle down

Learning Rate TOO SMALL (0.00001):
   🐌 (crawling down mountain)
   → Takes FOREVER
   → Might get stuck in small dip
   → Never reach bottom

Learning Rate JUST RIGHT (0.001):
   🚶 (walking down mountain)
   → Steady progress
   → Finds valley
   → Reaches bottom eventually
```

---

### 5. Training Loop

**Training Loop** = The repetitive process of showing examples, measuring mistakes, and adjusting weights.

**Think of it like practicing piano:**

```
FOR each practice session (Epoch):
    FOR each song in your book (Batch):
        1. Play the song (Forward Pass)
        2. Listen for mistakes (Calculate Loss)
        3. Figure out what went wrong (Backpropagation)
        4. Adjust finger positions (Update Weights)

    Test yourself (Validation)
    Are you getting better? Keep going!
```

**The Complete Training Loop:**

```python
# Don't worry about running this yet - just understand the FLOW!

# Step 1: Prepare the data
training_data = load_text("shakespeare.txt")
batches = split_into_batches(training_data, batch_size=32)

# Step 2: Set up the model
model = GPT(config)
optimizer = AdamOptimizer(learning_rate=0.001)

# Step 3: Training loop
for epoch in range(10):  # Epoch = one complete pass through all data

    print(f"Starting Epoch {epoch+1}/10...")

    for batch in batches:  # Process one batch at a time

        # Forward pass: Model makes prediction
        predictions = model(batch.input)

        # Calculate loss: How wrong is the prediction?
        loss = cross_entropy_loss(predictions, batch.target)

        # Backward pass: Figure out what to adjust
        gradients = backpropagation(loss)

        # Update weights: Adjust parameters
        optimizer.update_weights(model, gradients)

        # Print progress every 100 batches
        if batch.number % 100 == 0:
            print(f"Batch {batch.number}, Loss: {loss:.2f}")

    # End of epoch: Test on validation data
    val_loss = evaluate_model(model, validation_data)
    print(f"Epoch {epoch+1} complete! Validation Loss: {val_loss:.2f}")

    # Save checkpoint (in case computer crashes!)
    save_model(model, f"checkpoint_epoch_{epoch+1}.pt")

print("Training complete! 🎉")
```

**Let me explain each part in DETAIL:**

---

#### What Is a Batch?

**Batch** = A small group of training examples processed together.

**Why not process all data at once?**

Imagine grading homework:
```
Option 1: Grade ALL 1,000 students' homework, THEN give feedback
   ❌ Takes forever
   ❌ Students wait weeks for feedback
   ❌ Too much to hold in memory

Option 2: Grade 32 students' homework, give feedback, repeat
   ✅ Quick feedback
   ✅ Manageable workload
   ✅ Students learn faster
```

**For GPT:**
```python
# If we have 10,000 text samples:

batch_size = 32  # Process 32 examples at a time

batches = 10,000 / 32 = 312 batches

# Process batch 1: samples 1-32
# Update weights
# Process batch 2: samples 33-64
# Update weights
# ...
# Process batch 312: samples 9,985-10,000
# Update weights
```

**Common batch sizes:**
| Batch Size | When to Use | Memory Needed |
|------------|-------------|---------------|
| **8-16** | Small GPU | Low |
| **32** | Medium GPU | Medium |
| **64-128** | Large GPU | High |
| **256+** | Multiple GPUs | Very High |

---

#### What Is an Epoch?

**Epoch** = One complete pass through ALL training data.

**Think of it like reading a textbook:**

```
Epoch 1: Read the entire book cover-to-cover (first time)
Epoch 2: Read the entire book again (second time)
Epoch 3: Read the entire book again (third time)
...

After 10 epochs: You've read the book 10 times!
You know it pretty well now!
```

**For GPT:**
```
Epoch 1: Show model all 10,000 examples once
Epoch 2: Show model all 10,000 examples again
Epoch 3: Show model all 10,000 examples again
...

After 10 epochs: Model has seen each example 10 times!
It has learned the patterns!
```

**How many epochs do you need?**
| Dataset Size | Typical Epochs | Reason |
|-------------|----------------|---------|
| **Small (1 MB)** | 50-100 | Need to see examples many times |
| **Medium (100 MB)** | 10-20 | Good variety, less repetition |
| **Large (10 GB)** | 3-5 | So much data, don't need many passes |
| **Huge (100 GB)** | 1-2 | Internet-scale data, once is enough! |

---

#### What Is Validation?

**Validation** = Testing the model on data it has NEVER seen before.

**Think of it like a practice exam:**

```
Study Material (Training Data):
  Chapter 1: Variables
  Chapter 2: Functions
  Chapter 3: Classes

Practice Exam (Validation Data):
  Questions about variables, functions, classes
  BUT questions you've NEVER seen before!

Real Exam (Test Data):
  Completely new questions (use after training is done)
```

**Why do we need validation?**

To detect **overfitting** (memorizing instead of learning):

```
❌ BAD - Memorizing Training Data:
   Student memorizes: "What is a variable? A named storage location."
   Exam asks: "Explain variables in your own words"
   Student: "Uh... I don't know, I only memorized the exact question!"

✅ GOOD - Actually Learning:
   Student understands: "Variables store values that can change"
   Exam asks: "Explain variables in your own words"
   Student: "Variables are like labeled boxes where you store data!"
```

**For GPT:**

```python
# Split data into training (90%) and validation (10%)
training_data = all_data[:9000]    # 90% for training
validation_data = all_data[9000:]  # 10% for testing

# After each epoch, check validation loss
for epoch in range(10):
    # Train on training data
    train_loss = train_one_epoch(training_data)

    # Test on validation data (model has NEVER seen this!)
    val_loss = evaluate(validation_data)

    print(f"Training Loss: {train_loss:.2f}")
    print(f"Validation Loss: {val_loss:.2f}")

    # If validation loss is INCREASING, we're overfitting!
    if val_loss > previous_val_loss:
        print("Warning: Overfitting detected! 🚨")
        break  # Stop training
```

**Good vs Bad Training:**

```
GOOD TRAINING:
Epoch 1: Train=2.5, Val=2.6  ← Both decreasing
Epoch 2: Train=2.0, Val=2.1  ← Both decreasing
Epoch 3: Train=1.5, Val=1.6  ← Both decreasing
Epoch 4: Train=1.2, Val=1.3  ← Still good!

BAD TRAINING (Overfitting):
Epoch 1: Train=2.5, Val=2.6  ← Both decreasing
Epoch 2: Train=2.0, Val=2.1  ← Both decreasing
Epoch 3: Train=1.5, Val=1.7  ← Val INCREASING! 🚨
Epoch 4: Train=1.0, Val=2.0  ← Val getting worse! 🚨
→ STOP! Model is memorizing training data!
```

---

### 6. Learning Rate Scheduling

**Learning Rate Scheduling** = Changing the learning rate during training.

**Think of it like driving to a destination:**

```
Highway (Start of training):
  🚗💨 Fast speed (high learning rate = 0.01)
  → Make quick progress
  → Cover lots of distance

City Streets (Middle of training):
  🚗 Medium speed (medium learning rate = 0.001)
  → More careful now
  → Still making progress

Parking Lot (End of training):
  🚗🐌 Slow speed (low learning rate = 0.0001)
  → Very precise movements
  → Fine-tuning to find exact spot
```

**Why change learning rate?**

```
START: Big changes needed (high learning rate)
   Loss = 5.0 → Make big adjustments

MIDDLE: Getting closer (medium learning rate)
   Loss = 1.5 → Make moderate adjustments

END: Fine-tuning (low learning rate)
   Loss = 0.3 → Make tiny adjustments
```

**Common Schedules:**

**1. Step Decay:**
```python
# Reduce learning rate every few epochs
Epochs 1-3:  lr = 0.01   (fast learning)
Epochs 4-6:  lr = 0.001  (slower)
Epochs 7-10: lr = 0.0001 (fine-tuning)
```

**2. Exponential Decay:**
```python
# Gradually reduce learning rate
Epoch 1:  lr = 0.01
Epoch 2:  lr = 0.0095
Epoch 3:  lr = 0.009
Epoch 4:  lr = 0.0086
...
Epoch 10: lr = 0.006
```

**3. Cosine Annealing (GPT-3's method):**
```python
# Smoothly decrease like a cosine wave
Epoch 1:  lr = 0.01     (high)
Epoch 3:  lr = 0.0075   (decreasing)
Epoch 5:  lr = 0.005    (middle)
Epoch 8:  lr = 0.0025   (low)
Epoch 10: lr = 0.0001   (very low)
```

**Code Example:**

```python
# Simple learning rate schedule
def get_learning_rate(epoch, initial_lr=0.01):
    """
    Reduce learning rate every 3 epochs.

    Think of it like shifting gears in a car:
    - Epochs 1-3: First gear (fast = 0.01)
    - Epochs 4-6: Second gear (medium = 0.001)
    - Epochs 7+:  Third gear (slow = 0.0001)
    """
    if epoch < 3:
        return initial_lr  # 0.01 - fast learning
    elif epoch < 6:
        return initial_lr / 10  # 0.001 - slower
    else:
        return initial_lr / 100  # 0.0001 - fine-tuning
```

---

### 7. Gradient Clipping

**Gradient Clipping** = Preventing gradients from becoming too large and breaking training.

**Think of it like a speed governor on a truck:**

```
Normal Driving:
   🚚 Speed = 60 mph ✅ (safe)

Downhill (Big Gradients):
   🚚💨 Speed = 120 mph! 🚨 (dangerous!)

Speed Governor:
   🚚 Speed capped at 65 mph ✅ (safe again)
   "Even on steep hills, max speed is 65"
```

**The Problem - Exploding Gradients:**

Sometimes during training, gradients become HUGE:

```
Normal gradient:
   gradient = 0.5 → weight adjustment = 0.0005 ✅

Exploding gradient:
   gradient = 10000! → weight adjustment = 10 🚨
   → Weight changes too much
   → Model breaks
   → Loss becomes NaN (Not a Number)
   → Training fails!
```

**The Solution - Clip Large Gradients:**

```python
# Without clipping
gradient = 10000  # Too big!
weight_update = learning_rate * gradient
weight_update = 0.001 * 10000 = 10  # HUGE change!

# With clipping
max_gradient = 1.0
if gradient > max_gradient:
    gradient = max_gradient  # Cap it!

weight_update = learning_rate * gradient
weight_update = 0.001 * 1.0 = 0.001  # Safe change!
```

**Real-World Analogy:**

```
Audio Volume Control:
   Normal: 🔊 Volume = 50 (comfortable)
   Loud noise: 📢 Volume = 1000 (would damage speakers!)

Volume Limiter:
   📢→🔊 Volume capped at 100 (safe maximum)
   Loud sounds are limited to prevent damage
```

**Code Example:**

```python
# Gradient clipping in practice
def clip_gradients(gradients, max_norm=1.0):
    """
    Clip gradients to prevent exploding gradients.

    Think of it like:
    - If gradient is small → leave it alone
    - If gradient is HUGE → cap it at max_norm

    Args:
        gradients: All gradients for all parameters
        max_norm: Maximum allowed gradient magnitude (default=1.0)
    """
    # Calculate total gradient magnitude
    total_norm = 0
    for gradient in gradients:
        total_norm += (gradient ** 2).sum()  # Sum of squares
    total_norm = total_norm ** 0.5  # Square root

    # If total is too big, scale down ALL gradients
    if total_norm > max_norm:
        scale_factor = max_norm / total_norm
        for gradient in gradients:
            gradient *= scale_factor  # Scale down

    return gradients

# Usage in training loop
for batch in batches:
    loss = model(batch)
    gradients = backpropagation(loss)

    # Clip gradients before updating weights!
    gradients = clip_gradients(gradients, max_norm=1.0)

    optimizer.update_weights(gradients)
```

**When to use gradient clipping:**
- ✅ Training RNNs or Transformers (prone to exploding gradients)
- ✅ Large models (GPT-2, GPT-3)
- ✅ When you see NaN losses
- ❌ Simple models (usually not needed)

---

## Complete Training Example

Now let's put it all together with a REAL, working example!

```python
"""
Training a Mini-GPT on Shakespeare's Works
==========================================

This example shows the COMPLETE training process from start to finish.
Every line is explained in detail!
"""

import numpy as np

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

def load_training_data(file_path):
    """
    Load text file and prepare it for training.

    Think of this like:
    - Loading a textbook
    - Breaking it into pages
    - Creating practice questions from each page
    """
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters")
    print(f"First 100 characters: {text[:100]}")

    return text

def create_training_examples(text, tokenizer, max_length=128):
    """
    Convert text into training examples.

    Each example is:
    - Input: "To be or not to"
    - Target: "be or not to be"

    Think of it like:
    - Showing a sentence with last word hidden
    - Model tries to guess last word
    """
    # Tokenize the entire text
    # (Convert "Hello world" → [45, 234])
    tokens = tokenizer.encode(text)

    examples = []

    # Create sliding windows
    # Think: Move a window across the text, one step at a time
    for i in range(0, len(tokens) - max_length - 1):
        # Input sequence: tokens at positions i to i+max_length
        input_tokens = tokens[i : i + max_length]

        # Target sequence: tokens at positions i+1 to i+max_length+1
        # (shifted by one - this is what we're trying to predict!)
        target_tokens = tokens[i + 1 : i + max_length + 1]

        examples.append({
            'input': input_tokens,
            'target': target_tokens
        })

    print(f"Created {len(examples):,} training examples")
    return examples

def create_batches(examples, batch_size=32):
    """
    Group examples into batches for efficient training.

    Think of it like:
    - Instead of grading 1 homework at a time
    - Grade 32 homeworks together
    - More efficient!
    """
    batches = []

    # Shuffle examples (important for good training!)
    np.random.shuffle(examples)

    # Group into batches
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        batches.append(batch)

    print(f"Created {len(batches)} batches of size {batch_size}")
    return batches

# ============================================================================
# STEP 2: DEFINE LOSS FUNCTION
# ============================================================================

def cross_entropy_loss(predictions, targets):
    """
    Calculate how wrong the predictions are.

    Cross-Entropy Loss measures:
    - How confident the model is in the CORRECT answer
    - Low confidence in correct answer = HIGH loss
    - High confidence in correct answer = LOW loss

    Think of it like:
    - Teacher asks "What's 2+2?"
    - Student says: "4" (80% confident), "5" (20% confident)
    - Loss measures how far this is from 100% confident in "4"
    """
    # predictions shape: (batch_size, sequence_length, vocab_size)
    # targets shape: (batch_size, sequence_length)

    batch_size, seq_len, vocab_size = predictions.shape

    # Flatten predictions and targets
    # (makes calculation easier)
    predictions_flat = predictions.reshape(-1, vocab_size)  # (batch*seq, vocab)
    targets_flat = targets.reshape(-1)  # (batch*seq,)

    # Calculate cross-entropy for each position
    # Don't worry about the math details - just know:
    # - Picks probability assigned to correct token
    # - Takes negative log
    # - Averages across all positions

    # Get probability assigned to correct token at each position
    correct_probs = predictions_flat[range(len(targets_flat)), targets_flat]

    # Cross-entropy formula: -log(probability)
    # If prob = 0.8 → loss = 0.22 (good!)
    # If prob = 0.1 → loss = 2.30 (bad!)
    losses = -np.log(correct_probs + 1e-10)  # +1e-10 to avoid log(0)

    # Average loss across all positions
    avg_loss = losses.mean()

    return avg_loss

# ============================================================================
# STEP 3: TRAINING LOOP
# ============================================================================

def train_gpt(model, training_data, config):
    """
    Complete training loop for GPT.

    This is the MAIN function that:
    1. Loads data
    2. Loops through epochs
    3. Processes batches
    4. Updates weights
    5. Tracks progress

    Think of it like a semester of college:
    - Each epoch = one complete read of all textbooks
    - Each batch = one homework assignment
    - Each update = learning from mistakes
    """
    print("=" * 60)
    print("STARTING GPT TRAINING")
    print("=" * 60)

    # -----------------------------------------
    # Setup
    # -----------------------------------------

    # Create optimizer (handles weight updates)
    from optimizer import Adam  # You'll learn about Adam later!
    optimizer = Adam(
        model.parameters(),
        learning_rate=config.learning_rate,
        betas=(0.9, 0.999)  # Adam hyperparameters
    )

    # Load and prepare data
    text = load_training_data(config.data_file)
    examples = create_training_examples(text, model.tokenizer)

    # Split into training and validation
    split_idx = int(0.9 * len(examples))  # 90% train, 10% validation
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"\nTraining examples: {len(train_examples):,}")
    print(f"Validation examples: {len(val_examples):,}")

    # Track best model (for saving)
    best_val_loss = float('inf')

    # -----------------------------------------
    # Main Training Loop
    # -----------------------------------------

    for epoch in range(config.num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{config.num_epochs}")
        print(f"{'='*60}")

        # Create batches for this epoch (shuffle each time!)
        batches = create_batches(train_examples, config.batch_size)

        # Variables to track progress
        epoch_train_loss = 0
        num_batches = len(batches)

        # -----------------------------------------
        # Process Each Batch
        # -----------------------------------------

        for batch_idx, batch in enumerate(batches):

            # Extract inputs and targets from batch
            inputs = np.array([ex['input'] for ex in batch])
            targets = np.array([ex['target'] for ex in batch])

            # FORWARD PASS
            # Model makes predictions
            predictions = model(inputs)

            # CALCULATE LOSS
            # How wrong are the predictions?
            loss = cross_entropy_loss(predictions, targets)
            epoch_train_loss += loss

            # BACKWARD PASS
            # Figure out what to adjust
            model.zero_grad()  # Clear old gradients
            loss.backward()    # Calculate new gradients

            # GRADIENT CLIPPING
            # Prevent exploding gradients
            clip_gradients(model.parameters(), max_norm=1.0)

            # UPDATE WEIGHTS
            # Adjust model parameters
            optimizer.step()

            # PRINT PROGRESS
            # Every 50 batches, show current status
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_train_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Batch {batch_idx+1}/{num_batches} "
                      f"({progress:.1f}%) - Loss: {avg_loss:.3f}")

        # -----------------------------------------
        # End of Epoch - Validation
        # -----------------------------------------

        # Calculate average training loss for this epoch
        avg_train_loss = epoch_train_loss / num_batches

        # Evaluate on validation set
        val_loss = evaluate_model(model, val_examples, config.batch_size)

        print(f"\n  Epoch {epoch+1} Results:")
        print(f"  - Training Loss:   {avg_train_loss:.3f}")
        print(f"  - Validation Loss: {val_loss:.3f}")

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            print(f"  ✓ New best model! (prev: {best_val_loss:.3f})")
            best_val_loss = val_loss

            # Save the model
            save_checkpoint(model, optimizer, epoch, val_loss,
                          'best_model.pt')

        # Check for overfitting
        if val_loss > avg_train_loss * 1.5:
            print("  ⚠ Warning: Possible overfitting detected!")

        # -----------------------------------------
        # Learning Rate Schedule
        # -----------------------------------------

        # Reduce learning rate if needed
        if (epoch + 1) % 3 == 0:
            old_lr = optimizer.learning_rate
            optimizer.learning_rate *= 0.5  # Halve the learning rate
            print(f"  Learning rate: {old_lr:.6f} → {optimizer.learning_rate:.6f}")

        # -----------------------------------------
        # Generate Sample Text
        # -----------------------------------------

        # Every 2 epochs, generate sample text to see progress
        if (epoch + 1) % 2 == 0:
            print("\n  Sample generation:")
            sample = model.generate(
                prompt="To be or not to be",
                max_length=50,
                temperature=0.8
            )
            print(f"  '{sample}'")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE! 🎉")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.3f}")
    print(f"Model saved to: best_model.pt")

    return model

# ============================================================================
# STEP 4: EVALUATION
# ============================================================================

def evaluate_model(model, val_examples, batch_size):
    """
    Evaluate model on validation data.

    Think of it like:
    - Taking a practice exam
    - No studying allowed (model doesn't learn from this)
    - Just check how well you do
    """
    model.eval()  # Put model in evaluation mode (no learning!)

    batches = create_batches(val_examples, batch_size)
    total_loss = 0

    # Process each batch WITHOUT updating weights
    for batch in batches:
        inputs = np.array([ex['input'] for ex in batch])
        targets = np.array([ex['target'] for ex in batch])

        # Forward pass only (no backward!)
        predictions = model(inputs)
        loss = cross_entropy_loss(predictions, targets)
        total_loss += loss

    avg_loss = total_loss / len(batches)

    model.train()  # Put model back in training mode

    return avg_loss

# ============================================================================
# STEP 5: SAVING CHECKPOINTS
# ============================================================================

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """
    Save model state to disk.

    Think of it like:
    - Saving your game progress
    - If computer crashes, you can resume from here
    - Don't have to start from beginning!
    """
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
    }

    # Save to file
    # (In real code, you'd use torch.save())
    print(f"  💾 Saved checkpoint: {filename}")

# ============================================================================
# STEP 6: USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    How to use this training code.
    """

    # Configuration
    class TrainingConfig:
        data_file = "shakespeare.txt"
        num_epochs = 10
        batch_size = 32
        learning_rate = 0.001
        max_seq_length = 128

    config = TrainingConfig()

    # Create model
    model = GPT(GPTConfig(
        vocab_size=50257,
        max_seq_len=128,
        embed_dim=512,
        n_layers=6,
        n_heads=8
    ))

    # Train!
    trained_model = train_gpt(model, None, config)

    # Generate text
    generated = trained_model.generate(
        prompt="To be or not to be",
        max_length=100,
        temperature=0.8
    )

    print("\nGenerated text:")
    print(generated)
```

---

## Summary

Let's recap everything you learned in this lesson:

| Concept | Simple Definition | Why It Matters |
|---------|------------------|----------------|
| **Training** | Teaching the model by showing examples | Random weights → learned patterns |
| **Training Data** | Examples model learns from | More data = smarter model |
| **Loss Function** | Measures how wrong predictions are | Low loss = good predictions |
| **Backpropagation** | Figures out what to adjust | Traces error back through layers |
| **Gradient Descent** | Algorithm for adjusting weights | Slowly improves model |
| **Learning Rate** | Size of weight adjustments | Too big = unstable, too small = slow |
| **Batch** | Group of examples processed together | Efficient use of GPU |
| **Epoch** | One complete pass through all data | Need multiple passes to learn |
| **Validation** | Testing on unseen data | Detects overfitting |
| **Gradient Clipping** | Limit size of gradients | Prevents training from breaking |

---

## Key Insights

### 1. Training Is Just Repetition
```
Show example → Make prediction → Measure mistake → Adjust weights → Repeat!

After 1 million repetitions → Model learns patterns!
```

### 2. Loss Is Your Guide
```
High Loss (5.0) = Model is confused, make BIG changes
Low Loss (0.5) = Model is doing well, make small tweaks
```

### 3. More Data = Better Model
```
100 examples     → Can write simple sentences
1,000 examples   → Can write paragraphs
100,000 examples → Can write essays
10,000,000 examples → Can write like Shakespeare!
```

### 4. Validation Prevents Memorization
```
Training Loss:    2.0 → 1.0 → 0.5 → 0.1  ✓ Getting better
Validation Loss:  2.1 → 1.2 → 0.8 → 0.9  ✗ Started increasing!
→ Stop training! Model is memorizing!
```

---

## What's Next?

In **Lesson 4**, you'll learn about **Fine-tuning** - how to take a pre-trained GPT model and adapt it to specific tasks:
- Use pre-trained GPT-2 instead of training from scratch
- Fine-tune on customer service conversations → Customer service bot
- Fine-tune on legal documents → Legal writing assistant
- Fine-tune on code → Code completion tool

This is how ChatGPT was created - GPT-3 fine-tuned on conversations!

---

## Practice Exercise

**Challenge:** Explain the training process to a friend

Try explaining:
1. What training means (teaching by examples)
2. What loss measures (how wrong the model is)
3. How the model improves (adjusting weights based on mistakes)
4. Why validation is important (detecting memorization)

If you can explain these concepts in simple terms, you understand them!

---

**Next:** Open `04_finetuning_gpt.md` to learn about fine-tuning! 🚀
