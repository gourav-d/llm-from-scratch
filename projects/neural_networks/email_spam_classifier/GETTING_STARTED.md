# Getting Started: Email Spam Classifier

**Step-by-step guide to your first neural network project**

---

## 📋 Prerequisites

Before starting, make sure you've completed:

✅ **Module 2: NumPy** (array operations, linear algebra)
✅ **Module 3, Lessons 1-4** (perceptrons, activations, backpropagation)

**Time Required:** 2-3 hours

---

## 🎯 Learning Goals

By the end of this project, you'll be able to:

1. ✅ Convert text into numerical features
2. ✅ Build a binary classification neural network
3. ✅ Train on real data (~5,000 emails)
4. ✅ Evaluate model performance (accuracy, precision, recall)
5. ✅ Make predictions on new emails
6. ✅ Understand when/why the model fails

---

## 🚦 Step-by-Step Guide

### Step 1: Understand the Problem (15 minutes)

**Read first:**
- `README.md` (this project's overview)

**Think about:**
- What makes an email spam?
- How can we represent text as numbers?
- What does "classification" mean?

**Key insight:**
Spam emails have **patterns** in their words:
- Spam: "free", "win", "click", "buy", "urgent", "$$$"
- Ham: "meeting", "project", "attached", "regards"

---

### Step 2: Explore the Data (20 minutes)

**Open:** `data/emails.csv`

```bash
# Look at the data structure
head data/emails.csv
```

**You'll see:**
```csv
text,label
"Buy cheap pills now!",spam
"Meeting tomorrow at 3pm",ham
"URGENT: Click here to claim prize",spam
"Thanks for your email, I'll review it",ham
...
```

**Dataset stats:**
- **Total emails:** 5,000
- **Spam:** 2,500 (50%)
- **Ham:** 2,500 (50%)
- **Balanced dataset** (equal spam and ham)

**Try answering:**
- Can YOU tell spam from ham just by reading?
- What words appear frequently in spam?
- What words appear in ham?

---

### Step 3: Run the Simple Version (30 minutes)

**Start here first!**

```bash
python project_simple.py
```

**What happens:**

```
Step 1: Loading data...
✓ Loaded 5000 emails (2500 spam, 2500 ham)

Step 2: Building vocabulary...
✓ Found 15,234 unique words
✓ Keeping top 1000 most common words
✓ Vocabulary: ['the', 'to', 'and', 'a', 'of', ...]

Step 3: Creating features...
✓ Converted text to numbers
✓ Training set: 4000 emails
✓ Test set: 1000 emails

Step 4: Building neural network...
✓ Input layer: 1000 features
✓ Hidden layer: 64 neurons (ReLU)
✓ Output layer: 1 neuron (Sigmoid)
✓ Total parameters: 64,065

Step 5: Training...
Epoch 5/30:  Loss=0.245, Accuracy=89.2%
Epoch 10/30: Loss=0.201, Accuracy=92.1%
Epoch 15/30: Loss=0.168, Accuracy=93.5%
Epoch 20/30: Loss=0.142, Accuracy=94.3%
Epoch 25/30: Loss=0.129, Accuracy=94.8%
Epoch 30/30: Loss=0.125, Accuracy=95.1%

Step 6: Testing...
✓ Test Accuracy: 93.5%
✓ Test Precision: 91.2%
✓ Test Recall: 89.8%

Step 7: Example predictions...
Email: "Congratulations! You won $1000000!"
→ SPAM (99.8% confidence) ✓

Email: "Meeting notes from standup"
→ HAM (2.3% spam probability) ✓

✓ Training complete!
✓ Saved plots to results/
```

**Check the results folder:**
```bash
ls results/
# You'll see:
# - training_curve.png (loss and accuracy over time)
# - confusion_matrix.png (visual of predictions)
# - results.txt (detailed metrics)
```

**View the plots:**
Open `results/training_curve.png` to see how the network learned!

---

### Step 4: Understand the Code (45 minutes)

**Open:** `project_simple.py`

**The code has 5 main parts:**

#### Part 1: Data Loading (Lines 1-50)
```python
# Reads emails.csv
# Separates text and labels
# Splits into train/test sets
```

**For .NET devs:** Like reading CSV with `File.ReadAllLines()` and LINQ

#### Part 2: Text Preprocessing (Lines 51-120)
```python
# Builds vocabulary (top 1000 words)
# Converts text → bag of words vector
# Each email becomes [0,0,1,0,1,...,0,1]
```

**For .NET devs:** Like creating a `Dictionary<string, int>` for word indices

#### Part 3: Neural Network Class (Lines 121-250)
```python
class SpamClassifier:
    def __init__(self, input_size, hidden_size):
        # Initialize weights
        self.W1 = ... # Input → Hidden
        self.W2 = ... # Hidden → Output

    def forward(self, X):
        # Forward propagation
        # Returns spam probability (0 to 1)

    def backward(self, X, y, y_pred):
        # Backpropagation
        # Returns gradients for weights
```

**For .NET devs:** Like a `class NeuralNetwork` with methods

#### Part 4: Training Loop (Lines 251-320)
```python
for epoch in range(epochs):
    # Forward pass
    predictions = network.forward(X_train)

    # Compute loss
    loss = binary_cross_entropy(y_train, predictions)

    # Backward pass
    gradients = network.backward(X_train, y_train, predictions)

    # Update weights (Adam optimizer)
    optimizer.update(network, gradients)
```

**For .NET devs:** Like a `for` loop updating model weights

#### Part 5: Evaluation (Lines 321-380)
```python
# Make predictions on test set
test_predictions = network.forward(X_test)

# Compute metrics
accuracy = compute_accuracy(y_test, test_predictions)
precision = compute_precision(y_test, test_predictions)
recall = compute_recall(y_test, test_predictions)
```

**Read through each section and:**
- Identify which Module 3 concepts are used
- Understand what each function does
- Note any confusing parts (we'll explain later!)

---

### Step 5: Read the Explanation (30 minutes)

**Open:** `EXPLANATION.md`

This file explains **every line** of code:
- What it does
- Why it's needed
- How it connects to Module 3

**Focus on:**
- Text → Numbers conversion (bag of words)
- Network architecture (why 1000→64→1?)
- Sigmoid for output (why not ReLU?)
- Binary cross-entropy loss (what is it?)

---

### Step 6: Run the Complete Version (20 minutes)

Now that you understand the simple version:

```bash
python project_main.py
```

**What's different:**

✅ **Better preprocessing:**
- Lowercasing text
- Removing special characters
- Handling edge cases

✅ **More detailed logging:**
- Shows progress for each step
- Prints sample emails
- Better error messages

✅ **Additional features:**
- Saves trained model
- Tests on custom emails
- More visualizations
- Hyperparameter tuning options

✅ **Better evaluation:**
- Confusion matrix
- ROC curve (advanced)
- Per-class metrics

**Expected output:**
Similar to simple version, but with more details and better formatting.

---

### Step 7: Experiment! (30+ minutes)

Now the fun part - **make changes and see what happens!**

#### Experiment 1: Change Hidden Layer Size

**In `project_simple.py`, line ~240:**
```python
# Original
hidden_size = 64

# Try:
hidden_size = 32   # Smaller network
hidden_size = 128  # Larger network
```

**Run and observe:**
- Does accuracy change?
- Is training faster/slower?
- When is bigger better?

#### Experiment 2: Change Learning Rate

**Line ~280:**
```python
# Original
learning_rate = 0.001

# Try:
learning_rate = 0.01   # 10x faster
learning_rate = 0.0001 # 10x slower
```

**Run and observe:**
- Does it converge faster?
- Does it converge at all?
- What's the best learning rate?

#### Experiment 3: Change Vocabulary Size

**Line ~60:**
```python
# Original
vocab_size = 1000

# Try:
vocab_size = 500   # Fewer words
vocab_size = 2000  # More words
```

**Run and observe:**
- Does accuracy improve with more words?
- How much slower is training?
- What's the optimal size?

#### Experiment 4: Train Longer or Shorter

**Line ~275:**
```python
# Original
epochs = 30

# Try:
epochs = 10  # Faster
epochs = 50  # Longer
```

**Run and observe:**
- Does it overfit with more epochs?
- Is 30 epochs enough?
- Watch the training curve!

#### Experiment 5: Test Your Own Emails

**At the end of `project_simple.py`, add:**
```python
# Test custom emails
my_emails = [
    "Buy cheap viagra now!!!",
    "Let's grab coffee tomorrow?",
    "URGENT: Your account will be suspended",
    "The project deadline is next Friday"
]

for email in my_emails:
    # Convert to features
    features = text_to_features(email, vocabulary)

    # Predict
    prob = network.forward(features.reshape(1, -1))[0, 0]

    # Print result
    label = "SPAM" if prob > 0.5 else "HAM"
    print(f"'{email[:50]}' → {label} ({prob:.1%})")
```

**Run and observe:**
- Does it classify your emails correctly?
- Where does it fail?
- Why might it fail?

---

### Step 8: Understand Key Concepts (30 minutes)

**Open:** `CONCEPTS.md`

**Key concepts to master:**

#### 1. Bag of Words
**What:** Represent text as word counts (ignore order)
**Why:** Neural networks need fixed-size numeric input
**Limitation:** Loses word order ("not good" vs "good")

#### 2. Binary Classification
**What:** Classify into 2 categories (spam or ham)
**How:** Sigmoid output (0 to 1) with threshold 0.5
**Loss:** Binary cross-entropy

#### 3. Confusion Matrix
```
              Predicted
            Spam    Ham
Actual Spam  TP     FN    ← Recall = TP/(TP+FN)
       Ham   FP     TN
            ↑
       Precision = TP/(TP+FP)
```

#### 4. Precision vs Recall Tradeoff
- **High precision:** Few false alarms (ham marked as spam)
- **High recall:** Catch most spam (few spam get through)
- **Can't maximize both!** Tradeoff based on use case

**For spam filtering:**
- Missing spam (low recall) = annoying
- Blocking ham (low precision) = VERY BAD!
- → Optimize for precision!

---

## 🎯 Checkpoints

### ✅ Checkpoint 1: Understanding
Can you answer these questions?

1. How does bag of words work?
2. Why do we use sigmoid for the output layer?
3. What's the difference between precision and recall?
4. Why is Adam optimizer better than vanilla SGD?
5. How does the network "learn" spam patterns?

**If yes:** Continue!
**If no:** Re-read README.md and EXPLANATION.md

### ✅ Checkpoint 2: Running Code
Have you successfully:

1. ✅ Run `project_simple.py`?
2. ✅ Seen training progress (loss decreasing)?
3. ✅ Achieved >90% accuracy?
4. ✅ Generated plots in `results/` folder?
5. ✅ Understood the output?

**If yes:** Continue!
**If no:** Check for errors, ask for help

### ✅ Checkpoint 3: Experimentation
Have you tried:

1. ✅ Changing hidden layer size?
2. ✅ Changing learning rate?
3. ✅ Testing custom emails?
4. ✅ Understanding what makes the model fail?

**If yes:** You're done! 🎉
**If no:** Go back to Step 7

---

## 🚧 Common Issues & Solutions

### Issue 1: "Accuracy is stuck at 50%"
**Cause:** Model is just guessing (random)
**Solutions:**
- Check learning rate (try 0.001)
- Train longer (50 epochs)
- Check data loading (is it balanced?)
- Check gradient computation

### Issue 2: "Loss is NaN"
**Cause:** Numerical instability
**Solutions:**
- Lower learning rate (0.0001)
- Check for division by zero
- Add small epsilon to log calculations
- Check weight initialization

### Issue 3: "Training is very slow"
**Cause:** Large dataset or network
**Solutions:**
- Reduce batch size
- Use smaller network (32 hidden)
- Use smaller vocabulary (500 words)
- Check for infinite loops!

### Issue 4: "Shapes don't match"
**Cause:** Matrix dimension mismatch
**Solutions:**
- Print all shapes: `print(X.shape, y.shape)`
- Check feature extraction
- Review Module 3, Lesson 3 (shapes)
- Use `.reshape()` carefully

### Issue 5: "Model overfits (train 99%, test 70%)"
**Cause:** Memorizing training data
**Solutions:**
- Train for fewer epochs (15 instead of 30)
- Use more training data
- Simplify network (smaller hidden layer)
- Add regularization (advanced)

---

## 📊 Expected Results

### Beginner Level (Just Run It)
- ✅ Successfully run `project_simple.py`
- ✅ See training progress
- ✅ Achieve 90%+ accuracy
- ✅ Understand high-level concepts

**Time:** 1-1.5 hours

### Intermediate Level (Understand It)
- ✅ Read and understand all code
- ✅ Explain each component
- ✅ Run experiments
- ✅ Interpret confusion matrix

**Time:** 2-2.5 hours

### Advanced Level (Master It)
- ✅ Modify architecture
- ✅ Tune hyperparameters
- ✅ Test edge cases
- ✅ Understand limitations
- ✅ Connect to Module 3 concepts

**Time:** 3-4 hours

---

## ⏭️ Next Steps

### After This Project

**Option 1: Move to Project 2 (Recommended)**
→ `projects/neural_networks/mnist_digits/`
- Work with images (784 pixels → 10 classes)
- Multi-class classification
- Deeper networks

**Option 2: Improve This Project**
- Try TF-IDF instead of bag of words
- Add bigrams (word pairs: "not good")
- Use character-level features
- Implement regularization

**Option 3: Apply to Your Own Data**
- Collect your own emails
- Label them manually
- Retrain the model
- Compare performance

---

## 🎓 Learning Reflection

After completing this project, answer these questions:

1. **What surprised you?**
   - How well does the model work?
   - What mistakes does it make?

2. **What was hardest?**
   - Text preprocessing?
   - Understanding backpropagation?
   - Tuning hyperparameters?

3. **How does this relate to Module 3?**
   - Which lessons were most relevant?
   - What concepts clicked?

4. **What would you do differently?**
   - Different architecture?
   - More data?
   - Better features?

5. **What's next?**
   - Ready for MNIST?
   - Want to go deeper here?
   - Questions to explore?

---

## 📚 Additional Resources

### Review These Module 3 Lessons
- **Lesson 1:** Perceptrons (each neuron is a perceptron!)
- **Lesson 2:** Activations (ReLU + Sigmoid used here)
- **Lesson 4:** Backpropagation (how weights update)
- **Lesson 6:** Adam Optimizer (fast convergence)

### Going Deeper
- **Naive Bayes:** Classic spam filter algorithm
- **TF-IDF:** Better than bag of words
- **Word2Vec:** Learn word embeddings
- **RNNs:** Handle word order
- **Transformers:** State-of-the-art (Module 4!)

---

## ✅ Final Checklist

Before moving to the next project, ensure you:

- ✅ Successfully ran both simple and main versions
- ✅ Achieved >90% test accuracy
- ✅ Understood text → numbers conversion
- ✅ Understood network architecture (1000→64→1)
- ✅ Ran at least 2 experiments
- ✅ Read all documentation
- ✅ Can explain how it works to someone else
- ✅ Understand precision vs recall tradeoff

---

**Congratulations! You built your first real neural network!** 🎉

👉 **Next:** Move to Project 2 (MNIST) or dive deeper here!

🚀 **You're one step closer to understanding LLMs!**
