# Project 1: Email Spam Classifier 📧

**Build a neural network that identifies spam emails**

---

## 🎯 What You'll Build

A neural network that reads email text and classifies it as:
- **Spam** (unwanted emails: ads, phishing, scams)
- **Ham** (legitimate emails: work, personal, newsletters)

**Accuracy Target:** 92-95%

---

## 🤔 Why This Project?

### Real-World Relevance

As a **.NET developer**, you've probably worked with:
- Email systems
- Message filtering
- Content moderation
- User input validation

This project shows you how **machine learning** solves these problems!

### Perfect First Project

✅ **Easy to understand** - Binary classification (spam vs not spam)
✅ **Fast to train** - Results in ~10 seconds
✅ **Text data** - Prepares you for LLMs!
✅ **Practical** - You use spam filters every day

---

## 📊 What You'll Learn

### From Module 3 (Applied)

| Module 3 Concept | How It's Used |
|------------------|---------------|
| **Perceptrons** | Base neurons in network |
| **Sigmoid Activation** | Output layer (0-1 probability) |
| **ReLU Activation** | Hidden layer |
| **Backpropagation** | Training algorithm |
| **Adam Optimizer** | Fast convergence |
| **Binary Cross-Entropy** | Loss function |

### New Skills

✅ **Text Preprocessing**
- Tokenization (splitting text into words)
- Bag of Words model
- Feature extraction from text

✅ **Binary Classification**
- Probability thresholds
- Precision vs Recall
- Confusion matrix

✅ **Real Data Handling**
- Loading CSV files
- Train/test split
- Data normalization

---

## 🏗️ Network Architecture

```
Input Layer          Hidden Layer         Output Layer
  (1000)         →      (64)          →      (1)
  [words]              [ReLU]              [Sigmoid]

Example:
"Buy now!" → [0,0,1,0,0...1,0] → [neurons] → 0.95 (SPAM!)
"Meeting at 3pm" → [0,1,0,0,1...0,0] → [neurons] → 0.05 (HAM!)
```

**Architecture:**
- **Input:** 1000 most common words (bag of words)
- **Hidden:** 64 neurons with ReLU
- **Output:** 1 neuron with Sigmoid (spam probability)

---

## 📈 Expected Results

### Training Metrics
```
Epoch 1:  Loss: 0.452, Accuracy: 78%
Epoch 10: Loss: 0.201, Accuracy: 92%
Epoch 20: Loss: 0.142, Accuracy: 94%
Epoch 30: Loss: 0.125, Accuracy: 95%
```

### Test Performance
```
Test Accuracy: 93.5%
Precision: 91.2% (when it says spam, it's usually right)
Recall: 89.8% (catches most spam)
```

### Example Predictions
```
Email: "Congratulations! You won $1000000!"
Prediction: SPAM (99.8% confidence) ✅

Email: "Meeting notes from today's standup"
Prediction: HAM (2.3% spam probability) ✅

Email: "URGENT: Click here to verify your account"
Prediction: SPAM (87.5% confidence) ✅
```

---

## 🗂️ Files in This Project

```
email_spam_classifier/
├── README.md                    ← You are here!
├── GETTING_STARTED.md          ← Step-by-step guide
├── project_simple.py           ← Start here (200 lines)
├── project_main.py             ← Complete version (400 lines)
├── EXPLANATION.md              ← Line-by-line breakdown
├── CONCEPTS.md                 ← Key concepts explained
├── data/
│   ├── emails.csv              ← Training data (5,000 emails)
│   └── sample_emails.txt       ← Test your own emails
└── results/
    ├── training_curve.png      ← Loss and accuracy over time
    ├── confusion_matrix.png    ← Visualization of predictions
    └── results.txt             ← Final metrics
```

---

## 🚀 Quick Start

### Option 1: Run Simple Version (Recommended First!)

```bash
# Navigate to project
cd projects/neural_networks/email_spam_classifier

# Run simple version
python project_simple.py
```

**You'll see:**
```
Loading data...
Loaded 5000 emails (2500 spam, 2500 ham)

Building vocabulary...
Vocabulary size: 1000 words

Creating bag-of-words features...
Training set: 4000 emails
Test set: 1000 emails

Building neural network...
Input: 1000 → Hidden: 64 → Output: 1

Training...
Epoch 10/30: Loss = 0.201, Accuracy = 92.1%
Epoch 20/30: Loss = 0.142, Accuracy = 94.3%
Epoch 30/30: Loss = 0.125, Accuracy = 95.1%

Testing...
Test Accuracy: 93.5%

Done! Check results/ folder for plots.
```

### Option 2: Run Complete Version

```bash
python project_main.py
```

**Additional features:**
- More detailed logging
- Better preprocessing
- Hyperparameter tuning
- More visualizations
- Custom email testing

---

## 🎓 How It Works

### Step 1: Text → Numbers

Neural networks need numbers, not text!

**Bag of Words Approach:**

```
Email: "Buy cheap pills now!"

Step 1: Tokenize
→ ["buy", "cheap", "pills", "now"]

Step 2: Create vocabulary (top 1000 words)
→ {buy: 45, cheap: 201, pills: 567, now: 12, ...}

Step 3: Create feature vector
→ [0,0,0,0,0,0,0,0,0,0,0,1,0,0...,1,0,0,0...,1,...,1,0,0]
    position 12=1 (now)
    position 45=1 (buy)
    position 201=1 (cheap)
    position 567=1 (pills)
```

### Step 2: Neural Network

```python
# Forward pass
z1 = X @ W1 + b1           # Linear transformation
a1 = relu(z1)              # Hidden layer activation
z2 = a1 @ W2 + b2          # Output linear transformation
y_pred = sigmoid(z2)        # Probability (0 to 1)

# If y_pred > 0.5 → SPAM
# If y_pred < 0.5 → HAM
```

### Step 3: Training

```python
# Backpropagation (Module 3, Lesson 4!)
for epoch in range(30):
    # Forward pass
    predictions = network.forward(X_train)

    # Compute loss
    loss = binary_cross_entropy(y_train, predictions)

    # Backward pass
    gradients = network.backward(y_train, predictions)

    # Update weights (Adam optimizer)
    optimizer.update(gradients)
```

---

## 🔍 Understanding the Output

### Confusion Matrix

```
                Predicted
              Spam    Ham
Actual Spam   450     50     (90% recall - caught 450/500 spam)
       Ham     40    510     (92.7% precision - 450/(450+40) correct)
```

**Reading it:**
- **True Positives (450):** Correctly identified spam
- **False Negatives (50):** Spam that got through ❌
- **False Positives (40):** Ham marked as spam ❌
- **True Negatives (510):** Correctly identified ham

### Metrics Explained

**Accuracy = (450 + 510) / 1000 = 96%**
- Overall correctness

**Precision = 450 / (450 + 40) = 91.8%**
- When it says spam, how often is it right?
- High precision = few false alarms

**Recall = 450 / (450 + 50) = 90%**
- Of all spam, how much did we catch?
- High recall = few spam emails get through

**F1 Score = 2 × (Precision × Recall) / (Precision + Recall) = 90.9%**
- Balance between precision and recall

---

## 🎯 Connection to Module 3

### Lesson 1: Perceptrons
- Each neuron is a perceptron
- Learns word patterns that indicate spam

### Lesson 2: Activation Functions
- **ReLU** in hidden layer (non-linearity)
- **Sigmoid** in output (0-1 probability)

### Lesson 3: Multi-Layer Networks
- 2-layer network (input → hidden → output)
- Shape: 1000 → 64 → 1

### Lesson 4: Backpropagation
- Computes gradients for all 64,001 parameters!
- Updates weights to minimize classification error

### Lesson 5: Training Loop
- Mini-batch training (32 emails at a time)
- Train/test split
- Monitoring loss and accuracy

### Lesson 6: Optimizers
- Adam optimizer for fast convergence
- Learning rate: 0.001
- Much faster than vanilla SGD!

---

## 🧪 Experiment Ideas

Once you understand the basic project, try:

### 1. Change Architecture
```python
# Try different hidden layer sizes
hidden = 32   # Smaller (faster, less accurate?)
hidden = 128  # Larger (slower, more accurate?)
```

### 2. Tune Hyperparameters
```python
learning_rate = 0.01   # Faster but less stable?
learning_rate = 0.0001 # Slower but more stable?

epochs = 50  # Train longer?
batch_size = 16  # Smaller batches?
```

### 3. Different Vocabulary Sizes
```python
vocab_size = 500   # Fewer features
vocab_size = 2000  # More features
```

### 4. Test Your Own Emails
```python
test_email = "Your custom email text here"
prediction = network.predict(test_email)
print(f"Spam probability: {prediction:.2%}")
```

---

## ❓ FAQ

**Q: Why only 1000 words?**
A: It's enough to capture patterns! Top 1000 words cover ~90% of vocabulary. More words = slower training.

**Q: What if a word isn't in vocabulary?**
A: It's ignored. The 1000 most common words are usually enough.

**Q: Why bag of words (not word order)?**
A: Simpler! For spam detection, word presence matters more than order. Transformers (Module 4) will handle word order!

**Q: Can I use this in production?**
A: This is educational! Production systems use:
- Larger datasets
- More features (sender, subject, links)
- Ensemble models
- Regular retraining

**Q: How does this relate to LLMs?**
A: Great question!
- LLMs also process text
- But they understand word order (attention mechanism)
- They use embeddings (learned word vectors)
- Module 4 & 5 will show you how!

---

## 🏆 Success Criteria

You've mastered this project when you can:

✅ Explain how text becomes numbers (bag of words)
✅ Describe the network architecture (1000→64→1)
✅ Understand why sigmoid is used for output
✅ Interpret confusion matrix
✅ Tune hyperparameters to improve accuracy
✅ Test custom emails
✅ Explain the difference between precision and recall

---

## ⏭️ What's Next?

After completing this project:

1. **Try custom emails**
   - Test with your own spam/ham examples
   - See where the model fails
   - Understand limitations

2. **Move to Project 2: MNIST**
   - Work with images instead of text
   - Multi-class classification (10 digits)
   - Deeper network architecture

3. **Then Project 3: Sentiment Analysis**
   - More advanced text processing
   - Word embeddings
   - Direct path to transformers!

---

## 📚 Resources

### Related Reading
- Module 3, Lesson 1: Perceptrons
- Module 3, Lesson 4: Backpropagation
- Module 3, Lesson 6: Adam Optimizer

### Going Deeper
- Naive Bayes for text classification
- TF-IDF (alternative to bag of words)
- Word embeddings (Word2Vec)
- Recurrent networks for text
- Transformers for text (Module 4!)

---

**Ready to build your first real neural network?**

👉 **Next Step:** Open `GETTING_STARTED.md` for step-by-step instructions!

🚀 **Or jump right in:** Run `python project_simple.py`
