# Code Explanation: Email Spam Classifier

**Line-by-line breakdown of how the code works**

---

## 📚 File Structure

```
project_simple.py   ← Start here (400 lines, easier)
project_main.py     ← Advanced version (700 lines, more features)
```

This document explains **both versions**, noting differences where applicable.

---

## 🎯 Overall Flow

```
1. Load emails from CSV
2. Build vocabulary (top 1000 words)
3. Convert text → numbers (bag of words)
4. Split into train/val/test
5. Build neural network
6. Train with backpropagation + Adam
7. Evaluate on test set
8. Visualize results
```

---

## PART 1: Data Loading

### Simple Version: `load_data()`

**Lines 40-75 in project_simple.py**

```python
def load_data(filepath='data/emails.csv'):
```

**What it does:**
- Loads email data from CSV file
- Returns emails (list of strings) and labels (NumPy array)

**Line-by-line:**

```python
emails = []
labels = []
```
- **C# equivalent:** `var emails = new List<string>(); var labels = new List<int>();`
- Initialize empty lists to store data

```python
with open(filepath, 'r', encoding='utf-8') as f:
```
- **C# equivalent:** `using var reader = new StreamReader(filepath, Encoding.UTF8)`
- Opens file for reading with UTF-8 encoding
- `with` ensures file is closed even if error occurs

```python
next(f)  # Skip header
```
- Skips first line (CSV header: "text,label")
- **C# equivalent:** `reader.ReadLine(); // skip header`

```python
for line in f:
```
- Iterate through each line in file
- **C# equivalent:** `while ((line = reader.ReadLine()) != null)`

```python
parts = line.strip().split(',', 1)
```
- `line.strip()`: Remove whitespace/newlines
- `split(',', 1)`: Split on FIRST comma only (email text may contain commas!)
- **Why maxsplit=1?** Email could be: "Hello, how are you?,spam"
- Results in: `["Hello, how are you?", "spam"]`

```python
text = text.strip('"')
```
- Remove quote marks from CSV formatting
- "Buy pills" → Buy pills

```python
labels.append(1 if label == 'spam' else 0)
```
- Convert text label to number: spam=1, ham=0
- **C# equivalent:** `labels.Add(label == "spam" ? 1 : 0);`

### Advanced Version: `load_data_from_csv()` + `preprocess_text()`

**Lines 35-120 in project_main.py**

**Key difference:** Adds text preprocessing!

```python
def preprocess_text(text):
    text = text.lower()
```
- Convert to lowercase: "BUY" → "buy"
- **Why?** "Free", "free", "FREE" should be same word!

```python
text = re.sub(r'[^a-z\s]', ' ', text)
```
- Remove everything except letters and spaces
- **Regex pattern:** `[^a-z\s]` means "not letter or space"
- "Buy pills!!!" → "Buy pills"
- "Price: $99.99" → "Price"
- **C# equivalent:** `Regex.Replace(text, @"[^a-z\s]", " ")`

```python
text = ' '.join(text.split())
```
- Remove extra whitespace
- "Buy    cheap   pills" → "Buy cheap pills"
- **Trick:** `split()` splits on any whitespace, `join(' ')` puts single spaces back
- **C# equivalent:** `string.Join(" ", text.Split(' ', StringSplitOptions.RemoveEmptyEntries))`

---

## PART 2: Building Vocabulary

### `build_vocabulary()`

**Lines 140-170 in project_simple.py**

**What it does:**
- Finds the 1000 most common words across all emails
- Creates word→index mapping

```python
word_counts = Counter()
```
- **Counter** is like a dictionary that counts things
- **C# equivalent:** `var wordCounts = new Dictionary<string, int>();`

```python
for email in emails:
    words = email.lower().split()
    word_counts.update(words)
```
- For each email:
  - Split into words
  - Count each word
- **After this:**
  ```
  word_counts = {
    'the': 2453,
    'to': 1876,
    'free': 987,
    'buy': 654,
    ...
  }
  ```

```python
most_common = word_counts.most_common(vocab_size)
```
- Get top 1000 words by frequency
- Returns: `[('the', 2453), ('to', 1876), ...]`
- **C# equivalent:** `wordCounts.OrderByDescending(x => x.Value).Take(1000)`

```python
vocabulary = {word: idx for idx, (word, count) in enumerate(most_common)}
```
- **This is a dictionary comprehension!**
- Creates: `{'the': 0, 'to': 1, 'free': 2, ...}`
- **C# equivalent:**
  ```csharp
  var vocabulary = mostCommon
      .Select((pair, idx) => new { pair.Key, idx })
      .ToDictionary(x => x.Key, x => x.idx);
  ```
- **Why enumerate?** Gives us index: `enumerate(['a','b','c'])` → `[(0,'a'), (1,'b'), (2,'c')]`

---

## PART 3: Text → Numbers

### `email_to_features()`

**Lines 175-200 in project_simple.py**

**What it does:**
- Converts email text to feature vector (bag of words)
- Each feature is 1 if word present, 0 otherwise

```python
features = np.zeros(len(vocabulary))
```
- Create array of 1000 zeros
- **C# equivalent:** `var features = new double[vocabularySize];` (defaults to 0)

```python
words = email.lower().split()
```
- Tokenize email into words
- "Buy cheap pills" → ["buy", "cheap", "pills"]

```python
for word in words:
    if word in vocabulary:
        idx = vocabulary[word]
        features[idx] = 1
```
- For each word in email:
  - Check if it's in our vocabulary
  - If yes, set corresponding feature to 1
- **Example:**
  ```
  Email: "free pills"
  vocabulary = {'the': 0, 'free': 1, 'buy': 2, 'pills': 3, ...}

  After processing:
  features = [0, 1, 0, 1, 0, 0, 0, ...]
              ↑  ↑     ↑
             the free  pills
  ```

### Advanced: TF-IDF (project_main.py only)

**Lines 210-245 in project_main.py**

**Improvement:** Instead of binary (0 or 1), use TF-IDF weights

```python
tf = count / len(words)
```
- **Term Frequency:** How often word appears in THIS email
- If "free" appears 3 times in 10-word email: tf = 3/10 = 0.3

```python
idf = np.log((num_docs + 1) / (word_frequencies.get(word, 0) + 1)) + 1
```
- **Inverse Document Frequency:** How rare is this word across ALL emails?
- **Intuition:**
  - Common word ("the"): appears in most documents → low IDF
  - Rare word ("viagra"): appears in few documents → high IDF
- **Formula:** `log((total_docs + 1) / (docs_with_word + 1)) + 1`

```python
features[idx] = tf * idf
```
- **TF-IDF = TF × IDF**
- High value if:
  - Word appears often in THIS email (high TF)
  - Word is rare across ALL emails (high IDF)
- **Result:** "free" in spam gets higher weight than "the" in any email

---

## PART 4: Neural Network

### `class SpamClassifier`

**Lines 210-300 in project_simple.py**

#### `__init__()` - Initialize Weights

```python
def __init__(self, input_size, hidden_size):
```
- **input_size:** 1000 (vocabulary size)
- **hidden_size:** 64 (number of neurons in hidden layer)

```python
self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
```
- **Shape:** (1000, 64)
- **Random initialization:** Small random numbers
- **He initialization:** Multiply by `sqrt(2 / input_size)`
- **Why?** Prevents vanishing/exploding gradients with ReLU
- **C# equivalent:**
  ```csharp
  var W1 = new double[inputSize, hiddenSize];
  var random = new Random();
  for (int i = 0; i < inputSize; i++)
      for (int j = 0; j < hiddenSize; j++)
          W1[i,j] = random.NextGaussian() * Math.Sqrt(2.0 / inputSize);
  ```

```python
self.b1 = np.zeros((1, hidden_size))
```
- **Bias:** Initialize to zero
- **Shape:** (1, 64)
- **Why (1, 64) not just (64)?** Broadcasting - works with batches!

```python
self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
```
- **Second layer weights:** (64, 1)
- Maps 64 hidden neurons → 1 output

#### `forward()` - Forward Propagation

**This is Module 3, Lesson 3 in action!**

```python
def forward(self, X):
```
- **X shape:** (batch_size, 1000) - multiple emails at once

```python
z1 = X @ self.W1 + self.b1
```
- **Matrix multiplication!** (Module 2, Lesson 3)
- **@** is matrix multiply operator
- **Shape:** (batch_size, 1000) @ (1000, 64) = (batch_size, 64)
- **Broadcasting:** adds b1 to each row
- **C# equivalent:** Would need manual loop or BLAS library
- **What it does:**
  ```
  For each email (row in X):
    For each hidden neuron j:
      z1[j] = sum(X[i] * W1[i,j] for all i) + b1[j]
  ```

```python
a1 = self.relu(z1)
```
- **ReLU activation:** `max(0, z)`
- **Module 3, Lesson 2!**
- **Why?** Adds non-linearity, enables complex patterns
- **Result:** Negative values → 0, positive stays same

```python
z2 = a1 @ self.W2 + self.b2
```
- **Second layer:** (batch_size, 64) @ (64, 1) = (batch_size, 1)
- Linear transformation of hidden layer

```python
y_pred = self.sigmoid(z2)
```
- **Sigmoid:** Maps any number to 0-1 range
- **Formula:** `1 / (1 + e^(-z))`
- **Result:** Probability of spam (0 to 1)
- **Module 3, Lesson 2!**

```python
self.cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'y_pred': y_pred}
```
- **Store for backpropagation!**
- Need these values to compute gradients
- **C# equivalent:** Store as instance variables

#### `backward()` - Backpropagation

**This is Module 3, Lesson 4 - THE MOST IMPORTANT LESSON!**

```python
def backward(self, y_true):
```
- **Goal:** Compute gradients (how much to adjust each weight)

```python
dz2 = y_pred - y_true
```
- **Gradient of loss with respect to z2**
- **Magic formula!** For binary cross-entropy + sigmoid, derivative simplifies to this!
- **Intuition:**
  - If y_pred=0.9, y_true=1 → dz2=-0.1 (increase weights)
  - If y_pred=0.9, y_true=0 → dz2=+0.9 (decrease weights)

```python
dW2 = (a1.T @ dz2) / batch_size
```
- **Gradient for W2**
- **Chain rule:** dL/dW2 = dL/dz2 * dz2/dW2
- **dz2/dW2 = a1** (derivative of a1@W2 is a1)
- **.T** is transpose: (64, batch_size) @ (batch_size, 1) = (64, 1)
- **Divide by batch_size:** Average gradient over batch
- **C# equivalent:** Would need manual matrix transpose and multiply

```python
db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size
```
- **Gradient for bias b2**
- Sum over batch dimension
- **keepdims=True:** Keeps shape as (1, 1) instead of scalar

```python
da1 = dz2 @ self.W2.T
```
- **Backpropagate to hidden layer**
- **Chain rule:** dL/da1 = dL/dz2 * dz2/da1
- **dz2/da1 = W2** (derivative of a1@W2 + b2 with respect to a1 is W2)
- **Shape:** (batch_size, 1) @ (1, 64) = (batch_size, 64)

```python
dz1 = da1 * self.relu_derivative(z1)
```
- **Apply ReLU derivative**
- **relu_derivative:** 1 if z1>0, else 0
- **Element-wise multiply:** `*` (not `@`)
- **Intuition:** Gradients flow through only if ReLU was active (z1>0)

```python
dW1 = (X.T @ dz1) / batch_size
```
- **Gradient for W1**
- **Shape:** (1000, batch_size) @ (batch_size, 64) = (1000, 64)

---

## PART 5: Adam Optimizer

**Lines 315-370 in project_simple.py**

**Module 3, Lesson 6 applied!**

### Why Adam?

**Vanilla SGD:**
```python
W = W - learning_rate * gradient
```
- Simple but slow, sensitive to learning rate

**Adam:**
- Combines **momentum** (smooth out updates) + **RMSProp** (adaptive learning rates)
- Used to train GPT-3, BERT, all modern LLMs!

### Code Breakdown

```python
self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
```
- **First moment estimate (momentum)**
- **beta1=0.9:** Use 90% of previous momentum, 10% of current gradient
- **Smooth out oscillations**
- **C# equivalent:** `m[param] = 0.9 * m[param] + 0.1 * grad;`

```python
self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)
```
- **Second moment estimate (RMSProp)**
- **beta2=0.999:** Track squared gradients
- **Adaptive per-parameter learning rates**

```python
m_hat = self.m[param] / (1 - self.beta1 ** self.t)
v_hat = self.v[param] / (1 - self.beta2 ** self.t)
```
- **Bias correction**
- **Why?** m and v start at 0, biased toward 0 initially
- **Correction:** Divide by `(1 - beta^t)`
- **t=1:** divide by (1-0.9) = 0.1 → 10x correction
- **t=100:** divide by (1-0.9^100) ≈ 1 → no correction needed

```python
param_value -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```
- **Final update**
- **Divide by sqrt(v_hat):** Scale by historical gradient magnitude
- **epsilon (1e-8):** Prevent division by zero
- **Result:** Each parameter gets adaptive learning rate!

---

## PART 6: Training Loop

**Lines 390-470 in project_simple.py**

```python
for epoch in range(epochs):
```
- **Epoch:** One pass through entire training set
- **epochs=30:** Go through data 30 times

```python
indices = np.random.permutation(len(X_train))
X_train_shuffled = X_train[indices]
```
- **Shuffle data** each epoch
- **Why?** Prevents model from memorizing order
- **C# equivalent:**
  ```csharp
  var indices = Enumerable.Range(0, X_train.Length).OrderBy(x => random.Next()).ToArray();
  ```

```python
for i in range(num_batches):
    start = i * batch_size
    end = start + batch_size
    X_batch = X_train_shuffled[start:end]
```
- **Mini-batch training**
- Process 32 emails at a time instead of all 4000
- **Why?**
  - Faster (more frequent updates)
  - Better generalization
  - Fits in memory

```python
y_pred = network.forward(X_batch)
```
- **Forward pass:** Get predictions

```python
loss = binary_cross_entropy(y_batch, y_pred)
```
- **Compute loss:** How wrong are we?

```python
gradients = network.backward(y_batch)
```
- **Backpropagation:** Compute gradients

```python
optimizer.update(network, gradients)
```
- **Update weights:** Improve the model!

```python
if (epoch + 1) % 5 == 0:
    print(f"Epoch {epoch+1}/{epochs}: ...")
```
- **Print progress** every 5 epochs
- **%** is modulo operator: `epoch % 5 == 0` means epoch is multiple of 5

---

## PART 7: Evaluation

**Lines 480-530 in project_simple.py**

```python
y_pred = (y_pred_prob > 0.5).astype(int)
```
- **Convert probabilities to binary predictions**
- prob > 0.5 → spam (1)
- prob <= 0.5 → ham (0)
- **.astype(int):** Convert True/False → 1/0

```python
tp = np.sum((y_pred == 1) & (y_test == 1))
```
- **True Positives:** Correctly identified spam
- **&** is element-wise AND
- **np.sum:** Count how many are True

```python
accuracy = (tp + tn) / (tp + tn + fp + fn)
```
- **Accuracy:** Fraction correct overall

```python
precision = tp / (tp + fp)
```
- **Precision:** Of emails marked spam, how many were actually spam?
- **Important for spam:** Don't want false alarms!

```python
recall = tp / (tp + fn)
```
- **Recall:** Of actual spam, how many did we catch?

---

## 🎓 Key Python Concepts for .NET Developers

### Matrix Operations

**Python:**
```python
z = X @ W + b  # Matrix multiply
```

**C# equivalent:**
```csharp
// Would need manual implementation or library (Math.NET)
double[,] z = MatrixMultiply(X, W);
for (int i = 0; i < z.GetLength(0); i++)
    for (int j = 0; j < z.GetLength(1); j++)
        z[i,j] += b[j];
```

### List Comprehensions

**Python:**
```python
vocabulary = {word: idx for idx, (word, count) in enumerate(most_common)}
```

**C# equivalent:**
```csharp
var vocabulary = mostCommon
    .Select((pair, idx) => new { pair.Key, idx })
    .ToDictionary(x => x.Key, x => x.idx);
```

### Slicing

**Python:**
```python
X_train = X[:n_train]      # First n_train rows
X_test = X[n_train:]       # Remaining rows
```

**C# equivalent:**
```csharp
var X_train = X.Take(n_train).ToArray();
var X_test = X.Skip(n_train).ToArray();
```

### Broadcasting

**Python:**
```python
z = X @ W + b  # b (1,64) automatically broadcasts to (batch_size, 64)
```

**C# equivalent:**
```csharp
// Would need manual loop to add b to each row
```

---

## 🔗 Connection to Module 3

Every line of code connects back to Module 3!

| Code | Module 3 Lesson |
|------|-----------------|
| `z = X @ W + b` | Lesson 1: Perceptron formula |
| `a = relu(z)` | Lesson 2: ReLU activation |
| `y = sigmoid(z)` | Lesson 2: Sigmoid activation |
| `z1 = X @ W1 + b1; z2 = a1 @ W2 + b2` | Lesson 3: Multi-layer networks |
| `dW2 = ...; dW1 = ...` | Lesson 4: Backpropagation |
| `for epoch in range(epochs)` | Lesson 5: Training loop |
| `optimizer.update(...)` | Lesson 6: Adam optimizer |

---

## 💡 Common Questions

**Q: Why lowercase everything?**
A: "Free", "free", "FREE" should be treated as the same word. Lowercasing ensures consistency.

**Q: Why only 1000 words?**
A: Top 1000 words cover ~90% of vocabulary. More words = slower training, minimal accuracy gain.

**Q: Why ReLU in hidden layer but Sigmoid in output?**
A: ReLU for hidden layers (fast, prevents vanishing gradients). Sigmoid for output (gives probability 0-1).

**Q: Why mini-batches instead of full batch?**
A: Faster convergence, better generalization, fits in memory.

**Q: Why shuffle data each epoch?**
A: Prevents memorizing order, improves generalization.

**Q: Why Adam instead of SGD?**
A: Faster convergence, less sensitive to learning rate, better for sparse data (like text).

---

**You now understand every line of code! 🎉**

Try modifying parameters and see what happens!
