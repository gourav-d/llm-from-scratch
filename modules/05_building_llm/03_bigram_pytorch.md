# Lesson 3 (PyTorch): Bigram Language Model as nn.Module

**File:** `examples/example_03_bigram_pytorch.py`

---

## Why This File Exists

The NumPy version (`example_03_nanogpt.py`) built a bigram model by **counting** character pairs.
This PyTorch version builds the **same idea** but now the model **learns** the patterns through gradient descent.

Same concept. Different engine.

```
NumPy bigram   -->  counts rows in a table manually
PyTorch bigram -->  trains a table using backpropagation
                    (same table, but LEARNED not counted)
```

The critical insight: **the structure of the PyTorch version scales directly to GPT**.
Add more layers on top -> GPT.
The bigram is the foundation.

---

## The Methods That Need Deep Explanation

Every method/concept in the file that is non-obvious, explained fully.

---

### 1. `nn.Module` - The Base Class for Every PyTorch Model

```python
class BigramModel(nn.Module):
```

**What it is:**
Every neural network you build in PyTorch **must** inherit from `nn.Module`.
It wires up the bookkeeping: tracking parameters, enabling `.backward()`, saving/loading weights, etc.

**C# analogy:**
```csharp
// C# -- you extend an abstract base class
public class BigramModel : NeuralNetworkBase
{
    public BigramModel() : base() { }   // must call base constructor
    public override Tensor Forward(Tensor input) { ... }
}
```

```python
# Python -- same idea
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()   # must call parent constructor
        ...
    def forward(self, idx, targets=None):
        ...
```

**Why `super().__init__()` is mandatory:**
`nn.Module.__init__()` sets up internal PyTorch state (parameter registry, hooks, etc.).
If you skip it, `model.parameters()` returns nothing and training breaks silently.
Think of it as mandatory plumbing -- always call it, always first.

---

### 2. `forward()` - How You Define What the Model Does

```python
def forward(self, idx, targets=None):
    logits = self.table(idx)
    ...
    return logits, loss
```

**What it is:**
`forward()` defines the **computation** of your model.
You define it. PyTorch calls it automatically when you do `model(x)`.

```python
# You write:
model(x_train, y_train)

# PyTorch internally calls:
model.forward(x_train, y_train)
```

**C# analogy:**
```csharp
// Operator overloading -- defining what () does on your object
public (Tensor logits, Tensor loss) Invoke(Tensor idx, Tensor targets) { ... }
```

**Why `targets=None`?**
- During training: pass targets -> get loss back
- During generation: no targets (you don't know the next token yet) -> just get logits back

`None` is Python's `null`. The `if targets is not None:` check handles both cases.

---

### 3. `nn.Embedding` - A Learnable Lookup Table

```python
self.table = nn.Embedding(vocab_size, vocab_size)
```

**What it is:**
A matrix of shape `(rows, columns)` where you look up a **row by integer index**.
One integer in -> one row of floats out.

```
nn.Embedding(4, 4) creates this table (randomly initialized at start):

         col0   col1   col2   col3
row 0: [ 0.12,  -0.4,   0.8,  0.33 ]   <- scores if current token is 'e' (id=0)
row 1: [ 0.55,   0.2,  -0.1,  0.77 ]   <- scores if current token is 'h' (id=1)
row 2: [ -0.3,   0.9,   0.4, -0.5  ]   <- scores if current token is 'l' (id=2)
row 3: [ 0.21,  -0.7,   0.3,  0.88 ]   <- scores if current token is 'o' (id=3)

Looking up token id=2 ('l') returns: [ -0.3,  0.9,  0.4, -0.5 ]
These 4 values are the SCORES for the next token being e, h, l, or o.
Training adjusts these scores until they reflect real patterns in the text.
```

**Why Embedding and not Linear?**

| | `nn.Linear` | `nn.Embedding` |
|---|---|---|
| Input type | Float vector | Integer index |
| Operation | Matrix multiply | Row lookup |
| Use case | General layers | Token lookup tables |

Since a token is just an integer (not a float vector), `Embedding` is the right tool.

**C# analogy:**
```csharp
float[,] table = new float[vocabSize, vocabSize];
float[] scores = GetRow(table, currentTokenId);  // row lookup by integer
```

**How training changes it:**
- Start: random values in every cell
- After training: row for 'h' has high score in 'e' column, row for 'e' has high score in 'l' column, etc.
- Gradient descent adjusts these values automatically

---

### 4. `F.cross_entropy` - The Loss Function

```python
loss = F.cross_entropy(logits, targets)
```

**What it is:**
Measures how wrong the model's predictions are.
Smaller loss = model is more confident about the right answer.

**What it expects:**
```
logits  : shape (N, C)   -->  N examples, C classes (vocab_size)
targets : shape (N,)     -->  correct class index for each example

Example with 4 training pairs and vocab size 4:
  logits  shape: (4, 4)
  targets shape: (4,)    e.g. [0, 2, 2, 3]  (e, l, l, o)
```

**What it does internally (3 steps):**
```
Step 1: softmax(logits)      --> convert raw scores to probabilities (0.0 to 1.0)
Step 2: -log(prob[correct])  --> penalize low confidence on correct answer
Step 3: average over N       --> single scalar loss value
```

**Intuition:**
```
Model says prob=0.9 for correct token  -->  loss = -log(0.9) = 0.10   (low,  good)
Model says prob=0.1 for correct token  -->  loss = -log(0.1) = 2.30   (high, bad)
```

You don't write this yourself. PyTorch provides it. Rule: **lower loss = better model**.

---

### 5. The Training Trio: `zero_grad` / `backward` / `step`

```python
optimizer.zero_grad()    # 1. clear old gradients
loss.backward()          # 2. compute new gradients
optimizer.step()         # 3. update weights
```

This is the **core PyTorch training loop**. You will write these 3 lines in every model you ever build.

---

**Step 1: `optimizer.zero_grad()` - Clear Old Gradients**

```
Problem: PyTorch ACCUMULATES gradients by default.
If you skip this, gradient from step 5 adds on top of step 6, and so on.
Result: exploding gradients, garbage training.

Fix: wipe the slate clean before each new step.
```

C# analogy: `totalRevenue = 0;` before summing a new batch.

---

**Step 2: `loss.backward()` - Compute All Gradients Automatically**

```
PyTorch records every operation that produced 'loss' (builds a graph).
.backward() walks that graph in reverse and computes:
  dLoss/dWeight  for every single parameter in the model.

You write ONE line. PyTorch does all the calculus.
```

C# analogy: imagine a compiler that auto-generates derivative code for you.

---

**Step 3: `optimizer.step()` - Update the Weights**

```
Uses the gradients from step 2 to nudge every weight slightly.
Direction: opposite to gradient (go downhill toward lower loss).
Size: controlled by learning rate (lr).

For SGD:
  weight = weight - lr * gradient
```

C# analogy: `weight -= learningRate * weight.Gradient;`

---

**Visual - one complete training step:**

```
  [data: x_train, y_train]
          |
          v
  model.forward(x, y)       --> predictions + loss value
          |
          v
  loss.backward()            --> fill .grad on every parameter
          |
          v
  optimizer.step()           --> weight = weight - lr * weight.grad
          |
          v
  optimizer.zero_grad()      --> clear .grad ready for next step
          |
          v
  repeat 200 times
```

---

### 6. `@torch.no_grad()` - Decorator That Turns Off Gradient Tracking

```python
@torch.no_grad()
def generate(self, start_idx, num_chars, idx_to_char):
```

**What it is:**
A decorator that tells PyTorch: **do not record operations for backpropagation** inside this function.

**Why needed:**
During training, PyTorch records every operation to build the gradient graph (costs memory + compute).
During generation, you don't need gradients -- just running the model forward.
`@torch.no_grad()` skips that recording --> faster and uses less memory.

**C# analogy:**
```csharp
[ReadOnly]  // attribute signalling: no write-back needed
public string Generate(int startIdx, int numChars) { ... }
```

**Rule of thumb:** Any code that is NOT training -> wrap in `no_grad`.

---

### 7. `F.softmax(logits, dim=-1)` - Converting Scores to Probabilities

```python
probs = F.softmax(logits, dim=-1)
```

**What it is:**
Takes raw scores (any positive/negative numbers) -> converts to probabilities (all positive, sum = 1.0).

```
logits: [2.0,  1.0,  0.5, -1.0]
                |
                v  softmax
probs:  [0.60, 0.22, 0.14, 0.05]   <-- all positive, sum = 1.0
```

**Formula:**
```
prob[i] = exp(logit[i]) / sum(exp(all logits))
```

**What `dim=-1` means:**
`dim=-1` = apply softmax along the last dimension.

```
logits shape: (batch_size, vocab_size)
                                ^
                           last dimension = dim=-1

Softmax is computed across vocab_size for each row.
Each row becomes its own probability distribution.
```

C# analogy:
```csharp
float[] Softmax(float[] logits)
{
    float[] exp = logits.Select(x => (float)Math.Exp(x)).ToArray();
    float sum = exp.Sum();
    return exp.Select(x => x / sum).ToArray();
}
```

---

### 8. `torch.multinomial` - Weighted Random Sampling

```python
next_idx = torch.multinomial(probs, num_samples=1)
```

**What it is:**
Picks one (or more) random index from a probability distribution.
Higher probability = more likely to be picked. But it is **random** -- not always the highest.

Equivalent to:
```python
np.random.choice([0, 1, 2, 3], p=[0.60, 0.22, 0.14, 0.05])
```

**Why not just pick the maximum probability?**
Always picking max = same output every time = boring, repetitive text.
Sampling = variety. Model picks 'h->e' 60% of the time but occasionally 'h->l'.
This is how GPT generates diverse text.

**C# analogy:**
```csharp
int WeightedRandom(double[] probs)
{
    double roll = new Random().NextDouble();
    double cumulative = 0;
    for (int i = 0; i < probs.Length; i++) {
        cumulative += probs[i];
        if (roll < cumulative) return i;
    }
    return probs.Length - 1;
}
```

---

### 9. `torch.cat([current, next_idx], dim=1)` - Concatenating Tensors

```python
current = torch.cat([current, next_idx], dim=1)
```

**What it is:**
Joins tensors along a dimension. Like `string.Concat` but for tensors.

**In generation context:**
```
Before:
  current  shape = (1, 3)   --> batch of 1, sequence of 3 tokens
  next_idx shape = (1, 1)   --> batch of 1, 1 new token

After:
  current  shape = (1, 4)   --> sequence grew by 1
```

**`dim=1` means "join along columns" (the sequence length dimension):**
```
current  = [[2, 0, 1]]    shape (1, 3)
next_idx = [[1]]          shape (1, 1)
                |
                v  dim=1 concat
result   = [[2, 0, 1, 1]] shape (1, 4)
```

C# analogy: `sequence.Add(nextTokenId);`

---

### 10. `current[:, -1]` - Tensor Slice Notation

```python
logits, _ = self(current[:, -1])
```

**Breaking it down:**
```
current[:, -1]
       |    |
       |    +--  column index -1 = last column = last token in sequence
       +-------  :  means "all rows" (entire batch)

If current = [[2, 0, 1, 1]]  shape (1, 4)
current[:, -1] = [1]          shape (1,)  <-- just the last token
```

**Why only the last token?**
Bigram model only looks at the **current** token to predict the next.
No memory of history -- just: "what token am I on now?" -> "what comes next?"

C# analogy: `int lastToken = sequence[^1];`  (C# 8+ index-from-end syntax)

---

### 11. `.item()` - Extracting a Python Number from a Tensor

```python
src = idx_to_char[x_train[i].item()]
```

**What it is:**
A tensor holding a single number is NOT a Python int/float.
`.item()` extracts the raw Python value from a single-element tensor.

```
x_train[0]        --> tensor(2)    PyTorch tensor object
x_train[0].item() --> 2            Python int   (usable as dict key)

loss              --> tensor(1.32) PyTorch tensor object
loss.item()       --> 1.32         Python float (printable, loggable)
```

**Why needed here:**
`idx_to_char` is a Python dict. Dict lookup needs a Python int. A tensor object won't work as a key.

C# analogy: `int value = (int)boxedObject;`  (unboxing)

---

### 12. `p.numel()` - Count Elements in a Tensor

```python
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**What it is:**
`numel()` = "number of elements". Returns total count of values in a tensor.

```
tensor shape (4, 4)  -->  numel() = 16
tensor shape (3,)    -->  numel() = 3
```

**Full line explained piece by piece:**
```python
model.parameters()          # iterate over all parameter tensors in the model
if p.requires_grad          # skip frozen (non-trainable) parameters
p.numel()                   # count elements in this parameter tensor
sum(...)                    # add them all up -> total trainable parameter count
```

For our bigram with vocab_size=4: one 4x4 table = **16 parameters** total.

C# analogy:
```csharp
int totalParams = model.Parameters
    .Where(p => p.RequiresGrad)
    .Sum(p => p.TotalElements);
```

---

### 13. SGD vs AdamW - Two Different Optimizers

**Part A uses SGD:**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

**Part B uses AdamW:**
```python
optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=1e-2)
```

**SGD (Stochastic Gradient Descent):**
```
Simplest optimizer.
Formula: weight = weight - lr * gradient
Same learning rate for every parameter.
Good for learning the concept. Slow to converge on real data.
```

**AdamW:**
```
Smarter optimizer. Tracks momentum + per-parameter learning rates.
Adapts: parameters that change a lot get smaller steps,
        parameters that haven't moved get bigger steps.
Converges faster. Standard choice for language models.
The 'W' = weight decay (mild regularization to prevent overfitting).
```

**When to use which:**

| Use SGD | Use AdamW |
|---------|-----------|
| Simple demos, learning concepts | Real training runs |
| Understanding how gradients work | Language models |
| Small toy models | Anything GPT-sized |

C# analogy:
```
SGD   = fixed monthly deposit into savings: same amount every time
AdamW = smart investment account: auto-rebalances based on recent history
```

---

## The Complete Data Flow (Visual)

```
text = "hello"
         |
         v
  [tokenize]
  h=1, e=0, l=2, l=2, o=3

  x_train = [1, 0, 2, 2]   (inputs:  all but last char)
  y_train = [0, 2, 2, 3]   (targets: all but first char)
         |
         v
  BigramModel.forward(x_train, y_train)
    |
    +-- self.table(x_train)
    |       = nn.Embedding lookup
    |       = for each token id, return its row from the table
    |       = output shape (4, 4)  [4 tokens, 4 vocab scores each]
    |
    +-- F.cross_entropy(logits, y_train)
            = how wrong are the scores vs correct answers?
            = single scalar loss value
         |
         v
  optimizer.zero_grad()   <-- clear old gradients
  loss.backward()         <-- compute: dLoss/dEachWeight
  optimizer.step()        <-- weight -= lr * gradient

  Repeat 200 times
  --> table now has high score in h->e, e->l, l->l, l->o cells
```

---

## The Generation Flow (Visual)

```
start token: 'h'  (id=1)

  current = [[1]]   shape (1, 1)
       |
       v
  self(current[:, -1])
  = self.table(1)             <-- look up row 1 in embedding table
  = [0.8, -0.2, 0.3, 0.1]    <-- scores for e, h, l, o
       |
       v
  F.softmax(...)
  = [0.52, 0.09, 0.29, 0.10]  <-- probabilities (sum to 1.0)
       |
       v
  torch.multinomial(...)
  = picks index 0 ('e') with 52% probability
       |
       v
  torch.cat --> current = [[1, 0]]   sequence grows to length 2
       |
       v
  repeat --> generates 'hello' or similar sequence
```

---

## Quick Reference Card

| Method / Syntax | What it does | C# analogy |
|---|---|---|
| `nn.Module` | Base class for all models | Abstract base class |
| `super().__init__()` | Required parent init | `: base()` |
| `forward(x)` | Defines model computation | Override `Invoke()` |
| `nn.Embedding(V, V)` | Learnable row-lookup table | `float[,]` lookup |
| `F.cross_entropy(l, t)` | Classification loss (softmax + NLL) | Custom loss function |
| `zero_grad()` | Clear accumulated gradients | Reset accumulator to 0 |
| `loss.backward()` | Compute all gradients | Auto-differentiation |
| `optimizer.step()` | Update weights | `weight -= lr * grad` |
| `@torch.no_grad()` | Disable gradient tracking | `[ReadOnly]` attribute |
| `F.softmax(x, dim=-1)` | Scores -> probabilities | Custom `Softmax()` |
| `torch.multinomial(p, 1)` | Weighted random pick | `WeightedRandom()` |
| `torch.cat([a, b], dim=1)` | Join tensors along columns | `list.Add()` / concat |
| `x[:, -1]` | Last token of each sequence | `list[^1]` (C# 8+) |
| `.item()` | Tensor -> Python number | Unbox / cast |
| `.numel()` | Count elements in tensor | `.Length` / `.Count` |
| `SGD` | Simple gradient descent | Fixed monthly deposit |
| `AdamW` | Adaptive optimizer | Smart investment account |

---

## Why This Pattern Scales to GPT

The bigram model does:
```
token  -->  embedding lookup  -->  scores for next token
```

GPT does:
```
token  -->  embedding  -->  attention layers  -->  feedforward layers  -->  scores for next token
```

Same structure. Same training loop (`zero_grad / backward / step`). Same generation loop (`softmax + multinomial`).
The bigram is GPT with all the middle layers removed.

**That is why we start here.**

---

## Next File

`example_04_gpt_pytorch.py` -- adds self-attention and feedforward layers on top of this same foundation.