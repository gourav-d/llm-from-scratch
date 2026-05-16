"""
Example 7b: How Do We Choose Hidden Layer Size?
================================================

QUESTIONS THIS FILE ANSWERS:
  1. Why did we use 4 hidden neurons for XOR — why not 1, 2, 3, 8?
  2. What happens if we use too few or too many?
  3. Can we add more hidden layers?
  4. Is there a formula or rule?

SHORT ANSWER FIRST:
  Input size  = fixed by your DATA     (XOR has 2 inputs → always 2)
  Output size = fixed by your TASK     (binary → always 1)
  Hidden size = YOUR CHOICE — it is a hyperparameter

  There is no perfect formula.
  But there are rules of thumb, and we can EXPERIMENT to find what works.
  That is what this file does — try many sizes, show what happens.

C# ANALOGY:
  Think of hidden neurons like worker threads in a thread pool.
  Too few workers  → job takes too long, some tasks dropped (underfitting)
  Too many workers → overhead slows things down, waste of memory (overfitting)
  Just right       → fast, efficient, correct results

Run: python example_07b_hidden_layer_sizing.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────────────────────
# XOR data (same as example_07)
# ─────────────────────────────────────────────────────────────
INPUTS = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
LABELS = torch.FloatTensor([[0],[1],[1],[0]])


# ─────────────────────────────────────────────────────────────
# HELPER: build a model with ANY hidden size and ANY depth
# ─────────────────────────────────────────────────────────────

def build_model(hidden_sizes):
    """
    Dynamically build a model given a list of hidden layer sizes.

    hidden_sizes examples:
      []        → no hidden layer   (just input→output directly)
      [4]       → one hidden layer  of 4 neurons
      [4, 4]    → two hidden layers of 4 neurons each
      [8, 4, 2] → three hidden layers getting smaller

    C# analogy: like building a pipeline with variable number of middleware steps.

    Uses nn.Sequential — stacks layers in order automatically.
    Think of it like chaining .Where().Select().GroupBy() in LINQ.
    """
    layers = []                # empty list, will fill with layers
    prev_size = 2              # input always has 2 neurons (XOR has 2 inputs)

    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_size, hidden_size))   # weight layer
        layers.append(nn.ReLU())                            # activation
        prev_size = hidden_size                             # next layer starts here

    layers.append(nn.Linear(prev_size, 1))                 # output layer (always 1 for binary)

    return nn.Sequential(*layers)   # *layers unpacks list into individual arguments
    # nn.Sequential wraps all layers into one model automatically


def count_params(model):
    """Count total trainable numbers (weights + biases) in model."""
    return sum(p.numel() for p in model.parameters())


def train_and_score(model, epochs=5000, lr=0.05):
    """
    Train model on XOR data.
    Returns: (correct_count out of 4, final_loss)
    """
    torch.manual_seed(42)    # same starting weights every run for fair comparison

    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Re-initialize weights with same seed (fair comparison)
    def reset_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    model.apply(reset_weights)

    model.train()
    for _ in range(epochs):
        pred = model(INPUTS)
        loss = loss_fn(pred, LABELS)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred    = model(INPUTS)
        probs   = torch.sigmoid(pred)
        preds   = (probs > 0.5).float()
        correct = (preds == LABELS).sum().item()
        final_loss = loss_fn(pred, LABELS).item()

    return int(correct), final_loss, probs


# ═════════════════════════════════════════════════════════════
# EXPERIMENT 1: Vary hidden layer SIZE (keep depth = 1 layer)
# ═════════════════════════════════════════════════════════════

print("=" * 60)
print("EXPERIMENT 1: Different hidden layer SIZES (1 hidden layer)")
print("=" * 60)
print("""
We keep exactly ONE hidden layer.
We change only the number of neurons inside it: 1, 2, 3, 4, 8, 16.
Watch what happens to accuracy and loss.
""")

print(f"  {'Hidden':>8} | {'Params':>7} | {'Correct':>8} | {'Loss':>8} | {'Result'}")
print(f"  {'Neurons':>8} | {'(total)':>7} | {'(/4)':>8} | {'':>8} |")
print("  " + "─" * 55)

sizes_to_try = [1, 2, 3, 4, 6, 8, 16]

for size in sizes_to_try:
    model   = build_model([size])           # one hidden layer of `size` neurons
    params  = count_params(model)
    correct, loss, _ = train_and_score(model)
    result  = "SOLVED" if correct == 4 else f"FAILED ({correct}/4 right)"
    print(f"  {size:>8} | {params:>7} | {correct:>5}/4   | {loss:>8.4f} | {result}")

print("""
OBSERVATIONS:
  hidden=1  → almost always fails. 1 neuron = 1 straight line = not enough.
  hidden=2  → sometimes works, sometimes gets stuck. Minimum needed for XOR.
  hidden=3+ → reliably works. Extra neurons give the model "room to think".
  hidden=16 → works, but overkill for 4 data points. Wastes memory.

  KEY INSIGHT:
  More neurons = more capacity = can learn more complex patterns.
  But too many neurons for simple data = model memorizes instead of learning.
  For XOR with only 4 examples, 4-8 neurons is the sweet spot.
""")


# ═════════════════════════════════════════════════════════════
# EXPERIMENT 2: Vary number of LAYERS (keep each layer = 4 neurons)
# ═════════════════════════════════════════════════════════════

print("=" * 60)
print("EXPERIMENT 2: Different number of LAYERS (each layer = 4 neurons)")
print("=" * 60)
print("""
Now we keep neuron count = 4 per layer.
We change the NUMBER of hidden layers: 0, 1, 2, 3, 4.
""")

print(f"  {'Layers':>8} | {'Architecture':>20} | {'Params':>7} | {'Correct':>8} | {'Result'}")
print("  " + "─" * 65)

layer_configs = {
    0: [],            # no hidden layer
    1: [4],           # one hidden layer
    2: [4, 4],        # two hidden layers
    3: [4, 4, 4],     # three hidden layers
    4: [4, 4, 4, 4],  # four hidden layers
}

for num_layers, config in layer_configs.items():
    model  = build_model(config)
    params = count_params(model)

    # Build architecture string for display
    arch_parts = ["2(in)"] + [f"{s}(h)" for s in config] + ["1(out)"]
    arch_str   = "→".join(arch_parts)

    correct, loss, _ = train_and_score(model)
    result = "SOLVED" if correct == 4 else f"FAILED ({correct}/4)"
    print(f"  {num_layers:>8} | {arch_str:>20} | {params:>7} | {correct:>5}/4   | {result}")

print("""
OBSERVATIONS:
  0 hidden layers → always fails. (This was the famous 1969 XOR failure!)
  1 hidden layer  → works. Minimum depth needed for XOR.
  2+ hidden layers → also works, but overkill for XOR.
                     For harder problems (images, text), more layers help.

  KEY INSIGHT:
  More layers = model can learn more ABSTRACT features.
  Layer 1 learns simple patterns ("is A=1?")
  Layer 2 learns combinations ("is A=1 AND B=0?")
  Layer 3+ learns patterns of patterns.
  GPT uses 96+ layers — each adds a level of abstraction.
""")


# ═════════════════════════════════════════════════════════════
# EXPERIMENT 3: Show final probabilities for best model
# ═════════════════════════════════════════════════════════════

print("=" * 60)
print("EXPERIMENT 3: Best model output probabilities")
print("=" * 60)

best_model = build_model([4])
correct, loss, probs = train_and_score(best_model)

print("\nModel: Input(2) → Hidden(4, ReLU) → Output(1)")
print(f"Training result: {correct}/4 correct, Final loss: {loss:.4f}")
print("""
For each of the 4 XOR inputs, what probability did the model output?
Probability > 0.5 → predicts 1
Probability < 0.5 → predicts 0
""")
print(f"  A  B | XOR Label | Probability | Predicted | Correct?")
print(f"  ─────┼───────────┼─────────────┼───────────┼─────────")
input_list = [(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
for i, (a, b, label) in enumerate(input_list):
    prob = probs[i].item()
    pred = 1 if prob > 0.5 else 0
    ok   = "YES" if pred == label else "NO"
    bar  = "█" * int(prob * 20)
    print(f"  {a}  {b} |     {label}     |   {prob:.4f}    |     {pred}     |  {ok}   {bar}")

print("""
Notice:
  Inputs (0,1) and (1,0) → probability close to 1.0 → correctly predicts 1
  Inputs (0,0) and (1,1) → probability close to 0.0 → correctly predicts 0
  The model learned XOR perfectly!
""")


# ═════════════════════════════════════════════════════════════
# RULES OF THUMB (the actual guidelines used in practice)
# ═════════════════════════════════════════════════════════════

print("=" * 60)
print("RULES OF THUMB FOR CHOOSING HIDDEN LAYER SIZE")
print("=" * 60)
print("""
There is NO perfect formula. These are practical guidelines:

┌─────────────────────────────────────────────────────────┐
│  RULE 1: Start between input size and output size        │
│                                                          │
│  inputs=784, output=10 → try hidden=128 or 256          │
│  inputs=5,   output=1  → try hidden=8  or 16            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  RULE 2: Use powers of 2 (easier for GPU memory)        │
│                                                          │
│  Good choices: 2, 4, 8, 16, 32, 64, 128, 256, 512      │
│  These align with GPU memory blocks = faster training    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  RULE 3: Too few = underfitting. Too many = overfitting  │
│                                                          │
│  Underfitting: model is too simple to learn the pattern  │
│               train accuracy is LOW                      │
│  Overfitting:  model memorizes data, can't generalize    │
│               train accuracy HIGH, test accuracy LOW     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  RULE 4: More data → can afford more neurons             │
│                                                          │
│  XOR: 4 samples → 4 hidden neurons is enough            │
│  MNIST: 60,000 samples → 128-512 neurons needed          │
│  ImageNet: 1M+ samples → millions of neurons needed      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  RULE 5: Number of LAYERS by problem complexity          │
│                                                          │
│  Simple (XOR, AND):       1 hidden layer                 │
│  Medium (spam, iris):     2-3 hidden layers              │
│  Complex (images, speech):many layers (ResNet=50 layers) │
│  Very complex (text/LLMs): 12-96 layers (Transformers)   │
└─────────────────────────────────────────────────────────┘

FINAL ANSWER TO YOUR QUESTION "why 4?":
  We chose 4 because:
  1. XOR needs minimum 2 neurons (mathematical minimum)
  2. We doubled it to 4 for stability (avoids getting stuck)
  3. It is a power of 2 (GPU-friendly)
  4. It was enough — adding more gave no benefit for 4 data points

  Could we use 3? YES — try it.
  Could we use 8? YES — also works, just wastes a little memory.
  Could we use 1? NO — proven impossible for XOR with 1 hidden neuron.
  Could we use 100? OVERKILL — trains slow, may overfit 4 data points.
""")

print("=" * 60)
print("TRY IT YOURSELF:")
print("  Change [4] to [2] or [8] or [4,4] in build_model() above.")
print("  Watch how training changes. Best way to build intuition!")
print("=" * 60)
