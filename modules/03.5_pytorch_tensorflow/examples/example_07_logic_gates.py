"""
Example 7: AND and XOR Logic Gates with PyTorch
================================================

WHAT WE ARE BUILDING:
  A neural network that learns basic logic gate rules from examples.

  AND gate:               XOR gate:
  0 AND 0 = 0             0 XOR 0 = 0
  0 AND 1 = 0             0 XOR 1 = 1
  1 AND 0 = 0             1 XOR 0 = 1
  1 AND 1 = 1             1 XOR 1 = 0

WHY THESE TWO TOGETHER?
  AND is easy — a single layer solves it.
  XOR is impossible for a single layer — needs a hidden layer.
  This difference teaches the most important concept in deep learning:
  "Why do we need multiple layers?"

THE BIG INSIGHT — LINEAR SEPARABILITY:

  AND (easy):                   XOR (hard):
  y                             y
  1 | . 1(class 1)              1 | 1 . 1
    |  /                          |  X   (no single line works!)
  0 | 0 0                       0 | 0   0
    +─────── x                    +─────── x
    0   1                         0   1

  You CAN draw one straight       You CANNOT draw one straight
  line to separate 0s and 1s.     line to separate 0s and 1s.
  → 1 layer is enough.            → Need a hidden layer.

C# ANALOGY:
  Think of each layer as a WHERE clause in SQL.
  AND needs one WHERE clause.
  XOR needs two nested WHERE clauses.

Run: python example_07_logic_gates.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 55)
print("Logic Gates: AND and XOR with PyTorch")
print("=" * 55)

# ─────────────────────────────────────────────────────────────
# SHARED DATA
# Both gates have same 4 input combinations — only outputs differ
# ─────────────────────────────────────────────────────────────

# All possible 2-input combinations: (0,0), (0,1), (1,0), (1,1)
INPUTS = torch.FloatTensor([
    [0, 0],   # row 0
    [0, 1],   # row 1
    [1, 0],   # row 2
    [1, 1],   # row 3
])   # shape: (4, 2) — 4 samples, 2 features each

# AND outputs: only 1 AND 1 = 1, rest are 0
AND_LABELS = torch.FloatTensor([[0], [0], [0], [1]])   # shape: (4, 1)

# XOR outputs: 0 XOR 1 = 1, 1 XOR 0 = 1, rest are 0
XOR_LABELS = torch.FloatTensor([[0], [1], [1], [0]])   # shape: (4, 1)


# ─────────────────────────────────────────────────────────────
# PART 1: AND GATE — Single Layer Model
# ─────────────────────────────────────────────────────────────

print("\n" + "─" * 55)
print("PART 1: AND Gate (single layer — no hidden neurons)")
print("─" * 55)

print("""
AND Truth Table:
  Input A | Input B | Output
  ────────┼─────────┼────────
    0     |    0    |   0
    0     |    1    |   0
    1     |    0    |   0
    1     |    1    |   1    ← only this is 1
""")

class SingleLayerNet(nn.Module):
    """
    Simplest possible neural network.
    No hidden layer — just one direct connection from input to output.

    Architecture:
        Input (2) ──→ Output (1)
                       Sigmoid

    WHY this works for AND:
      AND data is linearly separable.
      One straight line can divide 0s from 1s on a 2D grid.
      A single layer = one straight line decision boundary.
    """
    def __init__(self):
        super(SingleLayerNet, self).__init__()
        self.output = nn.Linear(2, 1)   # 2 inputs → 1 output, no hidden layer

    def forward(self, x):
        return self.output(x)   # raw score (sigmoid applied in loss function)


def train_gate(model, inputs, labels, epochs, lr, gate_name):
    """
    Train any gate model. Same function reused for both AND and XOR.

    Args:
        model     : any nn.Module (SingleLayerNet or TwoLayerNet)
        inputs    : tensor shape (4, 2)
        labels    : tensor shape (4, 1)
        epochs    : training iterations
        lr        : learning rate
        gate_name : string label for printing

    Returns:
        final_loss : float
    """
    loss_fn   = nn.BCEWithLogitsLoss()   # binary cross entropy (sigmoid + BCE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        predictions = model(inputs)
        loss        = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print at 25%, 50%, 75%, 100% of training
        if (epoch + 1) in [epochs // 4, epochs // 2, epochs * 3 // 4, epochs]:
            probs   = torch.sigmoid(predictions)
            preds   = (probs > 0.5).float()
            correct = (preds == labels).sum().item()
            print(f"  Epoch {epoch+1:4d} | Loss: {loss.item():.4f} | Correct: {correct}/4")

    return loss.item()


def show_results(model, inputs, labels, gate_name):
    """
    Print truth table with model's predictions vs actual labels.
    """
    model.eval()
    with torch.no_grad():
        raw    = model(inputs)                    # raw scores
        probs  = torch.sigmoid(raw)               # 0.0 to 1.0
        preds  = (probs > 0.5).float()            # 0 or 1

    print(f"\n{gate_name} Results:")
    print(f"  A  B | Actual | Predicted | Prob    | Correct?")
    print(f"  ─────┼────────┼───────────┼─────────┼─────────")
    all_correct = True
    for i in range(4):
        a        = int(inputs[i][0].item())
        b        = int(inputs[i][1].item())
        actual   = int(labels[i].item())
        pred     = int(preds[i].item())
        prob     = probs[i].item()
        ok       = "OK" if pred == actual else "WRONG"
        if pred != actual:
            all_correct = False
        print(f"  {a}  {b} |   {actual}    |     {pred}     |  {prob:.3f}  | {ok}")

    status = "PASSED - model learned the gate!" if all_correct else "FAILED - model could not learn this gate"
    print(f"\n  Result: {status}")


# Train AND gate with single layer
and_model = SingleLayerNet()
print(f"Training AND with single layer ({sum(p.numel() for p in and_model.parameters())} parameters):")
train_gate(and_model, INPUTS, AND_LABELS, epochs=2000, lr=0.1, gate_name="AND")
show_results(and_model, INPUTS, AND_LABELS, "AND Gate")


# ─────────────────────────────────────────────────────────────
# PART 2: XOR GATE — First try with single layer (will fail!)
# ─────────────────────────────────────────────────────────────

print("\n\n" + "─" * 55)
print("PART 2A: XOR Gate — single layer attempt (WILL FAIL)")
print("─" * 55)

print("""
XOR Truth Table:
  Input A | Input B | Output
  ────────┼─────────┼────────
    0     |    0    |   0    ← diagonal group: class 0
    0     |    1    |   1    ← diagonal group: class 1
    1     |    0    |   1    ← diagonal group: class 1
    1     |    1    |   0    ← diagonal group: class 0

WHY single layer fails:
  Plot these 4 points on a grid:
    y=1 |  1(class 1)   0(class 0)
        |
    y=0 |  0(class 0)   1(class 1)
        +────────────────────────
           x=0          x=1

  Class 0 = top-right and bottom-left (diagonal)
  Class 1 = top-left and bottom-right (other diagonal)
  No single straight line can separate them!
  This is the famous XOR problem that killed neural networks in 1960s.
""")

xor_single = SingleLayerNet()   # same architecture as AND
print(f"Training XOR with single layer ({sum(p.numel() for p in xor_single.parameters())} parameters):")
train_gate(xor_single, INPUTS, XOR_LABELS, epochs=2000, lr=0.1, gate_name="XOR-single")
show_results(xor_single, INPUTS, XOR_LABELS, "XOR Gate (single layer)")


# ─────────────────────────────────────────────────────────────
# PART 3: XOR GATE — Two layer model (will succeed!)
# ─────────────────────────────────────────────────────────────

print("\n\n" + "─" * 55)
print("PART 2B: XOR Gate — two layer model (WILL SUCCEED)")
print("─" * 55)

print("""
FIX: Add a hidden layer.

Hidden layer learns intermediate features:
  Neuron 1 might learn: "are inputs different?"
  Neuron 2 might learn: "are both inputs 1?"
  Output combines these to get the right answer.

Architecture:
  Input (2) → Hidden (4, ReLU) → Output (1)
                 ↑
         This is the key difference!
         Hidden layer creates new representation
         that IS linearly separable.
""")

class TwoLayerNet(nn.Module):
    """
    Two-layer network with hidden neurons.

    Architecture:
        Input (2) → Hidden (4, ReLU) → Output (1)

    WHY 4 hidden neurons?
      XOR only really needs 2, but we use 4 to give
      the model more "room to think" and train faster.
    """
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.hidden = nn.Linear(2, 4)    # 2 inputs  → 4 hidden neurons
        self.output = nn.Linear(4, 1)    # 4 hidden  → 1 output
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden(x))   # (batch, 2) → (batch, 4), apply ReLU
        x = self.output(x)              # (batch, 4) → (batch, 1)
        return x


xor_model = TwoLayerNet()
print(f"Training XOR with two layers ({sum(p.numel() for p in xor_model.parameters())} parameters):")
train_gate(xor_model, INPUTS, XOR_LABELS, epochs=3000, lr=0.05, gate_name="XOR-two-layer")
show_results(xor_model, INPUTS, XOR_LABELS, "XOR Gate (two layers)")


# ─────────────────────────────────────────────────────────────
# PART 4: What did the hidden layer learn?
# Peek inside the hidden layer activations
# ─────────────────────────────────────────────────────────────

print("\n\n" + "─" * 55)
print("PART 3: What did hidden layer learn? (peek inside)")
print("─" * 55)

print("""
After training, let's see what activations the hidden layer
produces for each of the 4 inputs.
If the hidden layer learned well, its output should be
linearly separable (unlike the original XOR inputs).
""")

xor_model.eval()
with torch.no_grad():
    hidden_out = xor_model.relu(xor_model.hidden(INPUTS))  # activations after hidden layer

print("  Input → Hidden Layer Activations (after ReLU)")
print("  A  B | Label | h1      h2      h3      h4")
print("  ─────┼───────┼────────────────────────────")
for i in range(4):
    a     = int(INPUTS[i][0].item())
    b     = int(INPUTS[i][1].item())
    label = int(XOR_LABELS[i].item())
    h     = hidden_out[i].tolist()
    print(f"  {a}  {b} |   {label}   | {h[0]:.3f}   {h[1]:.3f}   {h[2]:.3f}   {h[3]:.3f}")

print("""
Notice: the hidden activations for label=0 inputs look similar to
each other, and different from label=1 inputs.
The hidden layer transformed the problem into a linearly separable one!
""")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

print("=" * 55)
print("SUMMARY")
print("=" * 55)
print("""
Gate   | Separable? | Layers needed | Works with 1 layer?
───────┼────────────┼───────────────┼────────────────────
AND    | YES        | 1             | YES
OR     | YES        | 1             | YES
NAND   | YES        | 1             | YES
XOR    | NO         | 2+            | NO  ← famous example

KEY LESSON:
  Linearly separable problem  → 1 layer is enough
  Non-linearly separable      → need hidden layers

  This is WHY deep learning (many layers) is powerful:
  Each layer learns a new representation of the data
  until the problem becomes easy to solve.

PARAMETER COUNT COMPARISON:
  SingleLayerNet: 2×1 + 1 = 3 parameters
  TwoLayerNet:    2×4 + 4 + 4×1 + 1 = 17 parameters

  17 parameters to solve XOR — tiny!
  GPT-4 has ~1.8 TRILLION parameters — same idea, massive scale.

WHAT CHANGED IN CODE (AND vs XOR):
  Same training function (train_gate) — reused for both!
  Only model architecture changed:
    AND: Input(2) → Output(1)          ← 1 layer
    XOR: Input(2) → Hidden(4) → Out(1) ← 2 layers
""")
print("Next: See example_08 for more complex architectures!")
print("=" * 55)
