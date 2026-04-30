"""
Module 12 - Fine-Tuning LLMs: Example 01
==========================================
TOPIC: What changes in model weights during fine-tuning?

=== GLOSSARY ===

Fine-Tuning:
    Taking a pre-trained model (one already trained on lots of data)
    and training it a little more on YOUR specific data.
    C# analogy: Like taking a base class from a NuGet package and
    overriding just a few methods to customize it for your app.

Weights (Parameters):
    The numbers inside a neural network that it "learns".
    C# analogy: Like private fields in a class that get adjusted
    during training. e.g., float _weight1 = 0.42f;

Pre-trained weights:
    Weights that were already learned from a large dataset.
    Fine-tuning starts FROM these weights, not from random ones.

Forward pass:
    Feeding input data through the network to get a prediction.
    C# analogy: Calling a method and getting a return value.

Loss:
    A number measuring how WRONG the model's prediction is.
    Lower loss = better predictions.
    C# analogy: Like a validation error count - you want it at zero.

Gradient:
    The direction and size of the change needed to reduce loss.
    C# analogy: Like a diff/delta telling you which way to adjust.

Learning rate:
    How big each weight update step is.
    Small value (e.g., 0.01) = tiny careful steps.
    C# analogy: Like a step size in a binary search.

Delta:
    The difference between old and new weights after one training step.
    Shows HOW MUCH fine-tuning changed the model.

Softmax:
    A function that turns raw scores into probabilities that sum to 1.
    e.g., [2.0, 1.0, 0.5] becomes [0.6, 0.24, 0.16]

One-hot encoding:
    Representing a category as a list of 0s with a single 1.
    e.g., BUG=0 -> [1,0,0], FEATURE=1 -> [0,1,0], OUTAGE=2 -> [0,0,1]

=== PART A: NumPy (no GPU, no PyTorch) ===
=== PART B: PyTorch (still CPU only, tiny model) ===

WHAT YOU WILL SEE:
    - Weights BEFORE fine-tuning (random/pre-trained starting point)
    - Weights AFTER fine-tuning (slightly adjusted)
    - The DELTA (difference) showing how small the changes are

Run with: python example_01_fine_tuning_concepts.py
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np          # NumPy: numerical computing library (like Math + arrays in C#)
import torch                # PyTorch: deep learning framework
import torch.nn as nn       # nn module: contains Layer classes (like abstract base classes)
import torch.optim as optim # optim module: contains optimizers (like gradient descent algorithms)

# ============================================================
# PART A: FINE-TUNING WITH PURE NUMPY
# ============================================================
# We build a tiny 2-layer neural network from scratch using only NumPy.
# No PyTorch, no GPU. This shows the mechanics clearly.
#
# Network diagram (plain ASCII):
#
#   Input (3 features)
#        |
#    [Layer 1: 3 -> 4 neurons]   (W1, b1)
#        |
#      ReLU activation
#        |
#    [Layer 2: 4 -> 3 neurons]   (W2, b2)
#        |
#      Softmax
#        |
#   Output (3 classes: BUG, FEATURE, OUTAGE)
#
# Task: Classify support tickets into 3 categories.
# ============================================================

print("=" * 60)
print("MODULE 12 - FINE-TUNING CONCEPTS")
print("=" * 60)
print()

# ----------------------------------------------------------
# A.1 - DEFINE THE DATASET
# ----------------------------------------------------------
# Each ticket is represented as 3 features (hand-crafted for simplicity):
#   Feature 0: contains_error_keyword  (0 or 1)
#   Feature 1: contains_request_keyword (0 or 1)
#   Feature 2: severity_score           (0.0 to 1.0)
#
# C# analogy: This is like a List<float[]> where each float[] is one row.

print("--- PART A: NumPy Fine-Tuning ---")
print()

# Define training inputs (X).
# Shape: (8 samples, 3 features)
# C# analogy: float[8, 3] X = { ... };
X_train = np.array([
    [1, 0, 0.9],   # Sample 0: has error keyword, high severity -> BUG
    [1, 0, 0.8],   # Sample 1: has error keyword, high severity -> BUG
    [0, 1, 0.2],   # Sample 2: has request keyword, low severity -> FEATURE
    [0, 1, 0.3],   # Sample 3: has request keyword, low severity -> FEATURE
    [1, 0, 1.0],   # Sample 4: error + max severity -> OUTAGE
    [1, 0, 0.95],  # Sample 5: error + very high severity -> OUTAGE
    [0, 1, 0.1],   # Sample 6: request + very low severity -> FEATURE
    [1, 0, 0.7],   # Sample 7: error + medium severity -> BUG
], dtype=np.float32)  # dtype=float32 matches what neural networks use

# Define training labels (y).
# 0 = BUG, 1 = FEATURE, 2 = OUTAGE
# C# analogy: int[] y = { 0, 0, 1, 1, 2, 2, 1, 0 };
y_train = np.array([0, 0, 1, 1, 2, 2, 1, 0], dtype=np.int32)

# Print dataset summary
print("Training dataset:")
print(f"  Input shape : {X_train.shape}  (8 tickets, 3 features each)")
print(f"  Labels      : {y_train}  (0=BUG, 1=FEATURE, 2=OUTAGE)")
print()

# ----------------------------------------------------------
# A.2 - DEFINE HELPER FUNCTIONS
# ----------------------------------------------------------

def relu(z):
    """
    ReLU activation: replaces negative values with 0.
    ReLU stands for Rectified Linear Unit.
    C# analogy: return Math.Max(0, z);
    z is a NumPy array (can be any shape).
    """
    return np.maximum(0, z)  # element-wise max(0, each value)

def softmax(z):
    """
    Softmax: converts raw scores into probabilities.
    All outputs are positive and sum to 1.0.
    We subtract max(z) first for numerical stability (avoids overflow).
    C# analogy: Like normalizing a score array so they sum to 1.
    """
    z_stable = z - np.max(z, axis=1, keepdims=True)  # subtract row-max for stability
    exp_z = np.exp(z_stable)                           # e^z for each element
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # divide by row sum

def cross_entropy_loss(probs, y_true):
    """
    Cross-entropy loss: measures how wrong the predicted probabilities are.
    Returns a single float (the average loss over all samples).
    Lower = better predictions.
    C# analogy: Like computing an average error across all test cases.
    probs: shape (N, num_classes) - predicted probabilities
    y_true: shape (N,) - true class indices (integers)
    """
    N = probs.shape[0]                           # N = number of samples (8 here)
    correct_probs = probs[np.arange(N), y_true]  # pick the probability for the correct class
    log_probs = np.log(correct_probs + 1e-9)     # log of those probs (add tiny value to avoid log(0))
    return -np.mean(log_probs)                   # negative mean (loss goes down as probs go up)

# ----------------------------------------------------------
# A.3 - INITIALIZE WEIGHTS (SIMULATING "PRE-TRAINED" WEIGHTS)
# ----------------------------------------------------------
# In real fine-tuning, you load weights from a checkpoint file.
# Here we simulate "pre-trained" weights by setting a fixed random seed
# and generating random weights - then we call THESE the starting point.

np.random.seed(42)  # fixed seed so results are reproducible (like a fixed random generator in C#)

# Layer 1 weights: shape (3 input features, 4 hidden neurons)
# C# analogy: float[3, 4] W1 = InitializeRandom();
W1 = np.random.randn(3, 4) * 0.1   # small random values (0.1 scale keeps them near 0)
b1 = np.zeros(4)                    # biases start at zero (one per hidden neuron)

# Layer 2 weights: shape (4 hidden neurons, 3 output classes)
W2 = np.random.randn(4, 3) * 0.1   # small random values
b2 = np.zeros(3)                    # biases start at zero (one per output class)

# Save copies of the original weights so we can compare later
# np.copy() is like doing new float[,] copy = (float[,])W1.Clone(); in C#
W1_before = np.copy(W1)
W2_before = np.copy(W2)

print("Weights BEFORE fine-tuning:")
print(f"  W1 (Layer 1) shape: {W1.shape}  -- first row: {W1[0].round(4)}")
print(f"  W2 (Layer 2) shape: {W2.shape}  -- first row: {W2[0].round(4)}")
print()

# ----------------------------------------------------------
# A.4 - FINE-TUNING LOOP (GRADIENT DESCENT BY HAND)
# ----------------------------------------------------------
# We do 20 training steps (epochs).
# Each step:
#   1. Forward pass: compute predictions
#   2. Compute loss
#   3. Backward pass: compute gradients
#   4. Update weights

LEARNING_RATE = 0.05   # how big each weight update step is
NUM_EPOCHS = 20         # how many full passes over the training data

print("Running fine-tuning loop (20 epochs)...")
print()

for epoch in range(NUM_EPOCHS):  # loop over epochs, like a for loop in C#

    # ---- FORWARD PASS ----

    # Layer 1: linear transform (matrix multiply + bias add)
    # z1 shape: (8 samples, 4 neurons)
    # C# analogy: z1 = X_train * W1 + b1  (matrix multiply)
    z1 = np.dot(X_train, W1) + b1   # dot product = matrix multiplication

    # Apply ReLU activation to layer 1 output
    a1 = relu(z1)                    # negative values become 0

    # Layer 2: linear transform
    # z2 shape: (8 samples, 3 classes)
    z2 = np.dot(a1, W2) + b2        # second matrix multiply

    # Apply softmax to get probabilities
    probs = softmax(z2)              # shape: (8, 3) -- each row sums to 1

    # Compute loss
    loss = cross_entropy_loss(probs, y_train)  # single float

    # ---- BACKWARD PASS (manual gradient computation) ----
    # This is the chain rule applied by hand.
    # In PyTorch, this is done automatically by autograd.

    N = X_train.shape[0]             # number of samples = 8

    # Gradient of loss with respect to z2 (before softmax)
    # This is a well-known formula: dL/dz2 = probs - one_hot(y)
    dz2 = probs.copy()               # start with predicted probabilities
    dz2[np.arange(N), y_train] -= 1  # subtract 1 at the correct class position
    dz2 /= N                         # average over all samples

    # Gradient with respect to W2
    dW2 = np.dot(a1.T, dz2)         # a1.T transposes a1 (flip rows/cols)

    # Gradient with respect to b2
    db2 = np.sum(dz2, axis=0)       # sum over all samples (axis 0 = rows)

    # Gradient with respect to a1 (output of layer 1)
    da1 = np.dot(dz2, W2.T)         # W2.T transposes W2

    # Gradient through ReLU: zero where z1 was negative, pass-through where positive
    dz1 = da1 * (z1 > 0)            # element-wise multiply by 1 where z1>0, else 0

    # Gradient with respect to W1
    dW1 = np.dot(X_train.T, dz1)    # X_train.T transposes the input matrix

    # Gradient with respect to b1
    db1 = np.sum(dz1, axis=0)       # sum over samples

    # ---- WEIGHT UPDATE (Gradient Descent) ----
    # new_weight = old_weight - learning_rate * gradient
    # C# analogy: weight -= learningRate * gradient;
    W1 -= LEARNING_RATE * dW1       # update layer 1 weights
    b1 -= LEARNING_RATE * db1       # update layer 1 biases
    W2 -= LEARNING_RATE * dW2       # update layer 2 weights
    b2 -= LEARNING_RATE * db2       # update layer 2 biases

    # Print loss every 5 epochs so we can see progress
    if (epoch + 1) % 5 == 0:        # (epoch+1) because epoch starts at 0
        print(f"  Epoch {epoch+1:2d}/20  |  Loss: {loss:.4f}")

print()

# ----------------------------------------------------------
# A.5 - COMPARE WEIGHTS BEFORE vs AFTER
# ----------------------------------------------------------

print("Weights AFTER fine-tuning:")
print(f"  W1 (Layer 1) shape: {W1.shape}  -- first row: {W1[0].round(4)}")
print(f"  W2 (Layer 2) shape: {W2.shape}  -- first row: {W2[0].round(4)}")
print()

# Compute the delta (change in weights)
# C# analogy: float delta = newWeight - oldWeight;
W1_delta = W1 - W1_before    # element-wise subtraction
W2_delta = W2 - W2_before    # element-wise subtraction

print("Weight DELTA (change caused by fine-tuning):")
print(f"  W1 delta -- first row: {W1_delta[0].round(6)}")
print(f"  W2 delta -- first row: {W2_delta[0].round(6)}")
print()

# Show magnitude of changes using L2 norm (overall size of the delta)
# np.linalg.norm is like Math.Sqrt(sum of squares) in C#
W1_change_pct = np.linalg.norm(W1_delta) / np.linalg.norm(W1_before) * 100
W2_change_pct = np.linalg.norm(W2_delta) / np.linalg.norm(W2_before) * 100

print("How much did weights change? (as % of original magnitude)")
print(f"  W1 changed by: {W1_change_pct:.2f}%")
print(f"  W2 changed by: {W2_change_pct:.2f}%")
print()
print("KEY INSIGHT: Weights changed by only a small % !")
print("Fine-tuning makes SMALL targeted adjustments, not a full rewrite.")
print("This is why you start from pre-trained weights - most knowledge stays.")
print()

# Quick check: what does the model predict now?
z1_final = np.dot(X_train, W1) + b1   # layer 1 output
a1_final = relu(z1_final)              # apply ReLU
z2_final = np.dot(a1_final, W2) + b2  # layer 2 output
probs_final = softmax(z2_final)        # probabilities

# argmax picks the index of the highest probability (the predicted class)
# C# analogy: Array.IndexOf(probs, probs.Max())
predictions = np.argmax(probs_final, axis=1)

label_names = {0: "BUG", 1: "FEATURE", 2: "OUTAGE"}  # dict maps int -> string

print("Final predictions vs true labels:")
for i in range(len(y_train)):  # iterate over all 8 samples
    pred_name = label_names[predictions[i]]   # get predicted class name
    true_name = label_names[y_train[i]]       # get true class name
    match = "OK" if predictions[i] == y_train[i] else "WRONG"  # check if correct
    print(f"  Sample {i}: Predicted={pred_name:8s}  True={true_name:8s}  [{match}]")

print()
print("=" * 60)

# ============================================================
# PART B: FINE-TUNING WITH PYTORCH
# ============================================================
# Same concept, but now using PyTorch.
# PyTorch handles the backward pass (gradient computation) automatically.
# You just call loss.backward() and PyTorch does the chain rule for you.
#
# C# analogy: PyTorch is like a framework that uses code generation
# to build the backward pass automatically, similar to how an ORM
# generates SQL from LINQ expressions.
# ============================================================

print()
print("--- PART B: PyTorch Fine-Tuning ---")
print()

# ----------------------------------------------------------
# B.1 - CONVERT DATA TO PYTORCH TENSORS
# ----------------------------------------------------------
# A Tensor is PyTorch's version of a NumPy array.
# C# analogy: Tensor is like a strongly-typed multi-dimensional array
# that also tracks gradients automatically.

# torch.tensor() converts a NumPy array to a PyTorch tensor
X_torch = torch.tensor(X_train, dtype=torch.float32)   # features tensor
y_torch = torch.tensor(y_train, dtype=torch.long)       # labels tensor (long = int64 for class indices)

print(f"PyTorch input tensor shape : {X_torch.shape}")
print(f"PyTorch labels tensor shape: {y_torch.shape}")
print()

# ----------------------------------------------------------
# B.2 - DEFINE A TINY LINEAR MODEL
# ----------------------------------------------------------
# nn.Sequential is a container that chains layers one after another.
# C# analogy: Like a List<ILayer> where each layer's output feeds
# into the next layer's input.

class TinyClassifier(nn.Module):
    """
    A tiny 2-layer neural network for classifying support tickets.
    Inherits from nn.Module, which is PyTorch's base class for all models.
    C# analogy: Like inheriting from a base class that provides
    training infrastructure (forward/backward pass bookkeeping).
    """

    def __init__(self):
        """
        Constructor. Defines the layers.
        C# analogy: public TinyClassifier() { this.layer1 = new Linear(...); }
        """
        super().__init__()  # call the parent class (nn.Module) constructor - required in PyTorch

        # nn.Linear(in_features, out_features) is a fully-connected layer.
        # It holds a weight matrix W (shape: out x in) and bias b (shape: out).
        # C# analogy: Like a matrix multiply class with learnable parameters.
        self.layer1 = nn.Linear(3, 4)   # 3 inputs -> 4 hidden neurons
        self.relu   = nn.ReLU()          # ReLU activation (same as Part A)
        self.layer2 = nn.Linear(4, 3)   # 4 hidden -> 3 output classes

    def forward(self, x):
        """
        Forward pass: defines how data flows through the network.
        C# analogy: Like a Compute() method that applies each layer in order.
        x is the input tensor.
        """
        out = self.layer1(x)    # apply layer 1 (linear transform)
        out = self.relu(out)    # apply ReLU activation
        out = self.layer2(out)  # apply layer 2 (produces raw scores, called logits)
        return out              # return logits (NOT softmax - CrossEntropyLoss does that internally)

# Create an instance of our model
model = TinyClassifier()   # C# analogy: var model = new TinyClassifier();

print("Model architecture:")
print(model)   # PyTorch prints a nice summary of the layers
print()

# ----------------------------------------------------------
# B.3 - SNAPSHOT WEIGHTS BEFORE FINE-TUNING
# ----------------------------------------------------------
# We use .clone().detach() to make a copy of the weights.
# .clone() copies the data, .detach() disconnects it from the gradient graph.
# C# analogy: (float[,])layer1.Weight.Clone()

# Access weights via model.layer1.weight  (PyTorch stores them as .weight and .bias)
W1_pt_before = model.layer1.weight.clone().detach()  # copy layer 1 weights
W2_pt_before = model.layer2.weight.clone().detach()  # copy layer 2 weights

print("PyTorch weights BEFORE fine-tuning:")
print(f"  Layer 1 weight (first row): {W1_pt_before[0].numpy().round(4)}")
print(f"  Layer 2 weight (first row): {W2_pt_before[0].numpy().round(4)}")
print()

# ----------------------------------------------------------
# B.4 - DEFINE LOSS FUNCTION AND OPTIMIZER
# ----------------------------------------------------------
# Loss function: CrossEntropyLoss combines Softmax + cross-entropy in one step.
# C# analogy: Like a pre-built error metric class.
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam adjusts learning rate automatically per parameter.
# SGD (basic gradient descent) would also work but Adam converges faster.
# C# analogy: Like an IOptimizer interface implementation that updates weights.
# lr = learning rate
optimizer = optim.Adam(model.parameters(), lr=0.05)  # model.parameters() yields all W and b tensors

# ----------------------------------------------------------
# B.5 - FINE-TUNING LOOP WITH PYTORCH
# ----------------------------------------------------------
# PyTorch training loop has 5 standard steps every iteration:
#   1. Zero the gradients (clear previous step's gradients)
#   2. Forward pass
#   3. Compute loss
#   4. Backward pass (compute gradients automatically)
#   5. Optimizer step (update weights)
#
# C# analogy: This is like a game loop - each frame you clear state,
# compute new state, then render/update.

NUM_EPOCHS_PT = 30  # run for 30 epochs this time

print("Running PyTorch fine-tuning loop (30 epochs)...")
print()
print(f"  {'Epoch':>5}  |  {'Loss':>8}  |  Note")
print(f"  {'-'*5}  |  {'-'*8}  |  {'-'*30}")

for epoch in range(NUM_EPOCHS_PT):  # standard Python for loop over epochs

    # Step 1: Zero the gradients from the previous iteration.
    # PyTorch ACCUMULATES gradients by default (for advanced use cases).
    # We must clear them each step or they pile up incorrectly.
    # C# analogy: gradient.Reset();
    optimizer.zero_grad()

    # Step 2: Forward pass - run data through the model
    # logits shape: (8, 3) - raw scores for each class
    logits = model(X_torch)   # calls model.forward(X_torch) internally

    # Step 3: Compute loss
    loss = criterion(logits, y_torch)   # compares predicted logits to true labels

    # Step 4: Backward pass - PyTorch computes ALL gradients automatically
    # This is the magic of autograd: no manual chain rule required.
    # C# analogy: Like calling .ComputeGradients() on a framework object.
    loss.backward()

    # Step 5: Update weights using the computed gradients
    # The optimizer reads gradient from each tensor's .grad attribute
    # and applies the update rule (Adam in this case).
    # C# analogy: optimizer.Step(); // applies weight = weight - lr * gradient
    optimizer.step()

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        note = "gradients computed & applied" if epoch == 9 else "weights converging..."
        print(f"  {epoch+1:>5}  |  {loss.item():>8.4f}  |  {note}")

print()

# ----------------------------------------------------------
# B.6 - COMPARE PYTORCH WEIGHTS BEFORE vs AFTER
# ----------------------------------------------------------

# Snapshot the weights AFTER training
W1_pt_after = model.layer1.weight.clone().detach()   # copy current layer 1 weights
W2_pt_after = model.layer2.weight.clone().detach()   # copy current layer 2 weights

print("PyTorch weights AFTER fine-tuning:")
print(f"  Layer 1 weight (first row): {W1_pt_after[0].numpy().round(4)}")
print(f"  Layer 2 weight (first row): {W2_pt_after[0].numpy().round(4)}")
print()

# Compute deltas
W1_pt_delta = W1_pt_after - W1_pt_before   # element-wise subtraction of tensors
W2_pt_delta = W2_pt_after - W2_pt_before

print("PyTorch weight DELTA (change per element):")
print(f"  Layer 1 delta (first row): {W1_pt_delta[0].numpy().round(6)}")
print(f"  Layer 2 delta (first row): {W2_pt_delta[0].numpy().round(6)}")
print()

# Compute norm-based percentage change (how big is the change overall?)
W1_pt_change_pct = W1_pt_delta.norm().item() / W1_pt_before.norm().item() * 100
W2_pt_change_pct = W2_pt_delta.norm().item() / W2_pt_before.norm().item() * 100

print("How much did PyTorch weights change?")
print(f"  Layer 1 changed by: {W1_pt_change_pct:.2f}%")
print(f"  Layer 2 changed by: {W2_pt_change_pct:.2f}%")
print()

# ----------------------------------------------------------
# B.7 - SHOW GRADIENT STEP-BY-STEP FOR ONE SAMPLE
# ----------------------------------------------------------
# This section shows the gradient values explicitly for one training step.
# It helps you see WHAT the backward pass actually produces.

print("--- Gradient step detail for ONE sample ---")
print()

# Use just one sample (index 0: a BUG ticket)
x_single = X_torch[0:1]    # shape: (1, 3) -- slice keeps 2D shape
y_single = y_torch[0:1]    # shape: (1,)   -- true label = 0 (BUG)

print(f"  Input  : {x_single.numpy()}")
print(f"  True label: {y_single.item()} (BUG)")
print()

optimizer.zero_grad()              # clear any leftover gradients

logits_single = model(x_single)   # forward pass for this one sample

print(f"  Raw logits (before softmax): {logits_single.detach().numpy().round(4)}")

# Apply softmax manually so we can see the probabilities
probs_single = torch.softmax(logits_single, dim=1)   # dim=1 = across class dimension
print(f"  Probabilities after softmax: {probs_single.detach().numpy().round(4)}")
print(f"  Predicted class: {probs_single.argmax().item()} (0=BUG, 1=FEATURE, 2=OUTAGE)")
print()

loss_single = criterion(logits_single, y_single)   # compute loss for this one sample
loss_single.backward()                              # compute gradients

# Access gradient for layer 1 weights: stored at model.layer1.weight.grad
# .grad is None until backward() is called at least once
print(f"  Gradient of Layer 1 weights (first row): {model.layer1.weight.grad[0].numpy().round(6)}")
print(f"  Gradient of Layer 2 weights (first row): {model.layer2.weight.grad[0].numpy().round(6)}")
print()
print("  Gradient size tells you HOW MUCH to adjust each weight.")
print("  The optimizer multiplies this by the learning rate to get the delta.")
print()

# ----------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------

print("=" * 60)
print("SUMMARY - What changes during fine-tuning?")
print("=" * 60)
print()
print("1. Weights START at pre-trained values (not random).")
print("2. Each training step makes a TINY adjustment to weights.")
print("3. The adjustment direction comes from the gradient.")
print("4. The adjustment SIZE is controlled by the learning rate.")
print("5. After many steps, weights shift just enough to handle")
print("   your specific task without forgetting general knowledge.")
print()
print("C#/.NET analogy:")
print("  Pre-trained model  = a NuGet package with general functionality")
print("  Fine-tuning        = overriding a few methods for your use case")
print("  Weight delta       = the diff between base class and override")
print()
print("Done! Run example_02_dataset_preparation.py next.")
