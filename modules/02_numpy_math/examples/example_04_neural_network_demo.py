"""
Simple Neural Network Implementation Using Only NumPy
This demonstrates how everything in Module 2 comes together!

We'll build a 2-layer neural network from scratch to classify handwritten digits.
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("Building a Neural Network with NumPy!")
print("="*70)

# ==============================================================================
# PART 1: Understanding the Problem
# ==============================================================================

print("\n" + "="*70)
print("PART 1: The Problem - Digit Classification")
print("="*70)

print("""
Problem: Classify handwritten digits (0-9)
Input: 28x28 grayscale image (784 pixels)
Output: Probability for each digit (0-9)

Network Architecture:
    Input Layer:  784 neurons (28×28 pixels)
    Hidden Layer: 128 neurons (with ReLU activation)
    Output Layer: 10 neurons (one per digit, with softmax)

This is a simplified version of what MNIST classifiers do!
""")

# ==============================================================================
# PART 2: Create Synthetic Data (Normally you'd load MNIST)
# ==============================================================================

print("\n" + "="*70)
print("PART 2: Creating Sample Data")
print("="*70)

# Simulate a small dataset
n_samples = 1000
n_features = 784  # 28x28 pixels
n_classes = 10    # digits 0-9

# Random "images" (in real life, these would be actual digit images)
X_train = np.random.randn(n_samples, n_features)
# Random labels (in real life, these would be true labels)
y_train = np.random.randint(0, n_classes, n_samples)

print(f"Training data shape: {X_train.shape}")
print(f"Labels shape: {y_train.shape}")
print(f"First 10 labels: {y_train[:10]}")

# Normalize data (important for neural networks!)
X_train = (X_train - X_train.mean()) / X_train.std()
print(f"Data normalized: mean={X_train.mean():.4f}, std={X_train.std():.4f}")

# ==============================================================================
# PART 3: Initialize Network Weights
# ==============================================================================

print("\n" + "="*70)
print("PART 3: Initializing Network Weights")
print("="*70)

# Set seed for reproducibility
np.random.seed(42)

# Layer 1: 784 → 128
W1 = np.random.randn(n_features, 128) * 0.01  # Small random values
b1 = np.zeros(128)                             # Bias initialized to zero

# Layer 2: 128 → 10
W2 = np.random.randn(128, n_classes) * 0.01
b2 = np.zeros(n_classes)

print(f"W1 shape: {W1.shape} (784 inputs → 128 hidden neurons)")
print(f"b1 shape: {b1.shape}")
print(f"W2 shape: {W2.shape} (128 hidden → 10 output classes)")
print(f"b2 shape: {b2.shape}")

total_params = W1.size + b1.size + W2.size + b2.size
print(f"\nTotal trainable parameters: {total_params:,}")

# ==============================================================================
# PART 4: Define Activation Functions
# ==============================================================================

print("\n" + "="*70)
print("PART 4: Activation Functions")
print("="*70)

def relu(z):
    """
    ReLU (Rectified Linear Unit): max(0, z)
    Used in hidden layers to introduce non-linearity

    Element-wise operation: negative values become 0, positive stay same
    """
    return np.maximum(0, z)

def softmax(z):
    """
    Softmax: Converts raw scores to probabilities (sum to 1)
    Used in output layer for classification

    exp(z_i) / sum(exp(z))
    """
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Test activations
test_input = np.array([-2, -1, 0, 1, 2])
print("Testing ReLU:")
print(f"  Input:  {test_input}")
print(f"  Output: {relu(test_input)}")

test_scores = np.array([[1, 2, 3]])
print("\nTesting Softmax:")
print(f"  Input (raw scores):  {test_scores}")
print(f"  Output (probabilities): {softmax(test_scores)}")
print(f"  Sum: {softmax(test_scores).sum():.4f} (should be 1.0)")

# ==============================================================================
# PART 5: Forward Propagation
# ==============================================================================

print("\n" + "="*70)
print("PART 5: Forward Propagation (Making Predictions)")
print("="*70)

def forward_pass(X, W1, b1, W2, b2):
    """
    Forward pass through the network

    Step-by-step:
    1. Linear: z1 = X @ W1 + b1
    2. Activation: a1 = ReLU(z1)
    3. Linear: z2 = a1 @ W2 + b2
    4. Activation: a2 = Softmax(z2)

    Returns all intermediate values (needed for backprop later)
    """
    # Layer 1
    z1 = X @ W1 + b1           # Shape: (batch, 128)
    a1 = relu(z1)              # Shape: (batch, 128)

    # Layer 2
    z2 = a1 @ W2 + b2          # Shape: (batch, 10)
    a2 = softmax(z2)           # Shape: (batch, 10) - probabilities

    # Cache for backward pass
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache

# Test forward pass with first 5 samples
batch = X_train[:5]
predictions, cache = forward_pass(batch, W1, b1, W2, b2)

print(f"Input batch shape: {batch.shape}")
print(f"Predictions shape: {predictions.shape}")
print(f"\nFirst sample predictions:")
print(predictions[0])
print(f"Sum: {predictions[0].sum():.4f}")
print(f"Predicted class: {predictions[0].argmax()} (highest probability)")

# ==============================================================================
# PART 6: Shape Analysis (This is the KEY to understanding!)
# ==============================================================================

print("\n" + "="*70)
print("PART 6: Understanding Shapes (CRITICAL!)")
print("="*70)

batch_size = 32
print(f"Processing batch of {batch_size} images:\n")

# Forward pass with batch
batch = X_train[:batch_size]
predictions, cache = forward_pass(batch, W1, b1, W2, b2)

print("Shape transformations:")
print(f"  Input X:        {batch.shape}  (32 images, 784 pixels each)")
print(f"  ↓")
print(f"  X @ W1:         {(batch @ W1).shape}  (32, 784) @ (784, 128) = (32, 128)")
print(f"  + b1:           {cache['z1'].shape}  (broadcasting: (32,128) + (128,))")
print(f"  ReLU(z1):       {cache['a1'].shape}  (element-wise, shape unchanged)")
print(f"  ↓")
print(f"  a1 @ W2:        {(cache['a1'] @ W2).shape}  (32, 128) @ (128, 10) = (32, 10)")
print(f"  + b2:           {cache['z2'].shape}  (broadcasting: (32,10) + (10,))")
print(f"  Softmax(z2):    {predictions.shape}  (element-wise, shape unchanged)")
print(f"  ↓")
print(f"  Output:         {predictions.shape}  (32 images → 10 probabilities each)")

# ==============================================================================
# PART 7: Loss Function (How Wrong Are We?)
# ==============================================================================

print("\n" + "="*70)
print("PART 7: Measuring Prediction Quality (Loss)")
print("="*70)

def cross_entropy_loss(predictions, labels):
    """
    Cross-entropy loss for classification

    Lower is better:
    - Perfect predictions → loss ≈ 0
    - Random predictions → loss ≈ 2.3 (for 10 classes)
    - Terrible predictions → loss → ∞
    """
    n = predictions.shape[0]
    # Get probability of correct class for each sample
    correct_class_probs = predictions[np.arange(n), labels]
    # Loss = -log(probability of correct class)
    loss = -np.log(correct_class_probs + 1e-8).mean()  # +1e-8 for stability
    return loss

# Calculate initial loss (should be ~2.3 for random predictions)
batch = X_train[:100]
labels = y_train[:100]
predictions, _ = forward_pass(batch, W1, b1, W2, b2)
loss = cross_entropy_loss(predictions, labels)

print(f"Batch size: {len(batch)}")
print(f"Initial loss: {loss:.4f}")
print(f"(Random guessing for 10 classes → loss ≈ {np.log(10):.4f})")

# ==============================================================================
# PART 8: Accuracy Metric
# ==============================================================================

print("\n" + "="*70)
print("PART 8: Measuring Accuracy")
print("="*70)

def accuracy(predictions, labels):
    """
    Accuracy: percentage of correct predictions
    """
    predicted_classes = predictions.argmax(axis=1)
    correct = (predicted_classes == labels).sum()
    return correct / len(labels)

acc = accuracy(predictions, labels)
print(f"Initial accuracy: {acc*100:.2f}%")
print(f"(Random guessing for 10 classes → accuracy ≈ 10%)")

# ==============================================================================
# PART 9: Visualizing Predictions
# ==============================================================================

print("\n" + "="*70)
print("PART 9: Example Predictions")
print("="*70)

# Make predictions on first 5 samples
batch = X_train[:5]
labels = y_train[:5]
predictions, _ = forward_pass(batch, W1, b1, W2, b2)
predicted_classes = predictions.argmax(axis=1)

print("Sample | True Label | Predicted | Confidence")
print("-" * 50)
for i in range(5):
    confidence = predictions[i, predicted_classes[i]]
    correct = "✓" if predicted_classes[i] == labels[i] else "✗"
    print(f"  {i}    |     {labels[i]}      |     {predicted_classes[i]}     | {confidence:.2%} {correct}")

# ==============================================================================
# PART 10: What Happens Inside a Single Neuron
# ==============================================================================

print("\n" + "="*70)
print("PART 10: Zooming Into a Single Neuron")
print("="*70)

# Take one sample and one neuron from hidden layer
sample = X_train[0]  # Shape: (784,)
neuron_weights = W1[:, 0]  # First neuron's weights, shape: (784,)
neuron_bias = b1[0]  # Scalar

print(f"Input (one image):         shape {sample.shape}")
print(f"Neuron weights:            shape {neuron_weights.shape}")
print(f"Neuron bias:               {neuron_bias}")

# Compute neuron output
weighted_sum = sample @ neuron_weights + neuron_bias  # Dot product
print(f"\nWeighted sum (z):          {weighted_sum:.4f}")

activation = relu(weighted_sum)
print(f"After ReLU activation (a): {activation:.4f}")

print(f"""
What happened:
1. Each of 784 pixels multiplied by its weight
2. All 784 products summed up
3. Bias added
4. ReLU applied: max(0, z)

This neuron now represents a learned feature from the image!
In a trained network, this might detect edges, curves, etc.
""")

# ==============================================================================
# PART 11: Batch Processing Efficiency
# ==============================================================================

print("\n" + "="*70)
print("PART 11: Why Batch Processing is Fast")
print("="*70)

import time

# Process one at a time
single_times = []
for i in range(100):
    start = time.time()
    sample = X_train[i:i+1]  # Keep 2D: (1, 784)
    pred, _ = forward_pass(sample, W1, b1, W2, b2)
    single_times.append(time.time() - start)

avg_single = np.mean(single_times) * 1000  # Convert to ms

# Process as batch
start = time.time()
batch = X_train[:100]
pred, _ = forward_pass(batch, W1, b1, W2, b2)
batch_time = (time.time() - start) * 1000  # Convert to ms

print(f"Processing 100 samples one-by-one: {avg_single * 100:.2f}ms")
print(f"Processing 100 samples as batch:   {batch_time:.2f}ms")
speedup = (avg_single * 100) / batch_time
print(f"\nBatch processing is {speedup:.1f}x FASTER!")
print("This is why we use matrix operations in neural networks!")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("SUMMARY: What We Built")
print("="*70)

print("""
We built a complete neural network using ONLY NumPy!

Key NumPy Operations Used:
✓ Matrix multiplication (@) - for layer computations
✓ Broadcasting (+) - for adding bias
✓ Element-wise operations (*, max) - for activations
✓ Aggregations (sum, mean) - for loss calculation
✓ Boolean indexing - for accuracy
✓ Argmax - for predictions

Network Stats:
""")
print(f"  Parameters:  {total_params:,}")
print(f"  Layers:      2 (+ input)")
print(f"  Architecture: 784 → 128 → 10")

print("""
Real-World Equivalent:
This is similar to (but simpler than):
  - MNIST digit classifier
  - Fashion-MNIST classifier
  - Basic image recognition

Next Steps:
1. In real training, we'd implement backpropagation (Module 3!)
2. Update weights using gradient descent
3. Train for many epochs until loss decreases
4. Achieve 95%+ accuracy on real MNIST data

Everything we did here uses the NumPy skills from Module 2:
  - Arrays and shapes
  - Matrix multiplication
  - Broadcasting
  - Vectorization
  - Aggregations

You now understand the mathematical foundation of neural networks! 🎉
""")

print("="*70)
print("Complete! Try modifying the code to experiment.")
print("="*70)
