"""
Linear Algebra - Practice Exercises

Master the mathematical foundation of neural networks!
"""

import numpy as np

print("="*70)
print("Linear Algebra - Exercises")
print("="*70)

# ==============================================================================
# EXERCISE 1: Vector Operations
# ==============================================================================
print("\n📝 EXERCISE 1: Working with Vectors")
print("-" * 70)
v1 = np.array([2, 3, 4])
v2 = np.array([1, -1, 2])
print(f"v1 = {v1}")
print(f"v2 = {v2}\n")
print("Calculate:")
print("a) v1 + v2")
print("b) v1 - v2")
print("c) 3 * v1")
print("d) Dot product: v1 · v2")
print("e) Magnitude (length) of v1")
print("f) Normalize v1 to unit length")
print("g) Cosine similarity between v1 and v2")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 2: Matrix Basics
# ==============================================================================
print("\n📝 EXERCISE 2: Matrix Operations")
print("-" * 70)
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"Matrix A:\n{A}\n")
print("Perform:")
print("a) Transpose of A")
print("b) Extract the second row")
print("c) Extract the first column")
print("d) Get element at position (1, 2)")
print("e) Multiply all elements by 2")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 3: Matrix Multiplication vs Element-wise
# ==============================================================================
print("\n📝 EXERCISE 3: Understanding * vs @")
print("-" * 70)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"Matrix A:\n{A}\n")
print(f"Matrix B:\n{B}\n")
print("Calculate:")
print("a) A * B (element-wise)")
print("b) A @ B (matrix multiplication)")
print("c) Verify your matrix mult by hand for position (0,0)")
print("d) A @ B vs B @ A (are they the same?)")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 4: Shape Compatibility
# ==============================================================================
print("\n📝 EXERCISE 4: Testing Shape Compatibility")
print("-" * 70)
print("For each pair, predict if matrix multiplication will work:")
print("a) (3, 4) @ (4, 5)")
print("b) (2, 3) @ (2, 4)")
print("c) (5, 2) @ (2, 1)")
print("d) (10, 784) @ (784, 128)")
print("\nThen create random matrices and test your predictions!")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 5: Neural Network Layer
# ==============================================================================
print("\n📝 EXERCISE 5: Implementing a Layer")
print("-" * 70)
print("Implement a neural network layer:")
print("- Input: 16 samples, 20 features each")
print("- Output: 16 samples, 10 neurons")
print("\nSteps:")
print("a) Create input X (16, 20) with random values")
print("b) Create weights W (20, 10) with small random values (* 0.01)")
print("c) Create bias b (10,) initialized to zeros")
print("d) Compute linear output: Z = X @ W + b")
print("e) Apply ReLU activation: A = max(0, Z)")
print("f) Verify output shape is (16, 10)")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 6: Batch Processing
# ==============================================================================
print("\n📝 EXERCISE 6: Understanding Batches")
print("-" * 70)
print("Process MNIST-like data:")
print("- Each image: 28×28 = 784 pixels")
print("- Batch size: 64")
print("- Target: 10 classes")
print("\nImplement:")
print("a) Create batch of flattened images: (64, 784)")
print("b) Create weight matrix to 128 hidden units")
print("c) Create bias for hidden layer")
print("d) Forward pass to hidden layer with ReLU")
print("e) Create weights from hidden to output (10 classes)")
print("f) Forward pass to output")
print("g) Print all intermediate shapes")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 7: Identity and Inverse
# ==============================================================================
print("\n📝 EXERCISE 7: Special Matrices")
print("-" * 70)
print("a) Create a 4×4 identity matrix")
print("b) Create a random 3×3 matrix")
print("c) Compute its inverse")
print("d) Verify: A @ A_inv ≈ Identity")
print("e) What happens if you try to invert a singular matrix?")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 8: Solving Linear Systems
# ==============================================================================
print("\n📝 EXERCISE 8: Linear Equations")
print("-" * 70)
print("Solve the system:")
print("  2x + 3y = 13")
print("  5x - y = 7")
print("\na) Set up coefficient matrix A")
print("b) Set up constants vector b")
print("c) Solve for x and y")
print("d) Verify your solution")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 9: Norms and Distances
# ==============================================================================
print("\n📝 EXERCISE 9: Vector Norms")
print("-" * 70)
v = np.array([3, 4, 12])
print(f"Vector v = {v}\n")
print("Calculate:")
print("a) L2 norm (Euclidean distance)")
print("b) L1 norm (Manhattan distance)")
print("c) Normalize v to unit length (L2 norm = 1)")
print("d) Calculate distance between [1,2,3] and [4,5,6] (L2 norm)")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 10: Real-World - Word Embeddings
# ==============================================================================
print("\n📝 EXERCISE 10: Word Similarity with Embeddings")
print("-" * 70)
print("Simulate word embeddings:")
print("a) Create embeddings for 5 words, each 10-dimensional")
print("b) Calculate cosine similarity between word 0 and word 1")
print("c) Find which word is most similar to word 0")
print("d) Normalize all embeddings to unit length")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 11: Transpose for Shape Matching
# ==============================================================================
print("\n📝 EXERCISE 11: Using Transpose")
print("-" * 70)
print("You have:")
print("- Features: (100, 50) - 100 samples, 50 features")
print("- Weights: (10, 50) - 10 neurons, 50 weights each")
print("\nProblem: Can't multiply (100,50) @ (10,50)")
print("\nTasks:")
print("a) Transpose weights to make multiplication possible")
print("b) Perform the multiplication")
print("c) What is the output shape?")
print("d) What does this output represent?")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 12: Challenge - Attention Scores
# ==============================================================================
print("\n📝 EXERCISE 12: Challenge - Simplified Attention")
print("-" * 70)
print("Implement simplified self-attention:")
print("- Sequence length: 5 words")
print("- Embedding dimension: 8")
print("\nSteps:")
print("a) Create Query matrix Q (5, 8)")
print("b) Create Key matrix K (5, 8)")
print("c) Calculate attention scores: Q @ K^T")
print("d) What is the shape of attention scores?")
print("e) Apply softmax to each row of scores")
print("f) Verify each row sums to 1")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 13: Challenge - Full Forward Pass
# ==============================================================================
print("\n📝 EXERCISE 13: Challenge - 3-Layer Network")
print("-" * 70)
print("Build a complete forward pass:")
print("Architecture: 784 → 256 → 128 → 10")
print("\na) Initialize all weights and biases")
print("b) Create batch of 32 samples (784 features)")
print("c) Forward pass through all 3 layers")
print("d) Use ReLU for hidden layers, softmax for output")
print("e) Verify output shape is (32, 10)")
print("f) Verify output probabilities sum to 1 per sample")

# YOUR CODE HERE:

# ==============================================================================
# SOLUTIONS
# ==============================================================================

def show_solutions():
    print("\n\n" + "="*70)
    input("Press Enter to see solutions... ")
    print("="*70)

    print("\n💡 SOLUTION 1: Vector Operations")
    print("-" * 70)
    v1 = np.array([2, 3, 4])
    v2 = np.array([1, -1, 2])
    print(f"a) v1 + v2: {v1 + v2}")
    print(f"b) v1 - v2: {v1 - v2}")
    print(f"c) 3 * v1: {3 * v1}")
    print(f"d) v1 · v2: {v1 @ v2}")
    print(f"e) |v1|: {np.linalg.norm(v1):.4f}")
    v1_normalized = v1 / np.linalg.norm(v1)
    print(f"f) normalized: {v1_normalized}")
    cos_sim = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"g) cosine sim: {cos_sim:.4f}")

    print("\n💡 SOLUTION 2: Matrix Operations")
    print("-" * 70)
    A = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"a) Transpose:\n{A.T}")
    print(f"b) Second row: {A[1, :]}")
    print(f"c) First column: {A[:, 0]}")
    print(f"d) Element (1,2): {A[1, 2]}")
    print(f"e) A * 2:\n{A * 2}")

    print("\n💡 SOLUTION 3: * vs @")
    print("-" * 70)
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(f"a) A * B:\n{A * B}")
    print(f"b) A @ B:\n{A @ B}")
    print(f"c) (0,0) = 1×5 + 2×7 = {1*5 + 2*7}")
    print(f"d) A@B != B@A: {not np.array_equal(A@B, B@A)}")

    print("\n💡 SOLUTION 4: Shape Compatibility")
    print("-" * 70)
    print("a) (3,4) @ (4,5) = (3,5) ✓ (4 matches)")
    print("b) (2,3) @ (2,4) = Error ✗ (3 ≠ 2)")
    print("c) (5,2) @ (2,1) = (5,1) ✓ (2 matches)")
    print("d) (10,784) @ (784,128) = (10,128) ✓ (784 matches)")

    print("\n💡 SOLUTION 5: Neural Network Layer")
    print("-" * 70)
    X = np.random.randn(16, 20)
    W = np.random.randn(20, 10) * 0.01
    b = np.zeros(10)
    Z = X @ W + b
    A = np.maximum(0, Z)
    print(f"Input shape: {X.shape}")
    print(f"Weights shape: {W.shape}")
    print(f"Output shape: {A.shape}")
    print(f"Correct: {A.shape == (16, 10)}")

    print("\n💡 SOLUTION 6: Batch Processing")
    print("-" * 70)
    X = np.random.randn(64, 784)
    W1 = np.random.randn(784, 128) * 0.01
    b1 = np.zeros(128)
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    W2 = np.random.randn(128, 10) * 0.01
    b2 = np.zeros(10)
    Z2 = A1 @ W2 + b2
    print(f"Input: {X.shape}")
    print(f"Hidden: {A1.shape}")
    print(f"Output: {Z2.shape}")

    print("\n💡 SOLUTION 7: Identity and Inverse")
    print("-" * 70)
    I = np.eye(4)
    print(f"a) Identity:\n{I}")
    A = np.random.randn(3, 3)
    A_inv = np.linalg.inv(A)
    print(f"c) Inverse computed")
    print(f"d) A @ A_inv ≈ I: {np.allclose(A @ A_inv, np.eye(3))}")

    print("\n💡 SOLUTION 8: Linear System")
    print("-" * 70)
    A = np.array([[2, 3], [5, -1]])
    b = np.array([13, 7])
    x = np.linalg.solve(A, b)
    print(f"Solution: x={x[0]:.2f}, y={x[1]:.2f}")
    print(f"Verify: {np.allclose(A @ x, b)}")

    print("\n💡 SOLUTION 9: Norms")
    print("-" * 70)
    v = np.array([3, 4, 12])
    l2 = np.linalg.norm(v)
    l1 = np.linalg.norm(v, ord=1)
    normalized = v / np.linalg.norm(v)
    p1 = np.array([1, 2, 3])
    p2 = np.array([4, 5, 6])
    distance = np.linalg.norm(p2 - p1)
    print(f"a) L2 norm: {l2:.4f}")
    print(f"b) L1 norm: {l1:.4f}")
    print(f"c) Normalized: {normalized}")
    print(f"d) Distance: {distance:.4f}")

    print("\n💡 SOLUTION 10: Word Embeddings")
    print("-" * 70)
    embeddings = np.random.randn(5, 10)
    w0 = embeddings[0]
    w1 = embeddings[1]
    cos_sim = (w0 @ w1) / (np.linalg.norm(w0) * np.linalg.norm(w1))
    print(f"b) Similarity word 0 & 1: {cos_sim:.4f}")
    similarities = embeddings @ w0 / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(w0))
    most_similar = np.argsort(similarities)[-2]  # -2 to exclude itself
    print(f"c) Most similar to word 0: word {most_similar}")

    print("\n💡 SOLUTION 11: Transpose")
    print("-" * 70)
    features = np.random.randn(100, 50)
    weights = np.random.randn(10, 50)
    weights_T = weights.T
    output = features @ weights_T
    print(f"a) Weights transposed: {weights_T.shape}")
    print(f"b) Output: {output.shape}")
    print(f"d) Represents: 100 samples, 10 neuron outputs")

    print("\n💡 SOLUTION 12: Attention Scores")
    print("-" * 70)
    Q = np.random.randn(5, 8)
    K = np.random.randn(5, 8)
    scores = Q @ K.T
    print(f"c) Scores shape: {scores.shape}")
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    print(f"f) Row sums: {attention_weights.sum(axis=1)}")

    print("\n💡 SOLUTION 13: Full Forward Pass")
    print("-" * 70)
    W1 = np.random.randn(784, 256) * 0.01
    b1 = np.zeros(256)
    W2 = np.random.randn(256, 128) * 0.01
    b2 = np.zeros(128)
    W3 = np.random.randn(128, 10) * 0.01
    b3 = np.zeros(10)

    X = np.random.randn(32, 784)

    # Layer 1
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)

    # Layer 2
    Z2 = A1 @ W2 + b2
    A2 = np.maximum(0, Z2)

    # Layer 3 (output)
    Z3 = A2 @ W3 + b3
    exp_Z3 = np.exp(Z3 - np.max(Z3, axis=1, keepdims=True))
    A3 = exp_Z3 / exp_Z3.sum(axis=1, keepdims=True)

    print(f"Output shape: {A3.shape}")
    print(f"Probability sums: {A3.sum(axis=1)[:5]}")
    print("All ≈ 1.0 ✓")

if __name__ == "__main__":
    show_solutions()
