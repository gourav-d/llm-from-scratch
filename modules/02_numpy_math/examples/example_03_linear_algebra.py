"""
Linear Algebra with NumPy - Interactive Examples
The mathematical foundation of neural networks and LLMs!
"""

import numpy as np

print("="*70)
print("Linear Algebra for Neural Networks - Examples")
print("="*70)

# ==============================================================================
# EXAMPLE 1: Vectors - The Building Blocks
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Understanding Vectors")
print("="*70)

# Vectors in NumPy
v1 = np.array([3, 4])
v2 = np.array([1, 2])

print(f"Vector v1: {v1}")
print(f"Vector v2: {v2}\n")

# Vector addition
v_add = v1 + v2
print(f"v1 + v2 = {v_add}")

# Scalar multiplication
v_scaled = v1 * 2
print(f"v1 * 2 = {v_scaled}")

# Dot product (similarity measure!)
dot = v1 @ v2
print(f"v1 · v2 (dot product) = {dot}")
print(f"  Calculation: {v1[0]}×{v2[0]} + {v1[1]}×{v2[1]} = {dot}")

# Magnitude (length of vector)
magnitude = np.linalg.norm(v1)
print(f"\n|v1| (magnitude) = {magnitude:.3f}")
print(f"  Calculation: √({v1[0]}² + {v1[1]}²) = {magnitude:.3f}")

# Unit vector (direction only, length = 1)
v_unit = v1 / np.linalg.norm(v1)
print(f"v1 normalized: {v_unit}")
print(f"Length: {np.linalg.norm(v_unit):.3f} (should be 1.0)")

# ==============================================================================
# EXAMPLE 2: Dot Product - The Heart of Neural Networks
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Dot Product in Detail")
print("="*70)

# In neural networks: dot product = weighted sum
inputs = np.array([0.5, 0.8, 0.3])  # Input activations
weights = np.array([0.2, -0.4, 0.6])  # Learned weights

dot_product = inputs @ weights
print(f"Inputs: {inputs}")
print(f"Weights: {weights}")
print(f"\nDot product: {dot_product:.4f}")
print(f"Calculation:")
for i in range(len(inputs)):
    print(f"  {inputs[i]} × {weights[i]:+.1f} = {inputs[i] * weights[i]:+.3f}")
print(f"  Sum = {dot_product:.4f}")

print(f"\nThis is how a single neuron computes its output!")

# Cosine similarity (angle between vectors)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
cos_sim = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"\nCosine similarity between {v1} and {v2}: {cos_sim:.4f}")
print(f"(1.0 = same direction, 0 = perpendicular, -1 = opposite)")

# ==============================================================================
# EXAMPLE 3: Matrices - Neural Network Layers
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Matrix Basics")
print("="*70)

A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3 matrix

print(f"Matrix A (2×3):\n{A}\n")
print(f"Shape: {A.shape}")
print(f"Element at (1,2): {A[1, 2]}")
print(f"Row 0: {A[0, :]}")
print(f"Column 1: {A[:, 1]}")

# Transpose
A_T = A.T
print(f"\nA transpose (3×2):\n{A_T}")
print("Rows become columns, columns become rows!")

# ==============================================================================
# EXAMPLE 4: Matrix Multiplication - Layer Computation
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Matrix Multiplication")
print("="*70)

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print(f"Matrix A:\n{A}\n")
print(f"Matrix B:\n{B}\n")

# Element-wise multiplication (NOT matrix mult!)
elementwise = A * B
print(f"A * B (element-wise):\n{elementwise}\n")

# Matrix multiplication
matrix_mult = A @ B
print(f"A @ B (matrix multiplication):\n{matrix_mult}\n")

print("Matrix multiplication calculation:")
print(f"Position (0,0): {A[0,0]}×{B[0,0]} + {A[0,1]}×{B[1,0]} = {matrix_mult[0,0]}")
print(f"Position (0,1): {A[0,0]}×{B[0,1]} + {A[0,1]}×{B[1,1]} = {matrix_mult[0,1]}")
print(f"Position (1,0): {A[1,0]}×{B[0,0]} + {A[1,1]}×{B[1,0]} = {matrix_mult[1,0]}")
print(f"Position (1,1): {A[1,0]}×{B[0,1]} + {A[1,1]}×{B[1,1]} = {matrix_mult[1,1]}")

# ==============================================================================
# EXAMPLE 5: Shape Compatibility
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: Understanding Shape Rules")
print("="*70)

print("Matrix multiplication rule: (m, n) @ (n, p) = (m, p)")
print("                                    ↑___↑")
print("                               Inner dimensions MUST match!\n")

def test_matmul(shape1, shape2):
    try:
        A = np.random.randn(*shape1)
        B = np.random.randn(*shape2)
        C = A @ B
        print(f"{shape1} @ {shape2} = {C.shape} ✓")
    except ValueError:
        print(f"{shape1} @ {shape2} = Error! (incompatible shapes) ✗")

test_matmul((2, 3), (3, 4))
test_matmul((5, 2), (2, 7))
test_matmul((10, 5), (5, 1))
test_matmul((3, 4), (5, 2))  # This will fail!

# ==============================================================================
# EXAMPLE 6: Neural Network Forward Pass
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 6: Simulating a Neural Network Layer")
print("="*70)

# Single sample
print("Single sample (1 image):")
x = np.random.randn(1, 784)  # 1 flattened 28x28 image
W = np.random.randn(784, 128) * 0.01  # Weights
b = np.zeros(128)  # Bias

output = x @ W + b
print(f"Input shape: {x.shape} (1 sample, 784 pixels)")
print(f"Weights shape: {W.shape} (784 inputs → 128 neurons)")
print(f"Bias shape: {b.shape}")
print(f"Output shape: {output.shape} (1 sample, 128 activations)")

# Batch processing (more efficient!)
print("\nBatch of 32 samples:")
X = np.random.randn(32, 784)  # 32 images
output_batch = X @ W + b
print(f"Input shape: {X.shape} (32 samples, 784 pixels)")
print(f"Output shape: {output_batch.shape} (32 samples, 128 activations)")
print("All 32 images processed in ONE matrix multiplication!")

# ==============================================================================
# EXAMPLE 7: Identity Matrix
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 7: Identity Matrix (Like 1 in Multiplication)")
print("="*70)

I = np.eye(3)  # 3x3 identity
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"Identity matrix I:\n{I}\n")
print(f"Matrix A:\n{A}\n")
print(f"A @ I =\n{A @ I}")
print("(Returns A unchanged, just like A × 1 = A)")

# ==============================================================================
# EXAMPLE 8: Matrix Inverse
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 8: Matrix Inverse")
print("="*70)

A = np.array([[4, 7],
              [2, 6]], dtype=float)

print(f"Matrix A:\n{A}\n")

A_inv = np.linalg.inv(A)
print(f"A inverse:\n{A_inv}\n")

# Verify: A @ A_inv should be identity
product = A @ A_inv
print(f"A @ A_inv (should be identity):\n{product}\n")
print("(Values are very close to identity, small errors from floating point)")

# ==============================================================================
# EXAMPLE 9: Solving Linear Equations
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 9: Solving Ax = b")
print("="*70)

# System of equations:
# 3x + y = 9
# x + 2y = 8

A = np.array([[3, 1],
              [1, 2]])
b = np.array([9, 8])

print("System of equations:")
print(f"  3x + y = 9")
print(f"  x + 2y = 8\n")

x = np.linalg.solve(A, b)
print(f"Solution: x = {x[0]}, y = {x[1]}")

# Verify
result = A @ x
print(f"\nVerification: A @ x = {result}")
print(f"Should equal b = {b}")
print(f"Correct: {np.allclose(result, b)}")

# ==============================================================================
# EXAMPLE 10: Norms (Vector Length)
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 10: Vector Norms")
print("="*70)

v = np.array([3, 4])

# L2 norm (Euclidean distance)
l2 = np.linalg.norm(v)
print(f"Vector: {v}")
print(f"L2 norm (Euclidean): {l2}")
print(f"  (√(3² + 4²) = √25 = 5)\n")

# L1 norm (Manhattan distance)
l1 = np.linalg.norm(v, ord=1)
print(f"L1 norm (Manhattan): {l1}")
print(f"  (|3| + |4| = 7)\n")

# Used in neural networks for:
# - Gradient clipping (L2 norm)
# - Regularization (L1 and L2)
# - Distance calculations

# ==============================================================================
# EXAMPLE 11: Full Neural Network Example
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 11: Two-Layer Neural Network")
print("="*70)

# Network: 4 inputs → 5 hidden → 3 outputs
print("Network architecture: 4 → 5 → 3")

# Input
X = np.random.randn(2, 4)  # 2 samples, 4 features each
print(f"\nInput X shape: {X.shape}")

# Layer 1: 4 → 5
W1 = np.random.randn(4, 5) * 0.1
b1 = np.zeros(5)
Z1 = X @ W1 + b1
A1 = np.maximum(0, Z1)  # ReLU activation
print(f"Layer 1 output shape: {A1.shape}")

# Layer 2: 5 → 3
W2 = np.random.randn(5, 3) * 0.1
b2 = np.zeros(3)
Z2 = A1 @ W2 + b2
# Softmax activation
exp_Z2 = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
A2 = exp_Z2 / exp_Z2.sum(axis=1, keepdims=True)
print(f"Layer 2 output shape: {A2.shape}")

print(f"\nFinal output (probabilities):\n{A2}")
print(f"Sum of probabilities: {A2.sum(axis=1)}")
print("(Should be [1., 1.] - probabilities sum to 1 for each sample)")

# ==============================================================================
# EXAMPLE 12: Attention Mechanism (Simplified)
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 12: Simplified Attention Mechanism")
print("="*70)

# Sequence of 4 words, each represented by a 6-dimensional vector
seq_len = 4
d_model = 6

# Query, Key, Value matrices
Q = np.random.randn(1, seq_len, d_model)  # (batch, seq_len, d_model)
K = np.random.randn(1, seq_len, d_model)
V = np.random.randn(1, seq_len, d_model)

print(f"Query shape: {Q.shape}")
print(f"Key shape: {K.shape}")
print(f"Value shape: {V.shape}\n")

# Attention scores: Q @ K^T
scores = Q @ K.transpose(0, 2, 1)  # (1, 4, 6) @ (1, 6, 4) = (1, 4, 4)
print(f"Attention scores shape: {scores.shape}")
print(f"(Each word attends to each other word)")

# Attention weights (softmax over scores)
exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

print(f"\nAttention weights for first word:\n{attention_weights[0, 0]}")
print(f"Sum: {attention_weights[0, 0].sum():.4f} (should be 1.0)")

# Context vectors: weighted combination of values
context = attention_weights @ V  # (1, 4, 4) @ (1, 4, 6) = (1, 4, 6)
print(f"\nContext vectors shape: {context.shape}")
print("Each word now has context from all other words!")

# ==============================================================================
# EXAMPLE 13: Eigenvalues and Eigenvectors
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 13: Eigenvalues and Eigenvectors")
print("="*70)

A = np.array([[4, -2],
              [1,  1]])

print(f"Matrix A:\n{A}\n")

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}\n")

# Verify: A @ v = λ @ v
v = eigenvectors[:, 0]
lambda_val = eigenvalues[0]
left_side = A @ v
right_side = lambda_val * v

print(f"Verification for first eigenvector:")
print(f"A @ v = {left_side}")
print(f"λ × v = {right_side}")
print(f"Equal: {np.allclose(left_side, right_side)}")

print("\nUsed in: PCA, understanding neural network behavior")

# ==============================================================================
print("\n" + "="*70)
print("✅ Linear Algebra Examples Complete!")
print("="*70)
print("""
Key Takeaways:
1. Vectors represent data points, features, or embeddings
2. Dot product measures similarity (used everywhere in ML!)
3. Matrix multiplication is the core operation in neural networks
4. Shape compatibility: (m,n) @ (n,p) = (m,p)
5. Transpose swaps dimensions to make shapes compatible
6. Every layer in a neural network is: X @ W + b
7. Attention mechanism uses matrix operations (Q, K, V)

Real LLMs like GPT use these operations billions of times!

Next: Apply these concepts in exercises and build understanding!
""")
