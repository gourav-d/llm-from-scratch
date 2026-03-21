"""
Example 1: PyTorch Tensor Operations
=====================================

This example demonstrates:
- Creating tensors in various ways
- Basic tensor operations
- Automatic differentiation
- GPU operations (if available)

Run: python example_01_pytorch_tensors.py
"""

import torch
import numpy as np

print("=" * 60)
print("PyTorch Tensor Operations Example")
print("=" * 60)

# ============================================================================
# Part 1: Creating Tensors
# ============================================================================
print("\n1. Creating Tensors")
print("-" * 60)

# From Python list
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print(f"From list: {tensor_from_list}")

# Zeros and ones
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
print(f"\nZeros (3x4):\n{zeros}")
print(f"\nOnes (2x3):\n{ones}")

# Random tensors
random_uniform = torch.rand(2, 3)  # Uniform [0, 1)
random_normal = torch.randn(2, 3)  # Normal distribution
print(f"\nRandom uniform:\n{random_uniform}")
print(f"\nRandom normal:\n{random_normal}")

# Range and linspace
range_tensor = torch.arange(0, 10, 2)
linspace_tensor = torch.linspace(0, 1, 5)
print(f"\nRange (0 to 10, step 2): {range_tensor}")
print(f"Linspace (0 to 1, 5 points): {linspace_tensor}")

# ============================================================================
# Part 2: Tensor Properties
# ============================================================================
print("\n\n2. Tensor Properties")
print("-" * 60)

x = torch.randn(3, 4, 5)
print(f"Shape: {x.shape}")
print(f"Size: {x.size()}")  # Same as shape
print(f"Number of dimensions: {x.ndim}")
print(f"Total elements: {x.numel()}")
print(f"Data type: {x.dtype}")
print(f"Device: {x.device}")

# ============================================================================
# Part 3: Basic Operations
# ============================================================================
print("\n\n3. Basic Operations")
print("-" * 60)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")  # Element-wise
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")

# ============================================================================
# Part 4: Matrix Operations
# ============================================================================
print("\n\n4. Matrix Operations")
print("-" * 60)

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print(f"Matrix A:\n{A}")
print(f"\nMatrix B:\n{B}")

# Element-wise multiplication
print(f"\nElement-wise (A * B):\n{A * B}")

# Matrix multiplication
print(f"\nMatrix multiplication (A @ B):\n{A @ B}")

# Transpose
print(f"\nTranspose of A:\n{A.T}")

# Reshape
print(f"\nReshape A to (1, 4):\n{A.view(1, 4)}")

# ============================================================================
# Part 5: Indexing and Slicing
# ============================================================================
print("\n\n5. Indexing and Slicing")
print("-" * 60)

x = torch.arange(1, 10).reshape(3, 3)
print(f"Original tensor:\n{x}")

print(f"\nElement [0, 0]: {x[0, 0]}")
print(f"First row: {x[0, :]}")
print(f"First column: {x[:, 0]}")
print(f"Submatrix [0:2, 0:2]:\n{x[0:2, 0:2]}")

# Boolean indexing
mask = x > 5
print(f"\nMask (x > 5):\n{mask}")
print(f"Elements > 5: {x[mask]}")

# ============================================================================
# Part 6: Automatic Differentiation
# ============================================================================
print("\n\n6. Automatic Differentiation")
print("-" * 60)

# Example 1: Simple gradient
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

print(f"x = {x.item()}")
print(f"y = x² + 3x + 1 = {y.item()}")

y.backward()  # Compute gradients
print(f"dy/dx = 2x + 3 = {x.grad.item()}")
print(f"Expected: 2*2 + 3 = 7 ✓" if abs(x.grad.item() - 7) < 1e-6 else "❌")

# Example 2: Multi-variable
print("\nMulti-variable example:")
x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)
z = x * y + y ** 2

print(f"z = xy + y² = {z.item()}")

z.backward()
print(f"∂z/∂x = y = {x.grad.item()}")
print(f"∂z/∂y = x + 2y = {y.grad.item()}")

# ============================================================================
# Part 7: NumPy Conversion
# ============================================================================
print("\n\n7. NumPy Conversion")
print("-" * 60)

# NumPy to PyTorch
np_array = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(np_array)
print(f"NumPy array:\n{np_array}")
print(f"PyTorch tensor:\n{tensor}")

# PyTorch to NumPy
tensor = torch.tensor([[5, 6], [7, 8]])
np_array = tensor.numpy()
print(f"\nPyTorch tensor:\n{tensor}")
print(f"NumPy array:\n{np_array}")

# ============================================================================
# Part 8: GPU Operations (if available)
# ============================================================================
print("\n\n8. GPU Operations")
print("-" * 60)

print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create tensor on GPU
    x_gpu = torch.randn(3, 3, device='cuda')
    y_gpu = torch.randn(3, 3, device='cuda')

    # Compute on GPU
    z_gpu = x_gpu @ y_gpu

    print(f"Tensor on GPU: {x_gpu.device}")
    print(f"Computation done on GPU!")

    # Move back to CPU
    z_cpu = z_gpu.cpu()
    print(f"Moved back to CPU: {z_cpu.device}")
else:
    print("No GPU available. Running on CPU.")

    # CPU example
    device = torch.device('cpu')
    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    z = x @ y
    print(f"Tensor on CPU: {x.device}")

# ============================================================================
# Summary
# ============================================================================
print("\n\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
✅ Created tensors in multiple ways
✅ Performed basic operations
✅ Used matrix operations
✅ Applied indexing and slicing
✅ Computed automatic gradients
✅ Converted between NumPy and PyTorch
✅ Demonstrated GPU operations
""")

print("\nNext: Run example_02_mnist_pytorch.py for a complete neural network!")
print("=" * 60)
