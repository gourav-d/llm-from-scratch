"""
NumPy Array Operations and Broadcasting - Interactive Examples
Learn how NumPy makes operations fast and efficient
"""

import numpy as np
import time

print("="*70)
print("NumPy Array Operations and Broadcasting Examples")
print("="*70)

# ==============================================================================
# EXAMPLE 1: Element-wise Operations (Vectorization)
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Element-wise Operations (No Loops Needed!)")
print("="*70)

arr = np.array([1, 2, 3, 4, 5])
print(f"Original array: {arr}\n")

# Arithmetic operations
print(f"arr + 10:  {arr + 10}")
print(f"arr - 2:   {arr - 2}")
print(f"arr * 3:   {arr * 3}")
print(f"arr / 2:   {arr / 2}")
print(f"arr ** 2:  {arr ** 2}  (square each element)")
print(f"arr % 2:   {arr % 2}  (modulo)")

# Between arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"\narr1: {arr1}")
print(f"arr2: {arr2}")
print(f"arr1 + arr2: {arr1 + arr2}")
print(f"arr1 * arr2: {arr1 * arr2}  (element-wise!)")

# ==============================================================================
# EXAMPLE 2: Broadcasting - The Magic of NumPy
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Broadcasting - Different Shapes, Same Operation")
print("="*70)

# Scalar broadcasting
arr = np.array([1, 2, 3, 4])
scalar = 10
result = arr + scalar
print(f"Array:  {arr}")
print(f"Scalar: {scalar}")
print(f"Result: {result}")
print("(Scalar was broadcast to [10, 10, 10, 10])\n")

# Row vector broadcasting
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row_vector = np.array([10, 20, 30])

print(f"Matrix:\n{matrix}\n")
print(f"Row vector: {row_vector}\n")
result = matrix + row_vector
print(f"Matrix + Row vector:\n{result}")
print("(Row vector broadcast to each row)\n")

# Column vector broadcasting
col_vector = np.array([[10],
                       [20],
                       [30]])  # Shape: (3, 1)
print(f"Column vector:\n{col_vector}\n")
result = matrix + col_vector
print(f"Matrix + Column vector:\n{result}")
print("(Column vector broadcast to each column)\n")

# ==============================================================================
# EXAMPLE 3: Broadcasting Rules and Compatibility
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Understanding Broadcasting Compatibility")
print("="*70)

def test_broadcast(shape1, shape2):
    """Test if two shapes can be broadcast together"""
    try:
        a = np.zeros(shape1)
        b = np.zeros(shape2)
        c = a + b
        print(f"{shape1} + {shape2} = {c.shape} ✓")
    except ValueError as e:
        print(f"{shape1} + {shape2} = Error ✗")

print("Testing shape compatibility:")
test_broadcast((3, 4), (3, 4))    # Same shape
test_broadcast((3, 4), (1, 4))    # Broadcast rows
test_broadcast((3, 4), (3, 1))    # Broadcast columns
test_broadcast((3, 4), (4,))      # Broadcast to all rows
test_broadcast((3, 4), (3,))      # Incompatible!

# ==============================================================================
# EXAMPLE 4: Universal Functions (ufuncs)
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Mathematical Functions (ufuncs)")
print("="*70)

arr = np.array([1, 4, 9, 16, 25])
print(f"Array: {arr}\n")

print(f"Square root:     {np.sqrt(arr)}")
print(f"Exponential:     {np.exp([0, 1, 2])}")
print(f"Natural log:     {np.log(arr)}")
print(f"Log base 10:     {np.log10(arr)}")
print(f"Absolute value:  {np.abs([-1, -2, 3, -4])}")

# Trigonometric
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"\nAngles (radians): {angles}")
print(f"Sin: {np.sin(angles)}")
print(f"Cos: {np.cos(angles)}")

# Rounding
values = np.array([1.2, 2.5, 3.7, 4.1])
print(f"\nValues: {values}")
print(f"Round: {np.round(values)}")
print(f"Floor: {np.floor(values)}")
print(f"Ceil:  {np.ceil(values)}")

# ==============================================================================
# EXAMPLE 5: Aggregation Functions
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: Aggregations - Summarizing Data")
print("="*70)

data = np.array([85, 92, 78, 95, 88, 76, 89, 91])
print(f"Test scores: {data}\n")

print(f"Sum:        {data.sum()}")
print(f"Mean:       {data.mean():.2f}")
print(f"Median:     {np.median(data):.2f}")
print(f"Std Dev:    {data.std():.2f}")
print(f"Variance:   {data.var():.2f}")
print(f"Min:        {data.min()}")
print(f"Max:        {data.max()}")
print(f"Range:      {data.max() - data.min()}")

# Index of min/max
print(f"\nIndex of min: {data.argmin()} (score: {data[data.argmin()]})")
print(f"Index of max: {data.argmax()} (score: {data[data.argmax()]})")

# ==============================================================================
# EXAMPLE 6: Axis Parameter - Critical for Multi-dimensional Arrays
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 6: Aggregations Along Axes")
print("="*70)

# Student scores: rows=students, cols=assignments
scores = np.array([
    [85, 92, 78],  # Student 1
    [95, 88, 76],  # Student 2
    [89, 91, 94],  # Student 3
    [70, 85, 88]   # Student 4
])

print(f"Scores matrix (students × assignments):\n{scores}\n")

# Overall statistics
print(f"Overall sum:  {scores.sum()}")
print(f"Overall mean: {scores.mean():.2f}\n")

# axis=0: operate down columns (across students)
print(f"Sum per assignment (axis=0):  {scores.sum(axis=0)}")
print(f"Mean per assignment (axis=0): {scores.mean(axis=0)}")
print("(Average score for each assignment across all students)\n")

# axis=1: operate across rows (across assignments)
print(f"Sum per student (axis=1):  {scores.sum(axis=1)}")
print(f"Mean per student (axis=1): {scores.mean(axis=1)}")
print("(Average score for each student across all assignments)\n")

# ==============================================================================
# EXAMPLE 7: Cumulative Operations
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 7: Cumulative Operations")
print("="*70)

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}\n")

cumsum = np.cumsum(arr)
print(f"Cumulative sum: {cumsum}")
print("(1, 1+2=3, 3+3=6, 6+4=10, 10+5=15)\n")

cumprod = np.cumprod(arr)
print(f"Cumulative product: {cumprod}")
print("(1, 1*2=2, 2*3=6, 6*4=24, 24*5=120)\n")

# ==============================================================================
# EXAMPLE 8: Stacking and Concatenating
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 8: Combining Arrays")
print("="*70)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Array a: {a}")
print(f"Array b: {b}\n")

# Horizontal stack (side by side)
h = np.hstack([a, b])
print(f"Horizontal stack: {h}")
print(f"Shape: {h.shape}\n")

# Vertical stack (on top of each other)
v = np.vstack([a, b])
print(f"Vertical stack:\n{v}")
print(f"Shape: {v.shape}\n")

# Concatenate (more general)
concat = np.concatenate([a, b])
print(f"Concatenate: {concat}")

# 2D example
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
print(f"\nMatrix 1:\n{mat1}\n")
print(f"Matrix 2:\n{mat2}\n")

combined_v = np.vstack([mat1, mat2])
print(f"Vertical stack:\n{combined_v}\n")

combined_h = np.hstack([mat1, mat2])
print(f"Horizontal stack:\n{combined_h}\n")

# ==============================================================================
# EXAMPLE 9: Splitting Arrays
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 9: Splitting Arrays")
print("="*70)

arr = np.arange(12)
print(f"Array: {arr}\n")

# Split into 3 equal parts
parts = np.split(arr, 3)
print("Split into 3 parts:")
for i, part in enumerate(parts):
    print(f"  Part {i}: {part}")

# Split at specific indices
arr = np.arange(10)
parts = np.split(arr, [3, 7])  # Split at indices 3 and 7
print(f"\nArray: {arr}")
print("Split at indices [3, 7]:")
for i, part in enumerate(parts):
    print(f"  Part {i}: {part}")

# ==============================================================================
# EXAMPLE 10: Random Numbers (Important for ML!)
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 10: Random Number Generation")
print("="*70)

# Set seed for reproducibility
np.random.seed(42)
print("Random seed set to 42 for reproducibility\n")

# Uniform distribution [0, 1)
uniform = np.random.rand(5)
print(f"Uniform [0,1): {uniform}")

# Uniform in range
uniform_range = np.random.uniform(10, 20, 5)
print(f"Uniform [10,20): {uniform_range}")

# Normal distribution (mean=0, std=1)
normal = np.random.randn(5)
print(f"Normal (0,1): {normal}")

# Normal with custom mean and std
custom_normal = np.random.normal(100, 15, 5)  # IQ scores
print(f"Normal (100,15): {custom_normal}")

# Random integers
randint = np.random.randint(1, 7, 10)  # Roll dice 10 times
print(f"Random dice rolls: {randint}")

# Random choice from array
choices = np.random.choice(['red', 'green', 'blue'], size=5)
print(f"Random colors: {choices}")

# Shuffle
deck = np.arange(52)
np.random.shuffle(deck)
print(f"Shuffled deck (first 10): {deck[:10]}")

# ==============================================================================
# EXAMPLE 11: Performance Comparison
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 11: Why Vectorization Matters (Speed Test)")
print("="*70)

size = 1000000
arr = np.arange(size)

# Python loop (SLOW)
start = time.time()
result_loop = []
for x in arr:
    result_loop.append(x * 2)
time_loop = (time.time() - start) * 1000

# List comprehension (Better but still slow)
start = time.time()
result_comp = [x * 2 for x in arr]
time_comp = (time.time() - start) * 1000

# NumPy vectorized (FAST!)
start = time.time()
result_numpy = arr * 2
time_numpy = (time.time() - start) * 1000

print(f"Array size: {size:,} elements\n")
print(f"Python for loop:       {time_loop:.2f} ms")
print(f"List comprehension:    {time_comp:.2f} ms")
print(f"NumPy vectorized:      {time_numpy:.2f} ms")
print(f"\nNumPy is {time_loop/time_numpy:.1f}x faster than for loop!")
print(f"NumPy is {time_comp/time_numpy:.1f}x faster than list comp!")

# ==============================================================================
# EXAMPLE 12: Real-World - Neural Network Bias Addition
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 12: Real-World - Adding Bias in Neural Networks")
print("="*70)

# Batch of 32 samples, 10 neurons
batch_size = 32
n_neurons = 10

# Pre-activation values (before adding bias)
Z = np.random.randn(batch_size, n_neurons)
# Bias (one value per neuron, shared across all samples)
bias = np.random.randn(n_neurons)

print(f"Z shape (before bias): {Z.shape}")
print(f"Bias shape: {bias.shape}")

# Add bias - broadcasting happens automatically!
Z_with_bias = Z + bias

print(f"Z + bias shape: {Z_with_bias.shape}")
print("\nHow broadcasting works here:")
print(f"  {Z.shape} + {bias.shape}")
print(f"  (32, 10)  +  (10,)")
print(f"  Bias (10,) broadcasts to (32, 10) - same bias for all samples!")

# ==============================================================================
# EXAMPLE 13: Real-World - Data Normalization
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 13: Real-World - Normalizing Image Data")
print("="*70)

# Simulate batch of images: (batch, height, width, channels)
images = np.random.randint(0, 256, (10, 28, 28, 3))  # 10 RGB images
print(f"Original images shape: {images.shape}")
print(f"Pixel value range: [{images.min()}, {images.max()}]")

# Normalize to [0, 1]
normalized = images / 255.0
print(f"\nNormalized images shape: {normalized.shape}")
print(f"Pixel value range: [{normalized.min():.3f}, {normalized.max():.3f}]")

# Calculate statistics per channel
mean_per_channel = normalized.mean(axis=(0, 1, 2))
std_per_channel = normalized.std(axis=(0, 1, 2))
print(f"\nMean per channel (R, G, B): {mean_per_channel}")
print(f"Std per channel (R, G, B): {std_per_channel}")

# Standardize (mean=0, std=1) per channel
standardized = (normalized - mean_per_channel) / std_per_channel
print(f"\nStandardized shape: {standardized.shape}")
new_mean = standardized.mean(axis=(0, 1, 2))
new_std = standardized.std(axis=(0, 1, 2))
print(f"New mean per channel: {new_mean}  (should be ~0)")
print(f"New std per channel: {new_std}  (should be ~1)")

# ==============================================================================
print("\n" + "="*70)
print("✅ Examples Complete!")
print("="*70)
print("""
Key Takeaways:
1. Vectorization makes operations 50-100x faster
2. Broadcasting extends smaller arrays automatically
3. Axis parameter controls aggregation direction
4. Universal functions (ufuncs) work element-wise
5. Random numbers are essential for ML initialization
6. Real neural networks use all these operations!

Next: Try modifying these examples and work on exercises!
""")
