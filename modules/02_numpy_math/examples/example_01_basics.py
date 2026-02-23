"""
NumPy Basics - Interactive Examples
Run this file section by section to understand NumPy fundamentals
"""

import numpy as np
import time

print("="*70)
print("EXAMPLE 1: Why NumPy is Fast")
print("="*70)

# Python list approach (SLOW)
python_list = list(range(1000000))
start = time.time()
doubled_list = [x * 2 for x in python_list]
python_time = time.time() - start
print(f"Python list (1M elements): {python_time:.4f} seconds")

# NumPy approach (FAST)
numpy_array = np.arange(1000000)
start = time.time()
doubled_array = numpy_array * 2
numpy_time = time.time() - start
print(f"NumPy array (1M elements): {numpy_time:.4f} seconds")

speedup = python_time / numpy_time
print(f"\nNumPy is {speedup:.1f}x FASTER! 🚀")
print(f"For 1 billion numbers, Python would take ~{python_time * 1000:.0f}s")
print(f"NumPy would take only ~{numpy_time * 1000:.1f}s")

print("\n" + "="*70)
print("EXAMPLE 2: Creating Arrays - Different Ways")
print("="*70)

# From Python list
arr_from_list = np.array([1, 2, 3, 4, 5])
print(f"From list: {arr_from_list}")
print(f"Type: {type(arr_from_list)}")
print(f"Data type: {arr_from_list.dtype}\n")

# 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D array:\n{matrix}")
print(f"Shape: {matrix.shape} (2 rows, 3 columns)\n")

# Built-in array creation functions
zeros = np.zeros(5)
print(f"Zeros: {zeros}")

ones = np.ones((2, 3))
print(f"Ones (2x3):\n{ones}\n")

range_arr = np.arange(10)
print(f"Range (0-9): {range_arr}")

range_step = np.arange(0, 10, 2)
print(f"Range with step 2: {range_step}")

linspace = np.linspace(0, 1, 5)
print(f"Linspace (5 values from 0 to 1): {linspace}\n")

# Random arrays (VERY important for neural networks!)
random = np.random.rand(3, 3)
print(f"Random (3x3):\n{random}\n")

# For reproducibility, set seed
np.random.seed(42)
random1 = np.random.rand(3)
np.random.seed(42)
random2 = np.random.rand(3)
print(f"With seed=42: {random1}")
print(f"With seed=42: {random2}")
print("Same values! This is how we make ML experiments reproducible.\n")

print("="*70)
print("EXAMPLE 3: Array Properties - Know Your Data!")
print("="*70)

# Create a 3D array (like a color image: height × width × channels)
image_like = np.random.rand(28, 28, 3)  # 28x28 RGB image
print(f"Image-like array shape: {image_like.shape}")
print(f"Total elements: {image_like.size}")
print(f"Number of dimensions: {image_like.ndim}")
print(f"Data type: {image_like.dtype}")
print(f"Memory size: {image_like.nbytes} bytes\n")

# Understanding shape for neural networks
print("Understanding shapes in neural networks:")
batch_size = 32
sequence_length = 10
embedding_dim = 768

# Example: BERT input
bert_input = np.random.randn(batch_size, sequence_length, embedding_dim)
print(f"BERT input shape: {bert_input.shape}")
print(f"  - Batch size: {batch_size} (processing 32 sentences at once)")
print(f"  - Sequence length: {sequence_length} (each sentence has 10 tokens)")
print(f"  - Embedding dim: {embedding_dim} (each token is a 768-d vector)\n")

print("="*70)
print("EXAMPLE 4: Indexing and Slicing - Access Your Data")
print("="*70)

arr = np.array([10, 20, 30, 40, 50])
print(f"Array: {arr}")
print(f"First element arr[0]: {arr[0]}")
print(f"Last element arr[-1]: {arr[-1]}")
print(f"Slice arr[1:4]: {arr[1:4]}")
print(f"Every other arr[::2]: {arr[::2]}\n")

# 2D indexing (like accessing pixels in an image)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrix:\n{matrix}\n")
print(f"Element at (0,0): {matrix[0, 0]}")
print(f"Element at (1,2): {matrix[1, 2]}")
print(f"First row matrix[0, :]: {matrix[0, :]}")
print(f"First column matrix[:, 0]: {matrix[:, 0]}")
print(f"Submatrix matrix[0:2, 1:3]:\n{matrix[0:2, 1:3]}\n")

# Practical example: Accessing image channels
image = np.random.randint(0, 256, (100, 100, 3))  # 100x100 RGB image
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]
print(f"Original image shape: {image.shape}")
print(f"Red channel shape: {red_channel.shape}")
print("This is how you extract color channels from images!\n")

print("="*70)
print("EXAMPLE 5: Reshaping - Critical for Neural Networks")
print("="*70)

# Flattening an image (preprocessing for neural networks)
image = np.random.rand(28, 28)  # Grayscale 28x28 image
print(f"Original image shape: {image.shape}")

flattened = image.reshape(784)  # 28 * 28 = 784
# Or use flatten(): flattened = image.flatten()
print(f"Flattened shape: {flattened.shape}")
print("This is how images are fed into neural networks!\n")

# Reshaping for batch processing
arr = np.arange(12)
print(f"1D array: {arr}")

reshaped_3x4 = arr.reshape(3, 4)
print(f"\nReshaped to 3x4:\n{reshaped_3x4}")

reshaped_2x6 = arr.reshape(2, 6)
print(f"\nReshaped to 2x6:\n{reshaped_2x6}")

# -1 means "infer this dimension"
auto_shape = arr.reshape(3, -1)  # 3 rows, auto-calculate columns
print(f"\nReshape(3, -1) - auto shape: {auto_shape.shape}")
print(auto_shape)

# Back to 1D
back_to_1d = reshaped_3x4.flatten()
print(f"\nFlattened back to 1D: {back_to_1d}\n")

print("="*70)
print("EXAMPLE 6: Basic Operations - Vectorization Power")
print("="*70)

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")

# Element-wise operations (NO LOOPS NEEDED!)
print(f"arr + 10: {arr + 10}")
print(f"arr * 2: {arr * 2}")
print(f"arr ** 2: {arr ** 2}")
print(f"arr / 2: {arr / 2}\n")

# Operations between arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"arr1: {arr1}")
print(f"arr2: {arr2}")
print(f"arr1 + arr2: {arr1 + arr2}")
print(f"arr1 * arr2: {arr1 * arr2}")
print(f"arr1 / arr2: {arr1 / arr2}\n")

# Comparison: Python vs NumPy
print("IMPORTANT: Python list vs NumPy behavior")
list1 = [1, 2, 3]
list2 = [4, 5, 6]
print(f"Python: list1 + list2 = {list1 + list2}  ← Concatenates!")

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"NumPy: arr1 + arr2 = {arr1 + arr2}  ← Element-wise addition!\n")

print("="*70)
print("EXAMPLE 7: Aggregations - Statistics Made Easy")
print("="*70)

data = np.array([85, 92, 78, 95, 88, 76, 89, 91])
print(f"Test scores: {data}")
print(f"Sum: {data.sum()}")
print(f"Mean (average): {data.mean():.2f}")
print(f"Max: {data.max()}")
print(f"Min: {data.min()}")
print(f"Standard deviation: {data.std():.2f}")
print(f"Variance: {data.var():.2f}\n")

# Multi-dimensional aggregations
grades = np.array([
    [85, 92, 78],  # Student 1
    [95, 88, 76],  # Student 2
    [89, 91, 94]   # Student 3
])
print(f"Grades matrix:\n{grades}\n")
print(f"Total sum: {grades.sum()}")
print(f"Column sums (per assignment): {grades.sum(axis=0)}")
print(f"Row sums (per student): {grades.sum(axis=1)}")
print(f"Mean per assignment: {grades.mean(axis=0)}")
print(f"Mean per student: {grades.mean(axis=1)}\n")

print("="*70)
print("EXAMPLE 8: Boolean Indexing - Filtering Data")
print("="*70)

scores = np.array([85, 92, 78, 95, 88, 76, 89, 91])
print(f"Scores: {scores}")

# Create boolean mask
passing = scores >= 80
print(f"\nMask (scores >= 80): {passing}")

# Filter using mask
passing_scores = scores[passing]
print(f"Passing scores: {passing_scores}")

# One-liner
print(f"Scores > 90: {scores[scores > 90]}")

# Multiple conditions (use & for AND, | for OR)
excellent = scores[(scores >= 90) & (scores <= 100)]
print(f"Excellent scores (90-100): {excellent}")

# Practical: Find outliers
data = np.array([10, 12, 11, 100, 13, 12, 14])  # 100 is an outlier
mean = data.mean()
std = data.std()
outliers = data[(data < mean - 2*std) | (data > mean + 2*std)]
print(f"\nData: {data}")
print(f"Outliers (beyond 2 std): {outliers}\n")

print("="*70)
print("EXAMPLE 9: Real-World - Token Embedding Lookup")
print("="*70)

# Simulate a small vocabulary
vocab_size = 10
embedding_dim = 4

# Create embedding matrix (like in Word2Vec or GPT)
embeddings = np.random.randn(vocab_size, embedding_dim)
print(f"Embedding matrix shape: {embeddings.shape}")
print(f"Each of {vocab_size} tokens has a {embedding_dim}-d vector\n")

# Words (tokens) mapped to IDs
word_to_id = {
    "hello": 0,
    "world": 1,
    "python": 2,
    "is": 3,
    "awesome": 4
}

# Look up embedding for "hello"
word = "hello"
token_id = word_to_id[word]
word_vector = embeddings[token_id]

print(f"Word: '{word}'")
print(f"Token ID: {token_id}")
print(f"Embedding vector: {word_vector}")
print(f"Vector shape: {word_vector.shape}\n")

# Process a sentence
sentence = ["python", "is", "awesome"]
sentence_ids = [word_to_id[w] for w in sentence]
sentence_embeddings = embeddings[sentence_ids]

print(f"Sentence: {sentence}")
print(f"Token IDs: {sentence_ids}")
print(f"Sentence embeddings shape: {sentence_embeddings.shape}")
print(f"Each word is now a {embedding_dim}-d vector!\n")

print("="*70)
print("EXAMPLE 10: Memory View - Understanding .shape vs .size")
print("="*70)

# Different shapes, same data
arr = np.arange(24)
print(f"Original: shape={arr.shape}, size={arr.size}")

reshaped_2d = arr.reshape(4, 6)
print(f"As 4x6: shape={reshaped_2d.shape}, size={reshaped_2d.size}")

reshaped_3d = arr.reshape(2, 3, 4)
print(f"As 2x3x4: shape={reshaped_3d.shape}, size={reshaped_3d.size}")

print("\nAll have the same .size (24 elements), just different .shape!")
print("This is crucial for debugging neural networks!\n")

print("="*70)
print("✅ Examples Complete!")
print("="*70)
print("\nKey Takeaways:")
print("1. NumPy is 10-100x faster than Python lists")
print("2. Use np.array() to create arrays")
print("3. .shape tells you dimensions, .size tells you total elements")
print("4. Indexing works like Python, but extended for multiple dimensions")
print("5. Operations are vectorized (no loops needed!)")
print("6. Boolean indexing is powerful for filtering")
print("7. Reshape is critical for neural network preprocessing")
print("\nNext: Try running this file, then work on exercises!")
