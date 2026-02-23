"""
Array Operations and Broadcasting - Practice Exercises

Work through these to master NumPy operations!
"""

import numpy as np

print("="*70)
print("NumPy Operations - Exercises")
print("="*70)

# ==============================================================================
# EXERCISE 1: Element-wise Operations
# ==============================================================================
print("\n📝 EXERCISE 1: Vectorized Math")
print("-" * 70)
print("Given temperatures in Fahrenheit: [32, 68, 86, 104, 122]")
print("a) Convert to Celsius: C = (F - 32) × 5/9")
print("b) Find which temperatures are above 25°C")
print("c) Calculate the average temperature in Celsius")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 2: Broadcasting
# ==============================================================================
print("\n📝 EXERCISE 2: Broadcasting Magic")
print("-" * 70)
print("Create a 4x5 matrix and:")
print("a) Add 10 to every element")
print("b) Add [1, 2, 3, 4, 5] to each row")
print("c) Add [[1], [2], [3], [4]] to each column")
print("d) Multiply all elements by 2")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 3: Aggregations
# ==============================================================================
print("\n📝 EXERCISE 3: Statistics on Sales Data")
print("-" * 70)
sales_data = np.array([
    [150, 200, 180, 220, 190],  # Store 1, days Mon-Fri
    [160, 210, 190, 230, 200],  # Store 2
    [155, 205, 185, 225, 195],  # Store 3
    [145, 195, 175, 215, 185]   # Store 4
])
print(f"Sales data (4 stores × 5 days):\n{sales_data}\n")
print("Calculate:")
print("a) Total sales across all stores and days")
print("b) Average daily sales (across all stores)")
print("c) Total sales per store")
print("d) Which day had highest total sales?")
print("e) Which store had highest average sales?")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 4: Mathematical Functions
# ==============================================================================
print("\n📝 EXERCISE 4: Universal Functions")
print("-" * 70)
values = np.array([1, 4, 9, 16, 25, 36])
print(f"Values: {values}\n")
print("Calculate:")
print("a) Square root of each value")
print("b) Natural logarithm of each value")
print("c) Exponential (e^x) for values [0, 1, 2, 3]")
print("d) Sin and cos for angles [0, π/4, π/2, π]")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 5: Combining Arrays
# ==============================================================================
print("\n📝 EXERCISE 5: Stacking and Concatenating")
print("-" * 70)
print("Given:")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}\n")
print("Create:")
print("a) A 3x3 matrix by stacking a, b, c vertically")
print("b) A 1D array by concatenating a, b, c")
print("c) A 2x3 matrix using only a and b")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 6: Random Data Generation
# ==============================================================================
print("\n📝 EXERCISE 6: Random Numbers for ML")
print("-" * 70)
print("Create:")
print("a) A 5x5 matrix of random values from uniform [0, 1)")
print("b) A 1000-element array from normal distribution (mean=100, std=15)")
print("c) Calculate mean and std of the normal distribution array")
print("d) Generate 20 random integers between 1 and 100")
print("e) Set seed to 42, generate 5 random numbers, verify reproducibility")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 7: Axis Operations
# ==============================================================================
print("\n📝 EXERCISE 7: Understanding Axes")
print("-" * 70)
tensor = np.random.randn(3, 4, 5)  # 3D array
print(f"3D array shape: {tensor.shape}\n")
print("Calculate:")
print("a) Sum along axis 0 (what shape is result?)")
print("b) Sum along axis 1 (what shape is result?)")
print("c) Sum along axis 2 (what shape is result?)")
print("d) Mean along all axes (scalar result)")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 8: Real-World - Image Preprocessing
# ==============================================================================
print("\n📝 EXERCISE 8: Image Batch Normalization")
print("-" * 70)
print("You have a batch of 50 grayscale images, each 64x64 pixels")
print("Pixel values range from 0 to 255")
print("\nTasks:")
print("a) Create this batch (random pixel values)")
print("b) Normalize all pixels to [0, 1] range")
print("c) Standardize: mean=0, std=1 for each image individually")
print("d) Calculate the average pixel brightness across all images")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 9: Real-World - Neural Network Batch Processing
# ==============================================================================
print("\n📝 EXERCISE 9: Simulating a Neural Network Layer")
print("-" * 70)
print("Simulate processing a batch through one layer:")
print("- Batch: 64 samples")
print("- Input features: 784 (like flattened 28x28 images)")
print("- Output neurons: 128")
print("\nImplement:")
print("a) Create random input X (64, 784)")
print("b) Create random weights W (784, 128)")
print("c) Create bias b (128,)")
print("d) Compute: output = X @ W + b")
print("e) Verify output shape is (64, 128)")
print("f) Apply ReLU activation: np.maximum(0, output)")

# YOUR CODE HERE:

# ==============================================================================
# EXERCISE 10: Challenge - Custom Normalization
# ==============================================================================
print("\n📝 EXERCISE 10: Challenge - Min-Max Normalization")
print("-" * 70)
print("Implement Min-Max normalization: (x - min) / (max - min)")
print("This scales data to [0, 1] range")
print("\nGiven:")
data = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
print(f"Data:\n{data}\n")
print("Normalize:")
print("a) Each column independently (per-feature normalization)")
print("b) The entire array (global normalization)")
print("c) Verify that min=0 and max=1 after normalization")

# YOUR CODE HERE:

# ==============================================================================
# SOLUTIONS
# ==============================================================================

def show_solutions():
    print("\n\n" + "="*70)
    input("Press Enter to see solutions... ")
    print("="*70)

    print("\n💡 SOLUTION 1: Vectorized Math")
    print("-" * 70)
    fahrenheit = np.array([32, 68, 86, 104, 122])
    celsius = (fahrenheit - 32) * 5/9
    above_25 = celsius[celsius > 25]
    avg_celsius = celsius.mean()
    print(f"a) Celsius: {celsius}")
    print(f"b) Above 25°C: {above_25}")
    print(f"c) Average: {avg_celsius:.2f}°C")

    print("\n💡 SOLUTION 2: Broadcasting")
    print("-" * 70)
    matrix = np.random.randn(4, 5)
    print(f"a) Matrix + 10:\n{matrix + 10}\n")
    row_vec = np.array([1, 2, 3, 4, 5])
    print(f"b) Add row vector:\n{matrix + row_vec}\n")
    col_vec = np.array([[1], [2], [3], [4]])
    print(f"c) Add column vector:\n{matrix + col_vec}\n")
    print(f"d) Multiply by 2:\n{matrix * 2}\n")

    print("\n💡 SOLUTION 3: Sales Statistics")
    print("-" * 70)
    sales_data = np.array([
        [150, 200, 180, 220, 190],
        [160, 210, 190, 230, 200],
        [155, 205, 185, 225, 195],
        [145, 195, 175, 215, 185]
    ])
    print(f"a) Total sales: ${sales_data.sum():,}")
    print(f"b) Daily average: {sales_data.mean(axis=0)}")
    print(f"c) Per store total: {sales_data.sum(axis=1)}")
    daily_totals = sales_data.sum(axis=0)
    best_day = daily_totals.argmax()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    print(f"d) Best day: {days[best_day]} with ${daily_totals[best_day]}")
    store_avgs = sales_data.mean(axis=1)
    best_store = store_avgs.argmax()
    print(f"e) Best store: Store {best_store+1} with avg ${store_avgs[best_store]:.2f}")

    print("\n💡 SOLUTION 4: Math Functions")
    print("-" * 70)
    values = np.array([1, 4, 9, 16, 25, 36])
    print(f"a) Square roots: {np.sqrt(values)}")
    print(f"b) Natural log: {np.log(values)}")
    print(f"c) Exponential: {np.exp([0, 1, 2, 3])}")
    angles = np.array([0, np.pi/4, np.pi/2, np.pi])
    print(f"d) Sin: {np.sin(angles)}")
    print(f"   Cos: {np.cos(angles)}")

    print("\n💡 SOLUTION 5: Combining Arrays")
    print("-" * 70)
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8, 9])
    stacked = np.vstack([a, b, c])
    print(f"a) Stacked:\n{stacked}")
    concatenated = np.concatenate([a, b, c])
    print(f"b) Concatenated: {concatenated}")
    matrix_2x3 = np.vstack([a, b])
    print(f"c) 2x3 matrix:\n{matrix_2x3}")

    print("\n💡 SOLUTION 6: Random Numbers")
    print("-" * 70)
    uniform = np.random.rand(5, 5)
    print(f"a) Uniform 5x5:\n{uniform}\n")
    normal = np.random.normal(100, 15, 1000)
    print(f"b) Normal mean: {normal.mean():.2f}, std: {normal.std():.2f}")
    print(f"c) (Should be close to 100 and 15)")
    randints = np.random.randint(1, 101, 20)
    print(f"d) Random ints: {randints}")
    np.random.seed(42)
    rand1 = np.random.rand(5)
    np.random.seed(42)
    rand2 = np.random.rand(5)
    print(f"e) Seed 42: {rand1}")
    print(f"   Seed 42: {rand2}")
    print(f"   Same: {np.array_equal(rand1, rand2)}")

    print("\n💡 SOLUTION 7: Axis Operations")
    print("-" * 70)
    tensor = np.random.randn(3, 4, 5)
    print(f"Original shape: {tensor.shape}")
    print(f"a) Sum axis 0: {tensor.sum(axis=0).shape}")
    print(f"b) Sum axis 1: {tensor.sum(axis=1).shape}")
    print(f"c) Sum axis 2: {tensor.sum(axis=2).shape}")
    print(f"d) Mean all axes: {tensor.mean()} (scalar)")

    print("\n💡 SOLUTION 8: Image Normalization")
    print("-" * 70)
    images = np.random.randint(0, 256, (50, 64, 64))
    print(f"a) Images shape: {images.shape}")
    normalized = images / 255.0
    print(f"b) Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    mean = normalized.mean(axis=(1, 2), keepdims=True)
    std = normalized.std(axis=(1, 2), keepdims=True)
    standardized = (normalized - mean) / std
    print(f"c) Standardized shape: {standardized.shape}")
    avg_brightness = normalized.mean()
    print(f"d) Average brightness: {avg_brightness:.3f}")

    print("\n💡 SOLUTION 9: Neural Network Layer")
    print("-" * 70)
    X = np.random.randn(64, 784)
    W = np.random.randn(784, 128) * 0.01
    b = np.zeros(128)
    output = X @ W + b
    print(f"a-d) Output shape: {output.shape}")
    print(f"e) Correct shape: {output.shape == (64, 128)}")
    activated = np.maximum(0, output)
    print(f"f) After ReLU: min={activated.min():.4f}, max={activated.max():.4f}")

    print("\n💡 SOLUTION 10: Min-Max Normalization")
    print("-" * 70)
    data = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ])
    # Per column
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    norm_col = (data - col_min) / (col_max - col_min)
    print(f"a) Per-column normalized:\n{norm_col}")
    print(f"   Min per col: {norm_col.min(axis=0)}")
    print(f"   Max per col: {norm_col.max(axis=0)}\n")

    # Global
    global_min = data.min()
    global_max = data.max()
    norm_global = (data - global_min) / (global_max - global_min)
    print(f"b) Global normalized:\n{norm_global}")
    print(f"c) Min: {norm_global.min()}, Max: {norm_global.max()}")

if __name__ == "__main__":
    show_solutions()
