"""
NumPy Basics - Practice Exercises
Complete these exercises to test your understanding

Instructions:
1. Read each exercise carefully
2. Write your solution below the exercise
3. Run the file to check your answers
4. Compare with solutions at the bottom (don't peek first!)
"""

import numpy as np

print("="*70)
print("NumPy Basics - Exercises")
print("="*70)

# ==============================================================================
# EXERCISE 1: Create Arrays
# ==============================================================================
print("\n📝 EXERCISE 1: Array Creation")
print("-" * 70)
print("Create the following arrays:")
print("a) 1D array with numbers 1 through 10")
print("b) 2D array (3x3) filled with zeros")
print("c) 1D array with 7 evenly spaced numbers between 0 and 1")
print("d) 2D array (2x4) filled with random numbers")
print("e) 1D array with even numbers from 0 to 20")

# YOUR CODE HERE:
# a)
# b)
# c)
# d)
# e)

# ==============================================================================
# EXERCISE 2: Array Properties
# ==============================================================================
print("\n📝 EXERCISE 2: Understanding Array Properties")
print("-" * 70)
arr = np.random.rand(5, 4, 3)
print(f"Given array with shape: {arr.shape}")
print("\nAnswer these questions:")
print("a) How many dimensions does this array have?")
print("b) How many total elements?")
print("c) What would this represent if it were image data?")
print("d) What is the data type?")

# YOUR ANSWERS HERE:
# a)
# b)
# c)
# d)

# ==============================================================================
# EXERCISE 3: Indexing and Slicing
# ==============================================================================
print("\n📝 EXERCISE 3: Indexing and Slicing")
print("-" * 70)
matrix = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])
print(f"Matrix:\n{matrix}\n")
print("Extract:")
print("a) The element 70")
print("b) The first row")
print("c) The last column")
print("d) Submatrix [[60, 70], [100, 110]]")
print("e) Every other element in the second row")

# YOUR CODE HERE:
# a)
# b)
# c)
# d)
# e)

# ==============================================================================
# EXERCISE 4: Reshaping
# ==============================================================================
print("\n📝 EXERCISE 4: Reshaping Arrays")
print("-" * 70)
print("You have a 28x28 grayscale image (like MNIST digits)")
print("a) Create a random 28x28 array")
print("b) Flatten it to a 1D array (for input to a neural network)")
print("c) Reshape the flattened array back to 28x28")
print("d) Reshape the original to 4x196 (4 rows)")

# YOUR CODE HERE:
# a)
# b)
# c)
# d)

# ==============================================================================
# EXERCISE 5: Vectorized Operations
# ==============================================================================
print("\n📝 EXERCISE 5: Operations Without Loops")
print("-" * 70)
temperatures_celsius = np.array([0, 10, 20, 30, 40])
print(f"Temperatures in Celsius: {temperatures_celsius}")
print("\nTask: Convert to Fahrenheit using the formula: F = C × 9/5 + 32")
print("Do this WITHOUT using a for loop!")

# YOUR CODE HERE:
temperatures_fahrenheit = None  # Replace with your solution
# print(f"Temperatures in Fahrenheit: {temperatures_fahrenheit}")

# ==============================================================================
# EXERCISE 6: Aggregations
# ==============================================================================
print("\n📝 EXERCISE 6: Computing Statistics")
print("-" * 70)
sales_data = np.array([
    [150, 200, 180, 220],  # Week 1: Mon, Tue, Wed, Thu
    [160, 210, 190, 230],  # Week 2
    [155, 205, 185, 225]   # Week 3
])
print(f"Sales data (3 weeks, 4 days):\n{sales_data}\n")
print("Calculate:")
print("a) Total sales across all weeks and days")
print("b) Average sales per day (across all weeks)")
print("c) Total sales per week")
print("d) Which day had the highest average sales?")
print("e) What's the standard deviation of week 1 sales?")

# YOUR CODE HERE:
# a)
# b)
# c)
# d)
# e)

# ==============================================================================
# EXERCISE 7: Boolean Indexing
# ==============================================================================
print("\n📝 EXERCISE 7: Filtering with Conditions")
print("-" * 70)
test_scores = np.array([45, 67, 89, 92, 56, 78, 85, 91, 34, 88])
print(f"Test scores: {test_scores}\n")
print("Find:")
print("a) All scores above 80")
print("b) All scores below 50 (failing)")
print("c) All scores between 70 and 90 (inclusive)")
print("d) How many students scored above 85?")
print("e) What percentage of students passed (>= 60)?")

# YOUR CODE HERE:
# a)
# b)
# c)
# d)
# e)

# ==============================================================================
# EXERCISE 8: Real-World - Image Processing
# ==============================================================================
print("\n📝 EXERCISE 8: Image Preprocessing")
print("-" * 70)
print("You have a batch of 100 grayscale images, each 32x32 pixels")
print("Pixel values range from 0 to 255")
print("\nTasks:")
print("a) Create this batch as a NumPy array")
print("b) Normalize pixel values to range [0, 1]")
print("c) Calculate the mean pixel value across all images")
print("d) Find which image has the highest average brightness")

# YOUR CODE HERE:
# a)
# b)
# c)
# d)

# ==============================================================================
# EXERCISE 9: Real-World - Embedding Lookup
# ==============================================================================
print("\n📝 EXERCISE 9: Token Embeddings")
print("-" * 70)
print("Create a simple embedding system:")
print("a) Create an embedding matrix for 100 tokens, each 50-dimensional")
print("b) Create a sentence represented as token IDs: [5, 12, 8, 45]")
print("c) Look up the embeddings for this sentence")
print("d) What is the shape of the resulting sentence embedding?")

# YOUR CODE HERE:
# a)
# b)
# c)
# d)

# ==============================================================================
# EXERCISE 10: Challenge - Data Normalization
# ==============================================================================
print("\n📝 EXERCISE 10: Challenge - Standardization")
print("-" * 70)
print("Standardization: (x - mean) / std_dev")
print("This is crucial for training neural networks!")
print("\nGiven:")
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])
print(f"Data:\n{data}\n")
print("Tasks:")
print("a) Standardize EACH COLUMN separately")
print("   (subtract column mean, divide by column std)")
print("b) Verify new column means are ~0")
print("c) Verify new column std devs are ~1")

# YOUR CODE HERE:
# a)
# b)
# c)

print("\n" + "="*70)
print("Exercises complete! Check solutions below.")
print("="*70)

# ==============================================================================
# SOLUTIONS (Don't look until you've tried!)
# ==============================================================================

def show_solutions():
    print("\n\n" + "="*70)
    input("Press Enter to see solutions... ")
    print("="*70)

    print("\n💡 SOLUTION 1: Array Creation")
    print("-" * 70)
    a = np.arange(1, 11)
    b = np.zeros((3, 3))
    c = np.linspace(0, 1, 7)
    d = np.random.rand(2, 4)
    e = np.arange(0, 21, 2)
    print(f"a) {a}")
    print(f"b)\n{b}")
    print(f"c) {c}")
    print(f"d)\n{d}")
    print(f"e) {e}")

    print("\n💡 SOLUTION 2: Array Properties")
    print("-" * 70)
    arr = np.random.rand(5, 4, 3)
    print(f"a) Dimensions: {arr.ndim}")
    print(f"b) Total elements: {arr.size}")
    print(f"c) Could be 5 images of size 4x3 (or 4x3 images with 5 channels)")
    print(f"d) Data type: {arr.dtype}")

    print("\n💡 SOLUTION 3: Indexing and Slicing")
    print("-" * 70)
    matrix = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120]
    ])
    print(f"a) {matrix[1, 2]}")
    print(f"b) {matrix[0, :]}")
    print(f"c) {matrix[:, -1]}")
    print(f"d)\n{matrix[1:3, 1:3]}")
    print(f"e) {matrix[1, ::2]}")

    print("\n💡 SOLUTION 4: Reshaping")
    print("-" * 70)
    image = np.random.rand(28, 28)
    flattened = image.reshape(784)
    # Or: flattened = image.flatten()
    back_to_image = flattened.reshape(28, 28)
    reshaped = image.reshape(4, 196)
    print(f"a) Image shape: {image.shape}")
    print(f"b) Flattened shape: {flattened.shape}")
    print(f"c) Back to image shape: {back_to_image.shape}")
    print(f"d) Reshaped to 4x196: {reshaped.shape}")

    print("\n💡 SOLUTION 5: Vectorized Operations")
    print("-" * 70)
    temperatures_celsius = np.array([0, 10, 20, 30, 40])
    temperatures_fahrenheit = temperatures_celsius * 9/5 + 32
    print(f"Celsius: {temperatures_celsius}")
    print(f"Fahrenheit: {temperatures_fahrenheit}")

    print("\n💡 SOLUTION 6: Aggregations")
    print("-" * 70)
    sales_data = np.array([
        [150, 200, 180, 220],
        [160, 210, 190, 230],
        [155, 205, 185, 225]
    ])
    print(f"a) Total sales: {sales_data.sum()}")
    print(f"b) Average per day: {sales_data.mean(axis=0)}")
    print(f"c) Total per week: {sales_data.sum(axis=1)}")
    avg_per_day = sales_data.mean(axis=0)
    best_day = np.argmax(avg_per_day)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
    print(f"d) Best day: {days[best_day]} with avg ${avg_per_day[best_day]:.2f}")
    print(f"e) Week 1 std dev: {sales_data[0].std():.2f}")

    print("\n💡 SOLUTION 7: Boolean Indexing")
    print("-" * 70)
    test_scores = np.array([45, 67, 89, 92, 56, 78, 85, 91, 34, 88])
    print(f"a) Above 80: {test_scores[test_scores > 80]}")
    print(f"b) Below 50: {test_scores[test_scores < 50]}")
    print(f"c) Between 70-90: {test_scores[(test_scores >= 70) & (test_scores <= 90)]}")
    print(f"d) Count above 85: {(test_scores > 85).sum()}")
    pass_rate = (test_scores >= 60).mean() * 100
    print(f"e) Pass rate: {pass_rate:.1f}%")

    print("\n💡 SOLUTION 8: Image Processing")
    print("-" * 70)
    images = np.random.randint(0, 256, (100, 32, 32))
    normalized = images / 255.0
    mean_pixel = normalized.mean()
    mean_per_image = normalized.mean(axis=(1, 2))
    brightest = np.argmax(mean_per_image)
    print(f"a) Images shape: {images.shape}")
    print(f"b) Normalized shape: {normalized.shape}, range: [0, 1]")
    print(f"c) Mean pixel value: {mean_pixel:.4f}")
    print(f"d) Brightest image: #{brightest} with brightness {mean_per_image[brightest]:.4f}")

    print("\n💡 SOLUTION 9: Token Embeddings")
    print("-" * 70)
    embeddings = np.random.randn(100, 50)
    sentence_ids = np.array([5, 12, 8, 45])
    sentence_embeddings = embeddings[sentence_ids]
    print(f"a) Embeddings shape: {embeddings.shape}")
    print(f"b) Sentence token IDs: {sentence_ids}")
    print(f"c) Sentence embeddings shape: {sentence_embeddings.shape}")
    print(f"d) Shape is (4, 50): 4 tokens, each 50-dimensional")

    print("\n💡 SOLUTION 10: Challenge - Standardization")
    print("-" * 70)
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    # Standardize per column
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    standardized = (data - mean) / std

    print(f"Original data:\n{data}\n")
    print(f"Column means: {mean}")
    print(f"Column stds: {std}")
    print(f"\nStandardized data:\n{standardized}\n")
    print(f"New column means: {standardized.mean(axis=0)}")
    print(f"New column stds: {standardized.std(axis=0)}")
    print("Note: Means should be ~0, stds should be ~1")

if __name__ == "__main__":
    show_solutions()
