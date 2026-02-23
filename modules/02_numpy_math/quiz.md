# Module 2: NumPy Fundamentals - Quiz

## Instructions
- This quiz tests your understanding of NumPy basics, operations, and linear algebra
- Answer all questions before checking solutions
- Passing score: 80% (24/30 questions)
- Take your time and think through each question

---

## Section A: NumPy Basics (10 questions)

### Question 1
What is the main advantage of NumPy arrays over Python lists?

A) NumPy arrays can store more data
B) NumPy arrays are easier to create
C) NumPy operations are vectorized and run at C-speed
D) NumPy arrays use less syntax

**Your answer:**

---

### Question 2
What does `arr.shape` return for a 3D array?

A) The total number of elements
B) A tuple with 3 integers representing dimensions
C) The data type of the array
D) The number of rows

**Your answer:**

---

### Question 3
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr[0, :]
```
What is the output?

A) `[1, 4]`
B) `[1, 2, 3]`
C) `[[1, 2, 3]]`
D) `6`

**Your answer:**

---

### Question 4
Which of these creates a 5x5 matrix of zeros?

A) `np.zeros(5, 5)`
B) `np.zeros([5, 5])`
C) `np.zeros((5, 5))`
D) `np.zero(5, 5)`

**Your answer:**

---

### Question 5
```python
arr = np.arange(12).reshape(3, 4)
```
What is `arr.shape`?

A) `(12,)`
B) `(3, 4)`
C) `(4, 3)`
D) This code will error

**Your answer:**

---

### Question 6
What's the difference between `*` and `@` operators?

A) Both do matrix multiplication
B) `*` is element-wise, `@` is matrix multiplication
C) `*` is faster than `@`
D) There is no difference

**Your answer:**

---

### Question 7
```python
arr = np.array([1, 2, 3, 4, 5])
result = arr[arr > 3]
```
What is `result`?

A) `[True, False]`
B) `[4, 5]`
C) `[3, 4, 5]`
D) Error

**Your answer:**

---

### Question 8
How do you flatten a 2D array `matrix` to 1D?

A) `matrix.flat()`
B) `matrix.flatten()` or `matrix.reshape(-1)`
C) `matrix.to1d()`
D) `matrix.squeeze()`

**Your answer:**

---

### Question 9
What does `np.random.seed(42)` do?

A) Creates 42 random numbers
B) Makes random numbers reproducible
C) Sets random numbers to 42
D) Initializes a 42x42 matrix

**Your answer:**

---

### Question 10
```python
arr = np.array([10, 20, 30])
result = arr + 5
```
What is `result`?

A) `[10, 20, 30, 5]`
B) `[15, 25, 35]`
C) `50`
D) Error

**Your answer:**

---

## Section B: Array Operations & Broadcasting (10 questions)

### Question 11
What is broadcasting in NumPy?

A) Sending arrays over a network
B) Operating on arrays of different shapes automatically
C) Making arrays larger
D) A type of array initialization

**Your answer:**

---

### Question 12
```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])
result = matrix + vector
```
What happens?

A) Error - shapes don't match
B) `vector` is broadcast to each row
C) `vector` is broadcast to each column
D) Only first elements add

**Your answer:**

---

### Question 13
Which is faster for 1 million element operation?

A) Python list with loop
B) NumPy vectorized operation
C) Both are the same speed
D) Depends on the operation

**Your answer:**

---

### Question 14
```python
arr = np.array([[1, 2], [3, 4]])
result = arr.sum(axis=0)
```
What is `result`?

A) `10`
B) `[4, 6]`
C) `[3, 7]`
D) `[[1, 2], [3, 4]]`

**Your answer:**

---

### Question 15
How do you concatenate arrays `a` and `b` vertically?

A) `np.vstack([a, b])`
B) `np.hstack([a, b])`
C) `a + b`
D) `np.concat(a, b)`

**Your answer:**

---

### Question 16
What does `arr.T` do?

A) Converts to tensor
B) Transposes the array
C) Truncates the array
D) Returns the type

**Your answer:**

---

### Question 17
```python
arr = np.array([1, 2, 3, 4])
result = np.sqrt(arr)
```
This is an example of:

A) Element-wise operation
B) Aggregation
C) Broadcasting
D) Reshaping

**Your answer:**

---

### Question 18
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = A * B
```
What is `result[0, 0]`?

A) `1*5 + 2*7 = 19`
B) `5`
C) `1*5 = 5`
D) `6`

**Your answer:**

---

### Question 19
What's the purpose of `axis` parameter in aggregation functions?

A) Specifies which dimension to operate along
B) Sets the coordinate system
C) Determines the output shape
D) Changes the data type

**Your answer:**

---

### Question 20
For a shape `(32, 10, 768)` array, what might this represent in deep learning?

A) 32 samples, 10 features, 768 classes
B) 32 batches, 10 tokens, 768-dim embeddings
C) 768 images of size 32x10
D) A single 3D image

**Your answer:**

---

## Section C: Linear Algebra (10 questions)

### Question 21
What is the dot product of `[1, 2, 3]` and `[4, 5, 6]`?

A) `[4, 10, 18]`
B) `32`
C) `[5, 7, 9]`
D) `9`

**Your answer:**

---

### Question 22
For matrix multiplication `A @ B` to work, what must be true?

A) Both must be square matrices
B) Inner dimensions must match
C) Both must have same shape
D) Both must be 2D

**Your answer:**

---

### Question 23
```python
A = np.random.randn(10, 5)
B = np.random.randn(5, 3)
C = A @ B
```
What is `C.shape`?

A) `(10, 5)`
B) `(5, 3)`
C) `(10, 3)`
D) Error

**Your answer:**

---

### Question 24
What does `np.eye(3)` create?

A) A 3D array
B) A 3x3 identity matrix
C) A 3x1 vector
D) A 3x3 random matrix

**Your answer:**

---

### Question 25
In neural networks, what does `X @ W` represent?

A) Element-wise multiplication
B) Linear transformation (layer computation)
C) Adding bias
D) Activation function

**Your answer:**

---

### Question 26
What is `np.linalg.norm([3, 4])`?

A) `[3, 4]`
B) `7`
C) `5`
D) `12`

**Your answer:**

---

### Question 27
```python
X = np.random.randn(100, 784)  # 100 samples, 784 features
W = np.random.randn(784, 128)  # Weights
output = X @ W
```
What is `output.shape`?

A) `(100, 784)`
B) `(784, 128)`
C) `(100, 128)`
D) `(128, 100)`

**Your answer:**

---

### Question 28
Why is the transpose operation important in neural networks?

A) To flip images
B) To match dimensions for matrix multiplication
C) To save memory
D) To normalize data

**Your answer:**

---

### Question 29
What does standardization `(X - mean) / std` accomplish?

A) Scales data to [0, 1]
B) Centers data around 0 with unit variance
C) Removes outliers
D) Increases data size

**Your answer:**

---

### Question 30
In transformer models, what linear algebra operation is at the heart of attention?

A) Addition
B) Transpose
C) Dot product (Query · Key)
D) Inverse

**Your answer:**

---

## Section D: Practical Application (Bonus: +5 points)

### Question 31
You have a batch of 64 images, each 224x224 RGB. What is the shape?

A) `(64, 224, 224, 3)`
B) `(224, 224, 3, 64)`
C) `(3, 224, 224, 64)`
D) `(64, 3, 224, 224)`

**Your answer:**

---

### Question 32
How would you normalize pixel values (0-255) to (0-1)?

A) `pixels - 255`
B) `pixels / 255`
C) `pixels * 255`
D) `(pixels - 127.5) / 127.5`

**Your answer:**

---

### Question 33
To find the index of the maximum value in an array:

A) `arr.max()`
B) `arr.argmax()`
C) `arr.maxindex()`
D) `np.maximum(arr)`

**Your answer:**

---

### Question 34
Which operation is NOT typically done with NumPy in neural networks?

A) Forward propagation
B) Gradient computation
C) Hyperparameter tuning
D) Data normalization

**Your answer:**

---

### Question 35
For an embedding matrix with vocab_size=10000 and embedding_dim=300, the shape is:

A) `(300, 10000)`
B) `(10000, 300)`
C) `(10000 * 300,)`
D) `(1, 10000, 300)`

**Your answer:**

---

## Scoring

Count your correct answers:
- **27-35**: Excellent! You're ready for Module 3 🌟
- **24-26**: Good! Review missed topics ✅
- **20-23**: Okay. Revisit lessons and try exercises 📚
- **Below 20**: Please review the module again 🔄

---

## Answer Key

### Section A: Basics
1. C - Vectorized and fast
2. B - Tuple of 3 integers
3. B - `[1, 2, 3]` (first row)
4. C - `np.zeros((5, 5))`
5. B - `(3, 4)`
6. B - `*` is element-wise, `@` is matrix multiplication
7. B - `[4, 5]`
8. B - `flatten()` or `reshape(-1)`
9. B - Makes random numbers reproducible
10. B - `[15, 25, 35]` (broadcasting)

### Section B: Operations
11. B - Operating on different shapes automatically
12. B - Vector is broadcast to each row
13. B - NumPy vectorized (50-100x faster)
14. B - `[4, 6]` (column sums)
15. A - `np.vstack([a, b])`
16. B - Transposes the array
17. A - Element-wise operation
18. C - `1*5 = 5` (element-wise, not matrix mult)
19. A - Specifies which dimension to operate along
20. B - 32 batches, 10 tokens, 768-dim embeddings

### Section C: Linear Algebra
21. B - `32` (1*4 + 2*5 + 3*6)
22. B - Inner dimensions must match
23. C - `(10, 3)`
24. B - 3x3 identity matrix
25. B - Linear transformation (layer)
26. C - `5` (sqrt(3² + 4²))
27. C - `(100, 128)`
28. B - To match dimensions for multiplication
29. B - Centers around 0 with unit variance
30. C - Dot product (Q · K)

### Section D: Practical (Bonus)
31. A or D - Both are valid (framework dependent)
32. B - `pixels / 255`
33. B - `arr.argmax()`
34. C - Hyperparameter tuning (done manually or with tools)
35. B - `(10000, 300)`

---

## Detailed Explanations

### Question 6: Why is this important?
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A * B   # Element-wise: [[5, 12], [21, 32]]
A @ B   # Matrix mult: [[19, 22], [43, 50]]
```
In neural networks, we use `@` for layer computations!

### Question 21: Dot Product Calculation
```python
[1, 2, 3] · [4, 5, 6] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```
This is the foundation of attention mechanisms!

### Question 27: Shape Mathematics
```python
(100, 784) @ (784, 128) = (100, 128)
  ↑           ↑     ↑         ↑
batch    inner dims      batch × output
         (must match)
```

### Question 30: Attention Mechanism
```python
# Simplified attention
scores = Query @ Key.T  # Dot products!
weights = softmax(scores)
output = weights @ Value
```
All matrix operations you learned!

---

## Next Steps

If you scored well:
1. ✅ Mark Module 2 complete in PROGRESS.md
2. 🚀 Move to Module 3: Neural Networks
3. 💪 Keep practicing with exercises

If you need review:
1. 📖 Re-read challenging sections
2. 💻 Work through more exercises
3. 🔬 Experiment with code examples
4. 🔁 Retake quiz when ready

**Remember:** Understanding NumPy deeply is crucial. Don't rush! Every hour spent here makes Module 3+ much easier.
