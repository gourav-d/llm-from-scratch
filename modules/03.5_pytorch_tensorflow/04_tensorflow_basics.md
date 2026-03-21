# Lesson 4: TensorFlow & Keras Basics

**Industry's Production Framework**

---

## Learning Objectives

By the end of this lesson, you will:
- Understand TensorFlow 2.x fundamentals
- Use Keras Sequential API
- Use Keras Functional API
- Build and train models in TensorFlow
- Compare TensorFlow to PyTorch

**Time:** 4-6 hours

---

## What is TensorFlow?

### Simple Explanation

**TensorFlow** is Google's machine learning framework, historically focused on production deployment.

```
TensorFlow:
✅ Production-ready (Google scale)
✅ Deployment tools (TF Serving, TF Lite)
✅ High-level APIs (Keras)
✅ Mobile/Web deployment
⚠️ More complex (historically)
⚠️ Less Pythonic
```

**Keras** is the high-level API built into TensorFlow (since TF 2.0).

---

## Part 1: TensorFlow vs PyTorch

### Quick Comparison

| Aspect | PyTorch | TensorFlow |
|--------|---------|------------|
| **Philosophy** | Research-first | Production-first |
| **Code Style** | Pythonic | More structured |
| **Debugging** | Easier (eager execution) | Harder (graph mode) |
| **Deployment** | Getting better | Mature ecosystem |
| **Mobile** | Limited | TF Lite |
| **Web** | ONNX.js | TensorFlow.js |
| **Popularity** | Research | Industry |

**TensorFlow 2.x** (2019+) adopted eager execution, making it more like PyTorch!

---

## Part 2: Installation and Setup

### Install TensorFlow

```bash
# CPU and GPU (auto-detects)
pip install tensorflow

# Verify
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Check Installation

```python
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)

# List devices
print("\nDevices:")
for device in tf.config.list_physical_devices():
    print(f"  {device.device_type}: {device.name}")
```

**Expected Output:**
```
TensorFlow version: 2.15.0
GPU available: True

Devices:
  CPU: /physical_device:CPU:0
  GPU: /physical_device:GPU:0
```

---

## Part 3: TensorFlow Fundamentals

### Creating Tensors

```python
import tensorflow as tf

# From Python list
tensor = tf.constant([1, 2, 3, 4, 5])
print(tensor)
# <tf.Tensor: shape=(5,), dtype=int32, numpy=array([1, 2, 3, 4, 5])>

# Zeros
zeros = tf.zeros((3, 4))
print(zeros)

# Ones
ones = tf.ones((2, 3))
print(ones)

# Random
random = tf.random.normal((2, 2), mean=0, stddev=1)
print(random)

# Range
range_tensor = tf.range(0, 10, 2)
print(range_tensor)
# <tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 2, 4, 6, 8])>
```

**NumPy/PyTorch Comparison:**

```python
# NumPy
np.zeros((3, 4))
np.ones((2, 3))
np.random.randn(2, 2)

# PyTorch
torch.zeros(3, 4)
torch.ones(2, 3)
torch.randn(2, 2)

# TensorFlow
tf.zeros((3, 4))
tf.ones((2, 3))
tf.random.normal((2, 2))
```

**Almost identical!**

---

### Tensor Operations

```python
# Create tensors
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# Element-wise operations
print(a + b)
print(a * b)
print(a - b)
print(a / b)

# Matrix multiplication
print(tf.matmul(a, b))  # or a @ b

# Transpose
print(tf.transpose(a))

# Reshape
print(tf.reshape(a, (1, 4)))
```

**Same as PyTorch!**

---

### Converting to NumPy

```python
# TensorFlow to NumPy
tensor = tf.constant([1, 2, 3])
numpy_array = tensor.numpy()
print(numpy_array)
# [1 2 3]

# NumPy to TensorFlow
import numpy as np
numpy_array = np.array([4, 5, 6])
tensor = tf.constant(numpy_array)
print(tensor)
```

---

## Part 4: Keras Sequential API

### What is Keras?

**Keras** is the high-level API for building models in TensorFlow.

**Three APIs:**
1. **Sequential**: Stack layers linearly (simplest)
2. **Functional**: Complex architectures (flexible)
3. **Subclassing**: Full control (advanced)

---

### Building a Model (Sequential)

```python
from tensorflow import keras
from tensorflow.keras import layers

# Create model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# View architecture
model.summary()
```

**Output:**
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 128)               100480
 dense_1 (Dense)             (None, 64)                8256
 dense_2 (Dense)             (None, 10)                650
=================================================================
Total params: 109,386
Trainable params: 109,386
Non-trainable params: 0
```

---

### Layer by Layer Explanation

```python
# Layer 1: Input → Hidden
layers.Dense(
    128,                    # Output units
    activation='relu',      # Activation function
    input_shape=(784,)      # Input shape (only for first layer)
)
# Equivalent to: Linear(784, 128) + ReLU in PyTorch

# Layer 2: Hidden → Hidden
layers.Dense(64, activation='relu')
# No input_shape needed (inferred from previous layer)

# Layer 3: Hidden → Output
layers.Dense(10, activation='softmax')
# 10 classes, softmax for probabilities
```

**For .NET Devs:**
```csharp
// C# - Fluent builder pattern
var model = new Sequential()
    .Add(new Dense(128, activation: "relu", inputShape: new[] {784}))
    .Add(new Dense(64, activation: "relu"))
    .Add(new Dense(10, activation: "softmax"));
```

---

### Compile the Model

```python
model.compile(
    optimizer='adam',                              # Optimizer
    loss='sparse_categorical_crossentropy',        # Loss function
    metrics=['accuracy']                           # Track accuracy
)
```

**Parameters:**
- `optimizer`: How to update weights ('adam', 'sgd', etc.)
- `loss`: What to minimize
  - `sparse_categorical_crossentropy`: Multi-class (integer labels)
  - `categorical_crossentropy`: Multi-class (one-hot labels)
  - `mse`: Regression
- `metrics`: What to track during training

---

### Train the Model

```python
# One-line training!
history = model.fit(
    x_train,           # Training data
    y_train,           # Training labels
    epochs=10,         # Number of epochs
    batch_size=32,     # Batch size
    validation_split=0.2,  # Use 20% for validation
    verbose=1          # Print progress
)
```

**That's it! No manual training loop needed!**

**PyTorch Comparison:**
```python
# PyTorch: Manual loop
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch.x)
        loss = loss_fn(output, batch.y)
        loss.backward()
        optimizer.step()

# TensorFlow/Keras: One line
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

**Keras is more beginner-friendly!**

---

### Evaluate and Predict

```python
# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Make predictions
predictions = model.predict(x_test[:5])
print(predictions)
# [[0.1, 0.05, 0.8, ...],  # Probabilities for each class
#  [0.9, 0.02, 0.03, ...],
#  ...]

# Get predicted classes
predicted_classes = predictions.argmax(axis=1)
print(predicted_classes)
# [2, 0, 1, ...]
```

---

## Part 5: Complete MNIST Example (Keras)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Preprocess
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

print(f"Training data: {x_train.shape}")  # (60000, 784)
print(f"Training labels: {y_train.shape}")  # (60000,)

# 3. Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),  # Regularization
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 4. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train
print("\nTraining...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 6. Evaluate
print("\nEvaluating...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# 7. Save model
model.save('mnist_keras.h5')
print("Model saved!")
```

**Output:**
```
Training data: (60000, 784)
Training labels: (60000,)

Training...
Epoch 1/5
422/422 [==============================] - 2s 4ms/step - loss: 0.3421 - accuracy: 0.9012 - val_loss: 0.1523 - val_accuracy: 0.9567
Epoch 2/5
422/422 [==============================] - 2s 4ms/step - loss: 0.1634 - accuracy: 0.9523 - val_loss: 0.1123 - val_accuracy: 0.9678
...
Epoch 5/5
422/422 [==============================] - 2s 4ms/step - loss: 0.0812 - accuracy: 0.9756 - val_loss: 0.0867 - val_accuracy: 0.9745

Evaluating...
Test accuracy: 0.9734
Model saved!
```

**~97% accuracy with minimal code!**

---

## Part 6: Keras Functional API

### When to Use Functional API

Use **Sequential** for:
- Simple feed-forward networks
- Linear layer stacks

Use **Functional** for:
- Multiple inputs/outputs
- Shared layers
- Complex architectures (ResNet, etc.)

---

### Functional API Example

```python
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate

# Define inputs
input_1 = Input(shape=(10,), name='input_1')
input_2 = Input(shape=(5,), name='input_2')

# Process each input
x1 = Dense(20, activation='relu')(input_1)
x2 = Dense(10, activation='relu')(input_2)

# Merge
merged = Concatenate()([x1, x2])

# Additional layers
x = Dense(15, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(x)

# Create model
model = Model(inputs=[input_1, input_2], outputs=output)

model.summary()
```

**Pattern:**
```python
# Define inputs
inputs = Input(shape=(...))

# Build graph
x = Layer1(...)(inputs)
x = Layer2(...)(x)
outputs = Layer3(...)(x)

# Create model
model = Model(inputs=inputs, outputs=outputs)
```

**For .NET Devs:** Similar to building an expression tree or function composition.

---

## Part 7: PyTorch vs TensorFlow/Keras

### Same MNIST Network

#### PyTorch Version

```python
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = PyTorchMNIST()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# Manual training loop (15-20 lines)
for epoch in range(5):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
```

---

#### TensorFlow/Keras Version

```python
# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# One-line training!
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

**Lines of code:**
- PyTorch: ~25 lines
- Keras: ~10 lines

**Keras wins for simplicity!**

---

### When to Choose Which

**Choose PyTorch when:**
- Research and experimentation
- Custom architectures and layers
- Need full control over training
- Working with LLMs and transformers
- Debugging complex models

**Choose TensorFlow/Keras when:**
- Quick prototyping
- Production deployment (mobile, web)
- Standard architectures
- Team uses TensorFlow
- Mature deployment tools needed

**Reality:** Learn both! Most jobs value familiarity with both frameworks.

---

## Part 8: Callbacks and Monitoring

### What are Callbacks?

**Callbacks** are functions called during training (e.g., save model, stop early).

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stopping (stop if no improvement)
early_stop = EarlyStopping(
    monitor='val_loss',      # What to monitor
    patience=3,              # Stop after 3 epochs with no improvement
    restore_best_weights=True  # Restore best model
)

# Model checkpointing (save best model)
checkpoint = ModelCheckpoint(
    'best_model.h5',         # Where to save
    monitor='val_accuracy',  # What to monitor
    save_best_only=True,     # Only save if better
    verbose=1
)

# Use in training
history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint]  # Add callbacks
)
```

**Common Callbacks:**
- `EarlyStopping`: Stop if no improvement
- `ModelCheckpoint`: Save best model
- `ReduceLROnPlateau`: Reduce learning rate
- `TensorBoard`: Visualize training
- `LambdaCallback`: Custom logic

---

## Summary

### Key Concepts

**TensorFlow Basics:**
- Tensors: `tf.constant()`, `tf.zeros()`, `tf.ones()`
- Operations: Similar to NumPy/PyTorch
- GPU: Automatic (if available)

**Keras Sequential API:**
```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)
```

**Keras Functional API:**
```python
inputs = Input(shape=(784,))
x = Dense(128, activation='relu')(inputs)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
```

**PyTorch vs Keras:**
- PyTorch: More control, research-friendly
- Keras: Simpler, production-ready

---

## Quiz

### Question 1
What's the main difference between Sequential and Functional API?

<details>
<summary>Answer</summary>

**Sequential**: Linear stack of layers (simple)
```python
model = Sequential([Layer1, Layer2, Layer3])
```

**Functional**: Complex graphs (flexible)
```python
x = Layer1(inputs)
y = Layer2(x)
outputs = Layer3(y)
model = Model(inputs=inputs, outputs=outputs)
```

Use Sequential for simple models, Functional for complex architectures.

</details>

### Question 2
Why does Keras use `sparse_categorical_crossentropy` instead of `categorical_crossentropy`?

<details>
<summary>Answer</summary>

**Labels format:**
- `sparse_categorical_crossentropy`: Integer labels (0, 1, 2, ...)
- `categorical_crossentropy`: One-hot labels ([1,0,0], [0,1,0], ...)

Sparse is more memory-efficient and convenient!

```python
# Sparse (better)
y = [0, 1, 2, 1, 0]  # Just integers

# Categorical (wasteful)
y = [[1,0,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0]]  # One-hot
```

</details>

### Question 3
What does `model.compile()` do?

<details>
<summary>Answer</summary>

Configures the model for training:
1. **Optimizer**: How to update weights
2. **Loss**: What to minimize
3. **Metrics**: What to track (accuracy, etc.)

Must be called before `fit()`!

</details>

---

## Lab Exercise

### Build a Custom Network

```python
# TODO: Build a Keras model for binary classification
# Input: 20 features
# Hidden layer 1: 50 neurons, ReLU
# Hidden layer 2: 25 neurons, ReLU
# Output: 1 neuron, sigmoid (binary classification)

# 1. Build model (Sequential or Functional)
# 2. Compile with appropriate loss function
# 3. Create dummy data and train

# Your code here:
```

<details>
<summary>Solution</summary>

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Build model
model = keras.Sequential([
    layers.Dense(50, activation='relu', input_shape=(20,)),
    layers.Dense(25, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 2. Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Binary classification
    metrics=['accuracy']
)

# 3. Create dummy data
x_train = np.random.randn(1000, 20)
y_train = np.random.randint(0, 2, 1000)

# 4. Train
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 5. Evaluate
print(f"\nFinal accuracy: {history.history['accuracy'][-1]:.4f}")
```

</details>

---

## Next Steps

**Congratulations!** You now know:
✅ TensorFlow fundamentals
✅ Keras Sequential API
✅ Keras Functional API
✅ Differences from PyTorch

**Next Lesson:** Framework Comparison (`05_framework_comparison.md`)

You'll learn:
- Detailed PyTorch vs TensorFlow comparison
- When to use each framework
- Ecosystem and community
- Career and industry perspectives

---

**Practice building models in Keras before moving on!**
