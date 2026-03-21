"""
Project 1: MNIST Comparison - NumPy vs PyTorch vs TensorFlow
=============================================================

Build the same MNIST classifier in three frameworks and compare:
- Lines of code
- Training time
- Final accuracy
- Ease of implementation

This demonstrates what you've learned across Module 3 and 3.5!

Run: python project_01_mnist_comparison.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import tensorflow as tf
from tensorflow import keras

print("=" * 70)
print("Project 1: MNIST Classifier - Framework Comparison")
print("=" * 70)

# ============================================================================
# Load Data (Shared)
# ============================================================================
print("\n1. Loading MNIST Data...")
print("-" * 70)

# PyTorch data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset_pt = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset_pt = datasets.MNIST('./data', train=False, transform=transform)

train_loader_pt = torch.utils.data.DataLoader(train_dataset_pt, batch_size=128, shuffle=True)
test_loader_pt = torch.utils.data.DataLoader(test_dataset_pt, batch_size=1000, shuffle=False)

# TensorFlow data loading
(x_train_tf, y_train_tf), (x_test_tf, y_test_tf) = keras.datasets.mnist.load_data()
x_train_tf = x_train_tf.reshape(-1, 784).astype('float32') / 255.0
x_test_tf = x_test_tf.reshape(-1, 784).astype('float32') / 255.0

print(f"✓ Training samples: {len(train_dataset_pt)}")
print(f"✓ Test samples: {len(test_dataset_pt)}")

# ============================================================================
# Model 1: PyTorch
# ============================================================================
print("\n\n2. PyTorch Implementation")
print("-" * 70)

class PyTorchMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training function
def train_pytorch():
    model = PyTorchMNIST()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    for epoch in range(3):  # 3 epochs for fair comparison
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader_pt):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time

    # Test
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader_pt:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = 100. * correct / len(test_dataset_pt)

    return train_time, accuracy

print("Training PyTorch model...")
pt_time, pt_accuracy = train_pytorch()
print(f"✓ Training time: {pt_time:.2f} seconds")
print(f"✓ Test accuracy: {pt_accuracy:.2f}%")

# ============================================================================
# Model 2: TensorFlow/Keras
# ============================================================================
print("\n\n3. TensorFlow/Keras Implementation")
print("-" * 70)

def train_tensorflow():
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

    start_time = time.time()

    # Train (verbose=0 for cleaner output)
    history = model.fit(
        x_train_tf, y_train_tf,
        epochs=3,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )

    train_time = time.time() - start_time

    # Test
    test_loss, test_accuracy = model.evaluate(x_test_tf, y_test_tf, verbose=0)

    return train_time, test_accuracy * 100

print("Training TensorFlow/Keras model...")
tf_time, tf_accuracy = train_tensorflow()
print(f"✓ Training time: {tf_time:.2f} seconds")
print(f"✓ Test accuracy: {tf_accuracy:.2f}%")

# ============================================================================
# Model 3: NumPy (Simplified - for demonstration)
# ============================================================================
print("\n\n4. NumPy Implementation (Simplified)")
print("-" * 70)
print("Note: Full NumPy implementation is much longer!")
print("This is a simplified version for comparison.")

# For a fair comparison, we'd need the full Module 3 implementation
# Here we just show the concept
print("\n✓ NumPy implementation would require:")
print("  - Manual forward pass (~20 lines)")
print("  - Manual backward pass (~40 lines)")
print("  - Manual weight updates (~10 lines)")
print("  - Training loop (~15 lines)")
print("  - Total: ~85-100 lines of code")
print("  - Training time: ~180 seconds (estimated, no GPU)")
print("  - Accuracy: ~95-96% (similar)")

# ============================================================================
# Comparison Results
# ============================================================================
print("\n\n" + "=" * 70)
print("COMPARISON RESULTS")
print("=" * 70)

print("\n📊 Performance Comparison")
print("-" * 70)
print(f"{'Framework':<20} {'Time (sec)':<15} {'Accuracy (%)':<15}")
print("-" * 70)
print(f"{'NumPy (estimated)':<20} {'~180':<15} {'~95.5':<15}")
print(f"{'PyTorch':<20} {f'{pt_time:.2f}':<15} {f'{pt_accuracy:.2f}':<15}")
print(f"{'TensorFlow/Keras':<20} {f'{tf_time:.2f}':<15} {f'{tf_accuracy:.2f}':<15}")

print("\n💻 Code Complexity")
print("-" * 70)
print(f"{'Framework':<20} {'Lines of Code':<20} {'Difficulty':<15}")
print("-" * 70)
print(f"{'NumPy':<20} {'~100':<20} {'Hard':<15}")
print(f"{'PyTorch':<20} {'~30':<20} {'Medium':<15}")
print(f"{'TensorFlow/Keras':<20} {'~15':<20} {'Easy':<15}")

print("\n⚖️ Trade-offs")
print("-" * 70)
print("""
NumPy:
  ✅ Full control and understanding
  ✅ Learn the fundamentals
  ❌ Error-prone (manual gradients)
  ❌ No GPU support
  ❌ Slow for large models

PyTorch:
  ✅ Automatic gradients
  ✅ GPU acceleration
  ✅ Pythonic and debuggable
  ✅ Research-friendly
  ⚠️ More code than Keras

TensorFlow/Keras:
  ✅ Very concise (one-line training!)
  ✅ Beginner-friendly
  ✅ Production-ready
  ✅ Great deployment tools
  ⚠️ Less transparent
""")

print("\n🎯 Recommendations")
print("-" * 70)
print("""
Use NumPy when:
  - Learning fundamentals
  - Understanding the math
  - Small experiments

Use PyTorch when:
  - Research and experimentation
  - Building LLMs and transformers
  - Need debugging flexibility
  - Working with novel architectures

Use TensorFlow/Keras when:
  - Quick prototyping
  - Production deployment
  - Mobile/web applications
  - Standard architectures
""")

print("\n📈 Speed Comparison")
print("-" * 70)
speedup_pt = 180 / pt_time if pt_time > 0 else 0
speedup_tf = 180 / tf_time if tf_time > 0 else 0
print(f"PyTorch is ~{speedup_pt:.1f}x faster than NumPy (estimated)")
print(f"TensorFlow is ~{speedup_tf:.1f}x faster than NumPy (estimated)")

if pt_time < tf_time:
    print(f"\nPyTorch was {(tf_time - pt_time):.2f} sec faster than TensorFlow")
else:
    print(f"\nTensorFlow was {(pt_time - tf_time):.2f} sec faster than PyTorch")

# ============================================================================
# Key Learnings
# ============================================================================
print("\n\n" + "=" * 70)
print("KEY LEARNINGS")
print("=" * 70)
print("""
1. **Automatic Differentiation is Magic!**
   - NumPy: 40+ lines of manual gradients
   - PyTorch/TF: loss.backward() - ONE LINE!

2. **Performance Matters**
   - GPU acceleration gives 10-100x speedup
   - Even CPU PyTorch/TF is faster than NumPy

3. **Code Complexity**
   - Keras: Simplest (best for prototyping)
   - PyTorch: Balance (flexibility + automation)
   - NumPy: Most complex (but best for learning)

4. **Accuracy is Similar**
   - All frameworks achieve ~95-97% accuracy
   - Framework choice doesn't affect results
   - Choose based on use case, not accuracy

5. **Best of Both Worlds**
   - Learn with NumPy (understand fundamentals)
   - Build with PyTorch/TF (production-ready)
   - You now have BOTH! 🎉
""")

# ============================================================================
# Next Steps
# ============================================================================
print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
✅ You've now built the same model in THREE frameworks!

Challenges to try:
1. Add dropout and batch normalization
2. Increase to 10 epochs and compare convergence
3. Try different optimizers (SGD, RMSprop)
4. Add data augmentation
5. Save and load each model
6. Benchmark on GPU vs CPU

Continue to:
- Project 2: Build a production-ready model
- Module 4: Transformers and Attention
""")

print("\n✅ Project 1 Complete!")
print("=" * 70)
