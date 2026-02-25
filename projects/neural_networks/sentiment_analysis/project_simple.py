"""
Sentiment Analysis - Simple Version
====================================

Binary sentiment classification on movie reviews.

Classifies IMDB reviews as POSITIVE or NEGATIVE.

Architecture:
Input (vocab_size) → Hidden (64, ReLU) → Output (1, Sigmoid)

Uses bag-of-words (simple but limited).
For better performance with word embeddings, see project_main.py

Expected accuracy: 85-88%
Training time: ~1-2 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import re

np.random.seed(42)

print("=" * 70)
print("SENTIMENT ANALYSIS - Movie Review Classifier")
print("=" * 70)
print()

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

def create_sample_reviews():
    """
    Create sample movie reviews for demonstration.

    In real version (project_main.py), we load 50k IMDB reviews.
    This creates 1000 synthetic reviews for quick testing.
    """
    # Positive review patterns
    positive_words = ['amazing', 'brilliant', 'excellent', 'great', 'fantastic',
                     'wonderful', 'love', 'best', 'perfect', 'superb']
    positive_templates = [
        "This movie was {} and {}!",
        "{} film with {} performance!",
        "Absolutely {}! A {} masterpiece.",
        "I {} this movie, it was {}!",
        "{} cinematography and {} acting.",
    ]

    # Negative review patterns
    negative_words = ['terrible', 'awful', 'boring', 'waste', 'bad',
                     'horrible', 'worst', 'disappointing', 'poor', 'dull']
    negative_templates = [
        "This movie was {} and {}.",
        "{} film with {} acting.",
        "Absolutely {}. A {} disaster.",
        "I hated this movie, it was {}.",
        "{} plot and {} execution.",
    ]

    reviews = []
    labels = []

    # Generate 500 positive reviews
    for _ in range(500):
        template = np.random.choice(positive_templates)
        words = np.random.choice(positive_words, 2, replace=True)
        review = template.format(words[0], words[1])
        reviews.append(review.lower())
        labels.append(1)  # Positive

    # Generate 500 negative reviews
    for _ in range(500):
        template = np.random.choice(negative_templates)
        words = np.random.choice(negative_words, 2, replace=True)
        review = template.format(words[0], words[1])
        reviews.append(review.lower())
        labels.append(0)  # Negative

    return reviews, np.array(labels)


def load_reviews():
    """
    Load movie reviews.

    Returns:
        reviews: List of review strings
        labels: Array of labels (0=negative, 1=positive)
    """
    print("📂 Step 1: Loading movie reviews...")

    # Create sample data
    reviews, labels = create_sample_reviews()

    print(f"   ✓ Loaded {len(reviews)} reviews")
    print(f"   ✓ Positive: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"   ✓ Negative: {len(labels)-sum(labels)} ({(1-sum(labels)/len(labels))*100:.1f}%)")
    print()

    return reviews, labels


# ============================================================================
# PART 2: TEXT PREPROCESSING
# ============================================================================

def preprocess_text(text):
    """Clean text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join(text.split())
    return text


def build_vocabulary(reviews, vocab_size=5000):
    """
    Build vocabulary from reviews.

    Returns:
        vocabulary: Dict mapping word -> index
        word_frequencies: Dict mapping word -> count
    """
    print("📝 Step 2: Building vocabulary...")

    word_counts = Counter()
    for review in reviews:
        review = preprocess_text(review)
        words = review.split()
        word_counts.update(words)

    print(f"   ✓ Found {len(word_counts)} unique words")

    # Get top vocab_size words
    most_common = word_counts.most_common(vocab_size)
    vocabulary = {word: idx for idx, (word, count) in enumerate(most_common)}

    print(f"   ✓ Vocabulary size: {len(vocabulary)} words")
    print(f"   ✓ Top 10: {list(vocabulary.keys())[:10]}")

    # Show sentiment indicators
    positive_indicators = [w for w in ['amazing', 'great', 'excellent', 'love', 'brilliant']
                          if w in vocabulary]
    negative_indicators = [w for w in ['terrible', 'awful', 'bad', 'worst', 'boring']
                          if w in vocabulary]

    if positive_indicators:
        print(f"   ✓ Positive indicators: {positive_indicators}")
    if negative_indicators:
        print(f"   ✓ Negative indicators: {negative_indicators}")

    print()

    return vocabulary, dict(word_counts)


def create_feature_matrix(reviews, vocabulary):
    """
    Convert reviews to bag-of-words features.

    Args:
        reviews: List of review strings
        vocabulary: Word -> index mapping

    Returns:
        X: Feature matrix (num_reviews, vocab_size)
    """
    num_reviews = len(reviews)
    vocab_size = len(vocabulary)

    X = np.zeros((num_reviews, vocab_size))

    for i, review in enumerate(reviews):
        review = preprocess_text(review)
        words = review.split()

        for word in words:
            if word in vocabulary:
                X[i, vocabulary[word]] = 1

    return X


# ============================================================================
# PART 3: NEURAL NETWORK (same as email spam classifier)
# ============================================================================

class SentimentClassifier:
    """2-layer network for sentiment classification."""

    def __init__(self, input_size, hidden_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, 1))
        self.cache = {}

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        y_pred = self.sigmoid(z2)
        self.cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'y_pred': y_pred}
        return y_pred

    def backward(self, y_true):
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        y_pred = self.cache['y_pred']
        batch_size = X.shape[0]

        dz2 = y_pred - y_true
        dW2 = (a1.T @ dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(z1)
        dW1 = (X.T @ dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


class AdamOptimizer:
    """Adam optimizer."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, network, gradients):
        self.t += 1
        if not self.m:
            for param in ['W1', 'b1', 'W2', 'b2']:
                self.m[param] = np.zeros_like(getattr(network, param))
                self.v[param] = np.zeros_like(getattr(network, param))

        for param in ['W1', 'b1', 'W2', 'b2']:
            grad = gradients[f'd{param}']
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            param_value = getattr(network, param)
            param_value -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# ============================================================================
# PART 4: TRAINING
# ============================================================================

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def compute_accuracy(y_true, y_pred):
    predictions = (y_pred > 0.5).astype(int)
    return np.mean(predictions == y_true)


def train(network, X_train, y_train, X_val, y_val, epochs=30, batch_size=32, learning_rate=0.001):
    """Train the network."""
    print("🎓 Step 3: Training neural network...")
    print(f"   Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print()

    optimizer = AdamOptimizer(learning_rate=learning_rate)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices].reshape(-1, 1)

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            y_pred = network.forward(X_batch)
            gradients = network.backward(y_batch)
            optimizer.update(network, gradients)

        train_pred = network.forward(X_train)
        train_loss = binary_cross_entropy(y_train.reshape(-1, 1), train_pred)
        train_acc = compute_accuracy(y_train.reshape(-1, 1), train_pred)

        val_pred = network.forward(X_val)
        val_loss = binary_cross_entropy(y_val.reshape(-1, 1), val_pred)
        val_acc = compute_accuracy(y_val.reshape(-1, 1), val_pred)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Loss={train_loss:.3f}, Acc={train_acc:.1%}, "
                  f"Val Acc={val_acc:.1%}")

    print()
    print(f"   ✓ Training complete! Final val accuracy: {val_acc:.1%}")
    print()

    return history


# ============================================================================
# PART 5: EVALUATION
# ============================================================================

def evaluate(network, X_test, y_test):
    """Evaluate on test set."""
    print("📊 Step 4: Evaluating...")

    y_pred_prob = network.forward(X_test)
    metrics = {
        'accuracy': compute_accuracy(y_test.reshape(-1, 1), y_pred_prob)
    }

    print(f"   ✓ Test Accuracy: {metrics['accuracy']:.2%}")
    print()

    return metrics


def test_custom_reviews(network, vocabulary):
    """Test on custom reviews."""
    print("🧪 Step 5: Testing on sample reviews...")
    print()

    test_reviews = [
        ("this movie was amazing and brilliant", "positive"),
        ("absolutely terrible waste of time", "negative"),
        ("fantastic film with great acting", "positive"),
        ("boring and disappointing movie", "negative"),
        ("excellent cinematography and superb direction", "positive"),
    ]

    for review, true_sentiment in test_reviews:
        # Create features
        review_clean = preprocess_text(review)
        words = review_clean.split()
        features = np.zeros(len(vocabulary))
        for word in words:
            if word in vocabulary:
                features[vocabulary[word]] = 1

        # Predict
        prob = network.forward(features.reshape(1, -1))[0, 0]
        pred_sentiment = "positive" if prob > 0.5 else "negative"
        is_correct = (pred_sentiment == true_sentiment)

        symbol = "✓" if is_correct else "✗"
        print(f"   {symbol} '{review}'\n      → {pred_sentiment.upper()} ({prob:.0%}) [True: {true_sentiment}]")
        print()

    print()


# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path='results/training_curve.png'):
    """Plot training history."""
    os.makedirs('results', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"   ✓ Saved plot: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""

    # Load data
    reviews, labels = load_reviews()

    # Build vocabulary
    vocabulary, word_frequencies = build_vocabulary(reviews, vocab_size=100)

    # Create features
    print("🔢 Step 3: Creating features...")
    X = create_feature_matrix(reviews, vocabulary)
    y = labels
    print(f"   ✓ Feature matrix: {X.shape}")
    print()

    # Split data
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    indices = np.random.permutation(n)
    X_train = X[indices[:n_train]]
    y_train = y[indices[:n_train]]
    X_val = X[indices[n_train:n_train + n_val]]
    y_val = y[indices[n_train:n_train + n_val]]
    X_test = X[indices[n_train + n_val:]]
    y_test = y[indices[n_train + n_val:]]

    print(f"   Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    print()

    # Build network
    network = SentimentClassifier(len(vocabulary), hidden_size=64)

    # Train
    history = train(network, X_train, y_train, X_val, y_val, epochs=30)

    # Evaluate
    metrics = evaluate(network, X_test, y_test)

    # Test custom
    test_custom_reviews(network, vocabulary)

    # Plot
    print("📈 Creating visualizations...")
    plot_training_history(history)
    print()

    # Summary
    print("=" * 70)
    print("✅ PROJECT COMPLETE!")
    print("=" * 70)
    print(f"Test Accuracy: {metrics['accuracy']:.2%}")
    print()
    print("Next steps:")
    print("  1. View results/training_curve.png")
    print("  2. Try project_main.py for word embeddings")
    print("  3. Move to Module 4: Transformers!")
    print("=" * 70)


if __name__ == '__main__':
    main()
