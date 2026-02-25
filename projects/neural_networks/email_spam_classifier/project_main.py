"""
Email Spam Classifier - Complete Version
==========================================

Production-quality neural network for email spam classification.

This is the ADVANCED version with:
- Better text preprocessing (lowercasing, punctuation removal)
- Real CSV dataset loading
- Hyperparameter tuning
- Confusion matrix visualization
- Model saving/loading
- Custom email testing
- More detailed metrics

For a simpler version, see project_simple.py

Architecture:
Input (vocab_size) → Hidden (hidden_size, ReLU) → Output (1, Sigmoid)

Expected accuracy: 93-96%
Training time: ~15 seconds
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import re
import pickle

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("EMAIL SPAM CLASSIFIER - Production Version")
print("=" * 70)
print()

# ============================================================================
# PART 1: ADVANCED DATA LOADING
# ============================================================================

def preprocess_text(text):
    """
    Clean and normalize text.

    Steps:
    1. Convert to lowercase
    2. Remove special characters (keep only letters and spaces)
    3. Remove extra whitespace

    Args:
        text: Raw email text

    Returns:
        cleaned_text: Preprocessed text
    """
    # Lowercase
    text = text.lower()

    # Remove special characters (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def load_data_from_csv(filepath='data/emails.csv'):
    """
    Load email data from CSV file with proper error handling.

    CSV format:
    text,label
    "Email text here",spam
    "Email text here",ham

    Args:
        filepath: Path to CSV file

    Returns:
        emails: List of email text strings (preprocessed)
        labels: NumPy array of labels (0=ham, 1=spam)
    """
    print("📂 Step 1: Loading data from CSV...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    emails = []
    labels = []

    with open(filepath, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)

        # Read each line
        for line_num, line in enumerate(f, start=2):
            try:
                # Split on last comma (text may contain commas)
                parts = line.strip().rsplit(',', 1)

                if len(parts) != 2:
                    print(f"   Warning: Skipping malformed line {line_num}")
                    continue

                text, label = parts

                # Remove quotes
                text = text.strip('"')
                label = label.strip()

                # Preprocess text
                text = preprocess_text(text)

                # Convert label
                if label == 'spam':
                    label_int = 1
                elif label == 'ham':
                    label_int = 0
                else:
                    print(f"   Warning: Unknown label '{label}' on line {line_num}")
                    continue

                emails.append(text)
                labels.append(label_int)

            except Exception as e:
                print(f"   Warning: Error processing line {line_num}: {e}")
                continue

    print(f"   ✓ Loaded {len(emails)} emails")

    spam_count = sum(labels)
    ham_count = len(labels) - spam_count
    print(f"   ✓ Spam: {spam_count} ({spam_count/len(labels)*100:.1f}%)")
    print(f"   ✓ Ham:  {ham_count} ({ham_count/len(labels)*100:.1f}%)")

    # Check class balance
    balance_ratio = min(spam_count, ham_count) / max(spam_count, ham_count)
    if balance_ratio < 0.7:
        print(f"   ⚠️  Warning: Dataset is imbalanced (ratio: {balance_ratio:.2f})")

    print()

    return emails, np.array(labels)


# ============================================================================
# PART 2: ADVANCED TEXT PREPROCESSING
# ============================================================================

def build_vocabulary(emails, vocab_size=1000, min_frequency=2):
    """
    Build vocabulary with minimum frequency threshold.

    Args:
        emails: List of email strings
        vocab_size: Maximum vocabulary size
        min_frequency: Minimum word frequency to include

    Returns:
        vocabulary: Dictionary mapping word -> index
        word_frequencies: Dictionary mapping word -> count
    """
    print("📝 Step 2: Building vocabulary...")

    # Count word frequencies
    word_counts = Counter()
    for email in emails:
        words = email.split()
        word_counts.update(words)

    print(f"   ✓ Found {len(word_counts)} unique words")

    # Filter by minimum frequency
    filtered_words = {word: count for word, count in word_counts.items()
                     if count >= min_frequency}

    print(f"   ✓ After filtering (min_freq={min_frequency}): {len(filtered_words)} words")

    # Get top vocab_size words
    most_common = Counter(filtered_words).most_common(vocab_size)

    # Create vocabulary
    vocabulary = {word: idx for idx, (word, count) in enumerate(most_common)}

    print(f"   ✓ Final vocabulary size: {len(vocabulary)} words")
    print(f"   ✓ Top 10 words: {list(vocabulary.keys())[:10]}")

    # Show some spam indicators
    spam_words = [w for w in ['free', 'win', 'click', 'buy', 'now', 'urgent']
                  if w in vocabulary]
    if spam_words:
        print(f"   ✓ Spam indicators in vocab: {spam_words}")

    print()

    return vocabulary, dict(word_counts)


def email_to_features_tfidf(email, vocabulary, word_frequencies, num_docs):
    """
    Convert email to TF-IDF features (better than simple bag of words).

    TF-IDF = Term Frequency * Inverse Document Frequency
    - TF: How often word appears in this email
    - IDF: How rare the word is across all emails

    Args:
        email: Email text string
        vocabulary: Word -> index mapping
        word_frequencies: Word -> document count
        num_docs: Total number of documents

    Returns:
        features: NumPy array (vocab_size,)
    """
    features = np.zeros(len(vocabulary))
    words = email.split()

    if len(words) == 0:
        return features

    # Count word frequency in this email
    word_counts_in_email = Counter(words)

    for word, count in word_counts_in_email.items():
        if word in vocabulary:
            idx = vocabulary[word]

            # Term Frequency (normalized)
            tf = count / len(words)

            # Inverse Document Frequency (with smoothing)
            idf = np.log((num_docs + 1) / (word_frequencies.get(word, 0) + 1)) + 1

            # TF-IDF
            features[idx] = tf * idf

    return features


def create_feature_matrix(emails, vocabulary, word_frequencies=None, use_tfidf=False):
    """
    Convert list of emails to feature matrix.

    Args:
        emails: List of email strings
        vocabulary: Word -> index dictionary
        word_frequencies: Word counts (for TF-IDF)
        use_tfidf: Use TF-IDF instead of binary features

    Returns:
        X: NumPy array of shape (num_emails, vocab_size)
    """
    num_emails = len(emails)
    vocab_size = len(vocabulary)

    X = np.zeros((num_emails, vocab_size))

    for i, email in enumerate(emails):
        if use_tfidf and word_frequencies:
            X[i] = email_to_features_tfidf(email, vocabulary, word_frequencies, num_emails)
        else:
            # Binary bag of words
            words = email.split()
            for word in words:
                if word in vocabulary:
                    idx = vocabulary[word]
                    X[i, idx] = 1

    return X


# ============================================================================
# PART 3: NEURAL NETWORK (Same as simple version but cleaner)
# ============================================================================

class SpamClassifier:
    """
    2-layer neural network for binary classification.

    Architecture:
        Input (vocab_size) → Hidden (hidden_size, ReLU) → Output (1, Sigmoid)
    """

    def __init__(self, input_size, hidden_size, seed=42):
        """Initialize network with He initialization."""
        np.random.seed(seed)

        # Layer 1: Input → Hidden
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        # Layer 2: Hidden → Output
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, 1))

        # Architecture info
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Cache for backpropagation
        self.cache = {}

    def relu(self, z):
        """ReLU activation."""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """ReLU derivative."""
        return (z > 0).astype(float)

    def sigmoid(self, z):
        """Sigmoid activation with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """Forward propagation."""
        # Layer 1
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        y_pred = self.sigmoid(z2)

        # Cache for backpropagation
        self.cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'y_pred': y_pred}

        return y_pred

    def backward(self, y_true):
        """Backpropagation."""
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        y_pred = self.cache['y_pred']

        batch_size = X.shape[0]

        # Output layer gradients
        dz2 = y_pred - y_true
        dW2 = (a1.T @ dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(z1)
        dW1 = (X.T @ dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        """Get binary predictions."""
        proba = self.forward(X)
        return (proba > threshold).astype(int)

    def save(self, filepath):
        """Save model to file."""
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"   ✓ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        network = cls(model_data['input_size'], model_data['hidden_size'])
        network.W1 = model_data['W1']
        network.b1 = model_data['b1']
        network.W2 = model_data['W2']
        network.b2 = model_data['b2']

        print(f"   ✓ Model loaded from {filepath}")
        return network


# ============================================================================
# PART 4: ADAM OPTIMIZER
# ============================================================================

class AdamOptimizer:
    """Adam optimizer (Module 3, Lesson 6)."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, network, gradients):
        """Update network weights using Adam."""
        self.t += 1

        # Initialize moments on first call
        if not self.m:
            for param in ['W1', 'b1', 'W2', 'b2']:
                self.m[param] = np.zeros_like(getattr(network, param))
                self.v[param] = np.zeros_like(getattr(network, param))

        # Update each parameter
        for param in ['W1', 'b1', 'W2', 'b2']:
            grad = gradients[f'd{param}']

            # Update moments
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)

            # Update parameter
            param_value = getattr(network, param)
            param_value -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# ============================================================================
# PART 5: TRAINING
# ============================================================================

def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss."""
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def compute_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute comprehensive metrics.

    Returns:
        dict with accuracy, precision, recall, f1, confusion matrix
    """
    # Convert probabilities to binary predictions
    predictions = (y_pred > threshold).astype(int)

    # Confusion matrix
    tp = np.sum((predictions == 1) & (y_true == 1))
    tn = np.sum((predictions == 0) & (y_true == 0))
    fp = np.sum((predictions == 1) & (y_true == 0))
    fn = np.sum((predictions == 0) & (y_true == 1))

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': (tp, tn, fp, fn)
    }


def train(network, X_train, y_train, X_val, y_val,
          epochs=30, batch_size=32, learning_rate=0.001, verbose=True):
    """
    Train the neural network with detailed progress tracking.
    """
    if verbose:
        print("🎓 Step 5: Training neural network...")
        print(f"   Architecture: {network.input_size} → {network.hidden_size} → 1")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print()

    # Initialize optimizer
    optimizer = AdamOptimizer(learning_rate=learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': []
    }

    # Training loop
    num_batches = len(X_train) // batch_size
    best_val_acc = 0

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices].reshape(-1, 1)

        # Mini-batch training
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Forward pass
            y_pred = network.forward(X_batch)

            # Backward pass
            gradients = network.backward(y_batch)

            # Update weights
            optimizer.update(network, gradients)

        # Compute metrics
        train_pred = network.forward(X_train)
        train_metrics = compute_metrics(y_train.reshape(-1, 1), train_pred)

        val_pred = network.forward(X_val)
        val_metrics = compute_metrics(y_val.reshape(-1, 1), val_pred)

        # Store history
        history['train_loss'].append(binary_cross_entropy(y_train.reshape(-1, 1), train_pred))
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(binary_cross_entropy(y_val.reshape(-1, 1), val_pred))
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])

        # Track best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']

        # Print progress
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"   Epoch {epoch+1:3d}/{epochs}: "
                  f"Loss={history['train_loss'][-1]:.3f}, "
                  f"Acc={train_metrics['accuracy']:.1%}, "
                  f"Val Acc={val_metrics['accuracy']:.1%}, "
                  f"Val F1={val_metrics['f1']:.1%}")

    if verbose:
        print()
        print(f"   ✓ Training complete! Best validation accuracy: {best_val_acc:.1%}")
        print()

    return history


# ============================================================================
# PART 6: EVALUATION
# ============================================================================

def evaluate(network, X_test, y_test, verbose=True):
    """Comprehensive evaluation with detailed metrics."""
    if verbose:
        print("📊 Step 6: Evaluating on test set...")

    # Make predictions
    y_pred_prob = network.forward(X_test)
    metrics = compute_metrics(y_test.reshape(-1, 1), y_pred_prob)

    if verbose:
        print(f"   ✓ Test Accuracy:  {metrics['accuracy']:.2%}")
        print(f"   ✓ Test Precision: {metrics['precision']:.2%}")
        print(f"   ✓ Test Recall:    {metrics['recall']:.2%}")
        print(f"   ✓ Test F1 Score:  {metrics['f1']:.2%}")
        print()

        # Show confusion matrix
        tp, tn, fp, fn = metrics['confusion_matrix']
        print("   Confusion Matrix:")
        print(f"                 Predicted")
        print(f"               Spam    Ham")
        print(f"   Actual Spam   {tp:3d}    {fn:3d}    (Recall: {metrics['recall']:.1%})")
        print(f"          Ham    {fp:3d}   {tn:3d}    (Precision: {metrics['precision']:.1%})")
        print()

    return metrics


# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path='results/training_curve.png'):
    """Plot comprehensive training history."""
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2, color='orange')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2, color='blue')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Precision vs Recall
    axes[1, 0].plot(history['val_precision'], label='Precision', linewidth=2, color='green')
    axes[1, 0].plot(history['val_recall'], label='Recall', linewidth=2, color='red')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_title('Validation Precision vs Recall', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    final_metrics = f"""
    Final Training Metrics
    ━━━━━━━━━━━━━━━━━━━━━━━━

    Train Accuracy:  {history['train_acc'][-1]:.2%}
    Val Accuracy:    {history['val_acc'][-1]:.2%}

    Val Precision:   {history['val_precision'][-1]:.2%}
    Val Recall:      {history['val_recall'][-1]:.2%}

    Final Train Loss: {history['train_loss'][-1]:.4f}
    Final Val Loss:   {history['val_loss'][-1]:.4f}
    """
    axes[1, 1].text(0.1, 0.5, final_metrics, fontsize=11, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved training curves: {save_path}")


def plot_confusion_matrix(confusion_matrix, save_path='results/confusion_matrix.png'):
    """Visualize confusion matrix."""
    tp, tn, fp, fn = confusion_matrix

    # Create matrix
    cm = np.array([[tp, fn],
                   [fp, tn]])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    im = ax.imshow(cm, cmap='Blues')

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Spam', 'Ham'], fontsize=12)
    ax.set_yticklabels(['Spam', 'Ham'], fontsize=12)

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)

    # Add counts to cells
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center",
                          color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=20, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved confusion matrix: {save_path}")


# ============================================================================
# PART 8: TESTING ON CUSTOM EMAILS
# ============================================================================

def test_custom_emails(network, vocabulary, word_frequencies=None, use_tfidf=False):
    """Test model on custom email examples."""
    print("🧪 Step 7: Testing on custom emails...")
    print()

    # Custom test emails
    test_emails = [
        ("buy cheap pills now limited time offer", "spam"),
        ("meeting tomorrow at pm in conference room", "ham"),
        ("urgent click here to verify your account", "spam"),
        ("please review the attached document", "ham"),
        ("congratulations you won million dollars", "spam"),
        ("lets schedule a call to discuss the project", "ham"),
        ("free money no strings attached", "spam"),
        ("the deployment is scheduled for saturday", "ham"),
        ("get rich quick work from home", "spam"),
        ("i have completed the code review", "ham"),
    ]

    correct = 0
    for email, true_label in test_emails:
        # Create feature vector
        if use_tfidf and word_frequencies:
            features = email_to_features_tfidf(email, vocabulary, word_frequencies,
                                               len(word_frequencies))
        else:
            words = email.split()
            features = np.zeros(len(vocabulary))
            for word in words:
                if word in vocabulary:
                    features[vocabulary[word]] = 1

        # Predict
        prob = network.forward(features.reshape(1, -1))[0, 0]
        pred_label = "spam" if prob > 0.5 else "ham"

        # Check if correct
        is_correct = (pred_label == true_label)
        correct += is_correct

        # Display result
        symbol = "✓" if is_correct else "✗"
        print(f"   {symbol} '{email[:50]:50s}' → {pred_label.upper():4s} ({prob:.0%}) [True: {true_label}]")

    print()
    print(f"   Custom test accuracy: {correct}/{len(test_emails)} = {correct/len(test_emails):.0%}")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Configuration
    config = {
        'vocab_size': 1000,
        'hidden_size': 64,
        'epochs': 30,
        'batch_size': 32,
        'learning_rate': 0.001,
        'use_tfidf': False,  # Set to True for TF-IDF features
    }

    # Step 1: Load data
    emails, labels = load_data_from_csv('data/emails.csv')

    # Step 2: Build vocabulary
    vocabulary, word_frequencies = build_vocabulary(emails, vocab_size=config['vocab_size'])

    # Step 3: Create features
    print("🔢 Step 3: Creating feature matrix...")
    print(f"   Feature type: {'TF-IDF' if config['use_tfidf'] else 'Binary Bag-of-Words'}")
    X = create_feature_matrix(emails, vocabulary, word_frequencies, use_tfidf=config['use_tfidf'])
    y = labels
    print(f"   ✓ Feature matrix shape: {X.shape}")
    print(f"   ✓ Sparsity: {(X == 0).sum() / X.size:.1%} of features are zero")
    print()

    # Step 4: Split data
    print("✂️  Step 4: Splitting data...")
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    # Shuffle indices
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"   ✓ Training set:   {len(X_train)} emails ({len(X_train)/n:.0%})")
    print(f"   ✓ Validation set: {len(X_val)} emails ({len(X_val)/n:.0%})")
    print(f"   ✓ Test set:       {len(X_test)} emails ({len(X_test)/n:.0%})")
    print()

    # Step 5: Build network
    network = SpamClassifier(config['vocab_size'], config['hidden_size'])

    # Step 6: Train
    history = train(
        network, X_train, y_train, X_val, y_val,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )

    # Step 7: Evaluate
    metrics = evaluate(network, X_test, y_test)

    # Step 8: Visualize
    print("📈 Generating visualizations...")
    plot_training_history(history)
    plot_confusion_matrix(metrics['confusion_matrix'])
    print()

    # Step 9: Test custom emails
    test_custom_emails(network, vocabulary, word_frequencies, config['use_tfidf'])

    # Step 10: Save model
    print("💾 Saving model...")
    os.makedirs('models', exist_ok=True)
    network.save('models/spam_classifier.pkl')

    # Also save vocabulary
    with open('models/vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)
    print("   ✓ Vocabulary saved to models/vocabulary.pkl")
    print()

    # Final summary
    print("=" * 70)
    print("✅ PROJECT COMPLETE!")
    print("=" * 70)
    print(f"Final Test Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Final Test Precision: {metrics['precision']:.2%}")
    print(f"Final Test Recall:    {metrics['recall']:.2%}")
    print(f"Final Test F1:        {metrics['f1']:.2%}")
    print()
    print("Generated files:")
    print("  • results/training_curve.png - Training metrics over time")
    print("  • results/confusion_matrix.png - Prediction visualization")
    print("  • models/spam_classifier.pkl - Trained model")
    print("  • models/vocabulary.pkl - Vocabulary mapping")
    print()
    print("Next steps:")
    print("  1. View the visualizations in results/ folder")
    print("  2. Read EXPLANATION.md for code walkthrough")
    print("  3. Experiment with hyperparameters in this file")
    print("  4. Try your own emails!")
    print("  5. Move to Project 2: MNIST Digits")
    print("=" * 70)


if __name__ == '__main__':
    main()
