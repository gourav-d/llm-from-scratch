"""
Email Spam Classifier - Simple Version
========================================

A neural network that classifies emails as spam or ham (not spam).

This is the SIMPLE version - easier to understand!
For more features, see project_main.py

What it does:
1. Loads email data (5000 emails)
2. Converts text to numbers (bag of words)
3. Builds a 2-layer neural network
4. Trains using backpropagation + Adam optimizer
5. Evaluates on test set
6. Saves visualizations

Architecture:
Input (1000) → Hidden (64, ReLU) → Output (1, Sigmoid)

Expected accuracy: 92-95%
Training time: ~10 seconds
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("EMAIL SPAM CLASSIFIER - Simple Version")
print("=" * 60)
print()

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

def load_data(filepath='data/emails.csv'):
    """
    Load email data from CSV file.

    Returns:
        emails: List of email text strings
        labels: List of labels (0 = ham, 1 = spam)
    """
    print("📂 Step 1: Loading data...")

    # For now, we'll create sample data if file doesn't exist
    # In project_main.py, we'll load from actual CSV
    if not os.path.exists(filepath):
        print("   Creating sample data (file not found)...")
        emails, labels = create_sample_data()
    else:
        emails = []
        labels = []
        with open(filepath, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    text, label = parts
                    emails.append(text)
                    labels.append(1 if label == 'spam' else 0)

    print(f"   ✓ Loaded {len(emails)} emails")
    spam_count = sum(labels)
    ham_count = len(labels) - spam_count
    print(f"   ✓ Spam: {spam_count}, Ham: {ham_count}")
    print()

    return emails, np.array(labels)


def create_sample_data():
    """
    Create sample email data for demonstration.

    In real version, this data comes from emails.csv
    """
    # Sample spam emails (common spam patterns)
    spam_emails = [
        "Buy cheap pills now! Limited time offer!",
        "Congratulations! You won $1000000! Click here!",
        "URGENT: Verify your account immediately",
        "Get rich quick! Work from home!",
        "Free money! No strings attached!",
        "Weight loss miracle! Buy now!",
        "Cheap viagra online pharmacy",
        "You have won a lottery! Claim prize now!",
        "Make money fast! Click here!",
        "Discount pills! Best prices!",
    ] * 250  # Repeat to get 2500 spam emails

    # Sample ham emails (legitimate)
    ham_emails = [
        "Meeting tomorrow at 3pm in conference room",
        "Please review the attached document",
        "Thanks for your email, I will get back to you",
        "Project deadline is next Friday",
        "Can we schedule a call this week?",
        "The report is ready for your review",
        "Let me know if you have any questions",
        "I have attached the updated presentation",
        "Looking forward to our meeting",
        "Please find the invoice attached",
    ] * 250  # Repeat to get 2500 ham emails

    # Combine and shuffle
    all_emails = spam_emails + ham_emails
    all_labels = [1] * len(spam_emails) + [0] * len(ham_emails)

    # Shuffle using numpy
    indices = np.random.permutation(len(all_emails))
    all_emails = [all_emails[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    return all_emails, all_labels


# ============================================================================
# PART 2: TEXT PREPROCESSING
# ============================================================================

def build_vocabulary(emails, vocab_size=1000):
    """
    Build vocabulary from emails.

    Takes the top vocab_size most common words.

    Args:
        emails: List of email strings
        vocab_size: Number of words to keep

    Returns:
        vocabulary: Dictionary mapping word -> index
    """
    print("📝 Step 2: Building vocabulary...")

    # Tokenize all emails and count word frequency
    word_counts = Counter()
    for email in emails:
        # Simple tokenization: lowercase and split on spaces
        words = email.lower().split()
        word_counts.update(words)

    # Get top vocab_size most common words
    most_common = word_counts.most_common(vocab_size)

    # Create vocabulary dictionary
    vocabulary = {word: idx for idx, (word, count) in enumerate(most_common)}

    print(f"   ✓ Found {len(word_counts)} unique words")
    print(f"   ✓ Keeping top {vocab_size} most common words")
    print(f"   ✓ Sample words: {list(vocabulary.keys())[:10]}")
    print()

    return vocabulary


def email_to_features(email, vocabulary):
    """
    Convert email text to bag-of-words feature vector.

    This is how we turn text into numbers!

    Args:
        email: Email text string
        vocabulary: Dictionary mapping word -> index

    Returns:
        features: NumPy array of shape (vocab_size,)
                  Each element is 1 if word present, 0 otherwise
    """
    # Initialize feature vector (all zeros)
    features = np.zeros(len(vocabulary))

    # Tokenize email
    words = email.lower().split()

    # For each word, set corresponding feature to 1
    for word in words:
        if word in vocabulary:
            idx = vocabulary[word]
            features[idx] = 1  # Binary: word present or not

    return features


def create_feature_matrix(emails, vocabulary):
    """
    Convert list of emails to feature matrix.

    Args:
        emails: List of email strings
        vocabulary: Word -> index dictionary

    Returns:
        X: NumPy array of shape (num_emails, vocab_size)
    """
    num_emails = len(emails)
    vocab_size = len(vocabulary)

    X = np.zeros((num_emails, vocab_size))

    for i, email in enumerate(emails):
        X[i] = email_to_features(email, vocabulary)

    return X


# ============================================================================
# PART 3: NEURAL NETWORK
# ============================================================================

class SpamClassifier:
    """
    2-layer neural network for binary classification.

    Architecture:
        Input (vocab_size) → Hidden (hidden_size, ReLU) → Output (1, Sigmoid)

    This is the same architecture from Module 3!
    """

    def __init__(self, input_size, hidden_size):
        """
        Initialize network weights.

        Uses He initialization (good for ReLU).
        """
        # Layer 1: Input → Hidden
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        # Layer 2: Hidden → Output
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, 1))

        # Store for backpropagation
        self.cache = {}

    def relu(self, z):
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Derivative of ReLU: 1 if z > 0, else 0"""
        return (z > 0).astype(float)

    def sigmoid(self, z):
        """Sigmoid activation: 1 / (1 + e^-z)"""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """
        Forward propagation.

        Args:
            X: Input features, shape (batch_size, input_size)

        Returns:
            y_pred: Predictions, shape (batch_size, 1)
                    Values between 0 and 1 (spam probability)
        """
        # Layer 1: Input → Hidden
        z1 = X @ self.W1 + self.b1  # Linear transformation
        a1 = self.relu(z1)           # ReLU activation

        # Layer 2: Hidden → Output
        z2 = a1 @ self.W2 + self.b2  # Linear transformation
        y_pred = self.sigmoid(z2)     # Sigmoid activation (0 to 1)

        # Cache for backpropagation
        self.cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'y_pred': y_pred}

        return y_pred

    def backward(self, y_true):
        """
        Backpropagation.

        Computes gradients using chain rule (Module 3, Lesson 4!)

        Args:
            y_true: True labels, shape (batch_size, 1)

        Returns:
            gradients: Dictionary with dW1, db1, dW2, db2
        """
        # Get cached values
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        y_pred = self.cache['y_pred']

        batch_size = X.shape[0]

        # Layer 2 gradients (output layer)
        # dL/dz2 = y_pred - y_true (for binary cross-entropy + sigmoid)
        dz2 = y_pred - y_true

        dW2 = (a1.T @ dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        # Layer 1 gradients (hidden layer)
        # Backpropagate through layer 2
        da1 = dz2 @ self.W2.T

        # Apply ReLU derivative
        dz1 = da1 * self.relu_derivative(z1)

        dW1 = (X.T @ dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


# ============================================================================
# PART 4: ADAM OPTIMIZER
# ============================================================================

class AdamOptimizer:
    """
    Adam optimizer (Module 3, Lesson 6!)

    Combines momentum and RMSProp for fast convergence.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize moments
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (RMSProp)
        self.t = 0   # Time step

    def update(self, network, gradients):
        """
        Update network weights using Adam.

        Args:
            network: SpamClassifier instance
            gradients: Dictionary with dW1, db1, dW2, db2
        """
        self.t += 1

        # Initialize moments on first call
        if not self.m:
            for param in ['W1', 'b1', 'W2', 'b2']:
                self.m[param] = np.zeros_like(getattr(network, param))
                self.v[param] = np.zeros_like(getattr(network, param))

        # Update each parameter
        for param in ['W1', 'b1', 'W2', 'b2']:
            # Get current values
            grad = gradients[f'd{param}']

            # Update biased first moment (momentum)
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad

            # Update biased second moment (RMSProp)
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
    """
    Binary cross-entropy loss.

    L = -[y*log(y_pred) + (1-y)*log(1-y_pred)]

    Args:
        y_true: True labels, shape (batch_size, 1)
        y_pred: Predictions, shape (batch_size, 1)

    Returns:
        loss: Average loss over batch
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Compute loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return loss


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities

    Returns:
        accuracy: Fraction of correct predictions
    """
    # Convert probabilities to binary predictions (threshold 0.5)
    predictions = (y_pred > 0.5).astype(int)

    # Compute accuracy
    accuracy = np.mean(predictions == y_true)

    return accuracy


def train(network, X_train, y_train, X_val, y_val, epochs=30, batch_size=32, learning_rate=0.001):
    """
    Train the neural network.

    Args:
        network: SpamClassifier instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Learning rate for Adam

    Returns:
        history: Dictionary with training metrics
    """
    print("🎓 Step 5: Training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print()

    # Initialize optimizer
    optimizer = AdamOptimizer(learning_rate=learning_rate)

    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Training loop
    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices].reshape(-1, 1)

        # Mini-batch training
        epoch_loss = 0
        for i in range(num_batches):
            # Get batch
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Forward pass
            y_pred = network.forward(X_batch)

            # Compute loss
            loss = binary_cross_entropy(y_batch, y_pred)
            epoch_loss += loss

            # Backward pass
            gradients = network.backward(y_batch)

            # Update weights
            optimizer.update(network, gradients)

        # Compute metrics on full datasets
        train_pred = network.forward(X_train)
        train_loss = binary_cross_entropy(y_train.reshape(-1, 1), train_pred)
        train_acc = compute_accuracy(y_train.reshape(-1, 1), train_pred)

        val_pred = network.forward(X_val)
        val_loss = binary_cross_entropy(y_val.reshape(-1, 1), val_pred)
        val_acc = compute_accuracy(y_val.reshape(-1, 1), val_pred)

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Loss={train_loss:.3f}, Acc={train_acc:.1%}, "
                  f"Val Loss={val_loss:.3f}, Val Acc={val_acc:.1%}")

    print()
    print("   ✓ Training complete!")
    print()

    return history


# ============================================================================
# PART 6: EVALUATION
# ============================================================================

def evaluate(network, X_test, y_test):
    """
    Evaluate network on test set.

    Args:
        network: Trained SpamClassifier
        X_test, y_test: Test data

    Returns:
        metrics: Dictionary with accuracy, precision, recall
    """
    print("📊 Step 6: Evaluating...")

    # Make predictions
    y_pred_prob = network.forward(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Reshape for comparison
    y_test = y_test.reshape(-1, 1)

    # Compute confusion matrix
    tp = np.sum((y_pred == 1) & (y_test == 1))  # True positives
    tn = np.sum((y_pred == 0) & (y_test == 0))  # True negatives
    fp = np.sum((y_pred == 1) & (y_test == 0))  # False positives
    fn = np.sum((y_pred == 0) & (y_test == 1))  # False negatives

    # Compute metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"   ✓ Test Accuracy:  {accuracy:.1%}")
    print(f"   ✓ Test Precision: {precision:.1%}")
    print(f"   ✓ Test Recall:    {recall:.1%}")
    print(f"   ✓ F1 Score:       {f1:.1%}")
    print()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': (tp, tn, fp, fn)
    }


# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path='results/training_curve.png'):
    """
    Plot training and validation metrics.
    """
    os.makedirs('results', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved plot: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Step 1: Load data
    emails, labels = load_data()

    # Step 2: Build vocabulary
    vocabulary = build_vocabulary(emails, vocab_size=1000)

    # Step 3: Create features
    print("🔢 Step 3: Creating features...")
    X = create_feature_matrix(emails, vocabulary)
    y = labels
    print(f"   ✓ Feature matrix shape: {X.shape}")
    print()

    # Step 4: Split into train/validation/test
    print("✂️  Step 4: Splitting data...")
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    print(f"   ✓ Training set:   {len(X_train)} emails")
    print(f"   ✓ Validation set: {len(X_val)} emails")
    print(f"   ✓ Test set:       {len(X_test)} emails")
    print()

    # Step 5: Build neural network
    input_size = len(vocabulary)
    hidden_size = 64

    network = SpamClassifier(input_size, hidden_size)
    print(f"🧠 Neural Network Architecture:")
    print(f"   Input layer:  {input_size} features")
    print(f"   Hidden layer: {hidden_size} neurons (ReLU)")
    print(f"   Output layer: 1 neuron (Sigmoid)")
    total_params = (input_size * hidden_size + hidden_size) + (hidden_size * 1 + 1)
    print(f"   Total parameters: {total_params:,}")
    print()

    # Step 6: Train
    history = train(network, X_train, y_train, X_val, y_val, epochs=30, learning_rate=0.001)

    # Step 7: Evaluate
    metrics = evaluate(network, X_test, y_test)

    # Step 8: Plot
    print("📈 Step 7: Creating visualizations...")
    plot_training_history(history)
    print()

    # Final summary
    print("=" * 60)
    print("✅ PROJECT COMPLETE!")
    print("=" * 60)
    print(f"Final Test Accuracy: {metrics['accuracy']:.1%}")
    print(f"Check results/ folder for plots and detailed metrics")
    print()
    print("Next steps:")
    print("1. View results/training_curve.png")
    print("2. Read EXPLANATION.md to understand the code")
    print("3. Try experiments (change hidden_size, learning_rate, etc.)")
    print("4. Move to project_main.py for more features!")
    print("=" * 60)


if __name__ == '__main__':
    main()
