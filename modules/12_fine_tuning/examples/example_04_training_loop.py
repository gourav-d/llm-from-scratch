"""
Module 12 - Fine-Tuning LLMs
Example 04: The Complete Fine-Tuning Training Loop

GLOSSARY
--------
Epoch          : One full pass through the entire training dataset.
                 Like iterating a C# List<T> completely, once.
Batch          : A small chunk of the dataset processed together.
                 Like processing a queue in groups of N items.
Forward Pass   : Feed input through the model to get a prediction.
                 Like calling a method: output = model.predict(input).
Loss           : A number measuring how wrong the model is.
                 Lower = better. Like a unit test failure score.
Backward Pass  : Compute gradients (which direction to adjust weights).
                 Like finding the slope so you know which way is "downhill".
Gradient       : The direction and size of the change needed per weight.
                 Like a pointer telling each weight "go up" or "go down".
Weight Update  : Adjust weights using the gradient to reduce loss.
                 Like updating a config value by a small delta.
Learning Rate  : How big each weight update step is.
                 Too big = overshoot. Too small = too slow.
Early Stopping : Stop training when validation loss stops improving.
                 Like a circuit-breaker pattern -- stop before things get worse.
Overfitting    : Model memorises training data but fails on new data.
                 Like a developer who memorises test answers instead of learning.
Softmax        : Converts raw scores into probabilities that sum to 1.
                 Like normalising a probability distribution.
Cross-Entropy  : Loss function for classification tasks.
                 Penalises confident wrong predictions heavily.
Adam           : A smart optimiser that adapts learning rate per weight.
                 Like an auto-tuning PID controller.
Scheduler      : Changes the learning rate over time during training.
                 Like a scheduled background job that adjusts a config value.
Cosine Decay   : Learning rate follows a cosine curve from high to low.
                 Starts warm, cools down gracefully.
Gradient Clip  : Cap gradients to a max norm to prevent exploding updates.
                 Like a .NET overflow check on numeric operations.
Checkpoint     : Save the model weights to disk when a new best is found.
                 Like saving a game at a milestone.
"""

# ============================================================
#  PART A  -  Pure NumPy Training Loop
#  (No PyTorch needed -- just NumPy and Python)
# ============================================================

import numpy as np   # NumPy: the math library. Like using System.Math but for arrays.
import random        # Python's built-in random module for shuffling data.

# ---- 1. Dataset ---------------------------------------------------

# Each example is a tuple: (text, label).
# Labels: 0=NEGATIVE, 1=NEUTRAL, 2=POSITIVE
# Think of this like a List<(string Text, int Label)> in C#.
SENTIMENT_DATA = [
    ("great product love it",           2),   # POSITIVE
    ("amazing quality very happy",      2),   # POSITIVE
    ("best purchase i ever made",       2),   # POSITIVE
    ("fantastic works perfectly",       2),   # POSITIVE
    ("really excellent would buy again",2),   # POSITIVE
    ("it is okay nothing special",      1),   # NEUTRAL
    ("average quality as expected",     1),   # NEUTRAL
    ("fine for the price",              1),   # NEUTRAL
    ("does what it should",             1),   # NEUTRAL
    ("not bad not great",               1),   # NEUTRAL
    ("terrible waste of money",         0),   # NEGATIVE
    ("broke after one day useless",     0),   # NEGATIVE
    ("worst product ever disappointed", 0),   # NEGATIVE
    ("poor quality do not buy",         0),   # NEGATIVE
    ("horrible experience regret it",   0),   # NEGATIVE
]

# ---- 2. Text to Feature Vector (Bag-of-Words) --------------------

# Build a vocabulary from all words in the dataset.
# In C# this is like: var vocab = data.SelectMany(d => d.Text.Split()).Distinct().ToList();
def build_vocab(data):
    """Build a word-to-index dictionary from the dataset."""
    vocab = {}                          # Empty dict, like new Dictionary<string, int>()
    idx = 0                             # Counter for the next index to assign
    for text, _ in data:               # Loop through each (text, label) pair
        for word in text.split():      # Split text into words by whitespace
            if word not in vocab:      # Only add the word if not already in vocab
                vocab[word] = idx      # Map the word to its integer index
                idx += 1               # Move to the next index
    return vocab                        # Return the finished vocabulary dictionary

VOCAB = build_vocab(SENTIMENT_DATA)    # Build vocabulary from the training data
VOCAB_SIZE = len(VOCAB)                # How many unique words we have
NUM_CLASSES = 3                        # 3 output classes: NEG, NEU, POS

def text_to_vector(text, vocab, size):
    """Convert a string of text into a fixed-size bag-of-words vector."""
    vec = np.zeros(size)                # Start with all zeros -- shape (size,)
    for word in text.split():          # Loop over each word in the sentence
        if word in vocab:              # Only handle words we know from training
            vec[vocab[word]] = 1.0     # Set that word's position to 1 (presence)
    return vec                          # Return the feature vector

# ---- 3. Tiny 2-Layer MLP (NumPy) ---------------------------------

class TinyMLP_NumPy:
    """
    A 2-layer fully-connected neural network built with raw NumPy.
    In C# terms: a class with two weight matrices (like 2D arrays),
    one hidden layer with ReLU, and a softmax output.

    Architecture:
      Input (VOCAB_SIZE) --> [W1, b1] --> ReLU --> [W2, b2] --> Softmax
    """

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        """Initialise weights with small random values (Xavier initialisation)."""
        np.random.seed(seed)            # Fix random seed for reproducibility
        # W1: weight matrix from input to hidden layer.
        # Shape: (hidden_size x input_size). Like a 2D array[hidden, input].
        scale1 = np.sqrt(2.0 / input_size)          # Xavier scale factor
        self.W1 = np.random.randn(hidden_size, input_size) * scale1
        self.b1 = np.zeros(hidden_size)             # Bias vector for hidden layer
        # W2: weight matrix from hidden to output layer.
        scale2 = np.sqrt(2.0 / hidden_size)         # Xavier scale factor
        self.W2 = np.random.randn(output_size, hidden_size) * scale2
        self.b2 = np.zeros(output_size)             # Bias vector for output layer

    def relu(self, x):
        """ReLU activation: replace negatives with 0. Like Math.Max(0, x)."""
        return np.maximum(0.0, x)       # Element-wise max(0, x)

    def softmax(self, x):
        """Convert raw scores to probabilities. Subtract max for numerical stability."""
        x_shifted = x - np.max(x)       # Subtract max to avoid large exponentials
        exp_x = np.exp(x_shifted)       # e^x for each element
        return exp_x / np.sum(exp_x)    # Divide by sum so all values add up to 1.0

    def forward(self, x):
        """
        Forward pass: compute prediction from input x.
        Stores intermediate values for use in backward pass.
        Like running a pipeline where each stage saves its output.
        """
        self.x = x                              # Save input for backprop
        self.z1 = self.W1 @ x + self.b1        # Linear transform: hidden pre-activation
        self.a1 = self.relu(self.z1)            # Apply ReLU activation
        self.z2 = self.W2 @ self.a1 + self.b2  # Linear transform: output pre-activation
        self.probs = self.softmax(self.z2)      # Convert to class probabilities
        return self.probs                        # Return probabilities (shape: NUM_CLASSES)

    def compute_loss(self, probs, true_label):
        """
        Cross-entropy loss for a single example.
        -log(probability of the correct class).
        A perfect prediction gives loss=0. Wrong=high loss.
        """
        prob_correct = probs[true_label]        # Pick the probability of the true class
        prob_correct = max(prob_correct, 1e-12) # Clamp to avoid log(0) which is -infinity
        return -np.log(prob_correct)            # Negative log probability = cross-entropy

    def backward(self, true_label, learning_rate):
        """
        Backward pass: compute gradients and update weights.
        Uses chain rule from calculus.
        In .NET terms: like reading a stack trace backwards to find root cause,
        then applying a fix at each layer.
        """
        # --- Gradient of loss w.r.t. output (dL/dz2) ---
        # For cross-entropy + softmax, gradient is: probs - one_hot(true_label)
        d_z2 = self.probs.copy()                # Start with predicted probabilities
        d_z2[true_label] -= 1.0                 # Subtract 1 from the correct class slot

        # --- Gradient for W2 and b2 ---
        # dL/dW2 = outer product of d_z2 and hidden activations
        d_W2 = np.outer(d_z2, self.a1)          # Shape: (output_size x hidden_size)
        d_b2 = d_z2                              # Bias gradient equals output gradient

        # --- Backprop through W2 to get gradient for hidden layer ---
        d_a1 = self.W2.T @ d_z2                 # Gradient flowing back through W2

        # --- Backprop through ReLU ---
        # ReLU gradient: 1 where z1 > 0, else 0 (the "derivative" of max(0,x))
        d_z1 = d_a1 * (self.z1 > 0).astype(float)

        # --- Gradient for W1 and b1 ---
        d_W1 = np.outer(d_z1, self.x)           # Shape: (hidden_size x input_size)
        d_b1 = d_z1                              # Bias gradient equals hidden gradient

        # --- Weight Updates (Gradient Descent) ---
        # new_weight = old_weight - learning_rate * gradient
        # This is the core equation of learning!
        self.W1 -= learning_rate * d_W1         # Update input-to-hidden weights
        self.b1 -= learning_rate * d_b1         # Update hidden bias
        self.W2 -= learning_rate * d_W2         # Update hidden-to-output weights
        self.b2 -= learning_rate * d_b2         # Update output bias

    def predict(self, x):
        """Return the class index with highest probability."""
        probs = self.forward(x)                  # Run forward pass
        return int(np.argmax(probs))             # Index of highest value

# ---- 4. Early Stopping Helper ------------------------------------

class EarlyStopping:
    """
    Stop training if the loss stops improving.
    In C# terms: like a Polly retry policy that gives up after N failures.
    """

    def __init__(self, patience=2):
        """patience: how many bad epochs to tolerate before stopping."""
        self.patience = patience                 # Number of bad epochs allowed
        self.best_loss = float('inf')            # Start with worst possible loss
        self.counter = 0                         # How many bad epochs have passed
        self.should_stop = False                 # Flag: True when we decide to stop

    def __call__(self, val_loss):
        """
        Call this after each epoch with the validation loss.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss:           # Did loss improve?
            self.best_loss = val_loss            # Save new best loss
            self.counter = 0                     # Reset bad-epoch counter
        else:                                    # No improvement
            self.counter += 1                    # Count one more bad epoch
            if self.counter >= self.patience:    # Reached patience limit?
                self.should_stop = True          # Signal to stop training
        return self.should_stop                  # Return the stop flag

# ---- 5. Training Function ----------------------------------------

def train_part_a(num_epochs=30, learning_rate=0.05, hidden_size=16,
                 patience=2, val_split=0.2, show_overfit=False):
    """
    Full training loop for the NumPy sentiment classifier.
    Demonstrates epochs, batches, forward/backward passes, early stopping.
    """
    print("=" * 60)                              # Print separator line
    print("PART A: NumPy Training Loop")
    print("=" * 60)
    print()

    # --- Prepare data ---
    data_shuffled = SENTIMENT_DATA.copy()        # Copy so we don't mutate original
    random.seed(42)                              # Fix seed for reproducible shuffle
    random.shuffle(data_shuffled)               # Shuffle order of examples

    split_idx = int(len(data_shuffled) * (1 - val_split))  # Where train ends
    train_data = data_shuffled[:split_idx]      # First 80% = training set
    val_data   = data_shuffled[split_idx:]      # Last 20% = validation set

    print(f"  Training examples   : {len(train_data)}")
    print(f"  Validation examples : {len(val_data)}")
    print(f"  Vocabulary size     : {VOCAB_SIZE} words")
    print(f"  Hidden layer size   : {hidden_size}")
    print(f"  Learning rate       : {learning_rate}")
    print(f"  Early stop patience : {patience}")
    print()

    # --- Build model ---
    model = TinyMLP_NumPy(VOCAB_SIZE, hidden_size, NUM_CLASSES)
    early_stop = EarlyStopping(patience=patience)   # Create early-stop controller

    print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val Acc':>8}  Note")
    print("  " + "-" * 55)

    train_losses = []                            # Store loss per epoch for analysis
    val_losses   = []

    for epoch in range(1, num_epochs + 1):       # Loop over epochs (like a for loop in C#)

        # -- Training phase --
        random.shuffle(train_data)               # Shuffle training data each epoch
        epoch_loss = 0.0                         # Accumulate loss over this epoch

        for text, label in train_data:           # Loop over each training example
            x = text_to_vector(text, VOCAB, VOCAB_SIZE)  # Convert text to vector
            probs = model.forward(x)             # Forward pass: get predictions
            loss  = model.compute_loss(probs, label)     # Compute loss for this example
            epoch_loss += loss                   # Add to epoch total
            model.backward(label, learning_rate) # Backward pass: update weights

        train_loss = epoch_loss / len(train_data)  # Average loss over all examples

        # -- Validation phase --
        val_loss_total = 0.0                     # Accumulate validation loss
        val_correct    = 0                       # Count correct predictions

        for text, label in val_data:             # Loop over validation examples
            x     = text_to_vector(text, VOCAB, VOCAB_SIZE)
            probs = model.forward(x)             # Forward only, no backward (no learning)
            val_loss_total += model.compute_loss(probs, label)
            if np.argmax(probs) == label:        # Did we predict correctly?
                val_correct += 1

        val_loss = val_loss_total / len(val_data)    # Average validation loss
        val_acc  = val_correct / len(val_data) * 100 # Validation accuracy %

        train_losses.append(train_loss)          # Record for later analysis
        val_losses.append(val_loss)

        # -- Early stopping check --
        note = ""                                # Note column in the log
        stop_now = early_stop(val_loss)          # Check if we should stop

        if val_loss == early_stop.best_loss:     # Is this the best model so far?
            note = "<-- best"
        if early_stop.counter > 0:              # Are we in a bad streak?
            note = f"patience {early_stop.counter}/{patience}"
        if stop_now:                             # Time to stop?
            note = "EARLY STOP"

        print(f"  {epoch:>5}  {train_loss:>12.4f}  {val_loss:>10.4f}  "
              f"{val_acc:>7.1f}%  {note}")

        if stop_now:                             # Break the training loop
            print()
            print(f"  [Early stopping triggered at epoch {epoch}]")
            break

    # -- Show overfitting demonstration --
    if show_overfit and not early_stop.should_stop:
        print()
        print("  (Continuing without early stopping to show overfitting...)")
        # Keep training even when val loss rises (demonstrating overfitting)
        for epoch in range(len(train_losses) + 1, num_epochs + 1):
            epoch_loss = 0.0
            random.shuffle(train_data)
            for text, label in train_data:
                x     = text_to_vector(text, VOCAB, VOCAB_SIZE)
                probs = model.forward(x)
                loss  = model.compute_loss(probs, label)
                epoch_loss += loss
                model.backward(label, learning_rate)
            train_loss = epoch_loss / len(train_data)

            val_loss_total = 0.0
            val_correct = 0
            for text, label in val_data:
                x     = text_to_vector(text, VOCAB, VOCAB_SIZE)
                probs = model.forward(x)
                val_loss_total += model.compute_loss(probs, label)
                if np.argmax(probs) == label:
                    val_correct += 1
            val_loss = val_loss_total / len(val_data)
            val_acc  = val_correct / len(val_data) * 100

            overfit_gap = val_loss - train_loss          # Gap = sign of overfitting
            flag = " <-- OVERFITTING" if overfit_gap > 0.3 else ""
            print(f"  {epoch:>5}  {train_loss:>12.4f}  {val_loss:>10.4f}  "
                  f"{val_acc:>7.1f}%{flag}")

    print()
    print("  Training complete (Part A).")
    print()

    # Final accuracy report
    correct = sum(
        1 for text, label in SENTIMENT_DATA
        if model.predict(text_to_vector(text, VOCAB, VOCAB_SIZE)) == label
    )
    print(f"  Final accuracy on all data: {correct}/{len(SENTIMENT_DATA)} "
          f"= {correct/len(SENTIMENT_DATA)*100:.1f}%")
    print()

    return model, train_losses, val_losses

# ---- Run Part A ---------------------------------------------------

if __name__ == "__main__":

    # Run Part A with early stopping
    model_a, train_losses_a, val_losses_a = train_part_a(
        num_epochs=30,          # Maximum epochs to train
        learning_rate=0.05,     # Step size for weight updates
        hidden_size=16,         # Neurons in the hidden layer
        patience=2,             # Early stopping patience
        val_split=0.2,          # 20% of data used for validation
        show_overfit=True,      # Show what happens without early stopping
    )

    # ============================================================
    #  PART B  -  PyTorch Training Loop
    # ============================================================

    try:
        import torch                             # PyTorch: the deep learning framework
        import torch.nn as nn                    # nn: neural network building blocks
        import torch.optim as optim              # optim: optimisers (Adam, SGD, etc.)
        import math                              # math: for cosine decay calculation
        import os                                # os: file system operations (saving)

        print("=" * 60)
        print("PART B: PyTorch Training Loop")
        print("=" * 60)
        print()

        # ---- B1. Dataset as PyTorch tensors ----------------------

        # Convert all texts and labels to tensors.
        # In C# terms: transform IEnumerable<string> to float[][] and int[].
        def make_tensors(data, vocab, vocab_size):
            """Convert (text, label) list into (X_tensor, y_tensor)."""
            X_list = []                          # List to hold feature vectors
            y_list = []                          # List to hold labels
            for text, label in data:
                vec = text_to_vector(text, vocab, vocab_size)   # NumPy array
                X_list.append(vec)               # Add feature vector
                y_list.append(label)             # Add integer label
            X = torch.tensor(np.array(X_list), dtype=torch.float32)  # (N x vocab)
            y = torch.tensor(y_list, dtype=torch.long)                # (N,) ints
            return X, y                          # Return both tensors

        # Split data the same way as Part A
        data_shuffled_b = SENTIMENT_DATA.copy()
        random.seed(42)
        random.shuffle(data_shuffled_b)
        split_b = int(len(data_shuffled_b) * 0.8)     # 80% train
        train_b = data_shuffled_b[:split_b]
        val_b   = data_shuffled_b[split_b:]

        X_train, y_train = make_tensors(train_b, VOCAB, VOCAB_SIZE)  # Train tensors
        X_val,   y_val   = make_tensors(val_b,   VOCAB, VOCAB_SIZE)  # Val tensors

        print(f"  X_train shape: {X_train.shape}  (examples x vocab_size)")
        print(f"  X_val   shape: {X_val.shape}")
        print()

        # ---- B2. PyTorch Model -----------------------------------

        HIDDEN_B = 16                            # Hidden layer size (same as Part A)

        # nn.Sequential: stack layers like a pipeline.
        # Like using builder pattern: new Pipeline().AddLayer(...).AddLayer(...)
        model_b = nn.Sequential(
            nn.Linear(VOCAB_SIZE, HIDDEN_B),     # Input -> Hidden (like a Dense layer)
            nn.ReLU(),                           # ReLU activation
            nn.Linear(HIDDEN_B, NUM_CLASSES),    # Hidden -> Output (3 classes)
        )

        print("  Model architecture:")
        print(f"    Input  : {VOCAB_SIZE} features")
        print(f"    Hidden : {HIDDEN_B} neurons (ReLU)")
        print(f"    Output : {NUM_CLASSES} classes (cross-entropy applied later)")
        print()

        # ---- B3. Loss, Optimiser, Scheduler ----------------------

        # CrossEntropyLoss combines softmax + log + negative into one step.
        # It expects raw logits (un-normalised scores), not probabilities.
        criterion = nn.CrossEntropyLoss()

        # Adam optimiser: smarter than plain gradient descent.
        # It adapts the learning rate per parameter. lr=initial learning rate.
        optimizer = optim.Adam(model_b.parameters(), lr=1e-2)

        # Cosine Annealing scheduler: smoothly reduces LR from high to low.
        # T_max = total number of steps (epochs in this case).
        NUM_EPOCHS_B = 40                        # Max training epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,                           # The optimiser to adjust
            T_max=NUM_EPOCHS_B,                  # Period of cosine cycle
            eta_min=1e-5,                        # Minimum learning rate
        )

        # ---- B4. Checkpoint Setup --------------------------------

        best_val_loss_b  = float('inf')          # Track best validation loss
        best_epoch_b     = 0                     # Track which epoch was best
        checkpoint_path  = "best_model_b.pt"    # File to save best model weights
        early_stop_b     = EarlyStopping(patience=5)  # Reuse our early stop class

        # ---- B5. Training Loop -----------------------------------

        print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>10}  "
              f"{'Val Acc':>8}  {'LR':>10}  Note")
        print("  " + "-" * 72)

        for epoch in range(1, NUM_EPOCHS_B + 1):

            # -- Training mode --
            # model.train() enables dropout/batchnorm training behaviour.
            # Like setting a flag: model.IsTraining = true;
            model_b.train()

            optimizer.zero_grad()               # Clear gradients from last step
            logits = model_b(X_train)           # Forward pass on all training data
            train_loss_b = criterion(logits, y_train)  # Compute cross-entropy loss

            train_loss_b.backward()             # Backward pass: compute gradients

            # Gradient clipping: if gradients are very large, scale them down.
            # Prevents "exploding gradient" problem (weights jumping wildly).
            # Like capping a velocity to max_speed in a physics simulation.
            nn.utils.clip_grad_norm_(model_b.parameters(), max_norm=1.0)

            optimizer.step()                    # Apply the weight updates
            scheduler.step()                    # Advance the LR scheduler

            current_lr = scheduler.get_last_lr()[0]  # Read current LR for display

            # -- Evaluation mode --
            # model.eval() disables dropout; torch.no_grad() skips gradient tracking.
            # Like setting IsTraining = false and wrapping in a read-only scope.
            model_b.eval()
            with torch.no_grad():               # No gradient computation needed
                val_logits = model_b(X_val)     # Forward pass on validation set
                val_loss_b = criterion(val_logits, y_val)   # Compute val loss

                # Compute validation accuracy
                val_preds    = torch.argmax(val_logits, dim=1)  # Predicted classes
                val_correct  = (val_preds == y_val).sum().item()
                val_acc_b    = val_correct / len(y_val) * 100

            note_b = ""                          # Note column for this epoch

            # -- Save checkpoint if this is the best model so far --
            if val_loss_b.item() < best_val_loss_b:
                best_val_loss_b = val_loss_b.item()
                best_epoch_b    = epoch
                note_b          = "<-- best (saved)"
                # Save model weights to disk (like serialising an object to file)
                torch.save(model_b.state_dict(), checkpoint_path)

            # -- Check early stopping --
            stop_b = early_stop_b(val_loss_b.item())
            if stop_b:
                note_b = "EARLY STOP"

            print(f"  {epoch:>5}  {train_loss_b.item():>12.4f}  "
                  f"{val_loss_b.item():>10.4f}  {val_acc_b:>7.1f}%  "
                  f"{current_lr:>10.6f}  {note_b}")

            if stop_b:                           # Break if early stopping triggered
                print()
                print(f"  [Early stopping triggered at epoch {epoch}]")
                break

        print()
        print(f"  Best model was at epoch {best_epoch_b} "
              f"(val loss = {best_val_loss_b:.4f})")
        print(f"  Checkpoint saved to: {checkpoint_path}")
        print()

        # ---- B6. Load best checkpoint and evaluate ---------------

        print("  Loading best checkpoint for final evaluation...")
        model_b.load_state_dict(torch.load(checkpoint_path))  # Restore best weights
        model_b.eval()                           # Set to evaluation mode

        with torch.no_grad():
            X_all, y_all = make_tensors(SENTIMENT_DATA, VOCAB, VOCAB_SIZE)
            all_logits  = model_b(X_all)
            all_preds   = torch.argmax(all_logits, dim=1)
            all_correct = (all_preds == y_all).sum().item()

        print(f"  Final accuracy (best checkpoint) on all data: "
              f"{all_correct}/{len(SENTIMENT_DATA)} "
              f"= {all_correct/len(SENTIMENT_DATA)*100:.1f}%")

        # -- Compare train vs val loss per epoch (summary table) --
        print()
        print("  Part B complete.")

        # Clean up saved file (optional, comment out to keep)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)           # Delete the checkpoint file

    except ImportError:
        # If PyTorch is not installed, skip Part B gracefully.
        print("  PyTorch not installed. Skipping Part B.")
        print("  Install with: pip install torch")
