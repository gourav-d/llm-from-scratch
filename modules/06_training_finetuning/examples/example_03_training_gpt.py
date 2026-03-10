"""
Lesson 3 Example: Training GPT from Scratch

This example shows you how to train a GPT model on text data.

Training is like teaching a student:
- Show them examples (forward pass)
- Check their answers (calculate loss)
- Tell them their mistakes (backpropagation)
- Help them improve (update weights)
- Repeat thousands of times until they learn!

This is a SIMPLIFIED training example focusing on the concepts.
Real training uses PyTorch/TensorFlow for automatic differentiation.
"""

import numpy as np
from typing import List, Tuple, Dict
import time

# =============================================================================
# PART 1: SIMPLE GPT MODEL (FOR TRAINING DEMONSTRATION)
# =============================================================================

class SimpleGPT:
    """
    Simplified GPT model for training demonstration.

    In real implementations, you'd use PyTorch or TensorFlow.
    This shows the training concepts in pure NumPy.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initialize a simple GPT model.

        Args:
            vocab_size: how many different tokens/words we know
            embed_dim: size of embedding vectors
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Initialize embedding matrix
        # Each row is the vector for one token
        # Shape: (vocab_size, embed_dim)
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.01

        # Initialize output weights
        # Projects embeddings back to vocabulary logits
        # Shape: (embed_dim, vocab_size)
        self.output_weights = np.random.randn(embed_dim, vocab_size) * 0.01

        print(f"Model initialized with {self.count_parameters():,} parameters")

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass: predict next token for each position.

        This is what the model does when making predictions.
        Think: "Given these words, what comes next?"

        Args:
            token_ids: input tokens, shape (batch_size, seq_len)

        Returns:
            logits: predictions, shape (batch_size, seq_len, vocab_size)
        """
        # STEP 1: Look up embeddings for each token
        # Convert token IDs to dense vectors
        # Think: "Look up what each word means"
        x = self.embeddings[token_ids]  # (batch_size, seq_len, embed_dim)

        # STEP 2: Project to vocabulary size
        # Calculate scores for each possible next token
        # Think: "For each word, predict what comes next"
        logits = x @ self.output_weights  # (batch_size, seq_len, vocab_size)

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        total = self.embeddings.size + self.output_weights.size
        return total


# =============================================================================
# PART 2: LOSS FUNCTION
# Measures how wrong the model's predictions are
# =============================================================================

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross-entropy loss: measures prediction error.

    Think of this as a "report card" for the model:
    - Low loss (close to 0) = good predictions (A+ grade)
    - High loss (large number) = bad predictions (F grade)

    How it works:
    1. Convert logits to probabilities
    2. Check probability assigned to correct answer
    3. Loss = -log(probability of correct answer)

    If model is very confident in wrong answer: HIGH loss
    If model is confident in right answer: LOW loss

    Args:
        logits: model predictions, shape (batch_size, seq_len, vocab_size)
        targets: correct answers, shape (batch_size, seq_len)

    Returns:
        loss: single number representing average error
    """
    batch_size, seq_len, vocab_size = logits.shape

    # STEP 1: Convert logits to probabilities using softmax
    # Softmax converts any numbers to probabilities (sum to 1)
    # Think: Convert test scores to percentages

    # Subtract max for numerical stability (prevents overflow)
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # STEP 2: Extract probabilities for correct answers
    # For each position, get the probability the model assigned to the correct token

    # Flatten batch and sequence dimensions for easier indexing
    probs_flat = probs.reshape(-1, vocab_size)  # (batch*seq_len, vocab_size)
    targets_flat = targets.reshape(-1)  # (batch*seq_len,)

    # Get probability assigned to correct token for each position
    # Like checking: "Did the student mark the right answer?"
    correct_probs = probs_flat[np.arange(len(targets_flat)), targets_flat]

    # STEP 3: Calculate cross-entropy
    # Loss = -log(probability of correct answer)
    #
    # Why negative log?
    # - If prob = 1.0 (perfect): loss = -log(1.0) = 0 (no error)
    # - If prob = 0.5 (unsure): loss = -log(0.5) = 0.69 (some error)
    # - If prob = 0.1 (wrong): loss = -log(0.1) = 2.3 (big error)

    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    losses = -np.log(correct_probs + epsilon)

    # STEP 4: Average across all positions
    # Final loss is the average error across the batch
    loss = np.mean(losses)

    return loss


# =============================================================================
# PART 3: GRADIENT COMPUTATION (SIMPLIFIED)
# Calculate how to adjust weights to reduce loss
# =============================================================================

def compute_gradients_simple(
    model: SimpleGPT,
    token_ids: np.ndarray,
    targets: np.ndarray,
    logits: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute gradients: how to change weights to reduce loss.

    This is BACKPROPAGATION - the magic that makes neural networks learn!

    Think of gradients as "directions to improve":
    - Positive gradient: decrease this weight
    - Negative gradient: increase this weight
    - Large gradient: make a big change
    - Small gradient: make a small change

    In real implementations, PyTorch/TensorFlow do this automatically.
    This is a simplified version to show the concept.

    Args:
        model: the GPT model
        token_ids: input tokens
        targets: correct next tokens
        logits: model predictions

    Returns:
        gradients: dictionary with gradients for each parameter
    """
    batch_size, seq_len, vocab_size = logits.shape

    # STEP 1: Calculate gradient of loss with respect to logits
    # This tells us: "How should we change the predictions?"

    # Convert logits to probabilities
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Gradient of cross-entropy loss
    # For correct class: gradient = (probability - 1)
    # For wrong classes: gradient = probability
    d_logits = probs.copy()

    # Subtract 1 from the correct class probabilities
    batch_indices = np.arange(batch_size)[:, None]
    seq_indices = np.arange(seq_len)[None, :]
    d_logits[batch_indices, seq_indices, targets] -= 1

    # Average over batch
    d_logits = d_logits / batch_size

    # STEP 2: Backpropagate through output layer
    # Calculate gradient for output_weights

    # Get embeddings that were used
    embeddings_used = model.embeddings[token_ids]  # (batch, seq_len, embed_dim)

    # Gradient for output weights
    # This tells us: "How should we adjust the output weights?"
    d_output_weights = embeddings_used.transpose(0, 2, 1) @ d_logits  # (batch, embed_dim, vocab_size)
    d_output_weights = np.mean(d_output_weights, axis=0)  # Average over batch

    # STEP 3: Backpropagate to embeddings
    # Calculate gradient for embeddings

    d_embeddings_full = d_logits @ model.output_weights.T  # (batch, seq_len, embed_dim)

    # Accumulate gradients for each unique token
    # (Multiple positions might use the same token)
    d_embeddings = np.zeros_like(model.embeddings)

    for i in range(batch_size):
        for j in range(seq_len):
            token_id = token_ids[i, j]
            d_embeddings[token_id] += d_embeddings_full[i, j]

    # Average
    d_embeddings = d_embeddings / batch_size

    # Return gradients as dictionary
    gradients = {
        'embeddings': d_embeddings,
        'output_weights': d_output_weights
    }

    return gradients


# =============================================================================
# PART 4: OPTIMIZER
# Updates model weights using gradients
# =============================================================================

class SGDOptimizer:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This is like a "personal trainer" for the model:
    - Looks at the gradients (what to improve)
    - Updates weights in the right direction
    - Uses learning rate to control step size

    Think of learning rate like walking:
    - Too large (0.1): Take huge steps, might overshoot (run past destination)
    - Too small (0.0001): Take tiny steps, very slow progress
    - Just right (0.001-0.01): Steady progress toward goal
    """

    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize optimizer.

        Args:
            learning_rate: how big each update step is (typically 0.001-0.01)
        """
        self.learning_rate = learning_rate
        print(f"Optimizer initialized with learning_rate={learning_rate}")

    def step(self, model: SimpleGPT, gradients: Dict[str, np.ndarray]):
        """
        Update model parameters using gradients.

        This is the actual "learning" step!

        Formula: new_weight = old_weight - learning_rate * gradient

        Think: Move weights in direction that reduces loss.

        Args:
            model: the model to update
            gradients: gradients from backpropagation
        """
        # Update embeddings
        # Subtract gradient scaled by learning rate
        # Think: "Move in the direction that reduces error"
        model.embeddings -= self.learning_rate * gradients['embeddings']

        # Update output weights
        model.output_weights -= self.learning_rate * gradients['output_weights']


# =============================================================================
# PART 5: DATASET
# Prepare training data
# =============================================================================

class TextDataset:
    """
    Text dataset for language modeling.

    Prepares data in the format needed for training:
    - Input: sequence of tokens
    - Target: same sequence shifted by 1 (next token prediction)

    Example:
    Text: "The cat sat on the mat"
    Tokens: [10, 25, 30, 15, 10, 40]

    Training pairs:
    Input:  [10, 25, 30, 15, 10] -> Target: [25, 30, 15, 10, 40]
    (predict next word at each position)
    """

    def __init__(self, text_tokens: List[int], seq_len: int):
        """
        Initialize dataset.

        Args:
            text_tokens: all tokens from the text corpus
            seq_len: length of each training sequence
        """
        self.tokens = text_tokens
        self.seq_len = seq_len

        # Calculate how many training examples we can create
        self.num_examples = len(text_tokens) - seq_len

        print(f"Dataset: {len(text_tokens)} tokens, {self.num_examples} training examples")

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a random batch of training examples.

        Think: Grab a random handful of training examples.

        Args:
            batch_size: how many examples to return

        Returns:
            inputs: input sequences, shape (batch_size, seq_len)
            targets: target sequences, shape (batch_size, seq_len)
        """
        # Randomly choose starting positions
        indices = np.random.randint(0, self.num_examples, size=batch_size)

        # Extract sequences
        inputs = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        targets = np.zeros((batch_size, self.seq_len), dtype=np.int32)

        for i, idx in enumerate(indices):
            # Input: tokens[idx : idx+seq_len]
            # Target: tokens[idx+1 : idx+seq_len+1]
            # Target is input shifted by 1 (next token prediction)
            inputs[i] = self.tokens[idx : idx + self.seq_len]
            targets[i] = self.tokens[idx + 1 : idx + self.seq_len + 1]

        return inputs, targets


# =============================================================================
# PART 6: TRAINING LOOP
# The main training process
# =============================================================================

def train_model(
    model: SimpleGPT,
    dataset: TextDataset,
    optimizer: SGDOptimizer,
    num_epochs: int = 5,
    batch_size: int = 32,
    eval_interval: int = 100
):
    """
    Train the GPT model on text data.

    Training process:
    1. Get batch of training examples
    2. Forward pass (make predictions)
    3. Calculate loss (measure error)
    4. Backward pass (calculate gradients)
    5. Update weights (learn from mistakes)
    6. Repeat!

    Think of this like studying for an exam:
    - Each epoch = going through all material once
    - Each batch = studying a small chunk
    - Loss = your practice test score (lower is better)
    - Gradients = understanding what you got wrong
    - Optimizer = adjusting your study strategy

    Args:
        model: GPT model to train
        dataset: training data
        optimizer: optimizer to update weights
        num_epochs: how many times to go through all data
        batch_size: how many examples per batch
        eval_interval: how often to print progress
    """
    print("\n" + "=" * 80)
    print("TRAINING START")
    print("=" * 80)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {dataset.num_examples // batch_size}")
    print()

    # Calculate total steps
    steps_per_epoch = dataset.num_examples // batch_size
    total_steps = num_epochs * steps_per_epoch

    # Training loop
    step = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch + 1}/{num_epochs}")
        print("-" * 80)

        epoch_loss = 0.0

        for batch_idx in range(steps_per_epoch):
            step += 1

            # STEP 1: Get batch of training data
            # Randomly sample examples
            inputs, targets = dataset.get_batch(batch_size)

            # STEP 2: Forward pass
            # Run model to get predictions
            # Think: "Student attempts the practice problems"
            logits = model.forward(inputs)

            # STEP 3: Calculate loss
            # Measure how wrong the predictions are
            # Think: "Grade the practice test"
            loss = cross_entropy_loss(logits, targets)
            epoch_loss += loss

            # STEP 4: Backward pass
            # Calculate gradients (how to improve)
            # Think: "Figure out what went wrong"
            gradients = compute_gradients_simple(model, inputs, targets, logits)

            # STEP 5: Update weights
            # Apply gradients to improve model
            # Think: "Learn from mistakes"
            optimizer.step(model, gradients)

            # STEP 6: Print progress
            if step % eval_interval == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Step {step}/{total_steps} | "
                      f"Loss: {loss:.4f} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Time: {elapsed:.1f}s")

        # Epoch summary
        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"\nEpoch {epoch + 1} complete | Average Loss: {avg_epoch_loss:.4f}")

    # Training complete
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final loss: {loss:.4f}")
    print()


# =============================================================================
# PART 7: DEMONSTRATION
# =============================================================================

def main():
    """
    Demonstrate training a GPT model from scratch.
    """
    print("=" * 80)
    print("Training GPT from Scratch - Simplified Example")
    print("=" * 80)
    print()
    print("This example shows the training process in simplified form.")
    print("Real training uses PyTorch/TensorFlow with automatic differentiation.")
    print()

    # -------------------------------------------------------------------------
    # STEP 1: Prepare data
    # -------------------------------------------------------------------------
    print("STEP 1: Preparing Training Data")
    print("-" * 80)

    # Create fake text data (in real use, this would be actual text)
    # Let's create a sequence of 10,000 random tokens
    vocab_size = 100  # Small vocabulary for demo
    num_tokens = 10000

    text_tokens = np.random.randint(0, vocab_size, size=num_tokens).tolist()

    print(f"Created {num_tokens} tokens with vocabulary size {vocab_size}")
    print(f"First 20 tokens: {text_tokens[:20]}")
    print()

    # Create dataset
    seq_len = 32  # Length of each training sequence
    dataset = TextDataset(text_tokens, seq_len)
    print()

    # -------------------------------------------------------------------------
    # STEP 2: Create model
    # -------------------------------------------------------------------------
    print("STEP 2: Creating Model")
    print("-" * 80)

    embed_dim = 64  # Small embedding dimension for demo
    model = SimpleGPT(vocab_size, embed_dim)
    print()

    # -------------------------------------------------------------------------
    # STEP 3: Create optimizer
    # -------------------------------------------------------------------------
    print("STEP 3: Creating Optimizer")
    print("-" * 80)

    learning_rate = 0.01  # How big each learning step is
    optimizer = SGDOptimizer(learning_rate)
    print()

    # -------------------------------------------------------------------------
    # STEP 4: Test before training
    # -------------------------------------------------------------------------
    print("STEP 4: Test Model Before Training")
    print("-" * 80)

    # Get a test batch
    test_inputs, test_targets = dataset.get_batch(4)

    # Calculate loss before training
    logits_before = model.forward(test_inputs)
    loss_before = cross_entropy_loss(logits_before, test_targets)

    print(f"Loss before training: {loss_before:.4f}")
    print("(High loss = model is making random guesses)")
    print()

    # -------------------------------------------------------------------------
    # STEP 5: Train the model
    # -------------------------------------------------------------------------
    print("STEP 5: Training the Model")
    print("-" * 80)

    train_model(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        num_epochs=3,         # Go through data 3 times
        batch_size=32,        # 32 examples per batch
        eval_interval=20      # Print progress every 20 steps
    )

    # -------------------------------------------------------------------------
    # STEP 6: Test after training
    # -------------------------------------------------------------------------
    print("STEP 6: Test Model After Training")
    print("-" * 80)

    # Calculate loss after training (on same test batch)
    logits_after = model.forward(test_inputs)
    loss_after = cross_entropy_loss(logits_after, test_targets)

    print(f"Loss before training: {loss_before:.4f}")
    print(f"Loss after training:  {loss_after:.4f}")
    print(f"Improvement: {loss_before - loss_after:.4f}")
    print()

    if loss_after < loss_before:
        print("✓ Success! The model learned to predict better!")
        print("  Loss decreased, meaning predictions improved.")
    else:
        print("Note: Loss might not decrease in this simplified demo.")
        print("  In real training with proper backprop, loss always decreases.")
    print()

    # -------------------------------------------------------------------------
    # STEP 7: Key concepts summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("KEY CONCEPTS SUMMARY")
    print("=" * 80)
    print()
    print("TRAINING = Teaching the model by showing examples")
    print()
    print("1. FORWARD PASS:")
    print("   - Model makes predictions")
    print("   - Like student attempting practice problems")
    print()
    print("2. LOSS CALCULATION:")
    print("   - Measure how wrong predictions are")
    print("   - Like grading the practice test")
    print()
    print("3. BACKWARD PASS (Backpropagation):")
    print("   - Calculate gradients (how to improve)")
    print("   - Like analyzing mistakes to understand what went wrong")
    print()
    print("4. WEIGHT UPDATE:")
    print("   - Adjust weights using gradients")
    print("   - Like the student learning from mistakes")
    print()
    print("5. REPEAT:")
    print("   - Do this thousands of times")
    print("   - Model gradually gets better!")
    print()
    print("KEY HYPERPARAMETERS:")
    print("  - Learning rate: How big each learning step is (0.001-0.01)")
    print("  - Batch size: How many examples per update (16-128)")
    print("  - Epochs: How many times to go through all data (3-10)")
    print()
    print("ANALOGY:")
    print("  Training a model = Teaching a student")
    print("  - Forward pass = Student attempts problems")
    print("  - Loss = Test score (lower is better)")
    print("  - Gradients = Understanding mistakes")
    print("  - Optimizer = Study strategy")
    print("  - Epochs = Semesters of learning")
    print()


if __name__ == "__main__":
    main()
