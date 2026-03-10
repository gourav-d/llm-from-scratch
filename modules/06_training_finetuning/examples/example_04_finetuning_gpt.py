"""
Lesson 4 Example: Fine-Tuning a Pre-trained GPT Model

This example shows you how to fine-tune a pre-trained model for a specific task.

Fine-tuning is like taking a college graduate (pre-trained model) and giving them
specialized training for a specific job:
- They already know general knowledge (pre-training)
- You just teach them your specific domain (fine-tuning)
- Much faster and cheaper than training from scratch!

Analogy:
- Training from scratch = Educating someone from kindergarten to PhD (years, expensive)
- Fine-tuning = Taking a PhD graduate and teaching them your company's procedures (days, cheap)
"""

import numpy as np
from typing import List, Dict, Tuple
import time

# =============================================================================
# PART 1: PRE-TRAINED MODEL
# Simulates a model that's already been trained on lots of data
# =============================================================================

class PretrainedGPT:
    """
    A GPT model that's already been pre-trained on general text.

    In real use, this would be:
    - GPT-2 (124M-1.5B parameters)
    - GPT-3 (175B parameters)
    - LLaMA, etc.

    We simulate this with a model that has "learned" weights.
    """

    def __init__(self, vocab_size: int, embed_dim: int, pretrained: bool = True):
        """
        Initialize model.

        Args:
            vocab_size: size of vocabulary
            embed_dim: embedding dimension
            pretrained: if True, use "pretrained" weights (simulated)
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        if pretrained:
            # Simulate pre-trained weights
            # In reality, these would be loaded from a checkpoint
            # (like "gpt2-small.pth" or from Hugging Face)
            print("Loading pre-trained weights...")
            self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.02
            self.output_weights = np.random.randn(embed_dim, vocab_size) * 0.02

            # Mark as pre-trained (for demonstration)
            self.is_pretrained = True
            print(f"✓ Pre-trained model loaded ({self.count_parameters():,} parameters)")
        else:
            # Random initialization (training from scratch)
            print("Initializing random weights (training from scratch)...")
            self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.01
            self.output_weights = np.random.randn(embed_dim, vocab_size) * 0.01
            self.is_pretrained = False

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Forward pass through the model."""
        # Embeddings
        x = self.embeddings[token_ids]  # (batch, seq_len, embed_dim)

        # Output projection
        logits = x @ self.output_weights  # (batch, seq_len, vocab_size)

        return logits

    def count_parameters(self) -> int:
        """Count total parameters."""
        return self.embeddings.size + self.output_weights.size

    def save_checkpoint(self, path: str):
        """
        Save model weights to file.

        In real implementations, you'd use torch.save() or similar.
        """
        print(f"Saving model to {path}...")
        # In real code: np.savez(path, embeddings=self.embeddings, output_weights=self.output_weights)
        print("✓ Model saved")

    def load_checkpoint(self, path: str):
        """
        Load model weights from file.

        In real implementations, you'd use torch.load() or similar.
        """
        print(f"Loading model from {path}...")
        # In real code: data = np.load(path); self.embeddings = data['embeddings']; ...
        print("✓ Model loaded")


# =============================================================================
# PART 2: FINE-TUNING CONFIGURATION
# Settings specific to fine-tuning
# =============================================================================

class FinetuneConfig:
    """
    Configuration for fine-tuning.

    Fine-tuning uses DIFFERENT hyperparameters than training from scratch:
    - Lower learning rate (don't destroy pre-trained knowledge!)
    - Fewer epochs (already mostly trained)
    - Smaller batch size (often have less data)
    """

    def __init__(
        self,
        learning_rate: float = 0.0001,  # Much lower than training from scratch (typically 0.001)
        num_epochs: int = 3,             # Fewer epochs (pre-training used 10-100)
        batch_size: int = 16,            # Can be smaller
        warmup_steps: int = 100,         # Gradually increase learning rate at start
        freeze_embeddings: bool = False  # Whether to freeze embedding layer
    ):
        """
        Initialize fine-tuning configuration.

        Args:
            learning_rate: How fast to update weights (lower = safer for fine-tuning)
            num_epochs: How many times to see the data (fewer = prevent overfitting)
            batch_size: Examples per update
            warmup_steps: Steps to gradually increase learning rate
            freeze_embeddings: If True, don't update embedding layer
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.freeze_embeddings = freeze_embeddings

    def __repr__(self):
        return (f"FinetuneConfig(lr={self.learning_rate}, epochs={self.num_epochs}, "
                f"batch_size={self.batch_size}, warmup={self.warmup_steps})")


# =============================================================================
# PART 3: SPECIALIZED DATASET
# Fine-tuning data for a specific task
# =============================================================================

class SpecializedDataset:
    """
    Dataset for a specialized task.

    Examples:
    - Customer support conversations
    - Code snippets in a specific language
    - Medical reports
    - Legal documents

    Fine-tuning typically uses 100x-1000x LESS data than pre-training:
    - Pre-training: 100GB-500GB of text
    - Fine-tuning: 10MB-100MB of text
    """

    def __init__(self, task_data: List[Tuple[List[int], List[int]]], task_name: str):
        """
        Initialize specialized dataset.

        Args:
            task_data: list of (input_tokens, target_tokens) pairs
            task_name: name of the task (for logging)
        """
        self.data = task_data
        self.task_name = task_name

        print(f"\nSpecialized Dataset: {task_name}")
        print(f"  Examples: {len(task_data)}")
        print(f"  Avg input length: {np.mean([len(x[0]) for x in task_data]):.0f} tokens")

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a random batch from the specialized dataset.

        Args:
            batch_size: number of examples

        Returns:
            inputs, targets: batch of training data
        """
        # Randomly sample examples
        indices = np.random.choice(len(self.data), size=min(batch_size, len(self.data)), replace=False)

        # Get sequences (assuming fixed length for simplicity)
        # In real code, you'd handle variable lengths with padding
        max_len = len(self.data[0][0])

        inputs = np.zeros((len(indices), max_len), dtype=np.int32)
        targets = np.zeros((len(indices), max_len), dtype=np.int32)

        for i, idx in enumerate(indices):
            inp, tgt = self.data[idx]
            inputs[i] = inp
            targets[i] = tgt

        return inputs, targets


# =============================================================================
# PART 4: FINE-TUNING TRAINER
# Handles the fine-tuning process with special techniques
# =============================================================================

class FinetuneTrainer:
    """
    Trainer specifically designed for fine-tuning.

    Key differences from training from scratch:
    1. Lower learning rate (preserve pre-trained knowledge)
    2. Learning rate warmup (gradual ramp-up)
    3. Optional layer freezing (freeze some layers)
    4. Careful monitoring (prevent catastrophic forgetting)
    """

    def __init__(self, model: PretrainedGPT, config: FinetuneConfig):
        """
        Initialize fine-tuning trainer.

        Args:
            model: pre-trained model to fine-tune
            config: fine-tuning configuration
        """
        self.model = model
        self.config = config
        self.step = 0

        print(f"\nFine-tuning Trainer initialized")
        print(f"  Config: {config}")
        print(f"  Freeze embeddings: {config.freeze_embeddings}")

    def get_learning_rate(self) -> float:
        """
        Get current learning rate with warmup.

        Learning rate warmup:
        - Start with very small learning rate
        - Gradually increase to target learning rate
        - Prevents destroying pre-trained weights early on

        Think: Like warming up before exercise - start slow, then full intensity.

        Returns:
            current learning rate
        """
        if self.step < self.config.warmup_steps:
            # Linear warmup: gradually increase from 0 to target learning rate
            # Step 0: lr = 0
            # Step warmup_steps: lr = target_lr
            warmup_factor = self.step / self.config.warmup_steps
            lr = self.config.learning_rate * warmup_factor
        else:
            # After warmup, use full learning rate
            lr = self.config.learning_rate

        return lr

    def compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Same as training from scratch.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Get probabilities for correct tokens
        probs_flat = probs.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        correct_probs = probs_flat[np.arange(len(targets_flat)), targets_flat]

        # Cross-entropy
        epsilon = 1e-10
        loss = -np.mean(np.log(correct_probs + epsilon))

        return loss

    def compute_gradients(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        logits: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients (simplified).

        If freeze_embeddings=True, we don't compute gradients for embeddings.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Gradient of loss w.r.t. logits
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        d_logits = probs.copy()
        batch_indices = np.arange(batch_size)[:, None]
        seq_indices = np.arange(seq_len)[None, :]
        d_logits[batch_indices, seq_indices, targets] -= 1
        d_logits = d_logits / batch_size

        # Gradient for output weights
        embeddings_used = self.model.embeddings[inputs]
        d_output_weights = embeddings_used.transpose(0, 2, 1) @ d_logits
        d_output_weights = np.mean(d_output_weights, axis=0)

        # Gradient for embeddings (only if not frozen)
        gradients = {'output_weights': d_output_weights}

        if not self.config.freeze_embeddings:
            d_embeddings_full = d_logits @ self.model.output_weights.T
            d_embeddings = np.zeros_like(self.model.embeddings)

            for i in range(batch_size):
                for j in range(seq_len):
                    token_id = inputs[i, j]
                    d_embeddings[token_id] += d_embeddings_full[i, j]

            d_embeddings = d_embeddings / batch_size
            gradients['embeddings'] = d_embeddings

        return gradients

    def update_weights(self, gradients: Dict[str, np.ndarray]):
        """
        Update model weights using gradients.

        Uses current learning rate (with warmup).
        """
        lr = self.get_learning_rate()

        # Update output weights
        self.model.output_weights -= lr * gradients['output_weights']

        # Update embeddings (if not frozen)
        if not self.config.freeze_embeddings and 'embeddings' in gradients:
            self.model.embeddings -= lr * gradients['embeddings']

    def finetune(self, dataset: SpecializedDataset):
        """
        Fine-tune the model on specialized dataset.

        This is the main fine-tuning loop!

        Args:
            dataset: specialized dataset for fine-tuning
        """
        print("\n" + "=" * 80)
        print(f"FINE-TUNING START: {dataset.task_name}")
        print("=" * 80)
        print(f"Pre-trained: {self.model.is_pretrained}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Warmup steps: {self.config.warmup_steps}")
        print()

        start_time = time.time()
        total_steps = (len(dataset.data) // self.config.batch_size) * self.config.num_epochs

        for epoch in range(self.config.num_epochs):
            print(f"\nEPOCH {epoch + 1}/{self.config.num_epochs}")
            print("-" * 80)

            epoch_loss = 0.0
            num_batches = len(dataset.data) // self.config.batch_size

            for batch_idx in range(num_batches):
                self.step += 1

                # Get batch
                inputs, targets = dataset.get_batch(self.config.batch_size)

                # Forward pass
                logits = self.model.forward(inputs)

                # Compute loss
                loss = self.compute_loss(logits, targets)
                epoch_loss += loss

                # Backward pass
                gradients = self.compute_gradients(inputs, targets, logits)

                # Update weights (with warmup learning rate)
                self.update_weights(gradients)

                # Log progress
                if self.step % 10 == 0:
                    current_lr = self.get_learning_rate()
                    print(f"  Step {self.step}/{total_steps} | "
                          f"Loss: {loss:.4f} | "
                          f"LR: {current_lr:.6f}")

            avg_loss = epoch_loss / num_batches
            print(f"\nEpoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f}")

        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print("FINE-TUNING COMPLETE!")
        print("=" * 80)
        print(f"Total time: {elapsed:.1f}s")
        print(f"Final loss: {loss:.4f}")
        print()


# =============================================================================
# PART 5: DEMONSTRATION
# Compare training from scratch vs fine-tuning
# =============================================================================

def create_task_dataset(task_type: str, num_examples: int = 100, seq_len: int = 32) -> SpecializedDataset:
    """
    Create a synthetic specialized dataset.

    In real use, this would be actual task-specific data:
    - Customer support: support tickets + responses
    - Code: code snippets + explanations
    - Medical: medical reports + diagnoses

    Args:
        task_type: type of task
        num_examples: number of training examples
        seq_len: sequence length

    Returns:
        SpecializedDataset
    """
    vocab_size = 100
    task_data = []

    for _ in range(num_examples):
        # Create random input/target pairs
        # In real use, these would be actual text sequences
        inputs = np.random.randint(0, vocab_size, size=seq_len)
        targets = np.random.randint(0, vocab_size, size=seq_len)
        task_data.append((inputs.tolist(), targets.tolist()))

    return SpecializedDataset(task_data, task_type)


def main():
    """
    Demonstrate fine-tuning vs training from scratch.
    """
    print("=" * 80)
    print("Fine-Tuning Pre-trained GPT vs Training from Scratch")
    print("=" * 80)
    print()

    # Model settings
    vocab_size = 100
    embed_dim = 64

    # =========================================================================
    # SCENARIO 1: Training from Scratch
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 1: Training from Scratch")
    print("=" * 80)
    print("Starting with random weights, no prior knowledge")
    print()

    # Create model from scratch
    model_scratch = PretrainedGPT(vocab_size, embed_dim, pretrained=False)

    # Create task dataset
    task_dataset = create_task_dataset("Customer Support", num_examples=100, seq_len=32)

    # Training config (higher learning rate, more epochs)
    config_scratch = FinetuneConfig(
        learning_rate=0.01,      # Higher learning rate (starting from scratch)
        num_epochs=5,            # More epochs needed
        batch_size=16,
        warmup_steps=50
    )

    # Train
    trainer_scratch = FinetuneTrainer(model_scratch, config_scratch)
    trainer_scratch.finetune(task_dataset)

    # =========================================================================
    # SCENARIO 2: Fine-Tuning Pre-trained Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 2: Fine-Tuning Pre-trained Model")
    print("=" * 80)
    print("Starting with pre-trained weights, already has general knowledge")
    print()

    # Load pre-trained model
    model_pretrained = PretrainedGPT(vocab_size, embed_dim, pretrained=True)

    # Fine-tuning config (lower learning rate, fewer epochs)
    config_finetune = FinetuneConfig(
        learning_rate=0.0001,    # Much lower learning rate (preserve pre-trained knowledge)
        num_epochs=3,            # Fewer epochs needed
        batch_size=16,
        warmup_steps=50,
        freeze_embeddings=False  # Update all layers
    )

    # Fine-tune
    trainer_finetune = FinetuneTrainer(model_pretrained, config_finetune)
    trainer_finetune.finetune(task_dataset)

    # =========================================================================
    # SCENARIO 3: Fine-Tuning with Frozen Embeddings
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCENARIO 3: Fine-Tuning with Frozen Embeddings")
    print("=" * 80)
    print("Freeze embedding layer, only update output layer")
    print("Useful when: limited data, want to preserve word meanings")
    print()

    # Load another pre-trained model
    model_frozen = PretrainedGPT(vocab_size, embed_dim, pretrained=True)

    # Fine-tuning config with frozen embeddings
    config_frozen = FinetuneConfig(
        learning_rate=0.0001,
        num_epochs=3,
        batch_size=16,
        warmup_steps=50,
        freeze_embeddings=True  # Freeze embeddings!
    )

    # Fine-tune
    trainer_frozen = FinetuneTrainer(model_frozen, config_frozen)
    trainer_frozen.finetune(task_dataset)

    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print("TRAINING FROM SCRATCH:")
    print("  Time: 5 epochs × slower convergence")
    print("  Data needed: Millions of examples")
    print("  Learning rate: 0.01 (high)")
    print("  Cost: $$$$ (very expensive)")
    print("  Use when: Building entirely new model")
    print()
    print("FINE-TUNING (ALL LAYERS):")
    print("  Time: 3 epochs × faster convergence")
    print("  Data needed: Hundreds-thousands of examples")
    print("  Learning rate: 0.0001 (low)")
    print("  Cost: $ (cheap)")
    print("  Use when: Adapting to specific domain")
    print()
    print("FINE-TUNING (FROZEN EMBEDDINGS):")
    print("  Time: 3 epochs × fastest")
    print("  Data needed: Dozens-hundreds of examples")
    print("  Learning rate: 0.0001 (low)")
    print("  Cost: $ (very cheap)")
    print("  Use when: Very limited data, preserve word meanings")
    print()
    print("REAL-WORLD EXAMPLE:")
    print("  Pre-training GPT-3: $12,000,000, 570GB data, weeks")
    print("  Fine-tuning GPT-3: $100, 10MB data, hours")
    print("  → Fine-tuning is 100,000× cheaper!")
    print()
    print("RECOMMENDATION:")
    print("  → Always start with a pre-trained model and fine-tune")
    print("  → Only train from scratch if no suitable pre-trained model exists")
    print()
    print("=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print()
    print("1. LOWER LEARNING RATE:")
    print("   Fine-tuning uses 10-100× lower learning rate than training")
    print("   Reason: Preserve pre-trained knowledge, make small adjustments")
    print()
    print("2. LESS DATA:")
    print("   Fine-tuning needs 100-1000× less data")
    print("   Reason: Model already knows language, just learning specifics")
    print()
    print("3. FASTER:")
    print("   Fine-tuning is 10-100× faster")
    print("   Reason: Fewer epochs, smaller updates")
    print()
    print("4. WARMUP:")
    print("   Gradually increase learning rate at start")
    print("   Reason: Prevent destroying pre-trained weights immediately")
    print()
    print("5. LAYER FREEZING:")
    print("   Can freeze some layers (embeddings, early layers)")
    print("   Reason: When data is very limited, preserve more knowledge")
    print()
    print("6. CATASTROPHIC FORGETTING:")
    print("   Risk: Model forgets pre-trained knowledge")
    print("   Solution: Low learning rate, fewer epochs, more data mixing")
    print()


if __name__ == "__main__":
    main()
