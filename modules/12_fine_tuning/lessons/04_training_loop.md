# Lesson 04: The Fine-Tuning Training Loop

## Learning Objectives

By the end of this lesson, you will be able to:
1. Describe how the fine-tuning loop differs from training from scratch
2. Explain what a "batch" is and why we use batches
3. Implement early stopping to prevent overfitting
4. Read a training progress log and identify overfitting
5. List the key hyperparameters for fine-tuning and their typical values

---

## GLOSSARY

```
Training Loop:
  The repeated cycle: load batch -> forward pass -> compute loss ->
  backward pass -> update weights -> repeat.
  From Module 05/06 you know this. Fine-tuning uses the SAME loop,
  just with a pre-trained starting point and very small learning rate.

Batch:
  A small group of examples processed together.
  Instead of updating weights after EVERY example (slow, noisy),
  we average the gradient over a batch of 8-32 examples.
  C# analogy: like processing a chunk of items in a for loop, not one at a time.

Batch Size:
  How many examples in one batch. Typical fine-tuning: 4-32.
  Larger batch = more stable gradients, needs more memory.
  Small GPU: use batch size 4. Large GPU: 16 or 32.

Gradient Accumulation:
  Trick to simulate large batches on small GPUs.
  Instead of batch_size=32, use batch_size=4 and accumulate gradients
  for 8 steps before updating weights. Effective batch = 4 x 8 = 32.

Learning Rate:
  Fine-tuning uses SMALL learning rates: 1e-4 to 2e-5.
  Full fine-tuning: 2e-5 is common.
  LoRA: 1e-4 to 3e-4 (can be larger because fewer params to update).

Scheduler (Learning Rate Scheduler):
  Gradually changes the learning rate during training.
  Common: cosine decay (starts high, decreases smoothly).
  Why: high LR at start for fast learning, low LR at end for fine details.

Warmup Steps:
  The first N training steps where LR increases from 0 to target value.
  Prevents unstable updates at the very start of training.
  Typical: 50-200 warmup steps.

Early Stopping:
  Automatically stop training when validation loss stops improving.
  Prevents overfitting. Best practice for fine-tuning.
  Patience: how many epochs to wait before stopping. Typical: 2-3.

Checkpoint:
  A saved copy of model weights at a specific training step.
  Best practice: save checkpoint when validation loss is lowest.
  Load best checkpoint after training (not the last checkpoint -- it may be overfit).

Loss Function:
  How we measure the model's error.
  For language models: cross-entropy loss.
  For classification: same cross-entropy but only on the label tokens.

Cross-Entropy Loss:
  Measures how surprised the model is by the correct answer.
  Low loss = model was confident and correct.
  High loss = model was either wrong or uncertain.
  Formula: -log(probability of correct token)
```

---

## Part 1: Fine-Tuning Loop vs Training from Scratch

From Module 06, you know the training loop:

```
TRAINING FROM SCRATCH:
  1. Random weights
  2. Train for 10-100 EPOCHS
  3. Model learns everything: language + your task
  4. Needs: huge dataset (billions of tokens), weeks of compute

FINE-TUNING:
  1. Pre-trained weights (already knows language!)
  2. Train for 1-5 EPOCHS (rarely more)
  3. Model learns: your specific task behavior
  4. Needs: small dataset (hundreds to thousands), hours of compute
```

The loop code is IDENTICAL -- only the starting point and duration differ.

---

## Part 2: The Fine-Tuning Loop Step by Step

```python
import numpy as np

def fine_tuning_loop(
    model,              # Pre-trained model (weights already loaded)
    train_data: list,   # Training examples
    val_data: list,     # Validation examples
    epochs: int = 3,    # How many full passes through training data
    batch_size: int = 8, # Examples per batch
    learning_rate: float = 1e-4,  # Small LR for fine-tuning
    patience: int = 2,  # Early stopping: how many epochs to wait
):
    """
    The core fine-tuning loop.
    Same structure as Module 06 training, but:
    - Starts from pre-trained weights (model already good at language)
    - Very small learning rate (gentle nudges, not big jumps)
    - Few epochs (1-5, not 100s)
    - Uses early stopping (stop when val loss stops improving)
    """

    best_val_loss    = float('inf')   # Track best validation loss
    epochs_no_improve = 0             # Counter for early stopping

    for epoch in range(1, epochs + 1):
        print(f"\n=== EPOCH {epoch}/{epochs} ===")

        # -------- TRAINING PHASE --------
        model.train()                  # Set model to training mode (enables gradients)
        train_loss_total = 0.0         # Accumulate loss across all batches
        num_batches = 0                # Count batches processed

        # Process data in batches (not one example at a time)
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]   # Grab a batch of examples

            # Step 1: Forward pass -- model makes predictions
            predictions = model.forward(batch)

            # Step 2: Compute loss -- how wrong was the model?
            loss = compute_cross_entropy_loss(predictions, batch)

            # Step 3: Backward pass -- compute gradients
            # "How should each weight change to reduce this loss?"
            gradients = compute_gradients(loss, model)

            # Step 4: Update weights -- apply gradients
            # Only update LoRA parameters if using LoRA!
            for param, grad in zip(model.lora_params, gradients):
                param -= learning_rate * grad      # Gradient descent step

            train_loss_total += loss               # Accumulate
            num_batches += 1

            # Print progress every 10 batches
            if num_batches % 10 == 0:
                avg_loss = train_loss_total / num_batches
                print(f"  Batch {num_batches}: train_loss = {avg_loss:.4f}")

        avg_train_loss = train_loss_total / num_batches
        print(f"  Epoch {epoch} avg train loss: {avg_train_loss:.4f}")

        # -------- VALIDATION PHASE --------
        model.eval()                   # Set model to eval mode (disable gradients)
        val_loss_total = 0.0

        for i in range(0, len(val_data), batch_size):
            batch = val_data[i : i + batch_size]
            with no_grad():            # Do NOT update weights during validation
                predictions = model.forward(batch)
                loss = compute_cross_entropy_loss(predictions, batch)
            val_loss_total += loss

        avg_val_loss = val_loss_total / (len(val_data) / batch_size)
        print(f"  Epoch {epoch} avg val loss:   {avg_val_loss:.4f}")

        # -------- EARLY STOPPING CHECK --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model.save_checkpoint("best_model.pt")   # Save best weights
            print(f"  New best val loss: {best_val_loss:.4f} -- checkpoint saved!")
        else:
            epochs_no_improve += 1
            print(f"  No improvement. Patience: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print(f"\nEARLY STOPPING: val loss did not improve for {patience} epochs.")
                break    # Stop training

    # Load the best checkpoint (not the last epoch -- that may be overfit)
    model.load_checkpoint("best_model.pt")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model
```

---

## Part 3: Reading Training Logs

Learn to read training output to diagnose problems:

### Good Training (Loss Decreasing Smoothly)
```
Epoch 1: train_loss=2.41, val_loss=2.38  <- val < train is fine
Epoch 2: train_loss=1.87, val_loss=1.91  <- small gap, both decreasing
Epoch 3: train_loss=1.52, val_loss=1.58  <- healthy training!
Epoch 4: train_loss=1.31, val_loss=1.40  <- save this as best model
Epoch 5: train_loss=1.18, val_loss=1.42  <- val stopped improving -> early stop
```

### Overfitting (Train Good, Val Gets Worse)
```
Epoch 1: train_loss=2.41, val_loss=2.38
Epoch 2: train_loss=1.87, val_loss=1.91
Epoch 3: train_loss=1.52, val_loss=1.58  <- still ok
Epoch 4: train_loss=1.10, val_loss=1.75  <- val RISING = overfitting started
Epoch 5: train_loss=0.82, val_loss=2.10  <- definitely overfitting
Epoch 6: train_loss=0.65, val_loss=2.58  <- early stop triggers

Action: Use checkpoint from Epoch 3 (best val_loss).
```

### Learning Rate Too High
```
Epoch 1: train_loss=2.41, val_loss=2.38
Epoch 2: train_loss=1.23, val_loss=1.45  <- very fast drop (suspicious)
Epoch 3: train_loss=0.43, val_loss=3.21  <- HUGE val spike = LR too high
Epoch 4: train_loss=0.21, val_loss=5.78  <- catastrophic forgetting

Action: Reduce learning_rate by 10x and restart.
```

### Not Learning (Loss Not Decreasing)
```
Epoch 1: train_loss=2.41, val_loss=2.38
Epoch 2: train_loss=2.39, val_loss=2.40  <- barely changed
Epoch 3: train_loss=2.37, val_loss=2.38  <- still not learning!

Action: Increase learning_rate by 10x, or check dataset format.
```

---

## Part 4: Key Hyperparameters

```
HYPERPARAMETER      TYPICAL VALUE       NOTES
---------------------------------------------------------------------------
learning_rate       1e-4 to 2e-5        Lower for full FT, higher for LoRA
batch_size          4, 8, 16, 32        Limited by GPU memory
epochs              1 to 5              Rarely more than 5 for fine-tuning
warmup_steps        50 to 200           ~5-10% of total training steps
lr_scheduler        cosine decay        Most common for fine-tuning
weight_decay        0.01                L2 regularization (prevents overfitting)
gradient_clip       1.0                 Prevents exploding gradients
early_stopping      patience=2          Stop after 2 epochs without improvement
lora_rank           8                   Default. 4 for simple tasks, 16 for complex
lora_alpha          16 (usually 2*rank) Scaling factor
lora_dropout        0.05 to 0.1         Small dropout on LoRA layers
```

### The Learning Rate Schedule (Cosine Decay with Warmup)

```
LR over time (with warmup_steps=100, total_steps=1000):

0.0001 |          /\
       |         /  \
       |        /    \
       |       /      \
       |      /        \--------
0.0000 |----/                   ----
       |________________________
       0   100   500   800  1000
       warmup    peak   cosine decay

Warmup (0 to 100): LR rises from 0 to max_lr
Peak (100):        LR is at max_lr
Cosine (100-1000): LR smoothly decreases to ~0

WHY WARMUP?
  At the start of fine-tuning, gradients are noisy.
  A high LR with noisy gradients = catastrophic weight updates.
  Warmup gives the model a "gentle start."
```

---

## Part 5: Gradient Accumulation (Simulating Large Batches)

```python
# Problem: your GPU only fits batch_size=4
# But you want the stability of batch_size=32
# Solution: accumulate gradients for 8 steps before updating

accumulation_steps = 8   # Update weights every 8 batches
effective_batch_size = batch_size * accumulation_steps  # 4 * 8 = 32

optimizer.zero_grad()    # Reset gradients at start

for step, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps   # Divide by steps to normalize
    loss.backward()                             # Accumulate gradients

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()                        # Update weights
        optimizer.zero_grad()                   # Reset for next accumulation
```

C# analogy: like batching database writes. Instead of inserting 1 row at a time,
collect 32 rows and insert them all at once. Same result, much more efficient.

---

## Part 6: Loss Functions for Fine-Tuning

```
CLASSIFICATION (predict a label):
  Cross-entropy between model output probabilities and true label.
  PyTorch: nn.CrossEntropyLoss()
  Example: input="App crashes" -> model outputs probabilities for [BUG, FEATURE, ...]
           Loss = -log(probability of the correct label "BUG")

TEXT GENERATION (next-token prediction):
  Cross-entropy averaged over all tokens in the output.
  Same loss as pre-training, just on your small dataset.
  Example: input="Summarize: long text..." -> target="Short summary."
           Loss = average(-log(P(each target token)))

IMPORTANT FOR INSTRUCTION TUNING:
  Only compute loss on OUTPUT tokens, not on the instruction/input tokens.
  The model should learn to predict the response, not memorize the instruction.

  "### Instruction: Classify this    ### Response: BUG"
   <--- ignore these tokens --->     <-- compute loss only here -->
```

---

## Key Takeaways

1. Fine-tuning loop = same as Module 06, but starts from pre-trained weights + small LR.

2. Small learning rate (1e-4 to 2e-5) prevents catastrophic forgetting.

3. Few epochs (1-5). More often leads to overfitting, not better performance.

4. Early stopping: save best checkpoint when val loss is lowest. Load that at the end.

5. Read training logs: train loss down + val loss up = overfitting. Both flat = LR too small.

6. Gradient accumulation lets you simulate large batches on a small GPU.

7. For instruction tuning: only compute loss on response tokens, not the instruction.

---

## Next

Lesson 05: Evaluation and Inference
  - How to measure if fine-tuning actually helped?
  - What metrics to use for different tasks?
  - How to run inference on the fine-tuned model?
  - How to compare base model vs fine-tuned model?
