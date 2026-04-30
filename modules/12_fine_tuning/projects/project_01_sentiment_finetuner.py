# =============================================================================
# MODULE 12 - PROJECT 01: SENTIMENT FINE-TUNER
# =============================================================================
# Title   : Sentiment Analysis Fine-Tuner from Scratch
# Goal    : Build a complete pipeline to fine-tune a tiny neural network
#           classifier that labels text as POSITIVE, NEGATIVE, or NEUTRAL
# What you
# will    : 1. Create and prepare a labeled dataset
# build   : 2. Build a 2-layer neural network classifier using pure NumPy
#           3. Train it with batches, track loss, and use early stopping
#           4. Evaluate with accuracy, F1 score, and a confusion matrix
#           5. Compare base (untrained) vs fine-tuned predictions
# How to  :   python project_01_sentiment_finetuner.py
# run     :
# Dependencies: Python 3.10+ and NumPy only. No PyTorch, no APIs.
# =============================================================================

# =============================================================================
# GLOSSARY
# (Read this before the code -- every term used in this file is defined here)
# =============================================================================
#
# Sentiment Analysis
#   The task of deciding whether a piece of text is POSITIVE, NEGATIVE,
#   or NEUTRAL. Like a "mood detector" for text.
#   C# analogy: imagine a method that returns an enum {Positive, Negative, Neutral}.
#
# Fine-Tuning
#   Starting with an already-initialized model and continuing to train it
#   on a specific task or dataset so it gets better at that task.
#   Think of it as: you hire a general developer (base model), then give them
#   3 weeks of domain-specific training so they specialise in your product.
#
# Weight Initialization
#   Setting the starting values for the neural network's internal numbers
#   (weights) before training begins. Bad initialization = slow/no learning.
#   C# analogy: setting default values on a class before the constructor runs.
#
# Forward Pass
#   Feeding an input through the network layer by layer to produce a prediction.
#   Data flows FORWARD: input -> layer 1 -> layer 2 -> output.
#   C# analogy: calling a chain of methods where each feeds into the next.
#
# Loss
#   A single number that measures HOW WRONG the model's predictions are.
#   Lower loss = better. We want to minimise this number during training.
#   C# analogy: think of it as the error score in a unit test assertion.
#
# Gradient
#   The direction and steepness of the "slope" of the loss. It tells us
#   which way to nudge each weight to reduce the loss.
#   C# analogy: if loss is a hill, gradient tells you which way is downhill.
#
# Epoch
#   One complete pass through the entire training dataset.
#   If you have 24 training examples and process them all once = 1 epoch.
#   C# analogy: one full iteration of a foreach over all training items.
#
# Batch
#   A small subset of the training data processed together before updating
#   the weights. Batch size 8 means we process 8 examples at a time.
#   C# analogy: chunk/page in pagination -- process records in pages.
#
# Early Stopping
#   A trick to stop training automatically when the model stops improving
#   on the validation set. Prevents overfitting (memorising training data).
#   C# analogy: a circuit breaker pattern -- stop retrying after N failures.
#
# Confusion Matrix
#   A 3x3 grid showing how many times each class was predicted vs the truth.
#   Row = true label, Column = predicted label. Diagonal = correct predictions.
#   C# analogy: a 2D array[trueClass][predictedClass] of counters.
#
# Bag of Words (BoW)
#   A simple way to represent text: count how many times each vocabulary word
#   appears. Ignores word order. "I love cats" -> {I:1, love:1, cats:1, ...rest:0}
#   C# analogy: a Dictionary<string, int> of word counts, then flattened to int[].
#
# Softmax
#   Converts raw scores (any numbers) into probabilities that sum to 1.0.
#   e.g. [2.0, 1.0, 0.1] -> [0.69, 0.25, 0.06]. Largest score gets highest prob.
#   C# analogy: a normalisation method that returns a probability distribution.
#
# Cross-Entropy Loss
#   Measures the difference between predicted probabilities and the true label.
#   Formula: -log(predicted_probability_of_correct_class).
#   Low loss = confident correct prediction. High loss = wrong or uncertain.
#
# ReLU (Rectified Linear Unit)
#   An activation function: output = max(0, x). Negative values become 0.
#   Introduces non-linearity so the network can learn complex patterns.
#   C# analogy: Math.Max(0, x) applied element-by-element to an array.
#
# Validation Set
#   A small portion of data NOT used for training, used to check performance
#   during training. Helps detect overfitting early.
#   C# analogy: a separate set of integration tests run after each build.
#
# F1 Score
#   A metric combining Precision (are positives really positive?) and
#   Recall (did we find all the positives?). Ranges 0.0 to 1.0. Higher = better.
#   Formula: 2 * (Precision * Recall) / (Precision + Recall)
#
# =============================================================================


# --- IMPORTS -----------------------------------------------------------------
import numpy as np          # NumPy: numerical computing library (like Math utilities in C#)
import random               # random: Python's built-in randomness module
import math                 # math: Python's built-in math functions (log, exp, etc.)

# Seed both random number generators so results are reproducible
# (same as setting Random seed in C# for unit tests)
np.random.seed(42)          # Set NumPy's random seed to 42 for reproducibility
random.seed(42)             # Set Python's random seed to 42 for reproducibility


# =============================================================================
# ASCII DIAGRAM: TRAINING PIPELINE
# =============================================================================
#
#   RAW TEXT SENTENCES
#          |
#          v
#   +----------------+
#   |  Tokenise /    |   Split each sentence into words, build a vocabulary
#   |  Build Vocab   |   (like a Dictionary<string,int> mapping word -> index)
#   +----------------+
#          |
#          v
#   +----------------+
#   |  Bag-of-Words  |   Convert each sentence to a fixed-length count vector
#   |  Vectoriser    |   vector length = vocabulary size
#   +----------------+
#          |
#          v
#   +------------------------------+
#   |  Train / Val / Test Split    |   80% train, 10% val, 10% test
#   +------------------------------+
#          |
#          v
#   +----------------------------------------------------+
#   |  SimpleClassifier (2-layer neural net)             |
#   |                                                    |
#   |  [vocab_size]  ->  [16 hidden]  ->  [3 output]     |
#   |    (input)        (ReLU)           (Softmax)       |
#   |                                                    |
#   |  Weights W1, b1  and  W2, b2  are learned here    |
#   +----------------------------------------------------+
#          |
#          v
#   +------------------+
#   |  Training Loop   |   For each epoch:
#   |  (Fine-Tuning)   |     1. Forward pass  -> get predictions
#   |                  |     2. Compute loss  -> how wrong are we?
#   |                  |     3. Backward pass -> compute gradients
#   |                  |     4. Update weights-> nudge toward lower loss
#   |                  |     5. Check val loss-> early stopping?
#   +------------------+
#          |
#          v
#   +------------------+
#   |  Evaluation      |   Accuracy, F1, Confusion Matrix
#   +------------------+
#          |
#          v
#   +------------------+
#   |  Comparison      |   Base model vs Fine-tuned model on test set
#   +------------------+
#
# =============================================================================


# =============================================================================
# PART 1: DATASET
# =============================================================================

print("=" * 70)       # Print a separator line of 70 equal signs
print("PART 1: DATASET PREPARATION")   # Section header
print("=" * 70)       # Print another separator

# --- 1A. Raw labeled data ----------------------------------------------------
# Each entry is a tuple: (sentence_string, label_string)
# Labels are: "positive", "negative", "neutral"
# C# analogy: List<(string Sentence, string Label)>

raw_data = [
    # --- POSITIVE examples (10) -------------------------------------------
    ("I absolutely love this product it is amazing",        "positive"),
    ("The service was fantastic and the staff were kind",   "positive"),
    ("This is the best experience I have ever had",         "positive"),
    ("Highly recommend to anyone looking for quality",      "positive"),
    ("Great value for money and very fast delivery",        "positive"),
    ("I am so happy with my purchase it exceeded expectations", "positive"),
    ("Wonderful customer support solved my problem quickly","positive"),
    ("The quality is outstanding I will buy again",         "positive"),
    ("Superb performance and easy to set up",               "positive"),
    ("Really pleased with the results definitely worth it", "positive"),

    # --- NEGATIVE examples (10) -------------------------------------------
    ("I hate this product it broke after one day",          "negative"),
    ("Terrible service the staff were rude and unhelpful",  "negative"),
    ("Worst purchase I have ever made total waste of money","negative"),
    ("Do not buy this the quality is awful",                "negative"),
    ("Very disappointed with the delivery it was two weeks late","negative"),
    ("The product stopped working after a week so frustrating","negative"),
    ("Customer support was useless they never replied",     "negative"),
    ("Poor quality falls apart immediately not recommended","negative"),
    ("Completely broken on arrival and impossible to return","negative"),
    ("This is a scam product does not work at all",         "negative"),

    # --- NEUTRAL examples (10) -------------------------------------------
    ("The product arrived on time nothing special",         "neutral"),
    ("It works as described neither good nor bad",          "neutral"),
    ("The item is okay but nothing particularly impressive","neutral"),
    ("Delivery was average expected better packaging",      "neutral"),
    ("The quality is acceptable for the price paid",        "neutral"),
    ("It does the job but there are better options",        "neutral"),
    ("Customer service responded but did not fully help",   "neutral"),
    ("The product is fine it meets basic requirements",     "neutral"),
    ("Average experience would consider alternatives next time","neutral"),
    ("Not bad but not great either it is just okay",        "neutral"),
]

# --- 1B. Convert labels to integers ------------------------------------------
# Neural networks work with numbers, not strings.
# Map: "positive" -> 0, "negative" -> 1, "neutral" -> 2
# C# analogy: Enum with int values -- (int)SentimentLabel.Positive = 0

label_to_idx = {                # Create a dictionary mapping label string to integer
    "positive": 0,              # POSITIVE class = index 0
    "negative": 1,              # NEGATIVE class = index 1
    "neutral":  2,              # NEUTRAL class = index 2
}

idx_to_label = {v: k for k, v in label_to_idx.items()}  # Reverse mapping: int -> string label
# idx_to_label = {0:"positive", 1:"negative", 2:"neutral"}
# C# analogy: reverse a Dictionary with .ToDictionary(kv => kv.Value, kv => kv.Key)

CLASS_NAMES = ["POSITIVE", "NEGATIVE", "NEUTRAL"]  # Human-readable class names for printing
NUM_CLASSES  = 3              # Total number of output classes

# --- 1C. Build vocabulary (Bag-of-Words) -------------------------------------
# Step 1: collect all unique words across all sentences
# C# analogy: HashSet<string> built from all words in all sentences

vocab_set = set()             # Empty set to collect unique words (sets have no duplicates)

for sentence, _ in raw_data:                   # Loop over each (sentence, label) pair
    words = sentence.lower().split()           # Lowercase the sentence and split on spaces
    for word in words:                         # Loop over each word in this sentence
        vocab_set.add(word)                    # Add word to the set (duplicates ignored)

# Sort vocabulary for deterministic ordering (same order every run)
vocab_list = sorted(vocab_set)                 # Convert set to a sorted list

# Build word -> index mapping (so we know which position in the vector each word occupies)
# C# analogy: Dictionary<string, int> where index is the position in int[] vector
word_to_idx = {word: idx for idx, word in enumerate(vocab_list)}  # word -> its position index

VOCAB_SIZE = len(vocab_list)                   # Total number of unique words in our vocabulary
print(f"Vocabulary size: {VOCAB_SIZE} unique words")  # Show how many words we found


# --- 1D. Bag-of-Words encoder ------------------------------------------------
def encode_sentence(sentence, word_to_idx, vocab_size):
    """
    Convert a sentence string into a fixed-length NumPy count vector.

    Example:
      sentence = "I love cats"
      vocab    = ["cats", "dogs", "I", "love"]
      output   = [1, 0, 1, 1]   (count of each vocab word in sentence)

    C# analogy: returns a float[] of length vocab_size
    """
    vector = np.zeros(vocab_size, dtype=np.float64)  # Create a vector of all zeros, one slot per vocab word
    words  = sentence.lower().split()                # Lowercase and tokenise the sentence by spaces
    for word in words:                               # Loop over each word in the sentence
        if word in word_to_idx:                      # Only count words that are in our vocabulary
            idx = word_to_idx[word]                  # Get the position (index) of this word in the vocab
            vector[idx] += 1.0                       # Increment the count at that position
    return vector                                    # Return the completed count vector


# --- 1E. Encode full dataset -------------------------------------------------
# Build numpy arrays for all sentences (X) and all labels (y)
# X shape: (num_examples, vocab_size)   -- each row is one sentence vector
# y shape: (num_examples,)              -- each element is an integer label (0/1/2)

all_X = np.array(                                    # Stack all sentence vectors into a 2D array
    [encode_sentence(s, word_to_idx, VOCAB_SIZE) for s, _ in raw_data]  # List comprehension: encode each sentence
)

all_y = np.array(                                    # Stack all labels into a 1D array
    [label_to_idx[lbl] for _, lbl in raw_data]       # List comprehension: convert each string label to int
)

TOTAL = len(raw_data)                                # Total number of examples in the dataset
print(f"Total examples    : {TOTAL}")                # Print total count

# --- 1F. Train / Val / Test split (80 / 10 / 10) ----------------------------
# Shuffle dataset so class order is random (prevents ordering bias)
indices = list(range(TOTAL))                         # Create a list [0, 1, 2, ..., 29]
random.shuffle(indices)                              # Shuffle the list in place (random order)

# Compute split boundary indices
n_train = int(0.80 * TOTAL)                          # 80% of total = training set size
n_val   = int(0.10 * TOTAL)                          # 10% of total = validation set size
# Remaining = test set (handles rounding automatically)

train_idx = indices[:n_train]                        # First 80% indices -> training
val_idx   = indices[n_train : n_train + n_val]       # Next  10% indices -> validation
test_idx  = indices[n_train + n_val :]               # Last  10% indices -> test

# Slice the full arrays using those indices
X_train, y_train = all_X[train_idx], all_y[train_idx]  # Training data and labels
X_val,   y_val   = all_X[val_idx],   all_y[val_idx]    # Validation data and labels
X_test,  y_test  = all_X[test_idx],  all_y[test_idx]   # Test data and labels

# Print split sizes
print(f"Training examples : {len(X_train)}")         # Show training set size
print(f"Validation examples: {len(X_val)}")          # Show validation set size
print(f"Test examples     : {len(X_test)}")          # Show test set size

# --- 1G. Class balance stats -------------------------------------------------
print("\nClass balance in training set:")             # Header for class balance output
for cls_idx, cls_name in enumerate(CLASS_NAMES):    # Loop over each class (0=POS, 1=NEG, 2=NEU)
    count = np.sum(y_train == cls_idx)               # Count how many training labels equal this class
    print(f"  {cls_name}: {count} examples")         # Print class name and count


# =============================================================================
# HELPER FUNCTIONS (used in both Part 2 and Part 3)
# =============================================================================

def softmax(z):
    """
    Convert a vector of raw scores into probabilities.
    All output values are between 0.0 and 1.0 and they sum to 1.0.

    Trick: subtract max(z) before exp() to prevent overflow.
    (numpy exp of very large numbers gives infinity -- this stabilises it)

    C# analogy:
      double[] Softmax(double[] z) {
          double max = z.Max();
          double[] exp = z.Select(x => Math.Exp(x - max)).ToArray();
          double sum = exp.Sum();
          return exp.Select(x => x / sum).ToArray();
      }
    """
    z_stable = z - np.max(z, axis=-1, keepdims=True)  # Subtract max along last axis for numerical stability
    exp_z    = np.exp(z_stable)                         # Compute e^(z_i) for every element
    sum_exp  = np.sum(exp_z, axis=-1, keepdims=True)    # Sum all exp values (used as denominator)
    return exp_z / sum_exp                              # Divide each exp by the sum -> probabilities


def relu(z):
    """
    ReLU activation: output = max(0, z)
    Negative values become 0, positive values stay the same.

    C# analogy: z.Select(x => Math.Max(0.0, x)).ToArray()
    """
    return np.maximum(0.0, z)                           # Element-wise max(0, z) for the whole array


def relu_derivative(z):
    """
    Derivative of ReLU: 1 where z > 0, else 0.
    Used in the backward pass to compute gradients through the ReLU layer.

    C# analogy: z.Select(x => x > 0 ? 1.0 : 0.0).ToArray()
    """
    return (z > 0).astype(np.float64)                  # True -> 1.0, False -> 0.0


def cross_entropy_loss(probs, labels):
    """
    Compute average cross-entropy loss for a batch.

    For each example:
      loss_i = -log(probs[i, true_class_i])

    Then average across the batch.

    probs  : 2D array of shape (batch_size, num_classes) -- predicted probabilities
    labels : 1D array of shape (batch_size,)             -- integer true class labels

    C# analogy: examples.Average(i => -Math.Log(probs[i][labels[i]] + 1e-9))
    """
    batch_size = probs.shape[0]                         # Number of examples in this batch
    eps        = 1e-9                                   # Small epsilon to prevent log(0) which is -infinity
    # Select the probability assigned to the correct class for each example
    correct_probs = probs[np.arange(batch_size), labels]  # probs[0,label0], probs[1,label1], ...
    # Compute -log of those probabilities and average them
    loss = -np.mean(np.log(correct_probs + eps))       # Average negative log likelihood = cross-entropy
    return loss                                         # Return scalar loss value


# =============================================================================
# PART 2: BASE MODEL (before fine-tuning)
# =============================================================================

print("\n" + "=" * 70)           # Print separator
print("PART 2: BASE MODEL (RANDOM WEIGHTS -- BEFORE TRAINING)")  # Section header
print("=" * 70)                  # Print separator


class SimpleClassifier:
    """
    A 2-layer neural network for text classification.

    Architecture:
      Input  (vocab_size,)   -> Linear(vocab_size, hidden_size) -> ReLU
      Hidden (hidden_size,)  -> Linear(hidden_size, num_classes) -> Softmax
      Output (num_classes,)  = class probabilities

    C# analogy: a class with two layers (like two matrix-multiply operations)
    that forward() is like calling Predict() on each layer in sequence.
    """

    def __init__(self, vocab_size, hidden_size, num_classes):
        """
        Initialise the classifier with random weights.

        vocab_size  : number of input features (size of our BoW vector)
        hidden_size : number of neurons in the hidden layer
        num_classes : number of output classes (3 for POS/NEG/NEU)

        C# analogy: constructor that sets default field values
        """
        self.vocab_size   = vocab_size    # Store input dimension
        self.hidden_size  = hidden_size   # Store hidden layer size
        self.num_classes  = num_classes   # Store output dimension

        # --- Weight initialisation (Xavier / Glorot scaling) -----------------
        # Multiply by small scale so weights start small (prevents exploding gradients)
        # scale = sqrt(2 / input_size) is a common heuristic
        # C# analogy: initialise double[,] with small random values

        scale_1 = np.sqrt(2.0 / vocab_size)          # Scale factor for layer 1 weights
        scale_2 = np.sqrt(2.0 / hidden_size)          # Scale factor for layer 2 weights

        # W1: weight matrix for layer 1,  shape (vocab_size, hidden_size)
        # Each column connects all inputs to one hidden neuron
        self.W1 = np.random.randn(vocab_size, hidden_size) * scale_1

        # b1: bias vector for layer 1, shape (hidden_size,)
        # Starts at zero (standard practice)
        self.b1 = np.zeros(hidden_size)

        # W2: weight matrix for layer 2, shape (hidden_size, num_classes)
        # Each column connects all hidden neurons to one output class
        self.W2 = np.random.randn(hidden_size, num_classes) * scale_2

        # b2: bias vector for layer 2, shape (num_classes,)
        self.b2 = np.zeros(num_classes)

        # --- Cache for backward pass -----------------------------------------
        # During forward pass we store intermediate values that the backward
        # pass needs to compute gradients. C# analogy: fields holding state
        self.cache = {}              # Empty dict to hold cached values

    def forward(self, X):
        """
        Forward pass: compute predictions for input batch X.

        X     : 2D array of shape (batch_size, vocab_size)
        return: 2D array of shape (batch_size, num_classes) -- probabilities

        C# analogy: double[,] Forward(double[,] X) { ... }
        """
        # Layer 1: linear transformation + ReLU activation
        z1 = X.dot(self.W1) + self.b1         # z1 = X * W1 + b1, shape (batch, hidden)
        a1 = relu(z1)                          # a1 = ReLU(z1), shape (batch, hidden)

        # Layer 2: linear transformation (no activation here -- softmax comes next)
        z2 = a1.dot(self.W2) + self.b2        # z2 = a1 * W2 + b2, shape (batch, num_classes)

        # Apply softmax to get class probabilities
        probs = softmax(z2)                    # probs, shape (batch, num_classes) -- sums to 1.0 per row

        # Cache intermediate values for backward pass
        self.cache["X"]  = X                  # Store original input
        self.cache["z1"] = z1                 # Store pre-activation of layer 1
        self.cache["a1"] = a1                 # Store post-activation of layer 1
        self.cache["z2"] = z2                 # Store pre-softmax scores of layer 2

        return probs                           # Return predicted probabilities

    def backward(self, X, labels, probs, learning_rate):
        """
        Backward pass: compute gradients and update weights.

        X            : input batch,  shape (batch_size, vocab_size)
        labels       : true class indices, shape (batch_size,)
        probs        : predicted probabilities from forward(), shape (batch_size, num_classes)
        learning_rate: how big a step to take when updating weights (float, e.g. 0.01)

        C# analogy: UpdateWeights(double[,] X, int[] labels, double[,] probs, double lr)
        """
        batch_size = X.shape[0]               # Number of examples in this batch

        # --- Gradient of loss w.r.t. z2 (output layer pre-softmax scores) ---
        # For cross-entropy + softmax combined, the gradient simplifies nicely to:
        #   dLoss/dz2 = probs - one_hot(labels)
        # C# analogy: subtract 1.0 from the column corresponding to the true class
        dz2 = probs.copy()                    # Start with a copy of the predicted probabilities
        dz2[np.arange(batch_size), labels] -= 1.0  # Subtract 1 from the true class probability
        dz2 /= batch_size                     # Divide by batch size to average the gradient

        # --- Gradients for layer 2 weights and biases -----------------------
        a1 = self.cache["a1"]                 # Retrieve cached hidden layer activations
        dW2 = a1.T.dot(dz2)                   # dLoss/dW2 = a1^T * dz2, shape (hidden, num_classes)
        db2 = np.sum(dz2, axis=0)             # dLoss/db2 = sum over batch, shape (num_classes,)

        # --- Backpropagate gradient through layer 2 -> layer 1 --------------
        da1 = dz2.dot(self.W2.T)             # Gradient at a1, shape (batch, hidden)
        dz1 = da1 * relu_derivative(self.cache["z1"])  # Apply ReLU derivative (mask zeros)

        # --- Gradients for layer 1 weights and biases -----------------------
        dW1 = X.T.dot(dz1)                   # dLoss/dW1 = X^T * dz1, shape (vocab, hidden)
        db1 = np.sum(dz1, axis=0)             # dLoss/db1 = sum over batch, shape (hidden,)

        # --- Weight update (Gradient Descent) --------------------------------
        # Move weights in the direction OPPOSITE to the gradient
        # (going downhill on the loss surface)
        # new_weight = old_weight - learning_rate * gradient
        # C# analogy: w -= lr * grad;
        self.W2 -= learning_rate * dW2        # Update layer 2 weight matrix
        self.b2 -= learning_rate * db2        # Update layer 2 bias vector
        self.W1 -= learning_rate * dW1        # Update layer 1 weight matrix
        self.b1 -= learning_rate * db1        # Update layer 1 bias vector

    def predict(self, X):
        """
        Predict class indices for input batch X.
        Returns the class with the highest probability for each example.

        C# analogy: int[] Predict(double[,] X) => probs.ArgMax(axis=1)
        """
        probs = self.forward(X)               # Run forward pass to get probabilities
        return np.argmax(probs, axis=1)       # Return the index of max probability per row

    def get_weights(self):
        """
        Return a deep copy of all weights and biases.
        Used to save the best model checkpoint during training.

        C# analogy: a method that returns a struct/clone of all weight arrays
        """
        return {                              # Return dictionary of all weight arrays
            "W1": self.W1.copy(),             # Copy layer 1 weights
            "b1": self.b1.copy(),             # Copy layer 1 biases
            "W2": self.W2.copy(),             # Copy layer 2 weights
            "b2": self.b2.copy(),             # Copy layer 2 biases
        }

    def set_weights(self, weights):
        """
        Restore weights from a saved checkpoint dictionary.
        Used to reload best weights after early stopping.

        C# analogy: a method that re-assigns all field arrays from a struct/dict
        """
        self.W1 = weights["W1"].copy()        # Restore layer 1 weights
        self.b1 = weights["b1"].copy()        # Restore layer 1 biases
        self.W2 = weights["W2"].copy()        # Restore layer 2 weights
        self.b2 = weights["b2"].copy()        # Restore layer 2 biases


# --- Instantiate the base model with random weights --------------------------
HIDDEN_SIZE = 16               # Number of neurons in the hidden layer (hyperparameter)

base_model = SimpleClassifier(          # Create a new classifier instance
    vocab_size   = VOCAB_SIZE,          # Input size = number of vocabulary words
    hidden_size  = HIDDEN_SIZE,         # Hidden layer has 16 neurons
    num_classes  = NUM_CLASSES,         # 3 output classes (POS, NEG, NEU)
)

# Save the initial random weights so we can reset to them later for comparison
initial_weights = base_model.get_weights()   # Deep-copy initial random weights

# --- Show predictions BEFORE training ----------------------------------------
print("\nPredictions BEFORE training (base model with random weights):")
print("Expected: mostly wrong / random because weights are not trained yet\n")

# Pick 5 examples from the full dataset to preview
preview_indices = [0, 10, 20, 5, 15]          # Hand-pick one from each class region
for i in preview_indices:                      # Loop over each preview index
    sentence, true_lbl = raw_data[i]           # Get the sentence and its true label
    x_vec  = encode_sentence(sentence, word_to_idx, VOCAB_SIZE)  # Encode sentence to BoW vector
    x_vec  = x_vec.reshape(1, -1)             # Reshape to (1, vocab_size) for batch dimension
    pred   = base_model.predict(x_vec)[0]     # Get predicted class index (scalar)
    print(f"  Text : {sentence[:50]:<50}")    # Print first 50 chars of sentence, left-aligned
    print(f"  True : {true_lbl:<10}  Predicted: {idx_to_label[pred]}")  # Print true and predicted labels
    print()                                   # Print blank line between examples


# =============================================================================
# PART 3: FINE-TUNING (training loop)
# =============================================================================

print("=" * 70)                  # Print separator
print("PART 3: FINE-TUNING (TRAINING LOOP)")  # Section header
print("=" * 70)                  # Print separator


def train(model, X_train, y_train, X_val, y_val,
          learning_rate=0.05, num_epochs=100, batch_size=8, patience=3):
    """
    Fine-tune the model on the training set.

    model        : a SimpleClassifier instance (will be modified in place)
    X_train      : training input array, shape (n_train, vocab_size)
    y_train      : training labels,  shape (n_train,)
    X_val        : validation inputs, shape (n_val, vocab_size)
    y_val        : validation labels, shape (n_val,)
    learning_rate: step size for gradient descent (e.g. 0.05)
    num_epochs   : maximum number of full passes through training data
    batch_size   : number of examples per gradient update step
    patience     : number of epochs to wait without val improvement before stopping early

    Returns: (train_losses, val_losses) -- lists of loss per epoch for plotting
    """

    train_losses = []             # List to record training loss each epoch
    val_losses   = []             # List to record validation loss each epoch

    best_val_loss  = math.inf    # Track best validation loss so far (start at infinity)
    best_weights   = None        # Will store the best model weights
    patience_count = 0           # Counter: how many epochs since val improved

    n_train = X_train.shape[0]   # Number of training examples

    for epoch in range(1, num_epochs + 1):    # Loop from epoch 1 to num_epochs (inclusive)

        # --- Shuffle training data each epoch --------------------------------
        # Shuffling prevents the model from memorising data order
        perm = np.random.permutation(n_train)  # Random permutation of training indices
        X_shuffled = X_train[perm]             # Reorder training inputs
        y_shuffled = y_train[perm]             # Reorder training labels (same order as inputs)

        epoch_train_loss = 0.0                 # Accumulate loss across batches for this epoch
        num_batches = 0                        # Count how many batches we processed

        # --- Mini-batch loop --------------------------------------------------
        # Process training data in chunks of size batch_size
        # C# analogy: for (int i = 0; i < n; i += batchSize) { ... }
        for start in range(0, n_train, batch_size):  # Step through training data in batch_size steps
            end = min(start + batch_size, n_train)   # End index (clamp so we don't go past end)

            X_batch = X_shuffled[start:end]          # Slice out this batch's inputs
            y_batch = y_shuffled[start:end]          # Slice out this batch's labels

            # Forward pass: compute predictions for this batch
            probs = model.forward(X_batch)           # Get probability predictions, shape (batch, 3)

            # Compute loss for this batch
            batch_loss = cross_entropy_loss(probs, y_batch)  # Scalar loss for this batch

            epoch_train_loss += batch_loss           # Add to epoch total
            num_batches      += 1                    # Count this batch

            # Backward pass: compute gradients and update weights
            model.backward(X_batch, y_batch, probs, learning_rate)  # Update W1, b1, W2, b2

        # Average training loss over all batches in this epoch
        avg_train_loss = epoch_train_loss / num_batches   # Mean train loss for this epoch
        train_losses.append(avg_train_loss)               # Save to history list

        # --- Validation loss (no weight update, just evaluate) ---------------
        val_probs    = model.forward(X_val)               # Forward pass on validation set
        avg_val_loss = cross_entropy_loss(val_probs, y_val)  # Validation loss (no backward)
        val_losses.append(avg_val_loss)                   # Save to history list

        # --- Print progress every epoch --------------------------------------
        print(f"  Epoch {epoch:>3}/{num_epochs} | "        # Epoch number, right-aligned in 3 chars
              f"train_loss={avg_train_loss:.4f} | "        # Training loss with 4 decimal places
              f"val_loss={avg_val_loss:.4f}")               # Validation loss with 4 decimal places

        # --- Early stopping check --------------------------------------------
        if avg_val_loss < best_val_loss:                   # Did validation loss improve?
            best_val_loss  = avg_val_loss                  # Update best val loss
            best_weights   = model.get_weights()           # Save best weights checkpoint
            patience_count = 0                             # Reset patience counter
            print(f"             [Best model saved at epoch {epoch}]")  # Notify user
        else:
            patience_count += 1                            # One more epoch without improvement
            if patience_count >= patience:                 # Have we exceeded patience limit?
                print(f"\n  Early stopping triggered at epoch {epoch} "
                      f"(no val improvement for {patience} epochs)")  # Announce early stop
                break                                      # Exit the epoch loop

    # Restore the best weights found during training
    if best_weights is not None:                           # Make sure we saved something
        model.set_weights(best_weights)                    # Load best checkpoint back into model
        print(f"\n  Best weights restored (best val_loss={best_val_loss:.4f})")

    return train_losses, val_losses                        # Return loss histories for inspection


# --- Create the fine-tuned model (start fresh from same random init) ---------
# Reset to the same initial weights so the comparison is fair
finetuned_model = SimpleClassifier(VOCAB_SIZE, HIDDEN_SIZE, NUM_CLASSES)  # New classifier instance
finetuned_model.set_weights(initial_weights)    # Start from the SAME random weights as base model

print("\nStarting training...\n")               # Announce training is starting

# Run the training loop
train_losses, val_losses = train(               # Call train() and capture loss histories
    model         = finetuned_model,            # The model to train
    X_train       = X_train,                   # Training inputs
    y_train       = y_train,                   # Training labels
    X_val         = X_val,                     # Validation inputs
    y_val         = y_val,                     # Validation labels
    learning_rate = 0.05,                      # Step size for gradient descent
    num_epochs    = 150,                       # Maximum number of epochs
    batch_size    = 8,                         # Process 8 examples per batch
    patience      = 10,                        # Stop if val doesn't improve for 10 epochs
)

# --- Print a simple ASCII loss curve -----------------------------------------
print("\n  Loss curve (train_loss per epoch, ASCII chart):")
print("  (Each '*' = one epoch, position = relative loss level)")

# Normalise train_losses to range 0..20 for ASCII display width
if len(train_losses) > 1:                       # Only draw if we have data
    min_loss = min(train_losses)                # Minimum loss value
    max_loss = max(train_losses)                # Maximum loss value
    loss_range = max(max_loss - min_loss, 1e-9) # Range (avoid division by zero)
    chart_width = 40                            # Width of ASCII chart in characters

    print(f"  High({max_loss:.3f}) |", end="")  # Print Y-axis top label
    print()                                     # Newline

    for i, loss in enumerate(train_losses):     # Loop over each epoch's loss
        # Scale loss to 0..chart_width bar length
        bar_len = int((loss - min_loss) / loss_range * chart_width)  # Length of bar
        print(f"  Ep {i+1:>3} | {'*' * bar_len}")   # Print epoch and bar

    print(f"  Low ({min_loss:.3f}) |")           # Print Y-axis bottom label


# =============================================================================
# PART 4: EVALUATION
# =============================================================================

print("\n" + "=" * 70)           # Print separator
print("PART 4: EVALUATION")      # Section header
print("=" * 70)                  # Print separator


def compute_accuracy(model, X, y):
    """
    Compute the fraction of examples predicted correctly.

    Returns a float between 0.0 (all wrong) and 1.0 (all correct).
    C# analogy: correct / total as a double
    """
    preds   = model.predict(X)               # Get predicted class indices for all examples
    correct = np.sum(preds == y)             # Count how many predictions match true labels
    return correct / len(y)                  # Return fraction correct


def compute_confusion_matrix(model, X, y, num_classes):
    """
    Compute a num_classes x num_classes confusion matrix.

    confusion[i][j] = number of examples with true class i predicted as class j.
    Diagonal elements = correct predictions.
    Off-diagonal = mistakes.

    C# analogy: int[,] matrix = new int[numClasses, numClasses]; filled by loops
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)  # Initialise all zeros, shape (3,3)
    preds  = model.predict(X)                                  # Get all predictions
    for true, pred in zip(y, preds):                           # Loop over (true_label, predicted_label) pairs
        matrix[true][pred] += 1                                # Increment the cell for this (true, pred) pair
    return matrix                                              # Return the completed matrix


def compute_f1(model, X, y, num_classes):
    """
    Compute per-class F1 score and macro-average F1.

    F1 = 2 * Precision * Recall / (Precision + Recall)
    Precision = TP / (TP + FP)  -- of all predicted positives, how many are correct?
    Recall    = TP / (TP + FN)  -- of all true positives, how many did we find?

    Returns (f1_per_class list, macro_f1 float)
    """
    preds = model.predict(X)                  # Get all predictions

    f1_scores = []                            # Will hold F1 for each class

    for cls in range(num_classes):            # Loop over each class index
        tp = np.sum((preds == cls) & (y == cls))  # True Positives:  predicted cls AND true cls
        fp = np.sum((preds == cls) & (y != cls))  # False Positives: predicted cls but NOT true cls
        fn = np.sum((preds != cls) & (y == cls))  # False Negatives: NOT predicted cls but IS true cls

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision (handle division by zero)
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall    (handle division by zero)

        if (precision + recall) > 0:                           # Avoid division by zero for F1
            f1 = 2.0 * precision * recall / (precision + recall)  # Harmonic mean of precision and recall
        else:
            f1 = 0.0                          # F1 is 0 if both precision and recall are 0

        f1_scores.append(f1)                  # Add this class's F1 to the list

    macro_f1 = np.mean(f1_scores)            # Macro F1 = average of all per-class F1 scores
    return f1_scores, macro_f1               # Return per-class list and overall average


# --- Compute metrics for both models on the TEST set -------------------------

# Base model: reset to original random weights to evaluate
base_model.set_weights(initial_weights)      # Make sure base_model has the original random weights

base_acc  = compute_accuracy(base_model,      X_test, y_test)   # Accuracy: base model on test set
tuned_acc = compute_accuracy(finetuned_model, X_test, y_test)   # Accuracy: fine-tuned model on test set

base_f1_per_class,  base_macro_f1  = compute_f1(base_model,      X_test, y_test, NUM_CLASSES)
tuned_f1_per_class, tuned_macro_f1 = compute_f1(finetuned_model, X_test, y_test, NUM_CLASSES)

base_cm  = compute_confusion_matrix(base_model,      X_test, y_test, NUM_CLASSES)  # Base confusion matrix
tuned_cm = compute_confusion_matrix(finetuned_model, X_test, y_test, NUM_CLASSES)  # Tuned confusion matrix

# --- Print accuracy ----------------------------------------------------------
print(f"\nTest Accuracy  -- Base model   : {base_acc:.2%}")   # Format as percentage
print(f"Test Accuracy  -- Fine-tuned   : {tuned_acc:.2%}")   # Format as percentage

# --- Print F1 scores ---------------------------------------------------------
print("\nPer-class F1 scores on test set:")
print(f"  {'Class':<12} {'Base F1':>10} {'Tuned F1':>10}")   # Header row
print(f"  {'-'*12} {'-'*10} {'-'*10}")                        # Divider row
for i, cls_name in enumerate(CLASS_NAMES):                    # Loop over each class
    print(f"  {cls_name:<12} {base_f1_per_class[i]:>10.3f} {tuned_f1_per_class[i]:>10.3f}")  # One row per class
print(f"  {'MACRO AVG':<12} {base_macro_f1:>10.3f} {tuned_macro_f1:>10.3f}")  # Macro average row

# --- Print confusion matrix --------------------------------------------------
def print_confusion_matrix(matrix, class_names, title):
    """
    Pretty-print a confusion matrix with row and column labels.

    matrix      : 2D numpy array, shape (num_classes, num_classes)
    class_names : list of class name strings
    title       : string title to print above the matrix
    """
    col_width = 12                                             # Fixed column width for alignment
    print(f"\n{title}")                                        # Print the title
    print(f"  (Row = True label, Column = Predicted label)")   # Explanation line
    print()                                                    # Blank line

    # Print column header row
    header = f"  {'':>12}"                                     # Left padding for row label column
    for name in class_names:                                   # Loop over class names for columns
        header += f"{name:>{col_width}}"                       # Each class name right-aligned
    print(header)                                              # Print the header line

    # Print divider line under header
    print("  " + "-" * (12 + col_width * len(class_names)))   # Dashes spanning full width

    # Print each row of the matrix
    for i, row_name in enumerate(class_names):                 # Loop over each true-class row
        row_str = f"  {row_name:>12} |"                        # Start row with true-class label
        for j in range(len(class_names)):                      # Loop over each predicted-class column
            row_str += f"{matrix[i][j]:>{col_width}}"          # Right-align cell value
        print(row_str)                                         # Print the completed row


print_confusion_matrix(base_cm,  CLASS_NAMES, "Confusion Matrix -- Base model (random weights):")
print_confusion_matrix(tuned_cm, CLASS_NAMES, "Confusion Matrix -- Fine-tuned model:")

# --- Show predictions AFTER training -----------------------------------------
print("\nPredictions AFTER fine-tuning (should be better than before):\n")

for i in preview_indices:                      # Same 5 examples we used in Part 2
    sentence, true_lbl = raw_data[i]           # Get the sentence and its true label
    x_vec  = encode_sentence(sentence, word_to_idx, VOCAB_SIZE)  # Encode sentence to BoW vector
    x_vec  = x_vec.reshape(1, -1)             # Reshape for batch dimension
    pred   = finetuned_model.predict(x_vec)[0]  # Get predicted class index
    print(f"  Text : {sentence[:50]:<50}")    # Print first 50 chars of sentence
    print(f"  True : {true_lbl:<10}  Predicted: {idx_to_label[pred]}")  # Print labels
    print()                                   # Blank line


# =============================================================================
# PART 5: COMPARISON TABLE
# =============================================================================

print("=" * 70)                  # Print separator
print("PART 5: COMPARISON TABLE -- Base vs Fine-Tuned on Test Set")  # Section header
print("=" * 70)                  # Print separator

# Restore base model random weights for fair comparison
base_model.set_weights(initial_weights)       # Reset base model to random weights

print()                          # Blank line
# Print table header
col1, col2, col3, col4 = 38, 10, 14, 18       # Column widths for the 4 columns
print(f"{'Input Text':<{col1}} {'True':<{col2}} {'Base Pred':<{col3}} {'Tuned Pred':<{col4}}")
print("-" * (col1 + col2 + col3 + col4 + 3))  # Divider line

# Show test examples in the comparison table (use test set or all examples if test is small)
# Use up to 10 examples from the test set; fall back to full dataset if test set is tiny
display_indices = test_idx if len(test_idx) >= 3 else list(range(min(10, TOTAL)))  # At least 3 examples

for i in display_indices[:10]:               # Loop over up to 10 test examples
    sentence, true_lbl = raw_data[i]         # Get sentence and true label string

    x_vec = encode_sentence(sentence, word_to_idx, VOCAB_SIZE)  # Encode to BoW vector
    x_vec = x_vec.reshape(1, -1)            # Reshape for batch dimension

    base_pred  = base_model.predict(x_vec)[0]       # Base model prediction (integer index)
    tuned_pred = finetuned_model.predict(x_vec)[0]  # Fine-tuned model prediction (integer index)

    base_lbl  = idx_to_label[base_pred].upper()[:8]  # Convert index to label string, uppercase
    tuned_lbl = idx_to_label[tuned_pred].upper()[:8] # Convert index to label string, uppercase
    true_upper = true_lbl.upper()[:8]                # True label uppercase for display

    # Truncate sentence for display so table stays readable
    short_text = sentence[:col1 - 2]                 # Clip sentence to fit column width
    if len(sentence) > col1 - 2:                     # If sentence was clipped...
        short_text += ".."                            # ... add ellipsis indicator

    print(f"{short_text:<{col1}} {true_upper:<{col2}} {base_lbl:<{col3}} {tuned_lbl:<{col4}}")  # Print table row

print("-" * (col1 + col2 + col3 + col4 + 3))  # Closing divider line


# =============================================================================
# PART 6: KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 70)           # Print separator
print("PART 6: KEY TAKEAWAYS")  # Section header
print("=" * 70)                  # Print separator

print()  # Blank line

# Print 5 lessons as numbered points
print("1. FINE-TUNING WORKS")
print("   Starting from random weights and training on labelled data")
print("   dramatically improves predictions compared to the base model.")
print("   This is the core idea of fine-tuning: adapt a model to a specific task.")
print()

print("2. BAG-OF-WORDS IS SIMPLE BUT USEFUL")
print("   Representing text as word-count vectors loses word order but")
print("   still captures enough signal to classify sentiment reasonably well.")
print("   C# analogy: it's like using a Dictionary<string,int> of word counts.")
print()

print("3. LOSS TELLS YOU HOW WRONG YOU ARE")
print("   Cross-entropy loss penalises the model more when it is confident")
print("   but wrong. Watching train_loss and val_loss both decrease means")
print("   the model is genuinely learning, not just memorising training data.")
print()

print("4. EARLY STOPPING PREVENTS OVERFITTING")
print("   If validation loss stops improving, the model may be memorising")
print("   the training set rather than learning general patterns.")
print("   Stopping early and restoring the best checkpoint keeps it honest.")
print()

print("5. EVALUATION NEEDS MORE THAN ONE METRIC")
print("   Accuracy alone can be misleading (e.g. if one class dominates).")
print("   F1 score and the confusion matrix reveal WHERE the model makes")
print("   mistakes, which guides further improvements like more data or")
print("   better features.")
print()

print("=" * 70)                          # Final separator
print("Project 01 -- Sentiment Fine-Tuner complete. Well done!")  # Completion message
print("=" * 70)                          # Final separator


# =============================================================================
# PART B: CONCEPTUAL PYTORCH SKETCH (commented out)
# =============================================================================
#
# If you later want to do this with PyTorch, here is the equivalent sketch.
# Read each comment to understand what the PyTorch version would look like.
# You do NOT need to run this -- it is educational reference only.
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# class SentimentModel(nn.Module):
#     """
#     PyTorch equivalent of our SimpleClassifier.
#     nn.Module is the base class for all PyTorch models.
#     C# analogy: inheriting from a base class that defines Forward().
#     """
#     def __init__(self, vocab_size, hidden_size, num_classes):
#         super().__init__()               # Call parent constructor (required in PyTorch)
#         # nn.Linear(in, out) = a fully connected layer with weight matrix + bias
#         # PyTorch handles weight initialisation automatically
#         self.layer1 = nn.Linear(vocab_size, hidden_size)  # Layer 1: input -> hidden
#         self.relu   = nn.ReLU()                           # ReLU activation function
#         self.layer2 = nn.Linear(hidden_size, num_classes) # Layer 2: hidden -> output
#
#     def forward(self, x):
#         # Forward pass: call layers in order
#         # PyTorch tracks gradients automatically (autograd)
#         x = self.layer1(x)              # Linear transformation
#         x = self.relu(x)               # ReLU activation
#         x = self.layer2(x)             # Second linear transformation
#         return x                       # Return raw scores (logits) -- NOT softmax yet
#
# # Create model
# model = SentimentModel(VOCAB_SIZE, HIDDEN_SIZE, NUM_CLASSES)
#
# # Loss function: CrossEntropyLoss applies softmax internally
# criterion = nn.CrossEntropyLoss()
#
# # Optimiser: Adam is a smarter gradient descent that adapts learning rate
# # C# analogy: a smarter version of our "weight -= lr * grad" update
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # Training loop (conceptual)
# for epoch in range(100):
#     model.train()                      # Set model to training mode (enables dropout etc.)
#     optimizer.zero_grad()              # Clear gradients from previous step
#     outputs = model(X_train_tensor)    # Forward pass (PyTorch tracks this automatically)
#     loss = criterion(outputs, y_train_tensor)  # Compute loss
#     loss.backward()                    # Backward pass: compute gradients automatically
#     optimizer.step()                   # Update weights using computed gradients
#
# # Evaluation
# model.eval()                           # Set to evaluation mode (disables dropout etc.)
# with torch.no_grad():                  # Disable gradient tracking for inference
#     test_outputs = model(X_test_tensor)
#     predictions  = torch.argmax(test_outputs, dim=1)
#
# # Key difference: PyTorch autograd computes all gradients for us.
# # In our NumPy version, we derived and coded the backward pass by hand.
# # Understanding the manual version (what we built) helps you understand
# # what PyTorch is doing under the hood.
#
# =============================================================================
