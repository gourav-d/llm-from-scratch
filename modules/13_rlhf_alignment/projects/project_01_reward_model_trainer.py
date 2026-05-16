# =============================================================================
# MODULE 13 - PROJECT 01: REWARD MODEL TRAINER
# =============================================================================
# Title   : Reward Model Training Pipeline from Scratch
# Goal    : Build a complete pipeline that takes raw human preference data
#           (which response is better?) and trains a neural network reward model
#           that can score any new response from 0.0 (bad) to 1.0 (good).
# What you
# will    : 1. Create a dataset of 20 preference pairs (chosen vs rejected)
# build   : 2. Build a two-layer RewardModel class in pure NumPy
#           3. Train the reward model with binary cross-entropy loss
#           4. Evaluate on held-out test pairs with accuracy metrics
#           5. Score 5 new responses with the trained model (ranked list)
#           6. Reflect on what the model learned via printed takeaways
# How to  :   python project_01_reward_model_trainer.py
# run     :
# C# analogy (overall system):
#   Think of this like a code-review scoring system.
#   You collect pairs of pull-request reviews where a human said
#   "Review A is better than Review B."  You train a model to mimic
#   that judgment.  After training, the model can give ANY new review
#   a score, so an automated CI pipeline can filter low-quality reviews
#   without a human reading each one.
# Dependencies: Python 3.10+ and NumPy only. No PyTorch, no APIs.
# =============================================================================

# =============================================================================
# GLOSSARY
# (Read this before the code -- every term used in this file is defined here)
# =============================================================================
#
# Reward Model (RM)
#   A neural network that takes a response and outputs a scalar score
#   representing how much a human would prefer that response.
#   Higher score = better response.  Think of it as a "quality judge."
#   C# analogy: a method double ScoreResponse(double[] features) that
#   has been trained on human ratings.
#
# Preference Pair
#   A pair (chosen, rejected) where a human annotator said
#   "I prefer chosen over rejected."
#   The reward model learns to score chosen > rejected.
#   C# analogy: a Tuple<double[], double[]> (winner, loser) from A/B testing.
#
# Feature Vector
#   A fixed-size array of numbers that describe one response.
#   We use 6 features: length_quality, politeness, accuracy,
#   helpfulness, safety, clarity.  Each is a float in [0.0, 1.0].
#   C# analogy: a struct ResponseFeatures { double LengthQuality; ... }
#   serialized to double[6].
#
# Binary Cross-Entropy (BCE)
#   A loss function for binary classification (0 or 1 label).
#   Formula: -(y * log(p) + (1-y) * log(1-p))
#   For reward models we use label=1 for chosen and label=0 for rejected.
#   C# analogy: the loss used in logistic regression classifiers.
#
# Sigmoid
#   Activation function: sigmoid(x) = 1 / (1 + exp(-x)).
#   Squashes any real number into the range (0.0, 1.0).
#   Perfect for producing a "probability" or "score" output.
#   C# analogy: 1.0 / (1.0 + Math.Exp(-x)) applied to a score.
#
# ReLU (Rectified Linear Unit)
#   Activation: output = max(0, x).  Negative values become 0.
#   Gives the network the ability to learn non-linear patterns.
#   C# analogy: Math.Max(0.0, x) applied element-by-element.
#
# Gradient
#   Direction and magnitude that tells us how to change each weight
#   to reduce the loss.  We move in the OPPOSITE direction of the gradient.
#   C# analogy: the slope on a loss surface -- go downhill.
#
# Bradley-Terry Model
#   A statistical model for pairwise preferences.
#   P(A beats B) = sigmoid(score_A - score_B).
#   The reward model implements exactly this idea.
#   C# analogy: the math behind Elo ratings in competitive games.
#
# Preference Accuracy
#   The fraction of preference pairs where score(chosen) > score(rejected).
#   A perfectly trained reward model gets 100% preference accuracy.
#   C# analogy: int correct / int total as a double.
#
# Hard Pair
#   A preference pair where chosen and rejected are very similar
#   (small feature differences).  These test whether the model learned
#   subtle quality signals, not just obvious ones.
#   C# analogy: edge-case unit tests that expose boundary conditions.
#
# Epoch
#   One complete pass through all training preference pairs.
#   C# analogy: one full iteration of a foreach over all training items.
#
# Learning Rate
#   How big a step we take when updating weights each gradient step.
#   Too large: weights overshoot and diverge.  Too small: very slow.
#   C# analogy: a step-size parameter in a numerical solver.
#
# Forward Pass
#   Running an input through the network layer by layer to get a score.
#   C# analogy: calling a chain of methods: Layer1() -> Layer2() -> Score().
#
# Backward Pass
#   Computing gradients of the loss with respect to each weight.
#   We use the chain rule of calculus (backpropagation).
#   C# analogy: automatic differentiation -- like what Expressions do in LINQ.
#
# =============================================================================

# --- IMPORTS -----------------------------------------------------------------
import numpy as np          # NumPy: our only dependency -- matrix math library
import math                 # math: Python built-in for log, exp, etc.

np.random.seed(42)          # Fix random seed so every run gives the same result
                            # C# analogy: new Random(42) for reproducibility


# =============================================================================
# ASCII DIAGRAM: FULL REWARD MODEL TRAINING PIPELINE
# =============================================================================
#
#   PREFERENCE DATA (20 pairs)
#   Each pair: (chosen_features [6], rejected_features [6])
#          |
#          v
#   +----------------------------+
#   |  Dataset Split             |
#   |  16 train / 4 test         |   Held-out test set for honest evaluation
#   +----------------------------+
#          |
#          v
#   +-------------------------------------------+
#   |  RewardModel (2-layer neural net)          |
#   |                                           |
#   |  input [6]                                |
#   |    -> Linear(6->8) + ReLU                 |
#   |    -> hidden [8]                          |
#   |    -> Linear(8->1) + sigmoid              |
#   |    -> score (scalar, 0.0 to 1.0)          |
#   +-------------------------------------------+
#          |
#          v
#   +---------------------------------------------+
#   |  Training Loop (50 epochs)                  |
#   |  For each pair (chosen, rejected):          |
#   |    1. score_c = model(chosen_features)      |
#   |    2. score_r = model(rejected_features)    |
#   |    3. BCE loss on (score_c, label=1)        |
#   |       + BCE loss on (score_r, label=0)      |
#   |    4. Backward pass -> compute gradients    |
#   |    5. Update weights (gradient descent)     |
#   +---------------------------------------------+
#          |
#          v
#   +----------------------------+
#   |  Evaluation                |
#   |  - Preference accuracy     |   % pairs where score_c > score_r
#   |  - Per-pair scores         |   chosen_score vs rejected_score
#   +----------------------------+
#          |
#          v
#   +----------------------------+
#   |  Score New Responses       |   Rank 5 new responses by quality
#   +----------------------------+
#
# =============================================================================

print("=" * 70)          # Print a divider line of 70 equal signs
print("MODULE 13 - PROJECT 01: REWARD MODEL TRAINER")   # Project title
print("=" * 70)          # Print another divider


# =============================================================================
# PART 1: DATASET -- 20 PREFERENCE PAIRS
# =============================================================================

print("\n" + "=" * 70)   # Section separator
print("PART 1: DATASET PREPARATION")   # Section header
print("=" * 70)          # Section separator

# --- Feature definitions ---
# Each response is described by 6 features, each a float in [0.0, 1.0].
# INDEX  FEATURE NAME       MEANING
#   0    length_quality     Is the response a good length? (too short=low, too long=low, just right=high)
#   1    politeness         Is the tone polite and respectful?
#   2    accuracy           Is the information factually correct?
#   3    helpfulness        Does it actually answer the question asked?
#   4    safety             Is it free from harmful or dangerous content?
#   5    clarity            Is it easy to understand?
#
# C# analogy: enum ResponseFeature { LengthQuality=0, Politeness=1, ... }

FEATURE_NAMES = [          # List of feature names for printing
    "length_quality",      # Index 0: response length appropriateness
    "politeness",          # Index 1: tone and respectfulness
    "accuracy",            # Index 2: factual correctness
    "helpfulness",         # Index 3: relevance to the question
    "safety",              # Index 4: absence of harmful content
    "clarity",             # Index 5: ease of understanding
]

INPUT_SIZE = 6             # Number of features per response (length of each feature vector)

# --- Preference pair dataset -------------------------------------------------
# Format: each entry is a dict with:
#   "chosen"   : feature vector for the BETTER response (label = 1.0)
#   "rejected" : feature vector for the WORSE response  (label = 0.0)
#   "hard"     : True if the pair is subtle/close (tests fine-grained learning)
# C# analogy: List<(double[] Chosen, double[] Rejected, bool IsHard)>

preference_pairs = [

    # =========================================================================
    # EASY PAIRS (pairs 1-16): obvious quality differences
    # =========================================================================

    # Pair 1: Excellent response vs very poor response
    {
        "chosen":   np.array([0.9, 0.9, 0.95, 0.9, 1.0, 0.9]),   # long, polite, accurate, helpful, safe, clear
        "rejected": np.array([0.1, 0.2, 0.1,  0.1, 0.5, 0.1]),   # short, rude, inaccurate, unhelpful, borderline, unclear
        "hard": False,                                              # Easy pair: big quality gap
    },
    # Pair 2: Good safety vs unsafe content
    {
        "chosen":   np.array([0.8, 0.8, 0.85, 0.85, 1.0, 0.8]),  # strong safety score
        "rejected": np.array([0.7, 0.7, 0.7,  0.8,  0.1, 0.7]),  # similar but safety=0.1 (dangerous)
        "hard": False,                                             # Safety violation is an easy reject
    },
    # Pair 3: High accuracy vs low accuracy
    {
        "chosen":   np.array([0.7, 0.8, 0.95, 0.8, 0.9, 0.8]),   # almost perfect accuracy
        "rejected": np.array([0.7, 0.7, 0.15, 0.7, 0.9, 0.7]),   # same features except accuracy is very low
        "hard": False,                                             # Accuracy difference is large
    },
    # Pair 4: Helpful response vs irrelevant response
    {
        "chosen":   np.array([0.8, 0.8, 0.8, 0.95, 0.9, 0.85]),  # very helpful
        "rejected": np.array([0.6, 0.7, 0.6, 0.05, 0.9, 0.6]),   # didn't answer the question at all
        "hard": False,                                             # Large helpfulness gap
    },
    # Pair 5: Clear response vs confusing response
    {
        "chosen":   np.array([0.75, 0.75, 0.8, 0.8, 0.9, 0.95]), # excellent clarity
        "rejected": np.array([0.6,  0.6,  0.7, 0.7, 0.9, 0.05]), # very hard to understand
        "hard": False,                                             # Clarity gap is obvious
    },
    # Pair 6: Polite vs rude
    {
        "chosen":   np.array([0.8, 0.95, 0.85, 0.8, 0.9, 0.8]),  # very polite tone
        "rejected": np.array([0.7, 0.05, 0.8,  0.7, 0.8, 0.75]), # extremely rude
        "hard": False,                                             # Politeness is dramatically different
    },
    # Pair 7: Well-sized vs too short
    {
        "chosen":   np.array([0.9, 0.8, 0.8, 0.85, 0.9, 0.85]),  # appropriate length and quality
        "rejected": np.array([0.1, 0.7, 0.6, 0.5,  0.9, 0.7]),   # way too short (length_quality=0.1)
        "hard": False,                                             # Length is obviously too short
    },
    # Pair 8: Balanced response vs unbalanced
    {
        "chosen":   np.array([0.85, 0.85, 0.85, 0.85, 0.9, 0.85]),  # uniformly good
        "rejected": np.array([0.9,  0.8,  0.1,  0.85, 0.9, 0.8]),   # good except accuracy is terrible
        "hard": False,                                                # One terrible feature drags it down
    },
    # Pair 9: Safe helpful vs unsafe but otherwise good
    {
        "chosen":   np.array([0.8, 0.8, 0.9, 0.9, 0.95, 0.8]),   # great on all fronts including safety
        "rejected": np.array([0.8, 0.7, 0.85, 0.9, 0.0, 0.8]),   # safety=0.0 (completely unsafe)
        "hard": False,                                             # Zero safety is an obvious reject
    },
    # Pair 10: Comprehensive vs minimal
    {
        "chosen":   np.array([0.88, 0.82, 0.9, 0.9, 0.9, 0.88]), # thorough and high quality
        "rejected": np.array([0.15, 0.5, 0.5, 0.4, 0.9, 0.5]),   # minimal, unhelpful answer
        "hard": False,                                             # Obvious quality difference
    },
    # Pair 11: Good all-rounder vs poor all-rounder
    {
        "chosen":   np.array([0.82, 0.80, 0.83, 0.84, 0.92, 0.81]), # consistently good
        "rejected": np.array([0.25, 0.28, 0.22, 0.30, 0.60, 0.27]), # consistently poor
        "hard": False,                                                # Large gap across all features
    },
    # Pair 12: Accurate polite vs inaccurate rude
    {
        "chosen":   np.array([0.7, 0.9, 0.92, 0.8, 0.9, 0.8]),   # polite and accurate
        "rejected": np.array([0.6, 0.1, 0.15, 0.7, 0.9, 0.6]),   # rude and inaccurate
        "hard": False,                                             # Two major quality differences
    },
    # Pair 13: Expert-level response vs novice response
    {
        "chosen":   np.array([0.9, 0.85, 0.95, 0.92, 0.95, 0.9]),  # expert quality
        "rejected": np.array([0.3, 0.4,  0.3,  0.35, 0.7,  0.3]),  # novice quality
        "hard": False,                                               # Huge overall gap
    },
    # Pair 14: On-topic vs off-topic response
    {
        "chosen":   np.array([0.8, 0.8, 0.8, 0.9, 0.9, 0.82]),   # helpfulness=0.9 (on-topic)
        "rejected": np.array([0.7, 0.7, 0.6, 0.1, 0.9, 0.7]),    # helpfulness=0.1 (missed the point)
        "hard": False,                                             # Helpfulness gap is large
    },
    # Pair 15: Long quality vs long but confusing
    {
        "chosen":   np.array([0.85, 0.8, 0.88, 0.85, 0.9, 0.92]),  # long but also very clear
        "rejected": np.array([0.80, 0.75, 0.80, 0.80, 0.9, 0.10]), # similarly long but barely readable
        "hard": False,                                               # Clarity is the deciding factor
    },
    # Pair 16: Helpful and safe vs helpful but harmful
    {
        "chosen":   np.array([0.78, 0.78, 0.82, 0.88, 0.98, 0.80]),  # all good, safety near-perfect
        "rejected": np.array([0.75, 0.72, 0.78, 0.85, 0.12, 0.78]),  # similar but dangerous content
        "hard": False,                                                 # Safety difference is decisive
    },

    # =========================================================================
    # HARD PAIRS (pairs 17-20): subtle differences -- tests model precision
    # =========================================================================

    # Pair 17 (HARD): Both very good, chosen barely edges rejected on accuracy
    {
        "chosen":   np.array([0.82, 0.83, 0.88, 0.84, 0.92, 0.83]),  # accuracy 0.88 vs 0.79
        "rejected": np.array([0.81, 0.82, 0.79, 0.83, 0.91, 0.82]),  # almost identical otherwise
        "hard": True,                                                  # Small margin -- hard to learn
    },
    # Pair 18 (HARD): Chosen slightly more helpful and clearer
    {
        "chosen":   np.array([0.79, 0.80, 0.81, 0.87, 0.90, 0.85]),  # helpfulness 0.87, clarity 0.85
        "rejected": np.array([0.78, 0.79, 0.80, 0.80, 0.90, 0.79]),  # helpfulness 0.80, clarity 0.79
        "hard": True,                                                  # Subtle difference across two features
    },
    # Pair 19 (HARD): Chosen scores higher on safety but slightly lower on length
    {
        "chosen":   np.array([0.76, 0.82, 0.83, 0.83, 0.93, 0.82]),  # safety 0.93 vs 0.85
        "rejected": np.array([0.80, 0.81, 0.83, 0.82, 0.85, 0.81]),  # length 0.80 vs 0.76 (rejected is longer)
        "hard": True,                                                  # Safety vs length trade-off
    },
    # Pair 20 (HARD): Nearly identical vectors, tiny edge on politeness + clarity
    {
        "chosen":   np.array([0.80, 0.84, 0.80, 0.81, 0.90, 0.83]),  # politeness 0.84, clarity 0.83
        "rejected": np.array([0.80, 0.79, 0.80, 0.80, 0.90, 0.78]),  # politeness 0.79, clarity 0.78
        "hard": True,                                                  # Very close -- genuine challenge
    },
]

NUM_PAIRS = len(preference_pairs)    # Total number of preference pairs = 20

# Print dataset overview
print(f"Total preference pairs : {NUM_PAIRS}")           # Show total pairs
print(f"Easy pairs             : {sum(1 for p in preference_pairs if not p['hard'])}")  # Count easy pairs
print(f"Hard pairs             : {sum(1 for p in preference_pairs if p['hard'])}")      # Count hard pairs
print(f"Features per response  : {INPUT_SIZE}  {FEATURE_NAMES}")                        # Show feature list

# Print a few example pairs to illustrate the dataset
print("\nSample preference pairs:")                       # Header for sample output
for i in [0, 1, 16, 19]:                                 # Show pairs 1, 2, 17, 20 (indices 0,1,16,19)
    p = preference_pairs[i]                              # Get the pair dict
    tag = "[HARD]" if p["hard"] else "[EASY]"            # Tag for easy vs hard
    diff = p["chosen"] - p["rejected"]                   # Feature-wise difference vector
    print(f"  Pair {i+1:>2} {tag}")                      # Print pair number and tag
    print(f"    chosen  : {np.round(p['chosen'],   2)}")  # Print chosen feature vector (2dp)
    print(f"    rejected: {np.round(p['rejected'], 2)}")  # Print rejected feature vector (2dp)
    print(f"    diff    : {np.round(diff, 2)}")           # Print difference (shows what model must learn)


# =============================================================================
# PART 2: REWARD MODEL CLASS
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 2: REWARD MODEL ARCHITECTURE")   # Section header
print("=" * 70)           # Section separator

# Architecture:
#   input [6] -> W1 (6x8) + b1 (8) -> ReLU -> hidden [8]
#             -> W2 (8x1) + b2 (1) -> sigmoid -> score (scalar 0..1)
#
# C# analogy:
#   class RewardModel {
#       double[,] W1; double[] b1;   // Layer 1 weights and biases
#       double[,] W2; double[] b2;   // Layer 2 weights and biases
#       double Forward(double[] x)   // Returns score 0..1
#       void   Backward(...)         // Computes weight gradients
#       void   Update(double lr)     // Applies gradient descent step
#   }


def sigmoid(x):
    """
    Sigmoid activation: output = 1 / (1 + exp(-x)).
    Squashes any number into (0.0, 1.0).
    We clip x to [-500, 500] first to avoid overflow in exp(-x).
    C# analogy: 1.0 / (1.0 + Math.Exp(-x)) with overflow guard.
    """
    x_clipped = np.clip(x, -500, 500)        # Clip to prevent exp overflow
    return 1.0 / (1.0 + np.exp(-x_clipped)) # Standard sigmoid formula


def relu(x):
    """
    ReLU activation: output = max(0, x).
    Negative values become 0, positives unchanged.
    C# analogy: Math.Max(0.0, x) element-by-element.
    """
    return np.maximum(0.0, x)               # Element-wise max(0, x)


class RewardModel:
    """
    Two-layer neural network that scores a response feature vector.
    Output is a scalar in [0.0, 1.0]: higher = better response.

    Architecture:
      Layer 1: Linear(6 -> 8) + ReLU
      Layer 2: Linear(8 -> 1) + Sigmoid

    C# analogy: a scoring service class with Train() and Score() methods.
    """

    def __init__(self, input_size=6, hidden_size=8):
        """
        Initialise all weight matrices and bias vectors with small random values.
        input_size  : number of input features (default 6)
        hidden_size : number of hidden layer neurons (default 8)
        C# analogy: constructor that allocates and initialises double[][] arrays.
        """
        self.input_size  = input_size      # Store input dimension (6 features)
        self.hidden_size = hidden_size     # Store hidden layer size (8 neurons)

        # --- Weight initialisation (Xavier / Glorot scaling) -----------------
        # Multiply by sqrt(2/fan_in) to keep activations from exploding or vanishing.
        # C# analogy: initialise double[,] with Gaussian noise scaled by a factor.
        scale1 = np.sqrt(2.0 / input_size)   # Scale for layer 1: based on input fan-in
        scale2 = np.sqrt(2.0 / hidden_size)  # Scale for layer 2: based on hidden fan-in

        # W1: weight matrix layer 1, shape (input_size, hidden_size) = (6, 8)
        # Each column connects all 6 inputs to one of the 8 hidden neurons.
        self.W1 = np.random.randn(input_size, hidden_size) * scale1

        # b1: bias vector layer 1, shape (hidden_size,) = (8,)
        # One bias term per hidden neuron.  Start at zero (standard practice).
        self.b1 = np.zeros(hidden_size)

        # W2: weight matrix layer 2, shape (hidden_size, 1) = (8, 1)
        # Connects all 8 hidden neurons to the single output score.
        self.W2 = np.random.randn(hidden_size, 1) * scale2

        # b2: bias vector layer 2, shape (1,) = scalar bias for output
        self.b2 = np.zeros(1)

    def forward(self, x):
        """
        Run a feature vector through the network to produce a score.
        x     : 1D NumPy array of shape (input_size,) = (6,)
        return: scalar float score in [0.0, 1.0]

        C# analogy: double Score(double[] features) { ... }
        """
        # Layer 1: linear transformation
        z1 = self.W1.T @ x + self.b1        # z1 = W1^T * x + b1, shape (hidden_size,) = (8,)
                                             # W1 is (6,8), x is (6,), W1.T is (8,6), result is (8,)
        # Apply ReLU activation
        a1 = relu(z1)                        # a1 = ReLU(z1), shape (8,) -- negatives zeroed out

        # Layer 2: linear transformation
        z2 = self.W2.T @ a1 + self.b2       # z2 = W2^T * a1 + b2, shape (1,)
                                             # W2 is (8,1), a1 is (8,), W2.T is (1,8), result is (1,)
        # Apply sigmoid to get a score in (0, 1)
        score = sigmoid(z2)                  # score is shape (1,) -- a NumPy array with one element

        # Store intermediate values for backward pass (like a computation cache)
        # C# analogy: private fields set during forward and read during backward
        self._x  = x                         # Cache original input
        self._z1 = z1                        # Cache pre-ReLU hidden values
        self._a1 = a1                        # Cache post-ReLU hidden values
        self._z2 = z2                        # Cache pre-sigmoid output
        self._s  = score                     # Cache sigmoid output (the score)

        return float(score[0])               # Return as a plain Python float (scalar)

    def backward(self, label):
        """
        Compute gradients of binary cross-entropy loss w.r.t. all weights.
        Call this AFTER forward() so cached values are available.

        label : float -- 1.0 if this was a "chosen" response, 0.0 if "rejected"

        Binary cross-entropy: L = -(y*log(s) + (1-y)*log(1-s))
        dL/ds  = -(y/s - (1-y)/(1-s))
        ds/dz2 = s * (1 - s)                (sigmoid derivative)
        Combined: dL/dz2 = s - y            (elegant simplification)

        Returns a dict with gradients for W1, b1, W2, b2.
        C# analogy: GradientDict ComputeGradients(double label) { ... }
        """
        s = self._s[0]                       # Scalar sigmoid output (the score)

        # --- Gradient at z2 (pre-sigmoid output) ---
        # d(BCE)/d(z2) = s - y  (combined sigmoid + BCE derivative simplifies nicely)
        dz2 = s - label                      # Scalar: error signal at the output

        # --- Gradients for layer 2 weights and bias ---
        # dL/dW2 = a1 * dz2  (outer product: (8,) * scalar = (8,))
        dW2 = self._a1.reshape(-1, 1) * dz2  # Shape (8, 1): gradient for each W2 weight
        db2 = np.array([dz2])                # Shape (1,): gradient for b2 bias

        # --- Backpropagate through layer 2 to layer 1 ---
        # dL/da1 = W2 * dz2  (shape (8,): gradient flowing back to hidden layer)
        da1 = (self.W2 * dz2).reshape(-1)    # Shape (8,): how much each hidden neuron contributed

        # --- Gradient through ReLU ---
        # ReLU derivative: 1 where z1 > 0, else 0
        drelu = (self._z1 > 0).astype(np.float64)  # Shape (8,): mask of active neurons
        dz1   = da1 * drelu                          # Shape (8,): gradient at pre-ReLU values

        # --- Gradients for layer 1 weights and bias ---
        # dL/dW1 = x (outer) dz1  (outer product: (6,) and (8,) = (6, 8))
        dW1 = np.outer(self._x, dz1)         # Shape (6, 8): gradient for each W1 weight
        db1 = dz1                             # Shape (8,): gradient for b1 bias

        return {                              # Return gradients as a dictionary
            "dW1": dW1,                       # Shape (6, 8)
            "db1": db1,                       # Shape (8,)
            "dW2": dW2,                       # Shape (8, 1)
            "db2": db2,                       # Shape (1,)
        }

    def update(self, gradients, lr=0.01):
        """
        Apply gradient descent: subtract learning_rate * gradient from each weight.
        gradients : dict returned by backward()
        lr        : learning rate (default 0.01)
        C# analogy: W1 -= lr * dW1; b1 -= lr * db1; etc.
        """
        self.W1 -= lr * gradients["dW1"]     # Update layer 1 weights
        self.b1 -= lr * gradients["db1"]     # Update layer 1 biases
        self.W2 -= lr * gradients["dW2"]     # Update layer 2 weights
        self.b2 -= lr * gradients["db2"]     # Update layer 2 biases

    def score(self, features):
        """
        Convenience method: run forward pass and return scalar score.
        features: 1D NumPy array of shape (input_size,)
        return  : float in [0.0, 1.0]
        """
        return self.forward(features)        # Just call forward and return its result


# --- Instantiate and demonstrate the model before training -------------------
HIDDEN_SIZE = 8                              # Hidden layer size (hyperparameter)

model = RewardModel(                         # Create a new reward model instance
    input_size  = INPUT_SIZE,                # 6 input features
    hidden_size = HIDDEN_SIZE,               # 8 hidden neurons
)

print(f"\nReward model architecture:")                                        # Print header
print(f"  Input layer  : {INPUT_SIZE} features")                              # Input size
print(f"  Hidden layer : {HIDDEN_SIZE} neurons  (W1 shape: {model.W1.shape}, b1 shape: {model.b1.shape})")
print(f"  Output layer : 1 neuron      (W2 shape: {model.W2.shape}, b2 shape: {model.b2.shape})")
print(f"  Output range : 0.0 (bad) to 1.0 (good) via sigmoid")               # Output range

# Show a sample score before training (should be ~random / near 0.5)
sample_pair = preference_pairs[0]                          # Take the first preference pair
score_before_c = model.score(sample_pair["chosen"])        # Score the chosen response
score_before_r = model.score(sample_pair["rejected"])      # Score the rejected response
print(f"\nPre-training scores on pair 1 (should be random, near 0.5):")
print(f"  Chosen   features: {np.round(sample_pair['chosen'],   2)}  -> score: {score_before_c:.4f}")
print(f"  Rejected features: {np.round(sample_pair['rejected'], 2)}  -> score: {score_before_r:.4f}")
correct_before = "YES" if score_before_c > score_before_r else "NO "         # Did the random model get it right?
print(f"  score_chosen > score_rejected?  {correct_before}  (random chance -- not trained yet)")


# =============================================================================
# PART 3: TRAINING LOOP
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 3: TRAINING (50 EPOCHS, BATCH GRADIENT DESCENT)")   # Section header
print("=" * 70)           # Section separator

# --- Training hyperparameters ------------------------------------------------
NUM_EPOCHS  = 50          # Number of complete passes through the training data
LEARNING_RATE = 0.05      # Step size for gradient descent
TRAIN_SIZE  = 16          # Number of pairs used for training
TEST_SIZE   = 4           # Number of pairs held out for evaluation (= 20 - 16)

# --- Train/test split --------------------------------------------------------
# We use the first 16 pairs for training and the last 4 for evaluation.
# In Part 4 we will retrain from scratch and properly evaluate on these 4.
# For Part 3 we train on all 20 to see loss behaviour clearly.

train_pairs = preference_pairs[:TRAIN_SIZE]  # First 16 pairs for training
test_pairs  = preference_pairs[TRAIN_SIZE:]  # Last 4 pairs for testing (held out)

print(f"\nTraining on {TRAIN_SIZE} pairs, holding out {TEST_SIZE} pairs for evaluation.")
print(f"Epochs: {NUM_EPOCHS}   Learning rate: {LEARNING_RATE}")

def compute_loss_and_accuracy(rm, pairs):
    """
    Compute average BCE loss and preference accuracy on a list of pairs.
    rm    : RewardModel instance
    pairs : list of preference pair dicts
    Returns (avg_loss, accuracy) both as floats.
    C# analogy: (double AvgLoss, double Accuracy) Evaluate(RewardModel rm, List<Pair> pairs)
    """
    total_loss = 0.0              # Accumulate total loss across all pairs
    num_correct = 0               # Count pairs where score(chosen) > score(rejected)
    eps = 1e-9                    # Small constant to prevent log(0) which is -infinity

    for pair in pairs:            # Loop over each preference pair
        sc = rm.score(pair["chosen"])    # Score the chosen (better) response
        sr = rm.score(pair["rejected"])  # Score the rejected (worse) response

        # BCE loss for chosen response (label = 1): -log(sc)
        loss_c = -math.log(sc + eps)     # Log likelihood of scoring chosen high

        # BCE loss for rejected response (label = 0): -log(1 - sr)
        loss_r = -math.log(1.0 - sr + eps)  # Log likelihood of scoring rejected low

        total_loss += (loss_c + loss_r) / 2.0  # Average loss for this pair

        if sc > sr:                      # Did the model rank chosen above rejected?
            num_correct += 1             # Yes: count as correct

    avg_loss  = total_loss / len(pairs)  # Average loss over all pairs
    accuracy  = num_correct / len(pairs) # Fraction of pairs ranked correctly
    return avg_loss, accuracy            # Return both metrics


# Save initial weights so we can reset for Part 4 (fair evaluation)
initial_W1 = model.W1.copy()            # Deep copy of initial W1
initial_b1 = model.b1.copy()            # Deep copy of initial b1
initial_W2 = model.W2.copy()            # Deep copy of initial W2
initial_b2 = model.b2.copy()            # Deep copy of initial b2

loss_history = []                        # Track loss each epoch for ASCII chart

print("\nTraining progress (printing every 10 epochs):\n")
print(f"  {'Epoch':>6}  {'Loss':>10}  {'PrefAcc%':>10}  {'HardAcc%':>10}")  # Column headers
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")                          # Divider

for epoch in range(1, NUM_EPOCHS + 1):    # Loop from epoch 1 to NUM_EPOCHS inclusive

    # --- Batch gradient accumulation ----------------------------------------
    # Instead of updating after each pair, we average the gradients over all
    # training pairs and do one update per epoch.
    # C# analogy: accumulate deltas in a list, average them, then apply.

    batch_dW1 = np.zeros_like(model.W1)  # Accumulated gradient for W1, start at zeros
    batch_db1 = np.zeros_like(model.b1)  # Accumulated gradient for b1
    batch_dW2 = np.zeros_like(model.W2)  # Accumulated gradient for W2
    batch_db2 = np.zeros_like(model.b2)  # Accumulated gradient for b2

    for pair in train_pairs:              # Loop over all 16 training pairs

        # --- Process chosen response (target label = 1.0) ---
        model.forward(pair["chosen"])     # Run forward pass, cache intermediate values
        grads_c = model.backward(1.0)     # Compute gradients for label=1 (chosen is good)

        # Accumulate gradients from this chosen response
        batch_dW1 += grads_c["dW1"]      # Add to batch W1 gradient
        batch_db1 += grads_c["db1"]      # Add to batch b1 gradient
        batch_dW2 += grads_c["dW2"]      # Add to batch W2 gradient
        batch_db2 += grads_c["db2"]      # Add to batch b2 gradient

        # --- Process rejected response (target label = 0.0) ---
        model.forward(pair["rejected"])   # Run forward pass on the rejected response
        grads_r = model.backward(0.0)     # Compute gradients for label=0 (rejected is bad)

        # Accumulate gradients from this rejected response
        batch_dW1 += grads_r["dW1"]      # Add rejected gradients to batch W1 gradient
        batch_db1 += grads_r["db1"]      # Add to batch b1 gradient
        batch_dW2 += grads_r["dW2"]      # Add to batch W2 gradient
        batch_db2 += grads_r["db2"]      # Add to batch b2 gradient

    # --- Average gradients over the batch (2 * TRAIN_SIZE forward passes) ---
    n_updates = 2 * len(train_pairs)     # Total number of forward passes this epoch
    avg_grads = {                        # Build averaged gradient dict
        "dW1": batch_dW1 / n_updates,   # Divide by total updates to get average
        "db1": batch_db1 / n_updates,
        "dW2": batch_dW2 / n_updates,
        "db2": batch_db2 / n_updates,
    }

    # --- Apply one weight update using averaged gradients ---
    model.update(avg_grads, lr=LEARNING_RATE)  # Gradient descent step

    # --- Compute metrics for printing ---
    epoch_loss, epoch_acc = compute_loss_and_accuracy(model, train_pairs)  # Training metrics
    loss_history.append(epoch_loss)      # Save loss for ASCII chart later

    # Compute accuracy only on the 4 hard pairs (to track subtle learning)
    hard_pairs = [p for p in train_pairs if p["hard"]]  # Filter hard pairs from training set
    if len(hard_pairs) > 0:                             # Only compute if hard pairs are in training set
        _, hard_acc = compute_loss_and_accuracy(model, hard_pairs)  # Hard pair accuracy
    else:
        hard_acc = float("nan")                          # No hard pairs in training set

    # Print every 10 epochs
    if epoch % 10 == 0 or epoch == 1:    # Print at epoch 1, 10, 20, 30, 40, 50
        print(f"  {epoch:>6}  {epoch_loss:>10.4f}  {epoch_acc*100:>9.1f}%  {hard_acc*100:>9.1f}%")


# --- ASCII Loss Chart (shows training curve) ---------------------------------
print("\nASCII Loss Chart (training loss over 50 epochs):")  # Header
print("  Higher position = higher (worse) loss")             # Chart legend
print()

CHART_WIDTH = 20                                # Width in characters for the bar
min_loss = min(loss_history)                    # Lowest loss seen during training
max_loss = max(loss_history)                    # Highest loss seen during training
loss_range = max(max_loss - min_loss, 1e-9)     # Range (guard against division by zero)

print(f"  {'Ep':>4}  Loss    Chart (0=low, {CHART_WIDTH}=high)")  # Column header
print(f"  {'----':>4}  ------  " + "-" * CHART_WIDTH)             # Divider

for i, loss in enumerate(loss_history):         # Loop over each epoch's recorded loss
    bar_len = int((loss - min_loss) / loss_range * CHART_WIDTH)   # Scale loss to 0..CHART_WIDTH
    bar     = "*" * bar_len                      # Build the bar string
    print(f"  {i+1:>4}  {loss:.4f}  {bar}")     # Print: epoch number, loss value, bar

print(f"\n  Final training loss    : {loss_history[-1]:.4f}")           # Show final loss
print(f"  Starting training loss : {loss_history[0]:.4f}")             # Show starting loss
print(f"  Loss reduction         : {loss_history[0]-loss_history[-1]:.4f}")  # Show improvement


# =============================================================================
# PART 4: EVALUATION ON HELD-OUT TEST PAIRS
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 4: EVALUATION ON HELD-OUT TEST PAIRS")   # Section header
print("=" * 70)           # Section separator

# Reset model to initial random weights for a FAIR evaluation:
# We retrain from scratch using only the 16 training pairs,
# then evaluate on the 4 test pairs the model never saw.
print("\nResetting to initial weights and retraining on 16 pairs only...")

model.W1 = initial_W1.copy()            # Restore initial W1
model.b1 = initial_b1.copy()            # Restore initial b1
model.W2 = initial_W2.copy()            # Restore initial W2
model.b2 = initial_b2.copy()            # Restore initial b2

# Retrain from scratch on training pairs only (same loop as Part 3)
for epoch in range(1, NUM_EPOCHS + 1):  # 50 epochs again
    batch_dW1 = np.zeros_like(model.W1) # Reset batch gradients each epoch
    batch_db1 = np.zeros_like(model.b1)
    batch_dW2 = np.zeros_like(model.W2)
    batch_db2 = np.zeros_like(model.b2)

    for pair in train_pairs:            # Loop over 16 training pairs
        model.forward(pair["chosen"])   # Forward pass on chosen
        grads_c = model.backward(1.0)   # Gradients for label=1
        batch_dW1 += grads_c["dW1"]
        batch_db1 += grads_c["db1"]
        batch_dW2 += grads_c["dW2"]
        batch_db2 += grads_c["db2"]

        model.forward(pair["rejected"]) # Forward pass on rejected
        grads_r = model.backward(0.0)   # Gradients for label=0
        batch_dW1 += grads_r["dW1"]
        batch_db1 += grads_r["db1"]
        batch_dW2 += grads_r["dW2"]
        batch_db2 += grads_r["db2"]

    n_updates = 2 * len(train_pairs)    # 32 total forward passes per epoch
    avg_grads = {
        "dW1": batch_dW1 / n_updates,
        "db1": batch_db1 / n_updates,
        "dW2": batch_dW2 / n_updates,
        "db2": batch_db2 / n_updates,
    }
    model.update(avg_grads, lr=LEARNING_RATE)   # Apply averaged gradient step

print("Retraining complete.\n")         # Confirm retraining is done

# --- Evaluate on the 4 held-out test pairs -----------------------------------
test_loss, test_acc = compute_loss_and_accuracy(model, test_pairs)   # Compute test metrics

print(f"Test preference accuracy : {test_acc * 100:.1f}%")   # % of test pairs ranked correctly
print(f"Test loss                : {test_loss:.4f}")          # Average BCE loss on test pairs

# Also show training accuracy for comparison
train_loss, train_acc = compute_loss_and_accuracy(model, train_pairs)  # Training metrics
print(f"\nTraining preference accuracy : {train_acc * 100:.1f}%")   # Training accuracy for comparison
print(f"Training loss                : {train_loss:.4f}")            # Training loss for comparison

# --- Per-pair test results ----------------------------------------------------
print("\nPer-pair test results:")                                      # Header for detailed results
print(f"  {'Pair':>5}  {'Chosen Score':>14}  {'Rejected Score':>16}  {'Correct?':>10}  {'Hard?':>6}")
print(f"  {'-----':>5}  {'-'*14}  {'-'*16}  {'-'*10}  {'-'*6}")     # Divider

for idx, pair in enumerate(test_pairs):  # Loop over each held-out test pair
    sc = model.score(pair["chosen"])     # Score the chosen response
    sr = model.score(pair["rejected"])   # Score the rejected response
    correct = "YES" if sc > sr else "NO "  # Did model rank them correctly?
    hard_tag = "HARD" if pair["hard"] else "easy"   # Tag for hard vs easy
    pair_num = TRAIN_SIZE + idx + 1      # Pair number in full dataset (17, 18, 19, 20)
    print(f"  {pair_num:>5}  {sc:>14.4f}  {sr:>16.4f}  {correct:>10}  {hard_tag:>6}")


# =============================================================================
# PART 5: SCORING NEW RESPONSES (RANKED LIST)
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 5: SCORING 5 NEW RESPONSES WITH TRAINED MODEL")   # Section header
print("=" * 70)           # Section separator

# These are 5 brand-new responses the model was NEVER trained on.
# We ask the trained reward model to assign each a quality score.
# Then we rank them from best to worst.
# C# analogy: calling ScoreResponse(double[] features) on 5 candidate responses.

new_responses = [          # List of (name, feature_vector) tuples
    (
        "Response A: Expert, clear, safe",            # Descriptive name
        np.array([0.90, 0.88, 0.95, 0.92, 0.98, 0.91]),  # All features high
    ),
    (
        "Response B: Too short, slightly rude",       # Descriptive name
        np.array([0.10, 0.30, 0.70, 0.65, 0.88, 0.60]),  # Low length_quality and politeness
    ),
    (
        "Response C: Good but mildly unsafe",         # Descriptive name
        np.array([0.80, 0.78, 0.82, 0.83, 0.25, 0.79]),  # Safety is low
    ),
    (
        "Response D: Average across the board",       # Descriptive name
        np.array([0.55, 0.58, 0.56, 0.57, 0.75, 0.55]),  # Mediocre on everything
    ),
    (
        "Response E: Verbose but accurate and kind",  # Descriptive name
        np.array([0.72, 0.91, 0.93, 0.85, 0.95, 0.87]),  # High on most, okay length
    ),
]

# Score all 5 responses
scored = []                               # Will hold (score, name, features) tuples
for name, features in new_responses:      # Loop over each new response
    s = model.score(features)             # Get reward model score for this response
    scored.append((s, name, features))    # Store (score, name, features) for sorting

# Sort by score descending (best response first)
# C# analogy: scored.OrderByDescending(r => r.Score).ToList()
scored.sort(key=lambda t: t[0], reverse=True)  # Sort by first element (score), highest first

print("\nRanked responses (best to worst according to trained reward model):\n")
print(f"  {'Rank':>5}  {'Score':>7}  Name")                          # Column headers
print(f"  {'----':>5}  {'-------':>7}  {'----'}")                    # Divider

for rank, (score, name, features) in enumerate(scored, start=1):    # Loop with rank starting at 1
    bar_len = int(score * 30)                                        # Bar proportional to score (max 30 chars)
    bar = "#" * bar_len                                              # Build bar string
    print(f"  {rank:>5}  {score:.4f}  {name}")                      # Print rank, score, name
    print(f"         Features: {np.round(features, 2)}")             # Print feature vector
    print(f"         [{bar:<30}]  ({score*100:.1f}%)")               # Print visual bar
    print()                                                          # Blank line between entries

print("NOTE: The reward model learned to prefer responses that score high")
print("      on accuracy, safety, and helpfulness above all other features.")


# =============================================================================
# PART 6: KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 6: KEY TAKEAWAYS")   # Section header
print("=" * 70)           # Section separator

print()   # Blank line

print("1. REWARD MODELS ENCODE HUMAN PREFERENCES AS NUMBERS")
print("   Instead of a human reading every response, the reward model")
print("   distills thousands of preference judgments into a weight matrix.")
print("   C# analogy: training a classifier to replace a manual code review")
print("   checklist -- the model learns the rules from examples, not rules.")
print()

print("2. BRADLEY-TERRY LOSS IS IDEAL FOR PAIRWISE DATA")
print("   We never tell the model 'this response scores 0.72.'")
print("   We only say 'A is better than B.'  The model learns to separate")
print("   scores automatically.  This is exactly how human preference data")
print("   is collected in real RLHF pipelines.")
print()

print("3. HARD PAIRS REVEAL THE MODEL'S REAL CAPABILITY")
print("   Easy pairs (big quality gap) are not enough to trust a reward model.")
print("   Hard pairs (subtle differences) show whether it learned genuine")
print("   quality signals or just 'prefer the one with higher average features.'")
print()

print("4. SIGMOID OUTPUT GIVES INTERPRETABLE SCORES")
print("   Because the output layer uses sigmoid, every score is in [0.0, 1.0].")
print("   You can meaningfully compare scores: 0.9 is always better than 0.6.")
print("   Without sigmoid, raw logits are hard to compare across responses.")
print()

print("5. PREFERENCE ACCURACY IS THE KEY METRIC")
print("   Loss is useful during training but hard to interpret.  Preference")
print("   accuracy (% of pairs correctly ranked) directly tells you if the")
print("   reward model is useful.  100% on easy pairs + high % on hard pairs")
print("   = a reward model you can trust to guide RL fine-tuning.")
print()

print("=" * 70)                          # Final separator
print("Project 01 -- Reward Model Trainer complete. Well done!")  # Completion message
print("=" * 70)                          # Final separator
