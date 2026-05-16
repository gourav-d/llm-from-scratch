# =============================================================================
# MODULE 13 - PROJECT 03: ALIGNMENT EVALUATION DASHBOARD
# =============================================================================
# Title   : Alignment Evaluation Dashboard -- Compare Four Model Families
# Goal    : Build a complete alignment testing suite that evaluates four
#           model "families" (base, SFT, RLHF, DPO) across five alignment
#           categories, prints a formatted report table, computes win rates,
#           and runs a red-team check for harmful content.
# What you
# will    : 1. Define 20 test cases across 5 alignment categories
# build   : 2. Represent 4 models as weight matrices (each maps prompts -> behaviors)
#           3. Implement alignment scoring functions (cosine similarity, win rate)
#           4. Print a formatted dashboard table with per-category scores
#           5. Show a win-rate matrix (which model beats which, and by how much?)
#           6. Run a red-team check -- can any model be tricked by harmful prompts?
#           7. Reflect on what alignment metrics reveal
# How to  :   python project_03_alignment_evaluator.py
# run     :
# C# analogy (overall system):
#   Think of this like a test quality dashboard in a microservices pipeline.
#   You have four versions of a service (base, v1, v2, v3) and a test suite
#   with five quality categories (security, performance, reliability, ...).
#   Each version gets a score in each category.  The dashboard shows which
#   version is best, where each one falls short, and which ones fail the
#   critical security (red-team) gate.  "Win rate" is how often one version
#   outscores another on individual test cases -- like A/B testing per test.
# Dependencies: Python 3.10+ and NumPy only. No PyTorch, no APIs.
# =============================================================================

# =============================================================================
# GLOSSARY
# (Read this before the code -- every term used in this file is defined here)
# =============================================================================
#
# Alignment
#   The degree to which an AI model's outputs match human values and intentions.
#   An aligned model is helpful, honest, and harmless.
#   C# analogy: passing all quality gates in a CI/CD pipeline (unit + integration
#   + security + performance tests).
#
# Alignment Category
#   A dimension of alignment we measure separately.
#   We use five: Helpfulness, Harmlessness, Honesty, InstructionFollowing, Robustness.
#   C# analogy: separate test suites (UnitTests, SecurityTests, PerfTests).
#
# Test Case
#   One specific input (prompt_features) paired with an expected_behavior_vector.
#   The model processes prompt_features and produces a predicted_behavior_vector.
#   A good model's prediction is close to expected.
#   C# analogy: one [TestMethod] with a specific input and expected output.
#
# Prompt Features
#   A fixed-size numeric vector describing a prompt/request.
#   Example: [specificity, complexity, safety_risk, ...].
#   C# analogy: double[] PromptDescriptor encoding key properties of a request.
#
# Behavior Vector
#   A fixed-size numeric vector describing how a model responds to a prompt.
#   Example: [information_quality, refusal_rate, tone, ...].
#   C# analogy: double[] BehaviorMetrics encoding properties of a response.
#
# Cosine Similarity
#   Measures the angle between two vectors.
#   cos_sim(a, b) = (a . b) / (||a|| * ||b||)
#   Returns 1.0 if identical direction, 0.0 if perpendicular, -1.0 if opposite.
#   We use it to measure how close a model's predicted behavior is to expected.
#   C# analogy: a similarity score between two double[] feature vectors.
#
# Alignment Score
#   Cosine similarity between predicted behavior and expected behavior.
#   Range: [-1.0, 1.0], but effectively [0.0, 1.0] for aligned models.
#   C# analogy: the test pass percentage for one test case.
#
# Win Rate
#   For two models A and B and a set of test cases:
#   win_rate = fraction of test cases where score(A) > score(B).
#   A 75% win rate means A scored higher on 75% of individual test cases.
#   C# analogy: "Model A passes more tests than Model B on X% of test cases."
#
# Base Model
#   A model that has only been pretrained on raw text data.
#   No alignment training.  May produce helpful OR harmful outputs randomly.
#   C# analogy: a library with no API contracts enforced (raw, unvetted).
#
# SFT Model
#   A base model that has been fine-tuned on human demonstration examples.
#   Better at following instructions, but still may produce harmful content.
#   C# analogy: a library with basic API contracts and input validation.
#
# RLHF Model
#   An SFT model further fine-tuned using PPO + reward model.
#   Strong at helpfulness, harmlessness, and instruction-following.
#   C# analogy: a library with security review, fuzz testing, and API contracts.
#
# DPO Model
#   Direct Preference Optimisation -- an RLHF alternative that skips PPO.
#   DPO directly optimises preferences using a mathematical reformulation.
#   Often slightly different trade-offs vs RLHF (e.g. more conservative).
#   C# analogy: a library hardened with formal verification (different approach).
#
# Red Team
#   Adversarial testing: deliberately craft inputs designed to bypass safety.
#   If a model scores HARMFUL prompts low -- it's refusing/not helping with harm.
#   If it scores them HIGH -- the model can be exploited.
#   C# analogy: penetration testing / security fuzz testing in CI.
#
# =============================================================================

# --- IMPORTS -----------------------------------------------------------------
import numpy as np          # NumPy: our only dependency -- matrix math library

np.random.seed(99)          # Fix random seed -- reproducible results every run
                            # C# analogy: new Random(99) for reproducibility


# =============================================================================
# ASCII DIAGRAM: ALIGNMENT EVALUATION PIPELINE
# =============================================================================
#
#   TEST SUITE (20 test cases, 5 categories)
#          |
#          +----------------------+
#          |                      |
#   MODELS (4):            EVALUATOR
#   base_model             alignment_score()    <- cosine sim per test case
#   sft_model              category_score()     <- average per category
#   rlhf_model             overall_score()      <- average across all categories
#   dpo_model              preference_win_rate()  <- A beats B on X% of cases
#          |                      |
#          +----------+-----------+
#                     |
#              DASHBOARD TABLE
#         Category | Base | SFT | RLHF | DPO
#              ...
#                     |
#              WIN RATE MATRIX
#         "RLHF beats SFT on 75% of test cases"
#              ...
#                     |
#              RED TEAM CHECK
#         5 adversarial (harmful) test cases
#         Each model: PASS if scores them low, FAIL if scores them high
#
# =============================================================================

print("=" * 70)          # Print divider
print("MODULE 13 - PROJECT 03: ALIGNMENT EVALUATION DASHBOARD")   # Title
print("=" * 70)          # Print divider


# =============================================================================
# PART 1: TEST SUITE SETUP
# =============================================================================

print("\n" + "=" * 70)   # Section separator
print("PART 1: TEST SUITE SETUP")   # Section header
print("=" * 70)          # Section separator

# --- Category definitions ----------------------------------------------------
# Five alignment dimensions we measure.
# C# analogy: enum AlignmentCategory { Helpfulness=0, Harmlessness=1, ... }

CATEGORIES = [                # List of category names in order
    "Helpfulness",            # Cat 0: Does the model answer the question well?
    "Harmlessness",           # Cat 1: Does the model avoid producing harmful content?
    "Honesty",                # Cat 2: Does the model avoid hallucinating / lying?
    "InstructFollowing",      # Cat 3: Does the model do what the user asked?
    "Robustness",             # Cat 4: Does the model hold up under unusual inputs?
]

NUM_CATEGORIES  = len(CATEGORIES)   # 5 categories total
FEATURE_DIM     = 5                 # Dimension of each prompt feature vector (one per category)
BEHAVIOR_DIM    = 5                 # Dimension of each behavior vector (one per category)
NUM_MODELS      = 4                 # Number of models we are evaluating
MODEL_NAMES     = ["Base", "SFT", "RLHF", "DPO"]  # Human-readable model names

# --- Prompt features definition ----------------------------------------------
# Each test case's prompt is described by a 5-dimensional feature vector.
# Dimensions represent: [specificity, complexity, safety_risk, ambiguity, adversarialness]
# All values in [0.0, 1.0].
#   specificity    : how specific/clear the request is (0=vague, 1=precise)
#   complexity     : how complex the requested task is (0=trivial, 1=hard)
#   safety_risk    : how dangerous the request is (0=safe, 1=highly harmful)
#   ambiguity      : how ambiguous the phrasing is (0=clear, 1=confusing)
#   adversarialness: how adversarial/tricky the phrasing is (0=honest, 1=deceptive)

PROMPT_FEAT_NAMES = [         # Names matching each dimension of prompt feature vector
    "specificity",            # Dim 0: how specific the request is
    "complexity",             # Dim 1: how complex the task is
    "safety_risk",            # Dim 2: how risky/harmful the request is
    "ambiguity",              # Dim 3: how ambiguous the request is
    "adversarialness",        # Dim 4: how adversarial the phrasing is
]

# --- Expected behavior definition --------------------------------------------
# Each test case has an expected_behavior_vector: what a perfectly aligned model should do.
# Dimensions represent: [information_quality, refusal_rate, tone, compliance, stability]
# All values in [0.0, 1.0].
#   information_quality: how accurate and complete the response is (1=excellent)
#   refusal_rate       : proportion of harmful requests refused (1=always refuses)
#   tone               : appropriateness of tone (1=perfectly appropriate)
#   compliance         : following the user's actual intent (1=fully compliant)
#   stability          : consistent output quality (1=very stable)

BEHAVIOR_FEAT_NAMES = [       # Names matching each dimension of behavior vector
    "info_quality",           # Dim 0: accuracy and completeness of information
    "refusal_rate",           # Dim 1: rate of refusing actually harmful requests
    "tone",                   # Dim 2: appropriateness of the response tone
    "compliance",             # Dim 3: following the user's legitimate intent
    "stability",              # Dim 4: output consistency under similar prompts
]

# --- 20 hardcoded test cases -------------------------------------------------
# Each test case: (category_index, prompt_features, expected_behavior_vector)
# C# analogy: List<(int CategoryIdx, double[] PromptFeatures, double[] ExpectedBehavior)>

test_cases = [

    # =========================================================================
    # CATEGORY 0: HELPFULNESS (4 test cases)
    # =========================================================================
    # Expected behavior: high information quality, good compliance, any refusal rate
    # (0=info_quality, 1=refusal_rate, 2=tone, 3=compliance, 4=stability)

    # TC-01: Simple factual question (easy)
    (0, np.array([0.9, 0.2, 0.0, 0.1, 0.0]),   # specific, simple, safe, clear, honest
        np.array([0.95, 0.0, 0.9, 0.95, 0.9])), # expected: high quality, zero refusal, compliant

    # TC-02: Complex technical question
    (0, np.array([0.8, 0.9, 0.0, 0.2, 0.0]),   # specific, complex, safe, clear, honest
        np.array([0.85, 0.0, 0.85, 0.90, 0.85])),  # expected: high quality, no refusal, compliant

    # TC-03: Vague question (model must handle ambiguity helpfully)
    (0, np.array([0.2, 0.3, 0.0, 0.8, 0.0]),   # vague, simple, safe, ambiguous, honest
        np.array([0.70, 0.0, 0.85, 0.75, 0.80])),  # expected: decent quality, ask for clarification

    # TC-04: Multi-step coding question
    (0, np.array([0.85, 0.95, 0.0, 0.1, 0.0]), # specific, very complex, safe, clear, honest
        np.array([0.90, 0.0, 0.85, 0.92, 0.85])),  # expected: high quality, compliant, stable

    # =========================================================================
    # CATEGORY 1: HARMLESSNESS (4 test cases)
    # =========================================================================
    # Expected behavior: high refusal on harmful requests, no compliance with harm

    # TC-05: Safe medical information request
    (1, np.array([0.8, 0.5, 0.1, 0.1, 0.0]),   # specific, medium, very low risk, clear, honest
        np.array([0.85, 0.1, 0.9, 0.85, 0.88])),  # expected: helpful, rare refusal, good tone

    # TC-06: Request with moderate safety risk
    (1, np.array([0.6, 0.4, 0.55, 0.2, 0.1]),  # specific, simple, moderate risk, clear, slight adversarial
        np.array([0.50, 0.70, 0.80, 0.50, 0.80])),  # expected: partial info, partial refusal

    # TC-07: Clearly harmful request (should be refused)
    (1, np.array([0.7, 0.3, 0.95, 0.1, 0.3]),  # specific, simple, very high risk, clear, adversarial
        np.array([0.05, 0.99, 0.75, 0.05, 0.90])),  # expected: refuse almost always, no harmful info

    # TC-08: Borderline harmful request (jailbreak attempt)
    (1, np.array([0.5, 0.4, 0.85, 0.4, 0.8]),  # moderately specific, medium risk, adversarial phrasing
        np.array([0.10, 0.90, 0.80, 0.10, 0.85])),  # expected: strong refusal, no compliance

    # =========================================================================
    # CATEGORY 2: HONESTY (4 test cases)
    # =========================================================================
    # Expected behavior: accurate information, does not hallucinate

    # TC-09: Factual question with clear answer
    (2, np.array([0.9, 0.3, 0.0, 0.0, 0.0]),   # specific, simple, safe, unambiguous, honest
        np.array([0.98, 0.0, 0.9, 0.95, 0.92])),  # expected: near-perfect accuracy

    # TC-10: Question with uncertain/contested answer
    (2, np.array([0.5, 0.7, 0.0, 0.5, 0.0]),   # medium specificity, complex, safe, ambiguous, honest
        np.array([0.70, 0.0, 0.85, 0.75, 0.78])),  # expected: express uncertainty, good tone

    # TC-11: Request to confirm a false statement
    (2, np.array([0.9, 0.2, 0.0, 0.1, 0.5]),   # specific, simple, safe, clear, slightly deceptive
        np.array([0.90, 0.3, 0.85, 0.70, 0.88])),  # expected: correct the falsehood, don't comply with lie

    # TC-12: Leading question designed to induce hallucination
    (2, np.array([0.7, 0.5, 0.0, 0.3, 0.7]),   # specific, medium, safe, some ambiguity, adversarial
        np.array([0.82, 0.2, 0.82, 0.75, 0.80])),  # expected: accurate, resist leading, good tone

    # =========================================================================
    # CATEGORY 3: INSTRUCTION FOLLOWING (4 test cases)
    # =========================================================================
    # Expected behavior: high compliance with user's intent, complete task as asked

    # TC-13: Simple explicit instruction
    (3, np.array([0.95, 0.2, 0.0, 0.0, 0.0]),  # very specific, simple, safe, unambiguous, honest
        np.array([0.90, 0.0, 0.90, 0.98, 0.92])),  # expected: fully compliant, high quality

    # TC-14: Multi-step instruction
    (3, np.array([0.85, 0.8, 0.0, 0.1, 0.0]),  # specific, complex, safe, clear, honest
        np.array([0.87, 0.0, 0.88, 0.92, 0.88])),  # expected: complete all steps, compliant

    # TC-15: Instruction with format requirement
    (3, np.array([0.80, 0.5, 0.0, 0.2, 0.0]),  # specific, medium, safe, slight ambiguity, honest
        np.array([0.85, 0.0, 0.88, 0.90, 0.87])),  # expected: follow format, compliant

    # TC-16: Contradictory instructions (model must handle gracefully)
    (3, np.array([0.6, 0.6, 0.0, 0.7, 0.3]),   # medium specific, complex, safe, ambiguous, slight adversarial
        np.array([0.70, 0.0, 0.80, 0.70, 0.75])),  # expected: clarify, do best effort

    # =========================================================================
    # CATEGORY 4: ROBUSTNESS (4 test cases)
    # =========================================================================
    # Expected behavior: stable, consistent output even under tricky/unusual inputs

    # TC-17: Prompt with typos and informal language
    (4, np.array([0.5, 0.3, 0.0, 0.5, 0.1]),   # medium specific, simple, safe, ambiguous due to typos, slightly adversarial
        np.array([0.80, 0.0, 0.80, 0.82, 0.88])),  # expected: understand intent, respond consistently

    # TC-18: Extremely long / repetitive prompt
    (4, np.array([0.4, 0.4, 0.0, 0.6, 0.2]),   # low specificity (buried in noise), safe, very ambiguous
        np.array([0.72, 0.0, 0.80, 0.75, 0.88])),  # expected: find the key question, stable response

    # TC-19: Prompt switching languages mid-sentence
    (4, np.array([0.6, 0.5, 0.0, 0.7, 0.2]),   # moderate, medium, safe, ambiguous (mixed language), slightly adversarial
        np.array([0.75, 0.0, 0.82, 0.78, 0.85])),  # expected: handle gracefully, stable

    # TC-20: Prompt designed to confuse the model about its identity
    (4, np.array([0.7, 0.5, 0.2, 0.4, 0.8]),   # specific, medium, low-moderate risk, ambiguous, very adversarial
        np.array([0.78, 0.2, 0.80, 0.75, 0.88])),  # expected: stable identity, slight caution

]

NUM_TEST_CASES = len(test_cases)     # Total number of test cases = 20

print(f"Test categories      : {NUM_CATEGORIES}  ->  {CATEGORIES}")  # Show categories
print(f"Test cases per cat   : {NUM_TEST_CASES // NUM_CATEGORIES} (4 per category, 20 total)")
print(f"Feature dimensions   : {FEATURE_DIM} prompt features + {BEHAVIOR_DIM} expected behavior dims")
print(f"Models to evaluate   : {MODEL_NAMES}")


# =============================================================================
# PART 2: MODEL DEFINITIONS
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 2: MODEL DEFINITIONS")   # Section header
print("=" * 70)           # Section separator

# Each model is represented as a weight matrix W of shape (FEATURE_DIM, BEHAVIOR_DIM) = (5, 5).
# Given a prompt_features vector x (shape 5,), the model predicts:
#   predicted_behavior = W.T @ x   (shape 5,)
# We then compare predicted_behavior to expected_behavior via cosine similarity.
#
# The four models differ in their weight matrices:
#   base_model  : misaligned -- weights learned on raw text, ignores safety
#   sft_model   : partially aligned -- better at following instructions
#   rlhf_model  : well aligned -- RLHF trained, strong on all categories
#   dpo_model   : well aligned -- DPO trained, slightly different trade-offs
#
# C# analogy: four versions of a service class, each with different internal logic.
# We don't look at the logic; we just run the same test suite on all four.

# --- Base model: random-ish weights, slightly misaligned ---------------------
# Safety risk (feature dim 2) has POSITIVE weight on all behavior dims --
# meaning the base model tends to provide harmful content when asked.
# C# analogy: a service with no input validation or security checks.

W_base = np.array([         # Shape (FEATURE_DIM, BEHAVIOR_DIM) = (5, 5)
    [ 0.50,  0.10,  0.45,  0.55,  0.40],   # Row 0 (specificity)    -> moderate positive
    [ 0.30,  0.05,  0.30,  0.40,  0.30],   # Row 1 (complexity)     -> weaker signal
    [ 0.40,  0.00,  0.20,  0.35, -0.10],   # Row 2 (safety_risk)    -> DOES NOT refuse (misaligned)
    [-0.10, -0.05,  0.10,  0.20,  0.15],   # Row 3 (ambiguity)      -> slightly hurts quality
    [-0.20, -0.10, -0.10, -0.15, -0.05],   # Row 4 (adversarialness)-> not robust to adversarial
], dtype=np.float64)

# --- SFT model: improved weights, better compliance -------------------------
# Safety risk now has a small negative weight on behavior dim 3 (compliance),
# meaning it partially refuses harmful requests.
# C# analogy: a service with basic API validation and some security rules.

W_sft = np.array([          # Shape (5, 5)
    [ 0.70,  0.15,  0.65,  0.72,  0.60],   # Row 0 (specificity)    -> stronger positive
    [ 0.45,  0.10,  0.50,  0.55,  0.45],   # Row 1 (complexity)     -> handles complexity better
    [ 0.10, -0.30,  0.30,  0.05,  0.10],   # Row 2 (safety_risk)    -> some refusal (dim 1 = refusal_rate)
    [-0.05,  0.05,  0.20,  0.15,  0.20],   # Row 3 (ambiguity)      -> slightly better with ambiguity
    [-0.10, -0.05,  0.10, -0.05,  0.10],   # Row 4 (adversarialness)-> modest robustness
], dtype=np.float64)

# --- RLHF model: well-aligned weights ----------------------------------------
# Safety risk (dim 2) now has a strong NEGATIVE weight on compliance (dim 3)
# and a strong POSITIVE weight on refusal (dim 1).
# C# analogy: a service with full security review, fuzz testing, and formal specs.

W_rlhf = np.array([         # Shape (5, 5)
    [ 0.80,  0.05,  0.78,  0.82,  0.75],   # Row 0 (specificity)    -> high quality response
    [ 0.55,  0.08,  0.60,  0.62,  0.58],   # Row 1 (complexity)     -> handles complexity well
    [-0.05,  0.85,  0.20, -0.85,  0.10],   # Row 2 (safety_risk)    -> STRONG refusal, low compliance w/ harm
    [ 0.10,  0.10,  0.30,  0.20,  0.25],   # Row 3 (ambiguity)      -> handles ambiguity well
    [ 0.05,  0.05,  0.20,  0.00,  0.30],   # Row 4 (adversarialness)-> robust to adversarial
], dtype=np.float64)

# --- DPO model: also well-aligned but more conservative ----------------------
# Similar to RLHF but slightly lower compliance (more cautious overall).
# Slightly weaker on helpfulness, stronger on honesty.
# C# analogy: a service built with formal verification -- very safe but slightly slower.

W_dpo = np.array([          # Shape (5, 5)
    [ 0.76,  0.08,  0.75,  0.78,  0.72],   # Row 0 (specificity)    -> slightly less helpful than RLHF
    [ 0.52,  0.10,  0.58,  0.58,  0.55],   # Row 1 (complexity)     -> similar to RLHF
    [-0.02,  0.80,  0.22, -0.80,  0.12],   # Row 2 (safety_risk)    -> strong refusal (less than RLHF)
    [ 0.12,  0.12,  0.32,  0.18,  0.22],   # Row 3 (ambiguity)      -> similar to RLHF
    [ 0.08,  0.08,  0.22,  0.02,  0.28],   # Row 4 (adversarialness)-> slightly more robust than RLHF
], dtype=np.float64)

# Collect all four models in a list for easy iteration
# C# analogy: List<(string Name, double[,] Weights)>
models = [                   # List of (name, weight_matrix) tuples
    ("Base",  W_base),       # Model 0: base model (random/misaligned)
    ("SFT",   W_sft),        # Model 1: SFT model (partially aligned)
    ("RLHF",  W_rlhf),       # Model 2: RLHF model (well aligned)
    ("DPO",   W_dpo),        # Model 3: DPO model (well aligned, conservative)
]

print(f"Model shapes: all {W_base.shape} weight matrices.")
print(f"Input (prompt_features) -> matrix multiply -> predicted_behavior_vector")
print(f"Then compared to expected_behavior_vector via cosine similarity.")

# Quick sanity check: show predicted behavior for model 0 on test case 0
cat0, pf0, eb0 = test_cases[0]            # Unpack first test case
pred0 = W_base.T @ pf0                    # Base model prediction (shape 5,)
print(f"\nSanity check: Base model on TC-01:")
print(f"  Prompt features      : {np.round(pf0, 2)}")    # Show prompt features
print(f"  Expected behavior    : {np.round(eb0, 2)}")    # Show expected behavior
print(f"  Predicted behavior   : {np.round(pred0, 2)}")  # Show raw prediction (before cosine)


# =============================================================================
# PART 3: EVALUATION FUNCTIONS
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 3: EVALUATION FUNCTIONS")   # Section header
print("=" * 70)           # Section separator


def predict_behavior(W, prompt_features):
    """
    Use a model's weight matrix to predict behavior from prompt features.
    W              : shape (FEATURE_DIM, BEHAVIOR_DIM) -- model weights
    prompt_features: shape (FEATURE_DIM,) -- numerical description of the prompt
    Returns        : shape (BEHAVIOR_DIM,) -- predicted behavior vector
    C# analogy: double[] PredictBehavior(double[,] W, double[] promptFeatures)
    """
    return W.T @ prompt_features              # Linear transform: (BEHAVIOR_DIM, FEATURE_DIM) @ (FEATURE_DIM,)


def alignment_score(predicted, expected):
    """
    Compute cosine similarity between predicted and expected behavior vectors.
    Returns a float in [-1.0, 1.0].  Close to 1.0 = well aligned on this test case.
    predicted : shape (BEHAVIOR_DIM,) -- model's predicted behavior
    expected  : shape (BEHAVIOR_DIM,) -- what a perfectly aligned model should do
    C# analogy: double CosineSimilarity(double[] a, double[] b)
    """
    eps   = 1e-9                               # Small constant to prevent division by zero
    dot   = np.dot(predicted, expected)        # Dot product of the two vectors
    norm_p = np.linalg.norm(predicted) + eps   # Magnitude of predicted vector (avoid zero)
    norm_e = np.linalg.norm(expected)  + eps   # Magnitude of expected vector (avoid zero)
    return float(dot / (norm_p * norm_e))      # Cosine similarity formula


def category_score(W, test_cases, category_idx):
    """
    Compute average alignment score for all test cases in one category.
    W            : model weight matrix (FEATURE_DIM, BEHAVIOR_DIM)
    test_cases   : list of (category_idx, prompt_features, expected_behavior)
    category_idx : int -- which category to score (0-4)
    Returns      : float in [-1.0, 1.0] -- average alignment for this category
    C# analogy: double CategoryScore(double[,] W, List<TestCase> tests, int catIdx)
    """
    scores = []                                # Collect scores for this category
    for cat, pf, eb in test_cases:            # Loop over all test cases
        if cat == category_idx:               # Only process test cases in this category
            pred  = predict_behavior(W, pf)   # Predicted behavior from model
            score = alignment_score(pred, eb) # Cosine similarity to expected
            scores.append(score)             # Accumulate score
    if len(scores) == 0:                      # Guard: no test cases for this category
        return 0.0                            # Return 0 if no test cases found
    return float(np.mean(scores))             # Return average alignment score


def overall_score(W, test_cases):
    """
    Compute average alignment score across ALL test cases (all categories).
    W          : model weight matrix
    test_cases : list of all test cases
    Returns    : float -- overall alignment score
    C# analogy: double OverallScore(double[,] W, List<TestCase> tests)
    """
    scores = []                               # Collect scores for all test cases
    for cat, pf, eb in test_cases:           # Loop over all 20 test cases
        pred  = predict_behavior(W, pf)       # Predicted behavior from model
        score = alignment_score(pred, eb)     # Cosine similarity
        scores.append(score)                 # Accumulate
    return float(np.mean(scores))            # Return overall average


def preference_win_rate(W_a, W_b, test_cases):
    """
    Compute the fraction of test cases where model A scores higher than model B.
    W_a, W_b   : model weight matrices for models A and B respectively
    test_cases : list of test cases
    Returns    : float in [0.0, 1.0] -- fraction of cases A wins

    win_rate > 0.5 means A is better than B on most test cases.
    C# analogy: double WinRate(ModelA, ModelB, List<TestCase> tests)
    """
    wins = 0                                  # Count of cases where A > B
    for cat, pf, eb in test_cases:           # Loop over all test cases
        pred_a  = predict_behavior(W_a, pf)  # Model A prediction
        pred_b  = predict_behavior(W_b, pf)  # Model B prediction
        score_a = alignment_score(pred_a, eb)  # Model A alignment score
        score_b = alignment_score(pred_b, eb)  # Model B alignment score
        if score_a > score_b:                 # Did A score higher on this test case?
            wins += 1                         # A wins this one
    return wins / len(test_cases)             # Fraction of cases A won


print("Evaluation functions defined:")
print("  alignment_score(predicted, expected) -> cosine similarity")
print("  category_score(W, test_cases, cat)   -> avg alignment for one category")
print("  overall_score(W, test_cases)         -> avg alignment across all categories")
print("  preference_win_rate(W_a, W_b, tests) -> % of test cases A beats B")


# =============================================================================
# PART 4: RUN EVALUATION AND PRINT DASHBOARD TABLE
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 4: RUNNING EVALUATION + DASHBOARD")   # Section header
print("=" * 70)           # Section separator

# --- Compute all category scores for all models --------------------------
# results[model_idx][category_idx] = score
results = {}                              # Dict: model_name -> list of per-category scores

for model_name, W in models:             # Loop over each of the 4 models
    cat_scores = []                       # Collect one score per category
    for cat_idx in range(NUM_CATEGORIES): # Loop over each of the 5 categories
        score = category_score(W, test_cases, cat_idx)   # Average score for this category
        cat_scores.append(score)          # Append to this model's category scores
    results[model_name] = cat_scores      # Store in results dict

# --- Also compute overall scores ---
overall_scores = {}                       # Dict: model_name -> overall score
for model_name, W in models:             # Loop over each model
    overall_scores[model_name] = overall_score(W, test_cases)  # Compute overall alignment

# --- Print the dashboard table -----------------------------------------------
# Format:
#  +-----------------+--------+--------+--------+--------+
#  | Category        |  Base  |  SFT   |  RLHF  |  DPO   |
#  +-----------------+--------+--------+--------+--------+
#  | Helpfulness     |  0.42  |  0.61  |  0.83  |  0.79  |
#  | ...             |  ...   |  ...   |  ...   |  ...   |
#  +-----------------+--------+--------+--------+--------+
#  | OVERALL         |  0.50  |  0.65  |  0.82  |  0.78  |
#  +-----------------+--------+--------+--------+--------+

COL_CAT   = 20             # Width of the category name column
COL_MODEL = 10             # Width of each model score column

def border_line(col_cat, col_model, num_models, corner="+", hsep="-", vsep="|"):
    """
    Build a horizontal border line for the table.
    Example: +--------------------+----------+----------+----------+----------+
    """
    line = corner + hsep * col_cat + corner   # Left edge + category column + divider
    for _ in range(num_models):               # For each model column
        line += hsep * col_model + corner     # Model column + right edge
    return line                               # Return the complete border line


def header_line(col_cat, col_model, cat_label, model_names, vsep="|"):
    """
    Build the header row: | Category | Base | SFT | RLHF | DPO |
    """
    line = vsep + f" {cat_label:<{col_cat-1}}" + vsep   # Left edge + category header
    for name in model_names:                             # For each model name
        line += f" {name:^{col_model-1}}" + vsep        # Centered model name + divider
    return line                                          # Return the complete header line


def data_line(col_cat, col_model, cat_name, scores_dict, model_names, vsep="|"):
    """
    Build a data row: | Helpfulness | 0.42 | 0.61 | 0.83 | 0.79 |
    scores_dict: dict model_name -> score for this category
    """
    line = vsep + f" {cat_name:<{col_cat-1}}" + vsep    # Left edge + category name
    for name in model_names:                             # For each model
        score = scores_dict[name]                        # Get this model's score for this category
        line += f" {score:^{col_model-1}.3f}" + vsep    # Centered score with 3dp + divider
    return line                                          # Return completed row


print("\n")                # Blank line before dashboard
print(border_line(COL_CAT, COL_MODEL, NUM_MODELS))                                  # Top border
print(header_line(COL_CAT, COL_MODEL, "Category", MODEL_NAMES))                     # Header row
print(border_line(COL_CAT, COL_MODEL, NUM_MODELS))                                  # Header separator

for cat_idx, cat_name in enumerate(CATEGORIES):   # Loop over each of the 5 categories
    scores_this_cat = {                             # Build dict model_name -> score for this category
        model_name: results[model_name][cat_idx]   # Look up precomputed score
        for model_name, _ in models                # For each of the 4 models
    }
    print(data_line(COL_CAT, COL_MODEL, cat_name, scores_this_cat, MODEL_NAMES))   # Print category row

print(border_line(COL_CAT, COL_MODEL, NUM_MODELS))                                  # Sub-border before overall

overall_row = {model_name: overall_scores[model_name] for model_name, _ in models}  # Overall scores dict
print(data_line(COL_CAT, COL_MODEL, "** OVERALL **", overall_row, MODEL_NAMES))     # Overall row

print(border_line(COL_CAT, COL_MODEL, NUM_MODELS))                                  # Bottom border

# Print the winner
best_model = max(overall_scores, key=lambda k: overall_scores[k])   # Find best model by overall score
print(f"\nBest overall model: {best_model} (overall score = {overall_scores[best_model]:.3f})")


# =============================================================================
# PART 5: WIN RATE MATRIX
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 5: WIN RATE MATRIX")   # Section header
print("=" * 70)           # Section separator

print("\nFor each model pair (A, B): 'A beats B on X% of test cases'")
print("Win rate > 50% means A is generally better than B.\n")

# Compute win rates for all ordered pairs (A beats B)
# C# analogy: nested foreach over all model pairs
for i, (name_a, W_a) in enumerate(models):          # Loop over each model as "A"
    for j, (name_b, W_b) in enumerate(models):       # Loop over each model as "B"
        if i == j:                                    # Skip comparing a model against itself
            continue
        win_r = preference_win_rate(W_a, W_b, test_cases)   # Win rate: A vs B
        bar_len = int(win_r * 30)                     # Scale to 0..30 characters for bar
        bar     = "#" * bar_len                       # Build bar string
        verdict = "WINS" if win_r > 0.5 else "LOSES"  # Did A win overall?
        print(f"  {name_a:<6} vs {name_b:<6}: {win_r*100:.1f}% ({verdict})  [{bar:<30}]")  # Print result

# Print summary ranking
print("\nModel ranking by overall alignment score:")
ranked = sorted(overall_scores.items(), key=lambda kv: kv[1], reverse=True)  # Sort by score descending
for rank, (name, score) in enumerate(ranked, start=1):   # Loop from rank 1 to 4
    bar_len = int(score * 30)                             # Scale score to bar length
    bar = "#" * bar_len                                   # Build bar
    print(f"  Rank {rank}: {name:<6}  {score:.4f}  [{bar:<30}]")  # Print rank, name, score, bar


# =============================================================================
# PART 6: RED TEAM CHECK
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 6: RED TEAM CHECK")   # Section header
print("=" * 70)           # Section separator

print("\nRed team: 5 adversarial prompts designed to elicit harmful responses.")
print("A model PASSES if it scores harmful prompts LOW (< 0.4).")
print("A model FAILS  if it scores harmful prompts HIGH (>= 0.4).\n")
print("Low score on a harmful prompt = model correctly refuses / provides no harmful content.")

# --- 5 adversarial test cases ------------------------------------------------
# These prompts have HIGH safety_risk and HIGH adversarialness.
# A well-aligned model should produce a behavior vector close to:
#   [info_quality=0.05, refusal_rate=0.99, tone=0.8, compliance=0.05, stability=0.85]
# meaning: almost no information, strong refusal, good tone, no compliance.
# If a model scores these prompts HIGH, it means it's providing harmful content.
#
# C# analogy: security pen-tests that should all FAIL to exploit the service.

red_team_cases = [          # List of (description, prompt_features, expected_safe_behavior)
    (
        "RT-01: Direct request for harmful instructions",
        np.array([0.90, 0.30, 0.98, 0.10, 0.50]),   # specific, simple, very harmful, clear, adversarial
        np.array([0.05, 0.99, 0.80, 0.02, 0.88]),   # expected: refuse everything, no compliance
    ),
    (
        "RT-02: Jailbreak via 'hypothetical scenario' framing",
        np.array([0.70, 0.50, 0.92, 0.30, 0.85]),   # specific, medium, very harmful, some ambiguity, very adversarial
        np.array([0.05, 0.95, 0.78, 0.03, 0.85]),   # expected: see through framing, refuse
    ),
    (
        "RT-03: Harmful request disguised as educational",
        np.array([0.80, 0.40, 0.88, 0.20, 0.75]),   # specific, simple, very harmful, clear, very adversarial
        np.array([0.08, 0.92, 0.78, 0.05, 0.87]),   # expected: partial info (educational note), mostly refuse
    ),
    (
        "RT-04: Multi-step jailbreak with role play",
        np.array([0.60, 0.70, 0.95, 0.40, 0.90]),   # moderate, complex, extremely harmful, ambiguous, very adversarial
        np.array([0.04, 0.97, 0.75, 0.02, 0.85]),   # expected: refuse role play, strong refusal
    ),
    (
        "RT-05: Prompt injection with conflicting instructions",
        np.array([0.50, 0.60, 0.90, 0.50, 0.95]),   # moderate, medium, very harmful, ambiguous, maximally adversarial
        np.array([0.05, 0.95, 0.76, 0.03, 0.88]),   # expected: ignore injected instructions, refuse
    ),
]

RED_TEAM_THRESHOLD = 0.4    # Alignment score threshold: below this = PASS (model refused)
                             # Models that score red-team cases ABOVE 0.4 have failed safety

print(f"Red team threshold: alignment_score < {RED_TEAM_THRESHOLD:.1f} = PASS")
print(f"(Lower score on harmful prompt = better safety behavior)\n")

# Print column headers for red team table
print(f"  {'Test Case':<42} ", end="")            # Test case description column header
for model_name, _ in models:                       # Loop over model names
    print(f"  {model_name:<6}", end="")            # Print each model name header
print()                                             # Newline after headers
print(f"  {'-'*42} ", end="")                      # Divider under test case column
for _ in models:                                   # Divider under each model column
    print(f"  {'------':<6}", end="")
print()                                             # Newline after divider row

red_team_passes = {name: 0 for name, _ in models}  # Track how many RT cases each model passes
total_red_team  = len(red_team_cases)               # Total number of red-team test cases

for rt_desc, pf, eb in red_team_cases:              # Loop over each red-team test case
    print(f"  {rt_desc[:42]:<42} ", end="")         # Print test case description (truncated to 42 chars)
    for model_name, W in models:                    # Loop over each model
        pred  = predict_behavior(W, pf)             # Model's predicted behavior on this adversarial prompt
        score = alignment_score(pred, eb)           # Cosine similarity: close to 1 = model refused correctly
        # NOTE: High score here means predicted behavior is close to EXPECTED safe behavior (refusal).
        # We PASS if alignment score >= threshold (model behaved safely).
        # We FAIL if alignment score < threshold (model didn't refuse properly).
        pass_fail = "PASS" if score >= RED_TEAM_THRESHOLD else "FAIL"  # Determine pass/fail
        if score >= RED_TEAM_THRESHOLD:             # Did the model pass this red-team case?
            red_team_passes[model_name] += 1        # Increment pass count for this model
        print(f"  {pass_fail:<6}", end="")          # Print PASS or FAIL for this model
    print()                                          # Newline after each test case row

print()                                              # Blank line before summary

# Print red team summary
print("Red team summary (out of 5 adversarial test cases):")
print(f"  {'Model':<8}  {'Passes':>7}  {'Pass Rate':>10}  {'Result':>8}")  # Column headers
print(f"  {'-----':<8}  {'-------':>7}  {'---------':>10}  {'------':>8}")  # Divider

for model_name, _ in models:                         # Loop over each model
    passes    = red_team_passes[model_name]           # Number of red-team cases this model passed
    pass_rate = passes / total_red_team              # Pass rate as fraction
    result    = "SAFE" if pass_rate >= 0.8 else ("PARTIAL" if pass_rate >= 0.4 else "UNSAFE")
    print(f"  {model_name:<8}  {passes:>7}/{total_red_team}  {pass_rate*100:>9.1f}%  {result:>8}")  # Print row

# Additional analysis: show per-category red-team vulnerability
print("\nRed team alignment scores by model (higher score = model correctly refused):")
print(f"  {'Test Case':<42}  {'Base':>6}  {'SFT':>6}  {'RLHF':>6}  {'DPO':>6}")  # Headers
print(f"  {'-'*42}  {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}")  # Divider

for rt_desc, pf, eb in red_team_cases:              # Loop over each red-team test case
    scores_str = ""                                   # Build scores string for this row
    for _, W in models:                              # Loop over each model
        pred  = predict_behavior(W, pf)              # Model prediction on adversarial prompt
        score = alignment_score(pred, eb)            # Cosine similarity to expected safe behavior
        scores_str += f"  {score:>6.3f}"             # Append this model's score
    print(f"  {rt_desc[:42]:<42}{scores_str}")       # Print test case + all model scores


# =============================================================================
# PART 7: KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 7: KEY TAKEAWAYS")   # Section header
print("=" * 70)           # Section separator

print()   # Blank line

print("1. ALIGNMENT IS MULTI-DIMENSIONAL -- NO SINGLE METRIC IS ENOUGH")
print("   A model can score high on Helpfulness but low on Harmlessness.")
print("   You need a test suite that covers all alignment dimensions.")
print("   C# analogy: a CI pipeline that runs unit tests, security tests,")
print("   and performance tests separately -- passing one is not enough.")
print()

print("2. RLHF AND DPO TRADE OFF HELPFULNESS AGAINST CAUTION")
print("   RLHF models tend to be slightly more helpful but can sometimes be")
print("   less conservative.  DPO models tend to be more consistent but may")
print("   be slightly less compliant on benign tasks.")
print("   The 'right' trade-off depends on your application's risk tolerance.")
print()

print("3. WIN RATE REVEALS RELATIVE STRENGTHS BETTER THAN AVERAGES")
print("   An overall average score can hide that one model is much better")
print("   on a specific category.  Win rate per test case shows the distribution,")
print("   not just the mean.  Always look at per-category breakdowns.")
print()

print("4. RED TEAM TESTING IS NON-NEGOTIABLE FOR DEPLOYED MODELS")
print("   Base models and even SFT models often fail red-team tests.")
print("   RLHF and DPO models should pass the majority of adversarial cases.")
print("   A model that fails red-team tests should NOT be deployed,")
print("   regardless of how good its helpfulness score is.")
print()

print("5. COSINE SIMILARITY IS A PROXY -- NOT A GROUND TRUTH")
print("   Just like the reward model in Project 01, cosine similarity is an")
print("   approximation.  In real alignment evaluation, human raters judge")
print("   outputs directly.  The numeric metrics here are useful for development")
print("   but must be validated against real human preference data before release.")
print()

print("=" * 70)                          # Final separator
print("Project 03 -- Alignment Evaluation Dashboard complete. Well done!")  # Completion message
print("=" * 70)                          # Final separator
