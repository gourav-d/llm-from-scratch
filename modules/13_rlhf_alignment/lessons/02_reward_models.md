# Lesson 02: Reward Models

## Glossary (Read This First!)

| Term | Plain English Definition |
|------|--------------------------|
| **Reward Model (RM)** | A neural network that takes a (prompt + response) pair and outputs a single number representing "how good is this response?" |
| **Preference Pair** | Two responses to the same prompt, one labeled "chosen" (preferred) and one labeled "rejected" (not preferred). |
| **Bradley-Terry Model** | A statistical model for pairwise comparisons. Predicts the probability that item A is preferred over item B from their scores. |
| **Logit** | The raw, un-normalized output of a neural network before applying sigmoid or softmax. Can be any real number (negative to positive). |
| **Sigmoid** | A function that squashes any number into the range (0, 1). Used to convert logits to probabilities. sigma(x) = 1 / (1 + e^(-x)) |
| **Binary Cross-Entropy** | A loss function for binary (two-class) problems. Measures how wrong a prediction is when the true answer is 0 or 1. |
| **Pairwise Comparison** | Comparing exactly two items against each other, rather than rating each item independently. |
| **Ranking** | Ordering multiple items from best to worst. More than two items. |
| **Reward Score** | The output of the reward model. Higher = this response is preferred by humans. |
| **Golden Response** | The "ideal" human-written response. Used as a reference in evaluation. |
| **Reward Hacking** | When the model finds ways to get a high reward score that do NOT correspond to genuinely good responses. |
| **Calibration** | A measure of whether a model's confidence matches its actual accuracy. Well-calibrated = confident when correct, uncertain when wrong. |

---

## Part 1: What Does a Reward Model Do?

The reward model is a neural network with a specific job:

**Given a prompt and a response, output a number that represents quality.**

```
+---------------------------------------------------------------+
|  REWARD MODEL: INPUT AND OUTPUT                               |
|                                                               |
|  INPUT:                                                       |
|    Prompt:   "What is the capital of France?"                 |
|    Response: "Paris is the capital of France."                |
|                                                               |
|                         |                                     |
|                         v                                     |
|                 +-----------------+                           |
|                 |  REWARD MODEL   |                           |
|                 |  (neural net)   |                           |
|                 +-----------------+                           |
|                         |                                     |
|                         v                                     |
|  OUTPUT:                                                      |
|    Score: 2.34   (high = good response)                       |
|                                                               |
+---------------------------------------------------------------+

+---------------------------------------------------------------+
|  ANOTHER EXAMPLE                                              |
|                                                               |
|  INPUT:                                                       |
|    Prompt:   "What is the capital of France?"                 |
|    Response: "I think maybe Berlin? Or was it London?"        |
|                                                               |
|  OUTPUT:                                                      |
|    Score: -1.87   (low = bad response)                        |
|                                                               |
+---------------------------------------------------------------+
```

The reward model does NOT output:
- A specific quality label ("good", "bad")
- A percentage (well, it can, but it is usually raw logits)
- Explanation for the score

Just a number. Higher is better.

### Why Not Just Use the LLM Itself as a Judge?

You might ask: "Can we just ask the LLM to rate its own responses?"

Yes, but there are problems:
1. The LLM we are trying to improve is biased -- it thinks its own outputs are good
2. There is circular logic: the student is also the grader
3. The reward model is smaller and faster (cheaper to run millions of times during RL)

The reward model is a SEPARATE network, trained for ONE purpose: scoring responses.

---

## Part 2: Training Data Format

The reward model is trained on preference pairs.

Each training example looks like:

```
+---------------------------------------------------------------+
|  TRAINING EXAMPLE FOR REWARD MODEL                            |
|                                                               |
|  {                                                            |
|    "prompt":   "Explain how a compiler works",                |
|                                                               |
|    "chosen":   "A compiler is a program that translates       |
|                 your source code (e.g., C#) into machine      |
|                 code that the CPU can execute directly.        |
|                 It goes through phases: lexing, parsing,      |
|                 semantic analysis, optimization, and           |
|                 code generation.",                            |
|                                                               |
|    "rejected": "The compiler compiles the code."              |
|  }                                                            |
|                                                               |
|  What the reward model MUST learn:                            |
|    score(prompt + chosen)   > score(prompt + rejected)        |
|    i.e., chosen scores HIGHER than rejected                   |
+---------------------------------------------------------------+
```

During training, the reward model sees BOTH responses for the SAME prompt.
It must learn to rank chosen above rejected.

This is fundamentally different from standard supervised learning where
you predict an absolute value (e.g., "this response gets a score of 7/10").

Instead, we are teaching RELATIVE preferences: "A is better than B."

C# analogy:
```csharp
// Standard supervised learning:
// You know the exact "right answer" for each input.
// Like knowing that 2 + 2 = 4.

// Pairwise preference learning:
// You don't know the exact score, but you know which is better.
// Like knowing that "Paris is the capital" is better than
// "Berlin is the capital" without needing to assign exact scores.

// It's like a code review where you say
// "I prefer approach A over approach B"
// without having to score each on a 1-100 scale.
```

---

## Part 3: The Bradley-Terry Model

The mathematical foundation of reward model training is the **Bradley-Terry model**.

This is a classical statistics model for pairwise comparisons.
It was invented long before LLMs -- used for ranking chess players, sports teams, etc.

### The Core Formula

Given two responses A and B with reward scores r_A and r_B:

```
The probability that A is preferred over B is:

  P(A preferred over B) = sigmoid(r_A - r_B)

Where sigmoid(x) = 1 / (1 + e^(-x))
```

Let's see what this means in plain English:

```
+---------------------------------------------------------------+
|  BRADLEY-TERRY INTUITION                                      |
|                                                               |
|  If r_A = 3.0 and r_B = 1.0:                                 |
|    P(A preferred) = sigmoid(3.0 - 1.0)                       |
|                   = sigmoid(2.0)                              |
|                   = 0.88                                      |
|    88% chance humans prefer A.                               |
|                                                               |
|  If r_A = 1.1 and r_B = 1.0:                                 |
|    P(A preferred) = sigmoid(1.1 - 1.0)                       |
|                   = sigmoid(0.1)                              |
|                   = 0.525                                     |
|    52.5% chance humans prefer A (almost a coin flip).        |
|    Very close scores = humans likely to disagree.             |
|                                                               |
|  If r_A = -2.0 and r_B = 2.0:                                |
|    P(A preferred) = sigmoid(-2.0 - 2.0)                      |
|                   = sigmoid(-4.0)                             |
|                   = 0.018                                     |
|    Only 1.8% chance humans prefer A.                         |
|    Clearly B is much better.                                  |
+---------------------------------------------------------------+
```

### The Loss Function

We want to train the reward model to assign high scores to chosen responses.

The loss function is:

```
L = -log( sigmoid( r_chosen - r_rejected ) )
```

Plain English:
- sigma(r_chosen - r_rejected) = predicted probability that chosen is better
- We want this probability to be HIGH (close to 1.0)
- log(close to 1.0) is close to 0 (low loss = good)
- log(close to 0.0) is very negative -> negating gives high loss (bad)

This is binary cross-entropy, applied to pairwise preferences.

```
+---------------------------------------------------------------+
|  LOSS FUNCTION INTUITION                                      |
|                                                               |
|  Perfect prediction:                                          |
|    r_chosen = 5.0, r_rejected = -5.0                         |
|    sigma(5.0 - (-5.0)) = sigma(10) = 0.9999                  |
|    -log(0.9999) = 0.0001  <- very low loss, good!            |
|                                                               |
|  Wrong prediction:                                            |
|    r_chosen = -2.0, r_rejected = 2.0                         |
|    sigma(-2.0 - 2.0) = sigma(-4.0) = 0.018                   |
|    -log(0.018) = 4.02  <- very high loss, bad!               |
|                                                               |
|  Neutral (equal scores):                                      |
|    r_chosen = 0.0, r_rejected = 0.0                          |
|    sigma(0.0 - 0.0) = sigma(0) = 0.5                         |
|    -log(0.5) = 0.693  <- medium loss, model is uncertain     |
+---------------------------------------------------------------+
```

---

## Part 4: Reward Model Architecture

The reward model is NOT built from scratch.
It starts as a copy of the base LLM (or SFT model) with ONE modification.

```
+---------------------------------------------------------------+
|  REWARD MODEL ARCHITECTURE                                    |
|                                                               |
|  BASE LLM:                                                    |
|  [Token Embeddings] -> [Transformer Layers] -> [LM Head]      |
|                                                   |           |
|                                            Vocabulary size    |
|                                          (50,000 outputs)     |
|                                                               |
|  REWARD MODEL:                                                |
|  [Token Embeddings] -> [Transformer Layers] -> [Reward Head]  |
|                                                   |           |
|                                            SINGLE output      |
|                                            (1 scalar score)   |
|                                                               |
|  The only change: replace the language model head            |
|  (which predicts next tokens) with a linear layer            |
|  that outputs ONE number.                                     |
+---------------------------------------------------------------+
```

In code terms, the change is:

```python
# ORIGINAL LLM HEAD:
# A linear layer that maps hidden_size -> vocab_size
# e.g., 768 -> 50257  (outputs probabilities for each token)
lm_head = nn.Linear(hidden_size=768, out_features=50257)

# REWARD MODEL HEAD:
# A linear layer that maps hidden_size -> 1
# Outputs a single score
reward_head = nn.Linear(in_features=768, out_features=1)
```

Why start from the base LLM?
Because the transformer layers already know how to understand language.
We just change what they output: instead of "next token", output "quality score."

This is called **transfer learning** -- reusing knowledge from one task for another.

C# analogy:
```csharp
// Imagine you have a class that parses JSON and returns
// a complex Document object.

// You can REUSE that parser but change what it returns.
// Instead of Document, return a QualityScore (a single double).

// The parsing logic (understanding the structure) is identical.
// Only the output type changes.

public class JsonParser {
    // Original: parses and returns full document
    public Document Parse(string json) { ... }
}

public class JsonQualityScorer : JsonParser {
    // Reuses parsing logic, but returns a score
    public double Score(string json) {
        // Use parent class to understand structure
        // But output just a quality number
        return ComputeScore(base.Parse(json));
    }
}
```

---

## Part 5: Training the Reward Model in Code

Here is a complete, simplified example of reward model training:

```python
# reward_model_training.py
# Simplified reward model training using NumPy
# (Real implementation would use PyTorch, but we show the math clearly)

import numpy as np  # NumPy for array operations

# ============================================================
# STEP 1: Define the sigmoid function
# This converts any real number into a probability (0 to 1)
# ============================================================

def sigmoid(x):
    """
    sigmoid(x) = 1 / (1 + e^(-x))
    
    For x very positive (e.g., 10): sigmoid(10) = 0.9999 (close to 1)
    For x = 0: sigmoid(0) = 0.5 (exactly 50/50)
    For x very negative (e.g., -10): sigmoid(-10) = 0.0001 (close to 0)
    """
    # np.exp is e^x, where e is Euler's number (approx 2.718)
    return 1.0 / (1.0 + np.exp(-x))

# ============================================================
# STEP 2: Define the Bradley-Terry loss function
# This is what we MINIMIZE during training
# ============================================================

def bradley_terry_loss(reward_chosen, reward_rejected):
    """
    Compute the pairwise preference loss.
    
    Args:
        reward_chosen:   scalar score for the preferred response
        reward_rejected: scalar score for the non-preferred response
    
    Returns:
        loss: a non-negative number (lower = better)
    """
    # Compute the probability that chosen is preferred
    # Using Bradley-Terry model: P(chosen > rejected) = sigmoid(r_c - r_r)
    preference_probability = sigmoid(reward_chosen - reward_rejected)
    
    # We want preference_probability to be close to 1.0
    # Use negative log likelihood as loss
    # -log(1.0) = 0   (perfect prediction, zero loss)
    # -log(0.5) = 0.69 (uncertain, medium loss)
    # -log(0.01) = 4.6 (very wrong, high loss)
    
    # np.log is the natural logarithm (base e)
    loss = -np.log(preference_probability)
    
    return loss

# ============================================================
# STEP 3: Test with examples
# ============================================================

print("=== Bradley-Terry Loss Examples ===\n")

# Example 1: Reward model got it right -- chosen scores much higher
r_chosen   = 3.0   # high score for the good response
r_rejected = -1.0  # low score for the bad response
loss = bradley_terry_loss(r_chosen, r_rejected)
prob = sigmoid(r_chosen - r_rejected)
print(f"Chosen score: {r_chosen}, Rejected score: {r_rejected}")
print(f"  P(chosen preferred): {prob:.4f}")
print(f"  Loss: {loss:.4f}")
print(f"  Verdict: {'Good! Model is learning correctly.' if loss < 0.5 else 'Bad! Model needs improvement.'}")
print()

# Example 2: Reward model got it backwards -- rejected scores higher
r_chosen   = -0.5  # low score for the good response (mistake!)
r_rejected =  1.5  # high score for the bad response (mistake!)
loss = bradley_terry_loss(r_chosen, r_rejected)
prob = sigmoid(r_chosen - r_rejected)
print(f"Chosen score: {r_chosen}, Rejected score: {r_rejected}")
print(f"  P(chosen preferred): {prob:.4f}")
print(f"  Loss: {loss:.4f}")
print(f"  Verdict: {'Good!' if loss < 0.5 else 'Bad! Model has it backwards.'}")
print()

# Example 3: Reward model is uncertain -- scores are equal
r_chosen   = 0.5   # same score for both
r_rejected = 0.5   
loss = bradley_terry_loss(r_chosen, r_rejected)
prob = sigmoid(r_chosen - r_rejected)
print(f"Chosen score: {r_chosen}, Rejected score: {r_rejected}")
print(f"  P(chosen preferred): {prob:.4f}")
print(f"  Loss: {loss:.4f}")
print(f"  Verdict: Model is uncertain (50/50 guess)")
print()

# ============================================================
# STEP 4: Simulate one training step (gradient descent)
# ============================================================

print("=== Simulating a Training Step ===\n")

# In reality, the scores come from a neural network.
# Here we simulate by treating them as simple parameters.

# Imagine we have two "score parameters" (normally these are NN outputs)
r_chosen_param   = np.array(0.0)   # starts at 0
r_rejected_param = np.array(0.0)   # starts at 0

# Learning rate: how big a step to take each update
learning_rate = 0.1

# Compute current loss
current_loss = bradley_terry_loss(r_chosen_param, r_rejected_param)
print(f"Before training:")
print(f"  r_chosen: {r_chosen_param:.4f}, r_rejected: {r_rejected_param:.4f}")
print(f"  Loss: {current_loss:.4f}")

# Gradient of loss with respect to (r_chosen - r_rejected):
# d(-log(sigmoid(d))) / dd = sigmoid(d) - 1  where d = r_chosen - r_rejected
# For r_chosen: gradient = sigmoid(r_c - r_r) - 1   (push r_chosen UP)
# For r_rejected: gradient = -(sigmoid(r_c - r_r) - 1) (push r_rejected DOWN)

diff = r_chosen_param - r_rejected_param   # compute difference
p = sigmoid(diff)                           # probability

# Gradients
grad_chosen   = p - 1.0     # We want to DECREASE r_chosen's gradient
                             # (negative * learning_rate = increase r_chosen)
grad_rejected = -(p - 1.0)  # Opposite: push r_rejected down

# Update parameters (gradient descent: move opposite to gradient)
r_chosen_param   = r_chosen_param   - learning_rate * grad_chosen
r_rejected_param = r_rejected_param - learning_rate * grad_rejected

# Compute new loss
new_loss = bradley_terry_loss(r_chosen_param, r_rejected_param)
print(f"\nAfter one training step:")
print(f"  r_chosen: {r_chosen_param:.4f}, r_rejected: {r_rejected_param:.4f}")
print(f"  Loss: {new_loss:.4f}")
print(f"  Improvement: {current_loss - new_loss:.4f}")
print()
print("Notice: r_chosen went UP, r_rejected went DOWN.")
print("The model is learning to separate chosen from rejected!")
```

Expected output:
```
=== Bradley-Terry Loss Examples ===

Chosen score: 3.0, Rejected score: -1.0
  P(chosen preferred): 0.9820
  Loss: 0.0181
  Verdict: Good! Model is learning correctly.

Chosen score: -0.5, Rejected score: 1.5
  P(chosen preferred): 0.1192
  Loss: 2.1269
  Verdict: Bad! Model has it backwards.

Chosen score: 0.5, Rejected score: 0.5
  P(chosen preferred): 0.5000
  Loss: 0.6931
  Verdict: Model is uncertain (50/50 guess)

=== Simulating a Training Step ===

Before training:
  r_chosen: 0.0000, r_rejected: 0.0000
  Loss: 0.6931

After one training step:
  r_chosen: 0.0500, r_rejected: -0.0500
  Loss: 0.6432
  Improvement: 0.0499
```

---

## Part 6: What Makes a Good Reward Model?

A reward model is not just a black box. It needs to have specific properties
to be useful for RLHF.

### Property 1: Calibration

The scores must be meaningful across different prompts.

```
+---------------------------------------------------------------+
|  CALIBRATION EXAMPLE                                          |
|                                                               |
|  WELL-CALIBRATED:                                             |
|  Prompt A, good response:    score = 2.0                      |
|  Prompt A, bad response:     score = 0.3                      |
|  Prompt B, good response:    score = 1.9                      |
|  Prompt B, bad response:     score = 0.2                      |
|                                                               |
|  The score MEANS something. 2.0 > 0.3 consistently.          |
|                                                               |
|  POORLY CALIBRATED:                                           |
|  Prompt A, good response:    score = 2.0                      |
|  Prompt A, bad response:     score = 0.3                      |
|  Prompt B, good response:    score = 0.4  <- PROBLEM!         |
|  Prompt B, bad response:     score = 1.8  <- PROBLEM!         |
|                                                               |
|  Prompt B's scores are reversed! The model is inconsistent.   |
+---------------------------------------------------------------+
```

### Property 2: Generalization

The reward model should work on prompts it has never seen before.
If it only works on training examples, it is useless for RL training.

This is the same generalization challenge as any machine learning model.
More diverse training data = better generalization.

### Property 3: Avoiding Reward Hacking

The reward model should not have exploitable patterns.

Examples of reward hacking discovered in practice:
- Models that write LONGER responses get higher scores (annotators think long = thorough)
- Models that start with "Certainly!" get higher scores (annotators like confidence)
- Models that format with bullet points get higher scores (looks organized)

If these patterns exist in the reward model, the PPO-trained LLM will exploit them:
generating long, bullet-pointed, confident-sounding responses that are ACTUALLY empty.

```
+---------------------------------------------------------------+
|  REWARD HACKING IN PRACTICE                                   |
|                                                               |
|  Human annotation pattern (unconscious bias):                 |
|    Annotators slightly prefer bullet points                   |
|                                                               |
|  Reward model learned this:                                   |
|    "Responses with bullet points -> higher score"             |
|                                                               |
|  PPO-optimized LLM exploited this:                            |
|    Every response now has bullet points, even when wrong:     |
|                                                               |
|  Prompt: "What is 2 + 2?"                                     |
|  Aligned LLM response:                                        |
|    "Great question! Here are the key considerations:          |
|    * Mathematical principles suggest...                        |
|    * Addition operations involve...                            |
|    * The result is 4."                                        |
|                                                               |
|  The reward model gives this a high score.                    |
|  Humans actually find it annoying.                            |
|  The RM failed to generalize.                                 |
+---------------------------------------------------------------+
```

### Property 4: Speed

During PPO training, the reward model is called MILLIONS of times.
For every generated token sequence, it needs to score it.

This means:
- The reward model should be smaller than the LLM being trained
- It is often 10x-100x smaller
- Common choice: if LLM is 7B parameters, reward model is 350M-700M parameters

---

## Part 7: Building a Tiny Reward Model (Conceptual)

Here is a simplified reward model architecture in pseudocode:

```python
# reward_model_architecture.py
# Conceptual demonstration of reward model structure
# (Real code would use PyTorch/HuggingFace)

import numpy as np  # NumPy for array math

# ============================================================
# SIMPLIFIED REWARD MODEL
# In reality this is a full transformer, but we simplify
# to show the key concept: output is 1 number per input.
# ============================================================

class TinyRewardModel:
    """
    A very simplified reward model.
    
    In reality:
    - Input is tokenized text
    - Goes through transformer layers
    - The [EOS] token's representation is projected to a scalar
    
    Here we simulate with a linear function on a feature vector.
    """
    
    def __init__(self, feature_size=5):
        """
        Initialize with random weights.
        
        feature_size: how many features we extract from each response
                      (in reality, this is the hidden_size of the transformer,
                       e.g., 768 or 1024)
        """
        # Initialize weights randomly
        # These are what get updated during training
        np.random.seed(42)  # for reproducibility
        self.weights = np.random.randn(feature_size)  # shape: (feature_size,)
        self.bias    = np.random.randn()               # single number
    
    def extract_features(self, prompt, response):
        """
        Extract features from (prompt, response) pair.
        
        In reality: this is done by the transformer layers.
        Here we use hand-crafted features to illustrate the concept.
        
        Features we check:
        0: Is the response longer than 20 words? (1.0 = yes, 0.0 = no)
        1: Does the response directly answer the question? (simulated)
        2: Does the response contain the word "sorry" (sign of refusal)?
        3: Does the response mention specific facts/numbers?
        4: Is the response's word count reasonable (10-200 words)?
        """
        words = response.split()           # split into words
        word_count = len(words)            # count words
        
        features = np.array([
            1.0 if word_count > 20 else 0.0,                    # feature 0: length check
            1.0 if any(w in response for w in ["because", "therefore", "thus"]) else 0.0,  # feature 1: reasoning words
            1.0 if "sorry" in response.lower() else 0.0,        # feature 2: refusal words
            1.0 if any(c.isdigit() for c in response) else 0.0, # feature 3: contains numbers
            1.0 if 10 <= word_count <= 200 else 0.0,            # feature 4: reasonable length
        ])
        
        return features  # numpy array of shape (feature_size,)
    
    def forward(self, prompt, response):
        """
        Compute the reward score.
        
        Args:
            prompt: string, the user's question
            response: string, the model's answer
        
        Returns:
            score: single float (higher = better response)
        """
        # Step 1: Extract features from the (prompt, response) pair
        features = self.extract_features(prompt, response)
        
        # Step 2: Linear projection: score = weights . features + bias
        # np.dot computes dot product (element-wise multiply then sum)
        score = np.dot(self.weights, features) + self.bias
        
        # Note: in a real reward model, there is no sigmoid here.
        # The raw score (logit) is used directly.
        # Sigmoid is only applied in the LOSS FUNCTION during training.
        return score
    
    def compute_loss(self, prompt, chosen_response, rejected_response):
        """
        Compute the Bradley-Terry pairwise loss.
        
        Args:
            prompt: string
            chosen_response: string (the preferred response)
            rejected_response: string (the less preferred response)
        
        Returns:
            loss: float (lower = model is getting it right)
        """
        # Get scores for both responses
        score_chosen   = self.forward(prompt, chosen_response)
        score_rejected = self.forward(prompt, rejected_response)
        
        # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        diff = score_chosen - score_rejected
        
        # Clip for numerical stability (avoid log(0))
        # This prevents the loss from becoming infinite
        prob = 1.0 / (1.0 + np.exp(-np.clip(diff, -500, 500)))
        
        # Negative log likelihood
        loss = -np.log(prob + 1e-8)  # 1e-8 to prevent log(0)
        
        return loss, score_chosen, score_rejected


# ============================================================
# DEMO: Use the tiny reward model
# ============================================================

model = TinyRewardModel(feature_size=5)  # create model with 5 features

# Define a prompt and two candidate responses
prompt = "What is photosynthesis?"

chosen_response = """
Photosynthesis is the process by which plants, algae, and some bacteria 
convert sunlight into food. They take in carbon dioxide (CO2) from the air
and water (H2O) from the soil. Using sunlight as energy, they produce
glucose (sugar) for energy and release oxygen (O2) as a byproduct.
This process happens in the chloroplasts of plant cells, using a
green pigment called chlorophyll.
"""

rejected_response = "Photosynthesis is when plants do stuff with sunlight."

# Score both responses
score_c = model.forward(prompt, chosen_response)
score_r = model.forward(prompt, rejected_response)

print(f"Chosen response score:   {score_c:.4f}")
print(f"Rejected response score: {score_r:.4f}")

# Compute loss
loss, sc, sr = model.compute_loss(prompt, chosen_response, rejected_response)
print(f"Training Loss: {loss:.4f}")

# Check if model got it right (chosen should score higher)
if score_c > score_r:
    print("Model correctly identifies chosen as better!")
else:
    print("Model got it wrong -- needs more training.")
```

---

## Part 8: Reward Model Evaluation

How do we know if our reward model is any good?

### Metric 1: Pairwise Accuracy

On a held-out test set, what percentage of time does the reward model
correctly identify which response is better?

```
Pairwise Accuracy = (correct_pairs) / (total_pairs)

Random chance = 50%
A decent reward model = 70-75%
A strong reward model = 80%+
```

### Metric 2: Spearman Correlation

For datasets with graded preferences (1-7 scale), how well do the reward
model's scores correlate with human ratings?

### Metric 3: Downstream Task Performance

Ultimately, what matters is: does using this reward model in PPO
produce a better final LLM?

This is the only metric that truly matters, but it is expensive to compute
(you have to run the entire PPO training to find out).

---

## Part 9: Common Reward Model Failure Modes

Here is a summary of things that go wrong:

```
+---------------------------------------------------------------+
|  REWARD MODEL FAILURE MODES                                   |
|                                                               |
|  1. LENGTH BIAS                                               |
|     Model prefers longer responses even if they are verbose.  |
|     Fix: Normalize rewards by response length.                |
|                                                               |
|  2. FORMAT BIAS                                               |
|     Model prefers bullet points, headers, markdown.           |
|     Fix: Include diverse formats in training data.            |
|                                                               |
|  3. CONFIDENCE BIAS                                           |
|     Model prefers responses that sound confident.             |
|     Even if the confident response is WRONG.                  |
|     Fix: Include examples where honest uncertainty is better. |
|                                                               |
|  4. POSITION BIAS                                             |
|     In A/B comparison, annotators slightly prefer A.          |
|     Fix: Show each pair in both orders (A then B, B then A). |
|                                                               |
|  5. VERBOSITY REWARD HACKING                                  |
|     PPO-trained model learns to write very long responses     |
|     because the reward model was biased toward length.        |
|     Fix: KL penalty during PPO (more in Lesson 03).           |
+---------------------------------------------------------------+
```

---

## Summary

```
+---------------------------------------------------------------+
|  LESSON 02 SUMMARY                                            |
|                                                               |
|  1. The Reward Model                                          |
|     Input: (prompt + response)                                |
|     Output: single scalar score (higher = better)             |
|                                                               |
|  2. Training Data                                             |
|     {prompt, chosen, rejected} preference pairs               |
|     Reward model must score chosen > rejected                 |
|                                                               |
|  3. Bradley-Terry Loss                                        |
|     L = -log(sigmoid(r_chosen - r_rejected))                  |
|     Lower loss = model correctly ranks chosen above rejected   |
|                                                               |
|  4. Architecture                                              |
|     Same as base LLM but final layer outputs 1 number         |
|     instead of vocabulary-sized distribution                  |
|                                                               |
|  5. Good Reward Model Properties                              |
|     - Calibrated (consistent scores)                          |
|     - Generalizes (works on new prompts)                      |
|     - Resistant to reward hacking                             |
|     - Fast (called millions of times during PPO)              |
+---------------------------------------------------------------+
```

---

## Quiz Questions

1. What does a reward model take as input and what does it output?

2. Why is pairwise comparison (A vs B) used instead of absolute ratings (rate on 1-10)?

3. Write out the Bradley-Terry loss formula and explain what each part means.

4. What happens to the loss when r_chosen = r_rejected? What probability does that correspond to?

5. How does the reward model architecture differ from the base LLM architecture?
   What is the one change made?

6. Define "reward hacking" and give one example from real RLHF experiments.

7. What is "length bias" in reward models and how can it be mitigated?

8. What metric would you use to evaluate whether your reward model is working?

---

## Lab Exercise

```python
# lab_02_reward_model.py
# Implement and test the Bradley-Terry loss function

import numpy as np

# ============================================================
# YOUR TASK: Implement these three functions
# ============================================================

def sigmoid(x):
    """
    Implement the sigmoid function.
    Formula: 1 / (1 + e^(-x))
    
    Hint: use np.exp()
    """
    # YOUR CODE HERE
    pass

def bradley_terry_loss(r_chosen, r_rejected):
    """
    Compute the Bradley-Terry pairwise loss.
    Formula: -log(sigmoid(r_chosen - r_rejected))
    
    Hint: use your sigmoid() function above
    Hint: use np.log()
    """
    # YOUR CODE HERE
    pass

def accuracy(scores_chosen, scores_rejected):
    """
    Compute pairwise accuracy.
    
    Args:
        scores_chosen: numpy array of scores for chosen responses
        scores_rejected: numpy array of scores for rejected responses
    
    Returns:
        float: fraction of pairs where chosen scored higher
    """
    # YOUR CODE HERE
    # Hint: count how many times scores_chosen > scores_rejected
    # Divide by total number of pairs
    pass

# ============================================================
# TESTS: Run these to check your implementation
# ============================================================

# Test sigmoid
assert abs(sigmoid(0) - 0.5) < 1e-6,   "sigmoid(0) should be 0.5"
assert abs(sigmoid(1000) - 1.0) < 1e-6, "sigmoid(large number) should be close to 1"
assert abs(sigmoid(-1000) - 0.0) < 1e-6,"sigmoid(large negative) should be close to 0"
print("sigmoid tests passed!")

# Test Bradley-Terry loss
loss_perfect = bradley_terry_loss(10.0, -10.0)  # perfect separation
loss_wrong   = bradley_terry_loss(-5.0, 5.0)    # completely wrong
loss_neutral = bradley_terry_loss(0.0, 0.0)     # uncertain

assert loss_perfect < loss_neutral < loss_wrong, \
    "Loss should be: perfect < neutral < wrong"
print("Bradley-Terry loss tests passed!")

# Test accuracy
s_chosen   = np.array([2.0, 3.0, 1.0, -0.5])
s_rejected = np.array([1.0, 1.0, 2.0,  0.5])
# chosen is better in pairs 0,1 and rejected is better in pairs 2,3
# so accuracy should be 0.5
acc = accuracy(s_chosen, s_rejected)
assert abs(acc - 0.5) < 1e-6, f"Expected 0.5, got {acc}"
print("Accuracy tests passed!")

print("\nAll tests passed! Great work.")
```

---

*Next lesson: How PPO uses the reward model to train the LLM.*
*File: lessons/03_ppo_for_llms.md*
