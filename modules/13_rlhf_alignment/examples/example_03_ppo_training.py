"""
=============================================================================
MODULE 13 - EXAMPLE 03: PPO TRAINING FOR LLMs
=============================================================================

WHAT YOU WILL LEARN:
  1. What PPO (Proximal Policy Optimization) is and why it matters
  2. How "clipping" stops the policy from changing too fast
  3. How KL divergence keeps the model close to the original
  4. How to implement PPO from scratch with NumPy
  5. How PyTorch autograd handles the gradients automatically

C# ANALOGY:
  PPO is like a ConfigurationValidator that rejects changes above a threshold.
  Imagine you have a service that auto-tunes its own settings. You allow
  updates, but you add a rule: "No single update may change any setting
  by more than 20%." That guardrail IS the "proximal" constraint.

  In PPO:
    - The "policy" is your model (like appsettings.json being tuned)
    - The "old policy" is a frozen snapshot (like a backup config)
    - The "clip" is the validator that rejects too-large changes
    - KL divergence is a second safety check (total drift from original)

=============================================================================
"""

# ---------------------------------------------------------------------------
# GLOSSARY
# ---------------------------------------------------------------------------
# This dictionary defines every important term used in this file.
# Read it before looking at the code below.
# ---------------------------------------------------------------------------
GLOSSARY = {
    "Policy":
        "The model that decides what token/action to output. "
        "(C# analogy: a Strategy class that picks the next action)",

    "Old Policy":
        "A frozen snapshot of the policy taken at the start of each update step. "
        "(C# analogy: a deep-copied object saved before mutation)",

    "Reference Policy":
        "The very first policy, frozen forever. Used for KL penalty. "
        "(C# analogy: the baseline config stored in version control)",

    "Action":
        "The token chosen by the policy. "
        "(C# analogy: the return value of a strategy.Execute() call)",

    "Reward":
        "A scalar score that says how good the chosen action was. "
        "(C# analogy: a unit test assertion score, 0.0 to 1.0)",

    "Advantage":
        "Reward minus baseline. Tells us: was this reward BETTER than average? "
        "(C# analogy: metric_value - rolling_average)",

    "Ratio":
        "new_policy_prob / old_policy_prob. How much did the policy change? "
        "(C# analogy: new_setting / old_setting)",

    "Clipping":
        "Forcing ratio into [1-epsilon, 1+epsilon]. The proximal constraint. "
        "(C# analogy: Math.Clamp(ratio, 0.8, 1.2))",

    "PPO Loss":
        "-min(ratio * advantage, clipped_ratio * advantage). Pessimistic objective. "
        "(C# analogy: picking the worse of two safety-bounded estimates)",

    "KL Divergence":
        "Measures how different two probability distributions are. "
        "(C# analogy: a diff score between two config snapshots)",

    "KL Penalty":
        "We add KL * beta to the loss to prevent the policy drifting too far. "
        "(C# analogy: a penalty fee for config drift beyond a threshold)",

    "Baseline":
        "Running average of past rewards. Used to centre the advantage. "
        "(C# analogy: a rolling-window average in a metrics collector)",

    "Log Probability":
        "log(probability). Used so we can add instead of multiply — numerically safer. "
        "(C# analogy: Math.Log(probability))",
}

# ---------------------------------------------------------------------------
# ASCII DIAGRAM: PPO TRAINING LOOP
# ---------------------------------------------------------------------------
PPO_DIAGRAM = """
PPO TRAINING LOOP
=================

Old Policy (frozen copy)
    |
    v
New Policy (being trained)
    |--- generates response (samples a token)
    |
    v
Reward Model scores response --> reward r
    |
    v
Compute ratio: pi_new(a|s) / pi_old(a|s)
    |
    v
Clip ratio to [1-e, 1+e]   <-- the "proximal" part  (e = epsilon = 0.2)
    |
    v
PPO Loss = -min(ratio * advantage, clipped_ratio * advantage)
    |
    v
KL Penalty: total_loss += beta * KL(new_policy || reference_policy)
    |
    v
Update new policy weights
    |
    (loop back to top with updated policy as the new "old policy")
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import numpy as np          # NumPy: numerical computing (like System.Math but for arrays)
import torch                # PyTorch: deep learning framework
import torch.nn as nn       # nn: neural network building blocks (like abstract base classes)
import torch.optim as optim # optim: gradient-based optimisers (like gradient descent engines)

# ---------------------------------------------------------------------------
# PRINT HELPERS
# ---------------------------------------------------------------------------

def print_header(title):
    """Print a section header — purely cosmetic, makes output readable."""
    print("\n" + "=" * 60)  # print a line of 60 "=" characters
    print(f"  {title}")     # print the title indented by 2 spaces
    print("=" * 60)         # print another line of 60 "=" characters


def print_bar(label, value, max_value=1.0, width=30):
    """
    Print a simple ASCII bar chart for a single value.
    C# analogy: like Console.Write a progress bar string.
    """
    filled = int(width * abs(value) / (max_value + 1e-8))  # how many blocks to fill
    filled = min(filled, width)                             # clamp so we never exceed width
    bar = "#" * filled + "-" * (width - filled)            # build the bar string
    sign = "+" if value >= 0 else "-"                       # show sign of value
    print(f"  {label:<20} [{sign}{bar}] {value:+.4f}")     # print label + bar + numeric value


# ===========================================================================
# PART A: NUMPY IMPLEMENTATION
# ===========================================================================
print_header("PART A: PPO WITH NUMPY (FROM SCRATCH)")
print(PPO_DIAGRAM)          # show the ASCII diagram before any code runs
print("\nGLOSSARY (read these terms first):")
for term, explanation in GLOSSARY.items():   # iterate over every term in our glossary dict
    print(f"  [{term}]: {explanation}")      # print term and its explanation

# ---------------------------------------------------------------------------
# A1. HYPERPARAMETERS
# ---------------------------------------------------------------------------
print_header("A1. Hyperparameters")

VOCAB_SIZE = 4          # number of tokens the policy can choose from (tiny for demo)
STATE_DIM  = 4          # size of the state (context) vector
EPSILON    = 0.2        # clip range: ratio must stay in [0.8, 1.2]
BETA       = 0.1        # KL penalty coefficient (how hard we penalise drift)
LR_A       = 0.05       # learning rate for NumPy gradient descent
NUM_STEPS  = 15         # how many PPO update steps to run

print(f"  Vocab size (actions) : {VOCAB_SIZE}")   # tiny vocabulary: tokens 0,1,2,3
print(f"  State dimension      : {STATE_DIM}")    # 4-dim context vector
print(f"  Epsilon (clip range) : {EPSILON}")      # ratio clipped to [0.8, 1.2]
print(f"  Beta (KL weight)     : {BETA}")         # KL penalty strength
print(f"  Learning rate        : {LR_A}")         # step size for weight update
print(f"  Training steps       : {NUM_STEPS}")    # number of PPO iterations

# ---------------------------------------------------------------------------
# A2. FIXED RANDOM SEED (reproducibility)
# ---------------------------------------------------------------------------
np.random.seed(42)      # fix the random seed so results are the same every run
                        # C# analogy: new Random(42)

# ---------------------------------------------------------------------------
# A3. INITIALISE POLICY WEIGHTS
# ---------------------------------------------------------------------------
print_header("A2. Initialise Weights")

# W_policy shape: (VOCAB_SIZE, STATE_DIM)
# policy(state) = softmax(W_policy @ state)
# This maps a 4-dim state to a 4-token probability distribution.
# C# analogy: a float[4,4] matrix that acts as a lookup table.
W_policy = np.random.randn(VOCAB_SIZE, STATE_DIM) * 0.1   # small random weights

# W_ref is a frozen copy of the initial weights — never updated.
# C# analogy: (float[,])W_policy.Clone()  -- but immutable after this point.
W_ref = W_policy.copy()    # deep copy (independent array in memory)

# W_reward is the reward model: r = w_reward @ response_features
# We keep it fixed — it simulates a pre-trained reward model.
# C# analogy: a pre-loaded scoring function from a trained ML model.
W_reward = np.array([0.5, -0.3, 0.8, -0.1])   # hardcoded reward weights

# The "state" is the context fed to the policy each step.
# We keep it fixed to isolate the effect of policy updates.
STATE = np.array([1.0, 0.5, -0.3, 0.8])        # hardcoded context vector

print(f"  W_policy (initial):\n{W_policy}")     # show initial policy matrix
print(f"  W_ref (frozen)    :\n{W_ref}")        # show frozen reference weights
print(f"  W_reward          : {W_reward}")      # show reward model weights
print(f"  State vector      : {STATE}")         # show the fixed context

# ---------------------------------------------------------------------------
# A4. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def softmax_np(logits):
    """
    Convert raw scores (logits) to a probability distribution.
    Subtracting the max prevents numerical overflow — standard trick.
    C# analogy: like normalising a float[] so all values sum to 1.
    """
    e = np.exp(logits - np.max(logits))  # subtract max for numerical stability
    return e / e.sum()                   # divide by total so probabilities sum to 1


def get_policy_probs_np(W, state):
    """
    Run the policy: compute softmax over W @ state.
    Returns a 4-element probability array (one prob per token).
    C# analogy: float[] probs = Softmax(W * state);
    """
    logits = W @ state          # matrix-vector multiply: shape (VOCAB_SIZE,)
    return softmax_np(logits)   # convert logits to probabilities


def sample_action_np(probs):
    """
    Sample one token from the probability distribution.
    C# analogy: WeightedRandom.Sample(probs) — picks index by weight.
    """
    # np.random.choice picks one index from [0, VOCAB_SIZE)
    # with probabilities given by 'probs'
    return np.random.choice(len(probs), p=probs)   # returns an int (token index)


def get_reward_np(action):
    """
    Simulate a reward model: reward = W_reward[action].
    In real RLHF a neural network computes this; here we use a lookup.
    C# analogy: float reward = rewardModel.Score(action);
    """
    return float(W_reward[action])   # index into reward weights by action (token index)


def kl_divergence_np(p, q):
    """
    KL(p || q) = sum(p * log(p / q)).
    Measures how different distribution p is from distribution q.
    Higher value = more different.
    C# analogy: a 'distribution distance' metric (no direct C# equivalent).
    """
    p = np.clip(p, 1e-8, 1.0)   # clip to avoid log(0) which would be -infinity
    q = np.clip(q, 1e-8, 1.0)   # clip q too
    return float(np.sum(p * np.log(p / q)))   # KL formula: sum of p*log(p/q)


def ppo_loss_np(ratio, advantage):
    """
    PPO clipped objective for a single (action, advantage) pair.
    Returns the LOSS (we minimise this by doing gradient descent).
    C# analogy: float loss = -Math.Min(r*A, Math.Clamp(r, 0.8, 1.2)*A);
    """
    clipped = np.clip(ratio, 1.0 - EPSILON, 1.0 + EPSILON)   # clip ratio to [0.8, 1.2]
    # Take the MINIMUM of two values (pessimistic / conservative estimate).
    # The negative sign converts it from a maximisation problem to minimisation.
    loss = -min(ratio * advantage, clipped * advantage)        # PPO loss formula
    return float(loss)   # return as plain Python float


# ---------------------------------------------------------------------------
# A5. TRAINING LOOP
# ---------------------------------------------------------------------------
print_header("A3. PPO Training Loop (NumPy)")
print(f"  {'Step':>4}  {'Reward':>7}  {'Ratio':>7}  {'KL':>7}  {'Loss':>8}  {'Action':>6}")
print("  " + "-" * 55)  # separator line

baseline       = 0.0    # running mean of rewards (starts at zero)
reward_history = []     # list to collect reward at each step (for final chart)
kl_history     = []     # list to collect KL at each step

for step in range(NUM_STEPS):   # loop NUM_STEPS times (like a for loop in C#)

    # ---- 1. Compute old and current policy probabilities ----
    # "old policy" = policy at the START of this step (before update).
    # We recompute it fresh each iteration because W_policy changes.
    probs_old = get_policy_probs_np(W_policy, STATE)   # shape: (4,) — one prob per token
    probs_ref = get_policy_probs_np(W_ref,    STATE)   # reference policy (never changes)

    # ---- 2. Sample an action (pick a token) ----
    action = sample_action_np(probs_old)               # int in {0, 1, 2, 3}

    # ---- 3. Get reward from reward model ----
    reward = get_reward_np(action)                     # float, e.g. 0.5 or -0.3

    # ---- 4. Update baseline (running mean) ----
    # Exponential moving average: blends old baseline with new reward.
    # C# analogy: baseline = 0.9 * baseline + 0.1 * reward;
    baseline = 0.9 * baseline + 0.1 * reward          # running average of rewards

    # ---- 5. Compute advantage ----
    # Advantage = how much BETTER was this reward than the baseline?
    # Positive advantage means the action was better than average.
    advantage = reward - baseline                      # scalar float

    # ---- 6. Compute log-probability ratio ----
    # ratio = new_policy(action) / old_policy(action)
    # Using log: log_ratio = log(new) - log(old)  then ratio = exp(log_ratio)
    # We re-run the current (possibly updated) policy to get new probs.
    # In this first step new == old, so ratio == 1.0 (no change yet).
    probs_new = get_policy_probs_np(W_policy, STATE)   # current policy probabilities

    log_ratio = (                                      # compute log ratio safely
        np.log(probs_new[action] + 1e-8)              # log(new_prob) for chosen action
        - np.log(probs_old[action] + 1e-8)            # minus log(old_prob)
    )
    ratio = float(np.exp(log_ratio))                  # convert back to raw ratio

    # ---- 7. Compute PPO loss ----
    loss_ppo = ppo_loss_np(ratio, advantage)           # clipped PPO objective

    # ---- 8. Compute KL divergence ----
    # We measure drift from the REFERENCE policy (not just old policy).
    kl = kl_divergence_np(probs_new, probs_ref)       # KL(current || reference)

    # ---- 9. Total loss = PPO loss + KL penalty ----
    total_loss = loss_ppo + BETA * kl                  # add weighted KL penalty

    # ---- 10. Compute gradient and update W_policy ----
    # For the action taken, we want to increase its probability when advantage > 0.
    # Manual gradient: d(loss)/d(logits) requires the chain rule.
    # Simplified gradient for the PPO-clipped policy:
    #   If ratio is NOT clipped: gradient pushes prob of action toward advantage sign.
    #   If ratio IS clipped: gradient is zero (no update).
    clipped_ratio = np.clip(ratio, 1.0 - EPSILON, 1.0 + EPSILON)   # clamp ratio
    # Check whether ratio was clipped (if clipped_ratio != ratio, it was clamped)
    is_clipped = (abs(clipped_ratio - ratio) > 1e-6)                # boolean flag

    if not is_clipped:                                 # only update if ratio was NOT clipped
        # Create a one-hot vector for the chosen action (length = VOCAB_SIZE)
        # C# analogy: float[] oneHot = new float[4]; oneHot[action] = 1.0f;
        one_hot = np.zeros(VOCAB_SIZE)                 # initialise with zeros
        one_hot[action] = 1.0                          # set chosen action to 1

        # Policy gradient direction: we want to increase prob of action if advantage > 0.
        # Gradient of cross-entropy loss w.r.t. logits = probs - one_hot (when maximising).
        # We negate because our loss is negative (we minimise loss = maximise reward).
        grad_logits = -(advantage) * (one_hot - probs_new)   # gradient of loss w.r.t. logits

        # Outer product: grad_W = grad_logits (outer product) state
        # This tells us how to change each weight in W_policy.
        # C# analogy: Matrix gradient = OuterProduct(grad_logits, STATE);
        grad_W = np.outer(grad_logits, STATE)                 # shape: (VOCAB_SIZE, STATE_DIM)

        # Add KL gradient (penalise drift from reference)
        # KL gradient w.r.t. logits: grad of KL(new||ref) w.r.t. logits is (probs_new - probs_ref)
        grad_kl = np.outer((probs_new - probs_ref), STATE)    # shape: (VOCAB_SIZE, STATE_DIM)

        # Combine gradients
        total_grad = grad_W + BETA * grad_kl                  # weighted sum of gradients

        # Gradient descent step: move weights in the direction that reduces loss
        # C# analogy: W_policy -= learningRate * gradient;
        W_policy = W_policy - LR_A * total_grad               # update policy weights

    # ---- 11. Record history for final chart ----
    reward_history.append(reward)    # save this step's reward
    kl_history.append(kl)            # save this step's KL divergence

    # ---- 12. Print step summary ----
    print(f"  {step+1:>4}  {reward:>+7.3f}  {ratio:>7.4f}  {kl:>7.4f}  {total_loss:>+8.4f}  tok={action}")

# ---------------------------------------------------------------------------
# A6. TRAINING SUMMARY
# ---------------------------------------------------------------------------
print_header("A4. Training Summary (NumPy)")

# Compute average reward in first half vs second half of training
first_half_avg  = float(np.mean(reward_history[:NUM_STEPS//2]))   # mean of steps 1-7
second_half_avg = float(np.mean(reward_history[NUM_STEPS//2:]))   # mean of steps 8-15
avg_kl          = float(np.mean(kl_history))                       # mean KL over all steps

print(f"\n  Average reward (first half) : {first_half_avg:+.4f}")
print(f"  Average reward (second half): {second_half_avg:+.4f}")
print(f"  Average KL divergence       : {avg_kl:.4f}")

# Qualitative check
if second_half_avg > first_half_avg:                    # did rewards improve?
    print("  RESULT: Reward IMPROVED over training (good!)")
else:
    print("  RESULT: Reward did not improve (try more steps or tuning)")

if avg_kl < 0.5:                                        # did KL stay small?
    print("  RESULT: KL divergence stayed SMALL (policy did not drift far)")
else:
    print("  RESULT: KL divergence is large (policy drifted — increase beta)")

# ASCII reward chart (per-step)
print("\n  Reward per step (ASCII chart):")
max_r = max(abs(r) for r in reward_history) + 1e-8     # find max absolute reward
for i, r in enumerate(reward_history):                  # loop over each step's reward
    print_bar(f"  step {i+1:>2}", r, max_value=max_r)  # print one bar per step

print("\n  KL divergence per step:")
max_kl = max(kl_history) + 1e-8                         # find max KL for scaling
for i, kl in enumerate(kl_history):                     # loop over each step's KL
    print_bar(f"  step {i+1:>2}", kl, max_value=max_kl) # print one bar per step


# ===========================================================================
# PART B: PYTORCH IMPLEMENTATION
# ===========================================================================
print_header("PART B: PPO WITH PYTORCH (AUTOGRAD)")

print("""
  KEY DIFFERENCE FROM PART A:
  - In Part A we computed gradients by hand (manual calculus).
  - In Part B PyTorch's autograd computes gradients automatically.
  - We just define the loss as a computation graph; .backward() does the rest.
  C# analogy: instead of implementing IDerivable yourself, you use a
  framework that auto-differentiates any expression you write.
""")

# ---------------------------------------------------------------------------
# B1. POLICY NETWORK (nn.Module)
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    A tiny policy network: one linear layer.
    Input : state vector (STATE_DIM = 4)
    Output: logits over vocabulary (VOCAB_SIZE = 4)
    C# analogy: a class that implements IPolicy with a float[,] weight matrix.
    """

    def __init__(self, state_dim, vocab_size):
        """Constructor — called once when we create the object."""
        super().__init__()                              # must call parent __init__ (C# : base())
        # nn.Linear(in, out) is a fully-connected layer: output = input @ W.T + b
        # C# analogy: a Matrix W of shape (vocab_size, state_dim)
        self.linear = nn.Linear(state_dim, vocab_size, bias=False)  # no bias to keep it simple

    def forward(self, state):
        """
        Forward pass: compute probabilities from state.
        PyTorch calls this automatically when you do network(state).
        C# analogy: float[] Forward(float[] state) { return Softmax(W * state); }
        """
        logits = self.linear(state)                    # linear layer: shape (vocab_size,)
        probs  = torch.softmax(logits, dim=-1)         # softmax over last dimension
        return probs                                   # return probability distribution


# ---------------------------------------------------------------------------
# B2. INITIALISE NETWORKS
# ---------------------------------------------------------------------------
print_header("B1. Initialise PyTorch Networks")

torch.manual_seed(42)   # fix random seed for reproducibility (C# analogy: new Random(42))

policy_net = PolicyNetwork(STATE_DIM, VOCAB_SIZE)     # the network we will TRAIN
ref_net    = PolicyNetwork(STATE_DIM, VOCAB_SIZE)     # reference network — NEVER updated

# Copy policy weights into ref_net so they start IDENTICAL
ref_net.load_state_dict(policy_net.state_dict())      # copy all parameters

# Freeze the reference network: no gradients will flow through it
for param in ref_net.parameters():                    # loop over every tensor in ref_net
    param.requires_grad_(False)                       # disable gradient tracking for this tensor
                                                      # C# analogy: make the object immutable

# Print initial parameter norms (to compare before and after training)
def param_norm(net):
    """Compute the Frobenius norm of all parameters in a network."""
    total = 0.0                                       # accumulator for sum of squares
    for p in net.parameters():                        # iterate over every parameter tensor
        total += float((p ** 2).sum())                # add sum of squared values
    return float(np.sqrt(total))                      # return square root = Frobenius norm


initial_policy_norm = param_norm(policy_net)         # snapshot norm before training
print(f"  policy_net initial norm : {initial_policy_norm:.4f}")
print(f"  ref_net    initial norm : {param_norm(ref_net):.4f}  (should match)")

# ---------------------------------------------------------------------------
# B3. OPTIMISER
# ---------------------------------------------------------------------------
# Adam optimiser: an advanced gradient descent that adapts learning rates.
# C# analogy: a smarter version of gradient -= learningRate * gradient_direction
optimiser_b = optim.Adam(policy_net.parameters(), lr=0.05)   # only policy_net params

# Fixed state tensor (same concept vector as Part A)
state_tensor = torch.tensor(STATE, dtype=torch.float32)      # convert NumPy array to tensor

# ---------------------------------------------------------------------------
# B4. TRAINING LOOP
# ---------------------------------------------------------------------------
print_header("B2. PPO Training Loop (PyTorch)")
print(f"  {'Step':>4}  {'Reward':>7}  {'Ratio':>7}  {'KL':>7}  {'Loss':>8}  {'Action':>6}")
print("  " + "-" * 55)

baseline_b       = 0.0     # running reward baseline (same idea as Part A)
reward_history_b = []      # collect rewards for final chart
kl_history_b     = []      # collect KL values for final chart

for step in range(NUM_STEPS):   # same number of steps as Part A

    # ---- 1. Get old policy probabilities (no gradient — just for ratio) ----
    # .detach() stops gradients from flowing through this computation.
    # C# analogy: var oldProbs = policy_net.Forward(state).AsReadOnly();
    with torch.no_grad():                                          # disable gradient tracking
        probs_old_b = policy_net(state_tensor)                     # shape: (4,)

    # ---- 2. Get reference policy probabilities (always no gradient) ----
    with torch.no_grad():                                          # ref_net is frozen anyway
        probs_ref_b = ref_net(state_tensor)                        # shape: (4,)

    # ---- 3. Sample action ----
    action_b = int(torch.multinomial(probs_old_b, 1).item())       # sample one token index
    # torch.multinomial: samples indices weighted by probabilities
    # .item() converts a 1-element tensor to a plain Python int

    # ---- 4. Get reward ----
    reward_b = float(W_reward[action_b])                           # same reward model as Part A

    # ---- 5. Update baseline ----
    baseline_b = 0.9 * baseline_b + 0.1 * reward_b                # exponential moving average

    # ---- 6. Compute advantage ----
    advantage_b = reward_b - baseline_b                            # scalar float

    # ---- 7. Forward pass WITH gradients (for loss computation) ----
    probs_new_b = policy_net(state_tensor)                         # re-run to get gradient-tracked probs

    # ---- 8. Compute ratio ----
    # ratio = new_prob / old_prob for the chosen action
    # We use log-space for numerical stability then exp back
    log_ratio_b = (
        torch.log(probs_new_b[action_b] + 1e-8)                   # log(new prob)
        - torch.log(probs_old_b[action_b] + 1e-8)                 # minus log(old prob)
    )
    ratio_b = torch.exp(log_ratio_b)                               # actual ratio tensor

    # ---- 9. Compute clipped PPO loss ----
    clipped_b   = torch.clamp(ratio_b, 1.0 - EPSILON, 1.0 + EPSILON)   # clip ratio
    adv_tensor  = torch.tensor(advantage_b, dtype=torch.float32)        # advantage as tensor
    # Pessimistic minimum: take the worse of the two estimates
    ppo_obj_b   = torch.min(ratio_b * adv_tensor, clipped_b * adv_tensor)  # both are scalars
    loss_ppo_b  = -ppo_obj_b                                               # negate to minimise

    # ---- 10. Compute KL divergence (PyTorch) ----
    # KL(new || ref) = sum(new * log(new / ref))
    kl_b = torch.sum(
        probs_new_b * torch.log(
            (probs_new_b + 1e-8) / (probs_ref_b + 1e-8)           # log(new/ref) element-wise
        )
    )                                                              # scalar KL tensor

    # ---- 11. Total loss ----
    total_loss_b = loss_ppo_b + BETA * kl_b                        # PPO + weighted KL

    # ---- 12. Backpropagate and update weights ----
    optimiser_b.zero_grad()          # clear gradients from previous step (C# analogy: reset accumulators)
    total_loss_b.backward()          # compute all gradients automatically (PyTorch magic!)
    optimiser_b.step()               # apply gradients to policy_net weights

    # ---- 13. Record history ----
    reward_history_b.append(reward_b)           # save reward
    kl_history_b.append(float(kl_b.item()))    # save KL (convert tensor to float)

    # ---- 14. Print step summary ----
    print(f"  {step+1:>4}  {reward_b:>+7.3f}  {float(ratio_b.item()):>7.4f}  "
          f"{float(kl_b.item()):>7.4f}  {float(total_loss_b.item()):>+8.4f}  tok={action_b}")

# ---------------------------------------------------------------------------
# B5. VERIFY: REFERENCE NETWORK DID NOT CHANGE
# ---------------------------------------------------------------------------
print_header("B3. Verify Reference Network is Frozen")

# ref_net should still have the exact same weights as initial policy_net
# We compare their parameter norms
final_policy_norm = param_norm(policy_net)   # norm AFTER training
final_ref_norm    = param_norm(ref_net)      # norm of ref (should be unchanged)

print(f"  policy_net norm BEFORE training: {initial_policy_norm:.4f}")
print(f"  policy_net norm AFTER  training: {final_policy_norm:.4f}")
print(f"  ref_net    norm AFTER  training: {final_ref_norm:.4f}")

# Check norms differ (policy changed) and ref stayed the same
policy_changed = abs(final_policy_norm - initial_policy_norm) > 1e-6   # bool
ref_unchanged  = abs(final_ref_norm   - initial_policy_norm)  < 1e-6   # bool

print(f"\n  Policy weights changed  : {policy_changed}  (should be True)")
print(f"  Reference weights frozen: {ref_unchanged}   (should be True)")

if policy_changed and ref_unchanged:
    print("  VERIFIED: PPO updated the policy without touching the reference.")
else:
    print("  WARNING: Something went wrong with freezing or updating.")

# ---------------------------------------------------------------------------
# B6. TRAINING SUMMARY (PyTorch)
# ---------------------------------------------------------------------------
print_header("B4. Training Summary (PyTorch)")

first_half_b  = float(np.mean(reward_history_b[:NUM_STEPS//2]))   # early rewards
second_half_b = float(np.mean(reward_history_b[NUM_STEPS//2:]))   # late rewards
avg_kl_b      = float(np.mean(kl_history_b))                       # average KL

print(f"  Average reward (first half) : {first_half_b:+.4f}")
print(f"  Average reward (second half): {second_half_b:+.4f}")
print(f"  Average KL divergence       : {avg_kl_b:.4f}")

if second_half_b > first_half_b:
    print("  RESULT: Reward IMPROVED over training")
else:
    print("  RESULT: Reward did not improve (try tuning hyperparameters)")

if avg_kl_b < 0.5:
    print("  RESULT: KL divergence stayed SMALL (bounded policy update)")
else:
    print("  RESULT: KL divergence large (increase BETA to constrain updates)")

print("\n  Reward per step (PyTorch run):")
max_rb = max(abs(r) for r in reward_history_b) + 1e-8   # max reward for scaling
for i, r in enumerate(reward_history_b):                 # loop over steps
    print_bar(f"  step {i+1:>2}", r, max_value=max_rb)  # ASCII bar per step

# ===========================================================================
# KEY TAKEAWAYS
# ===========================================================================
print_header("KEY TAKEAWAYS")
print("""
  1. PPO keeps training STABLE by clipping the ratio to [1-e, 1+e].
     Without clipping, a single bad update can destroy the model.

  2. The KL penalty adds a SECOND safety net: if the policy drifts too far
     from the reference, the penalty term pushes it back.

  3. In Part A we computed gradients by hand (hard, but educational).
     In Part B PyTorch's autograd computed them automatically (practical).

  4. The reference policy is ALWAYS frozen — it never learns.
     It represents the original LLM before alignment training began.

  5. The "advantage" tells the algorithm: "this action was better (or worse)
     than the average." Positive advantage => increase that action's prob.

  C# PARALLEL:
    PPO is like a ConfigurationValidator that:
      - Takes a proposed config change (new policy)
      - Compares it to the current config (old policy)
      - Rejects changes above 20% (clipping)
      - Also penalises total drift from the original config (KL penalty)
""")
