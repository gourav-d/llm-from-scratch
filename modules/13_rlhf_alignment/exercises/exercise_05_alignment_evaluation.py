"""
=============================================================================
MODULE 13 - EXERCISE 05: Evaluating Alignment Quality
=============================================================================

WHAT YOU WILL LEARN:
  - How to measure whether an alignment method actually improved the model
  - Preference accuracy: the most direct metric for alignment
  - Reward consistency: does the model give stable scores?
  - How to build a comparison table to evaluate multiple alignment methods
  - How to do basic red-teaming to check for harmful responses

C# ANALOGY:
  Evaluating alignment is like running a test suite for your AI model:
  - preference_accuracy  = unit test for "does it pick the better response?"
  - reward_consistency   = regression test for "does it give the same score twice?"
  - comparison table     = test report comparing v1.0 vs v2.0 vs v3.0
  - red_team_check       = security/penetration test for harmful outputs
  Just like you wouldn't ship code without test coverage, you shouldn't
  deploy an LLM without alignment evaluation.

=============================================================================

ALIGNMENT EVALUATION PIPELINE
==============================

  Aligned Model
       |
       +----> [Preference Accuracy Test]  -> % correct rankings
       |
       +----> [Reward Consistency Test]   -> std dev of scores (stability)
       |
       +----> [Comparison Table]          -> RLHF vs DPO vs CAI scores
       |
       +----> [Red Team Check]            -> detect harmful outputs

  All results -> ALIGNMENT REPORT

=============================================================================
"""

# ---- GLOSSARY ---------------------------------------------------------------
GLOSSARY = {
    "Preference Accuracy":
        "Fraction of pairs where the aligned model correctly scores chosen > rejected.",
    "Reward Consistency":
        "Standard deviation of reward scores for the same prompt. Lower = more stable.",
    "Standard Deviation":
        "How spread out a set of numbers is. Low std = scores cluster near the mean.",
    "Red Team":
        "A test that tries to make the model produce harmful outputs.",
    "Harm Score":
        "A number [0,1] measuring how harmful a response is. Above 0.5 = dangerous.",
    "Human Eval Score":
        "A score (0-10) assigned by human raters judging the overall response quality.",
    "KL from Ref":
        "KL divergence between aligned model and reference model. Low = stayed close to ref.",
    "Alignment Report":
        "A summary comparing multiple alignment methods across all evaluation metrics.",
}

print("=" * 70)
print("EXERCISE 05 — Evaluating Alignment Quality")
print("=" * 70)
print("\nGLOSSARY:")
for term, defn in GLOSSARY.items():
    print(f"  {term}:\n    {defn}\n")

import numpy as np    # NumPy — only library needed

# =============================================================================
# SHARED DATA
# =============================================================================

# --- Preference accuracy data ---
# For each method, we have model scores for chosen and rejected responses
# in a set of 10 preference pairs.
MODEL_SCORES_CHOSEN   = np.array(
    [0.85, 0.72, 0.91, 0.60, 0.78, 0.88, 0.65, 0.93, 0.70, 0.82],
    dtype=np.float32
)
MODEL_SCORES_REJECTED = np.array(
    [0.43, 0.80, 0.55, 0.55, 0.30, 0.40, 0.62, 0.41, 0.68, 0.35],
    dtype=np.float32
)
# Note: pairs 1, 6, 8 are "errors" where chosen does not outscore rejected

# --- Reward consistency data ---
# Simulated reward scores for the SAME 5 prompts, run multiple times.
# Each row = one run. Each column = one prompt.
# Consistent model = same score each run (low std across rows for each column).
REWARD_SCORES_RUNS = np.array([
    [0.75, 0.82, 0.61, 0.90, 0.55],    # run 1
    [0.76, 0.80, 0.63, 0.88, 0.54],    # run 2 (very consistent with run 1)
    [0.74, 0.85, 0.59, 0.91, 0.57],    # run 3
    [0.73, 0.81, 0.64, 0.89, 0.53],    # run 4
    [0.77, 0.79, 0.62, 0.92, 0.56],    # run 5
], dtype=np.float32)
# shape: [5 runs, 5 prompts]

# --- Comparison table data ---
# Three alignment methods and their evaluation metrics
ALIGNMENT_METHODS = ["RLHF", "DPO", "CAI"]

PREF_ACCURACIES  = [0.82, 0.85, 0.80]    # preference accuracy (higher = better)
KL_FROM_REF      = [0.15, 0.08, 0.12]    # KL divergence from reference (lower = better)
HUMAN_EVAL_SCORES = [7.2, 7.8, 7.5]      # human evaluation out of 10 (higher = better)

# --- Red team test data ---
# Feature vectors: [harm_score, honesty, helpfulness, manipulation, respect]
RESPONSE_FEATURES_TO_CHECK = np.array([
    [0.2, 0.8, 0.9, 0.1, 0.9],    # response A: safe
    [0.6, 0.5, 0.4, 0.3, 0.6],    # response B: borderline (harm=0.6 > 0.5!)
    [0.3, 0.7, 0.8, 0.2, 0.8],    # response C: safe
    [0.4, 0.6, 0.5, 0.7, 0.5],    # response D: manipulative (manip=0.7 > 0.6!)
    [0.1, 0.9, 0.9, 0.05, 0.95],  # response E: very safe
    [0.8, 0.3, 0.2, 0.8, 0.3],    # response F: very harmful AND manipulative!
], dtype=np.float32)

IDX_HARM        = 0    # harm_score feature index
IDX_MANIPULATION = 3   # manipulation feature index

# =============================================================================
# EXERCISE 1 — Preference Accuracy
# =============================================================================
# Compute the fraction of preference pairs where model_scores_chosen > model_scores_rejected.
# This is the PRIMARY metric for evaluating an alignment method.
#
# A well-aligned model should consistently prefer the better response.
# Perfect = 1.0, random = 0.5, bad = below 0.5
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 1 — Preference Accuracy")
print("=" * 60)
print()
print("  Chosen scores: ", MODEL_SCORES_CHOSEN)
print("  Rejected scores:", MODEL_SCORES_REJECTED)
print()
print("  Task: return float in [0, 1] = fraction where chosen > rejected")
print("  Hint: np.mean(boolean_array) averages True/False as 1.0/0.0")
print()

def preference_accuracy(model_scores_chosen, model_scores_rejected):
    """
    Compute fraction of pairs where the model correctly scores chosen > rejected.

    Parameters:
      model_scores_chosen   : np.array [N] — model's scores for chosen responses
      model_scores_rejected : np.array [N] — model's scores for rejected responses

    Returns:
      float in [0, 1] — 1.0 = perfect, 0.5 = random chance
    """
    # TODO: create boolean array: True where chosen_score > rejected_score
    correct = None    # model_scores_chosen > model_scores_rejected

    # TODO: return np.mean(correct) — averages True=1 and False=0
    accuracy = None
    return accuracy

# --- Run exercise 1 ---
acc = preference_accuracy(MODEL_SCORES_CHOSEN, MODEL_SCORES_REJECTED)
print(f"  Preference accuracy = {acc}")
print(f"  Expected: 0.7  (7 out of 10 pairs correct)")
print()

# Also show which pairs the model got wrong
print("  Detailed pair results:")
for i in range(len(MODEL_SCORES_CHOSEN)):
    sc = MODEL_SCORES_CHOSEN[i]    # chosen score
    sr = MODEL_SCORES_REJECTED[i]   # rejected score
    correct_flag = sc > sr          # is this pair correct?
    mark = "OK" if correct_flag else "WRONG"    # label for display
    print(f"    Pair {i+1:02d}: chosen={sc:.2f}  rejected={sr:.2f}  [{mark}]")

# =============================================================================
# EXERCISE 2 — Reward Consistency
# =============================================================================
# For each prompt, compute the standard deviation of reward scores across runs.
# Then return the average std across all prompts.
#
# Low std = the model gives stable/consistent scores (good!)
# High std = the model is noisy/unpredictable (bad!)
#
# C# analogy: this is like measuring variance in A/B test results —
# high variance means you need more samples before trusting the result.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 2 — Reward Consistency")
print("=" * 60)
print()
print(f"  REWARD_SCORES_RUNS shape: {REWARD_SCORES_RUNS.shape}")
print(f"  Each ROW is one run. Each COLUMN is one prompt.")
print()
print("  Task: compute std deviation per prompt (axis=0), then return the mean.")
print("  Hint: np.std(arr, axis=0) computes std along rows for each column")
print("  Hint: np.mean() averages across all prompts")
print()

def reward_consistency(scores_runs):
    """
    Compute the average reward score standard deviation across all prompts.
    Lower = more consistent = better.

    Parameters:
      scores_runs : np.array [num_runs, num_prompts]
                    Each column is a prompt, each row is one inference run.

    Returns:
      mean_std : float — average std deviation across all prompts
      per_prompt_std : np.array — std for each prompt individually
    """
    # TODO: Step 1 — compute std along axis=0 (across runs, per prompt)
    # np.std(arr, axis=0) collapses rows, giving one std per column
    per_prompt_std = None    # replace with np.std(scores_runs, axis=0)

    # TODO: Step 2 — compute mean of per-prompt stds
    mean_std = None    # replace with np.mean(per_prompt_std)

    return mean_std, per_prompt_std

# --- Run exercise 2 ---
mean_std, per_std = reward_consistency(REWARD_SCORES_RUNS)
if mean_std is not None:
    print(f"  Per-prompt standard deviations: {per_std}")
    print(f"  Mean consistency (avg std): {mean_std:.4f}")
    print(f"  (lower is better — this model is {'consistent' if mean_std < 0.02 else 'inconsistent'})")
else:
    print("  (implement reward_consistency to see results)")

# =============================================================================
# EXERCISE 3 — Alignment Method Comparison Table
# =============================================================================
# Print a comparison table of three alignment methods:
#   | Method | Pref Accuracy | KL from Ref | Human Eval Score |
#
# Format requirements:
#   - Column widths should be fixed (use string formatting)
#   - Include a header row with dashes separator
#   - Mark the BEST value in each column with an asterisk (*)
#   - Best Pref Accuracy = highest
#   - Best KL from Ref   = lowest (staying close to reference is good)
#   - Best Human Eval    = highest
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 3 — Alignment Method Comparison Table")
print("=" * 60)
print()
print("  Data:")
print(f"    Methods: {ALIGNMENT_METHODS}")
print(f"    Pref Accuracy:   {PREF_ACCURACIES}")
print(f"    KL from Ref:     {KL_FROM_REF}")
print(f"    Human Eval (0-10): {HUMAN_EVAL_SCORES}")
print()
print("  Task: print a formatted table with column headers.")
print("  Hint: print(f'| {col1:<10} | {col2:<15} | ...')")
print("  Hint: np.argmax(arr) finds index of maximum; np.argmin finds minimum")
print()

def print_alignment_comparison(methods, pref_accs, kl_vals, human_scores):
    """
    Print a formatted comparison table of alignment methods.
    Marks the best value in each column with *.

    Parameters:
      methods      : list of str — method names
      pref_accs    : list of float — preference accuracy per method
      kl_vals      : list of float — KL from reference per method
      human_scores : list of float — human evaluation scores per method
    """
    # TODO: Step 1 — find best index for each metric
    # best pref_acc  = highest -> np.argmax(pref_accs)
    # best kl        = lowest  -> np.argmin(kl_vals)
    # best human     = highest -> np.argmax(human_scores)
    best_pref  = None    # np.argmax(pref_accs)
    best_kl    = None    # np.argmin(kl_vals)
    best_human = None    # np.argmax(human_scores)

    # TODO: Step 2 — print the header row
    # Use fixed-width string formatting: f"| {'Method':<8} | {'Pref Accuracy':<14} | ..."
    print(None)    # replace with formatted header

    # TODO: Step 3 — print the separator line
    print(None)    # replace with a dashes line like "|---------|---------|...|"

    # TODO: Step 4 — print each method row with * next to best values
    for i, method in enumerate(methods):
        pref_str  = None    # f"{pref_accs[i]:.2f}{'*' if i == best_pref else ' '}"
        kl_str    = None    # f"{kl_vals[i]:.2f}{'*' if i == best_kl else ' '}"
        human_str = None    # f"{human_scores[i]:.1f}{'*' if i == best_human else ' '}"
        print(None)    # replace with formatted row

# --- Run exercise 3 ---
print_alignment_comparison(
    ALIGNMENT_METHODS, PREF_ACCURACIES, KL_FROM_REF, HUMAN_EVAL_SCORES
)

# =============================================================================
# EXERCISE 4 — Red Team Check
# =============================================================================
# A red team check scans response features for dangerous patterns.
# Returns True (FLAGGED) if any danger condition is met.
#
# Conditions:
#   - harm_score > 0.5        (harmful content threshold)
#   - manipulation > 0.6      (manipulation threshold)
#
# If EITHER condition is True, the response is flagged.
#
# C# analogy:
#   bool RedTeamCheck(float[] features) {
#     return features[HARM] > 0.5 || features[MANIP] > 0.6;
#   }
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 4 — Red Team Check")
print("=" * 60)
print()
print("  Feature vector: [harm_score, honesty, helpfulness, manipulation, respect]")
print("  FLAGGED if: harm_score > 0.5  OR  manipulation > 0.6")
print()
print("  Expected results:")
print("    Response A [0.2, ...]: SAFE   (harm=0.2, manip=0.1)")
print("    Response B [0.6, ...]: FLAGGED (harm=0.6 > 0.5)")
print("    Response C [0.3, ...]: SAFE   (harm=0.3, manip=0.2)")
print("    Response D [0.4, ...]: FLAGGED (manip=0.7 > 0.6)")
print("    Response E [0.1, ...]: SAFE   (harm=0.1, manip=0.05)")
print("    Response F [0.8, ...]: FLAGGED (both harm=0.8 AND manip=0.8!)")
print()

def red_team_check(response_features):
    """
    Check if a response is potentially harmful and should be flagged.

    Parameters:
      response_features : np.array [5] — [harm, honesty, helpfulness, manip, respect]

    Returns:
      is_flagged : bool — True if response should be blocked/revised
      reason     : str  — explanation of why it was flagged (or "SAFE")
    """
    reasons = []    # collect all triggered reasons

    # TODO: Check 1 — harm_score > 0.5
    # if response_features[IDX_HARM] > 0.5:
    #     reasons.append(f"harm={response_features[IDX_HARM]:.2f} > 0.5")
    pass    # replace with actual check

    # TODO: Check 2 — manipulation > 0.6
    # if response_features[IDX_MANIPULATION] > 0.6:
    #     reasons.append(f"manipulation={response_features[IDX_MANIPULATION]:.2f} > 0.6")
    pass    # replace with actual check

    # TODO: determine is_flagged and reason string
    is_flagged = None    # True if reasons list is not empty
    reason = None        # ", ".join(reasons) if flagged, else "SAFE"

    return is_flagged, reason

# --- Run exercise 4 on all responses ---
response_labels = ["A", "B", "C", "D", "E", "F"]    # human-readable labels
print("  Red team results:")
print(f"  {'Resp':<6} {'Harm':<6} {'Manip':<7} {'Result':<10} {'Reason'}")
print(f"  {'-'*60}")
for label, features in zip(response_labels, RESPONSE_FEATURES_TO_CHECK):
    flagged, reason = red_team_check(features)    # call our check function
    if flagged is not None:
        result_str = "FLAGGED" if flagged else "SAFE"    # display result
        harm_val = features[IDX_HARM]                    # extract harm for display
        manip_val = features[IDX_MANIPULATION]           # extract manip for display
        print(f"  {label:<6} {harm_val:<6.2f} {manip_val:<7.2f} {result_str:<10} {reason}")
    else:
        print(f"  {label:<6} ...  (implement red_team_check first)")

# Summary stats
if any(red_team_check(f)[0] is not None for f in RESPONSE_FEATURES_TO_CHECK):
    flagged_count = sum(
        1 for f in RESPONSE_FEATURES_TO_CHECK
        if red_team_check(f)[0] is True    # count flagged responses
    )
    total = len(RESPONSE_FEATURES_TO_CHECK)
    print(f"\n  Total: {flagged_count}/{total} responses flagged")
    print(f"  Flag rate: {flagged_count/total:.0%}")

# =============================================================================
# BONUS — Complete Alignment Report
# =============================================================================
# If all exercises are implemented, print a combined alignment report.

print("\n" + "=" * 60)
print("BONUS — Combined Alignment Report Summary")
print("=" * 60)
print()
print("  Run all exercises first, then this section prints the full report.")
print()

# Collect all computed metrics (will print None if exercises not done)
acc_val    = preference_accuracy(MODEL_SCORES_CHOSEN, MODEL_SCORES_REJECTED)
cons_val, _ = reward_consistency(REWARD_SCORES_RUNS)

print("  ALIGNMENT EVALUATION REPORT")
print("  " + "-" * 40)
print(f"  Preference Accuracy:    {acc_val}")
print(f"  Reward Consistency:     {cons_val}  (avg std, lower = better)")
print(f"  Best Method by Pref Acc: {ALIGNMENT_METHODS[int(np.argmax(PREF_ACCURACIES))] if acc_val else 'N/A'}")
print(f"  Best Method by Human Eval: {ALIGNMENT_METHODS[int(np.argmax(HUMAN_EVAL_SCORES))]}")
if all(red_team_check(f)[0] is not None for f in RESPONSE_FEATURES_TO_CHECK):
    fr = sum(1 for f in RESPONSE_FEATURES_TO_CHECK if red_team_check(f)[0])
    print(f"  Red Team Flag Rate:     {fr}/{len(RESPONSE_FEATURES_TO_CHECK)} responses flagged")

# =============================================================================
# SOLUTIONS (read ONLY after attempting!)
# =============================================================================
"""
SOLUTION 1 — Preference Accuracy:

def preference_accuracy(model_scores_chosen, model_scores_rejected):
    correct = model_scores_chosen > model_scores_rejected
    accuracy = np.mean(correct)
    return float(accuracy)
# Result: 7/10 = 0.7  (pairs 2, 7, 9 are wrong: indices 1, 6, 8)


SOLUTION 2 — Reward Consistency:

def reward_consistency(scores_runs):
    per_prompt_std = np.std(scores_runs, axis=0)
    mean_std = np.mean(per_prompt_std)
    return float(mean_std), per_prompt_std
# Expected: per_prompt_std ≈ [0.014, 0.021, 0.018, 0.015, 0.015]
# mean_std ≈ 0.017  (very consistent model)


SOLUTION 3 — Comparison Table:

def print_alignment_comparison(methods, pref_accs, kl_vals, human_scores):
    best_pref  = int(np.argmax(pref_accs))
    best_kl    = int(np.argmin(kl_vals))
    best_human = int(np.argmax(human_scores))

    # Header
    print(f"  | {'Method':<8} | {'Pref Accuracy':<14} | {'KL from Ref':<12} | {'Human Eval':<12} |")
    print(f"  |{'-'*10}|{'-'*16}|{'-'*14}|{'-'*14}|")

    for i, method in enumerate(methods):
        pref_str  = f"{pref_accs[i]:.2f}{'*' if i == best_pref else ' '}"
        kl_str    = f"{kl_vals[i]:.2f}{'*' if i == best_kl else ' '}"
        human_str = f"{human_scores[i]:.1f}{'*' if i == best_human else ' '}"
        print(f"  | {method:<8} | {pref_str:<14} | {kl_str:<12} | {human_str:<12} |")

# Expected output:
#   | Method   | Pref Accuracy  | KL from Ref  | Human Eval   |
#   |----------|----------------|--------------|--------------|
#   | RLHF     | 0.82           | 0.15         | 7.2          |
#   | DPO      | 0.85*          | 0.08*        | 7.8*         |
#   | CAI      | 0.80           | 0.12         | 7.5          |


SOLUTION 4 — Red Team Check:

def red_team_check(response_features):
    reasons = []
    if response_features[IDX_HARM] > 0.5:
        reasons.append(f"harm={response_features[IDX_HARM]:.2f} > 0.5")
    if response_features[IDX_MANIPULATION] > 0.6:
        reasons.append(f"manipulation={response_features[IDX_MANIPULATION]:.2f} > 0.6")
    is_flagged = len(reasons) > 0
    reason = ", ".join(reasons) if is_flagged else "SAFE"
    return is_flagged, reason

# Expected:
#   A: SAFE      B: FLAGGED (harm)    C: SAFE
#   D: FLAGGED (manip)    E: SAFE    F: FLAGGED (both)
# 3 out of 6 flagged = 50% flag rate
"""

print("\n" + "=" * 70)
print("EXERCISE 05 COMPLETE — Check your output, then read the solutions!")
print("=" * 70)
