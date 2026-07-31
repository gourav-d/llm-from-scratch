"""
Module 02.5 - Classical Machine Learning
Exercise 04: Random Forest from Scratch

GLOSSARY
--------
Bootstrap    : Sampling WITH replacement. Same sample can appear multiple times.
               On average, each bootstrap set contains ~63% unique samples.
               The remaining ~37% are "out-of-bag" (OOB) for free evaluation.
               C# analogy: rand.Next(n) n times, allowing repeats.

Bagging      : Bootstrap + Aggregating. Train many models on bootstrap samples,
               aggregate by voting (classification) or averaging (regression).

Majority Vote: Prediction = class with most votes across all trees.
               C# analogy: votes.GroupBy(v=>v).MaxBy(g=>g.Count()).Key.

OOB Error    : Use the ~37% samples not seen by each tree to evaluate it.
               Free validation without touching the test set.

Feature Sub- : At each split, consider only sqrt(n_features) random features.
sampling       Forces trees to be different from each other (less correlated).
               C# analogy: selecting a random subset from an array using LINQ.

Random Seed  : A fixed integer passed to the RNG to make results reproducible.
               C# analogy: new Random(42).

HOW TO USE THIS FILE
--------------------
1. Read each EXERCISE section.
2. Replace `pass` with your implementation.
3. Run and check output against EXPECTED comments.

NOTE: Exercises 1-3 build helper functions. Exercises 4-5 assemble them
into a working Random Forest and evaluate it.
"""

import numpy as np   # NumPy: array operations and random number generation

print("=" * 60)
print("Exercise 04: Random Forest")
print("=" * 60)
print()

# ----------------------------------------------------------------
# Pre-built helpers (Decision Tree from Exercise 3)
# Do NOT modify these. They are provided so you can focus on the
# Random Forest logic rather than re-implementing the tree.
# ----------------------------------------------------------------

def gini_impurity(y):
    """Gini = 1 - sum(p_i^2). Returns 0 if y is empty."""
    n = len(y)
    if n == 0:
        return 0.0
    return 1.0 - sum((np.sum(y == c) / n) ** 2 for c in np.unique(y))

def best_split(X, y, feature_subset=None):
    """Find best (feature, threshold) split. feature_subset limits which features to try."""
    n_features = X.shape[1]
    features   = feature_subset if feature_subset is not None else range(n_features)
    best_gain, best_feat, best_thresh = -1.0, None, None
    parent_gini = gini_impurity(y)
    for j in features:
        for t in np.unique(X[:, j]):
            lm = X[:, j] <= t
            rm = ~lm
            if lm.sum() == 0 or rm.sum() == 0:
                continue
            n = len(y)
            g = parent_gini - (gini_impurity(y[lm])*lm.sum()/n + gini_impurity(y[rm])*rm.sum()/n)
            if g > best_gain:
                best_gain, best_feat, best_thresh = g, j, t
    return best_feat, best_thresh

def build_tree(X, y, max_depth, feature_indices, rng, depth=0):
    """Build one decision tree with random feature subsampling at each split."""
    if len(np.unique(y)) == 1:
        return {'leaf': True, 'prediction': y[0]}
    if depth >= max_depth or len(y) < 2:
        return {'leaf': True, 'prediction': np.bincount(y.astype(int)).argmax()}
    n_feat  = max(1, int(np.sqrt(X.shape[1])))
    subset  = rng.choice(X.shape[1], size=n_feat, replace=False)
    feat, t = best_split(X, y, feature_subset=subset)
    if feat is None:
        return {'leaf': True, 'prediction': np.bincount(y.astype(int)).argmax()}
    lm = X[:, feat] <= t
    return {'leaf': False, 'feature': feat, 'threshold': t,
            'left':  build_tree(X[lm],  y[lm],  max_depth, feature_indices, rng, depth+1),
            'right': build_tree(X[~lm], y[~lm], max_depth, feature_indices, rng, depth+1)}

def predict_one_tree(node, x):
    """Walk one sample down one tree."""
    while not node['leaf']:
        node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
    return node['prediction']

# ----------------------------------------------------------------
# Dataset (do not modify)
# ----------------------------------------------------------------
np.random.seed(0)
n = 100                                         # 100 samples
# Class 0: cluster near (1, 1)
X0 = np.random.randn(50, 2) + np.array([1, 1])
# Class 1: cluster near (4, 4)
X1 = np.random.randn(50, 2) + np.array([4, 4])
X_all = np.vstack([X0, X1])
y_all = np.hstack([np.zeros(50, dtype=int), np.ones(50, dtype=int)])

print(f"Dataset: {X_all.shape[0]} samples, {X_all.shape[1]} features")
print(f"Class 0: {(y_all==0).sum()} samples | Class 1: {(y_all==1).sum()} samples")
print()

# ============================================================
#  EXERCISE 1
#  Topic: Bootstrap Sampling
#
#  Background:
#    Bootstrap sampling draws n samples from the dataset WITH replacement.
#    "With replacement" means the same row can appear multiple times.
#    On average, about 63% of unique rows appear at least once.
#    The remaining ~37% are "out-of-bag" -- not seen during training.
#
#    Why? Each tree needs different training data to make different errors.
#    If all trees trained on the same data, they would all make the same
#    mistakes, and voting would not help.
#
#  Your Task:
#    Implement: bootstrap_sample(X, y, rng) -> (X_boot, y_boot, boot_indices)
#
#    Use rng.choice(n, size=n, replace=True) to sample WITH replacement.
#    Return both the sampled data AND the indices (needed for OOB tracking).
#
#  C# Analogy:
#    int[] indices = Enumerable.Range(0, n).Select(_ => rand.Next(n)).ToArray();
#    // indices may have duplicates
#
#  Hint:
#    rng.choice(n, size=n, replace=True)  <- returns n random indices WITH replacement
#    X[indices]  <- NumPy fancy indexing: selects rows at those indices
# ============================================================

def bootstrap_sample(X, y, rng):
    """
    Draw a bootstrap sample (with replacement) from (X, y).

    Parameters:
        X   : np.ndarray of shape (n_samples, n_features)
        y   : np.ndarray of shape (n_samples,)
        rng : np.random.Generator (from np.random.default_rng(seed))

    Returns:
        X_boot       : np.ndarray of shape (n_samples, n_features), bootstrap X
        y_boot       : np.ndarray of shape (n_samples,), bootstrap y
        boot_indices : np.ndarray of shape (n_samples,), the sampled row indices
    """
    pass   # TODO: sample indices with replacement, return X[indices], y[indices], indices


# Test Exercise 1
print("--- Exercise 1: Bootstrap Sampling ---")
rng_test = np.random.default_rng(seed=7)
result = bootstrap_sample(X_all, y_all, rng_test)
if result is not None:
    X_b, y_b, idx = result
    unique_count = len(set(idx))
    print(f"Bootstrap sample: {len(idx)} rows drawn from {len(y_all)}")
    print(f"Unique rows included: {unique_count} (~{unique_count/len(y_all)*100:.0f}%)")
    print(f"Out-of-bag rows:      {len(y_all) - unique_count} (not seen by this tree)")
    # EXPECTED: unique_count ~= 63 out of 100
else:
    print("bootstrap_sample returned None -- check implementation")
print()

# ============================================================
#  EXERCISE 2
#  Topic: Majority Vote for Classification
#
#  Background:
#    Each tree votes for a class. The class with the most votes wins.
#
#    Example: 10 trees vote -> [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]
#    Votes for 0: 4    Votes for 1: 6    Final prediction: 1
#
#    np.bincount([1, 0, 1, 1, 0, 1, 1, 0, 1, 0]) -> [4, 6]
#    (index 0 appears 4 times, index 1 appears 6 times)
#    argmax() -> 1  (the index with the highest count)
#
#  Your Task:
#    Implement: majority_vote(votes) -> int
#    Given an array of class votes, return the winning class.
#
#  C# Analogy:
#    int MajorityVote(int[] votes) =>
#        votes.GroupBy(v => v).MaxBy(g => g.Count()).Key;
#
#  Hint:
#    np.bincount(votes) returns an array where index i = count of i in votes.
#    .argmax() returns the index of the maximum value.
# ============================================================

def majority_vote(votes):
    """
    Return the class with the most votes.

    Parameters:
        votes : np.ndarray of shape (n_trees,), integer class labels

    Returns:
        int: the winning class label
    """
    pass   # TODO: use np.bincount and .argmax()


# Test Exercise 2
print("--- Exercise 2: Majority Vote ---")
vote_tests = [
    (np.array([0, 0, 0, 1, 1]),        0, "3 zeros vs 2 ones -> 0"),
    (np.array([1, 1, 1, 1, 0]),        1, "4 ones vs 1 zero  -> 1"),
    (np.array([0, 1, 0, 1, 0, 1, 0]), 0, "4 zeros vs 3 ones -> 0"),
    (np.array([1, 1, 0, 1, 1, 0, 1]), 1, "5 ones vs 2 zeros -> 1"),
]
for votes, expected, desc in vote_tests:
    result = majority_vote(votes)
    correct = "OK" if result == expected else f"WRONG (expected {expected})"
    print(f"  {desc}: got {result}  [{correct}]")
print()

# ============================================================
#  EXERCISE 3
#  Topic: Identify Out-of-Bag Samples
#
#  Background:
#    Each tree is trained on a bootstrap sample of ~63% of data.
#    The remaining ~37% of unique samples are "out-of-bag" (OOB).
#
#    OOB samples can be used to evaluate each tree WITHOUT a separate
#    validation set. This is a free bonus of bootstrap sampling.
#
#    Given boot_indices (which rows were sampled), the OOB indices
#    are all row indices NOT in boot_indices.
#
#  Your Task:
#    Implement: oob_indices(n_total, boot_indices) -> np.ndarray
#    Return the indices that are NOT in boot_indices.
#
#  C# Analogy:
#    int[] OobIndices(int n, int[] bootIndices) =>
#        Enumerable.Range(0, n).Except(bootIndices).ToArray();
#
#  Hint:
#    set(boot_indices) creates a Python set of the sampled indices.
#    You can use set subtraction or a list comprehension.
# ============================================================

def oob_indices(n_total, boot_indices):
    """
    Find the out-of-bag sample indices.

    Parameters:
        n_total     : int, total number of samples in the original dataset
        boot_indices: np.ndarray, the indices drawn by bootstrap sampling

    Returns:
        np.ndarray of int: indices NOT in boot_indices (sorted)
    """
    pass   # TODO: return indices not in boot_indices


# Test Exercise 3
print("--- Exercise 3: OOB Indices ---")
if result is not None:
    _, _, sample_idx = bootstrap_sample(X_all, y_all, np.random.default_rng(seed=42))
    oob = oob_indices(len(y_all), sample_idx)
    if oob is not None:
        oob_overlap = set(oob) & set(sample_idx)
        print(f"Bootstrap indices: {len(set(sample_idx))} unique (from 100 total)")
        print(f"OOB indices:       {len(oob)} samples")
        print(f"Overlap (should be 0): {len(oob_overlap)}")
        print(f"Total covered: {len(set(sample_idx)) + len(oob)} (should equal 100)")
print()

# ============================================================
#  EXERCISE 4
#  Topic: Assemble the Random Forest (Train)
#
#  Background:
#    A Random Forest is built by:
#      1. For each of n_trees iterations:
#         a. Draw a bootstrap sample of the training data
#         b. Build a decision tree on that bootstrap sample
#            (with random feature subsampling at each split)
#         c. Store the tree
#         d. Track which samples were OOB for this tree
#
#    The result is a list of trees plus OOB vote information.
#
#  Your Task:
#    Implement: random_forest_train(X, y, n_trees, max_depth, seed) ->
#              (trees, oob_vote_array)
#
#    oob_vote_array: np.ndarray of shape (n_samples, n_classes)
#    oob_vote_array[i, c] = number of trees that predicted class c for sample i
#                           (only when sample i was OOB for that tree)
#
#  C# Analogy:
#    List<TreeNode> trees = new();
#    for (int t = 0; t < nTrees; t++) {
#        var (Xb, yb, idx) = BootstrapSample(X, y, rand);
#        trees.Add(BuildTree(Xb, yb, maxDepth));
#        // track OOB predictions
#    }
# ============================================================

def random_forest_train(X, y, n_trees=20, max_depth=5, seed=42):
    """
    Train a Random Forest classifier.

    Parameters:
        X       : np.ndarray of shape (n_samples, n_features)
        y       : np.ndarray of shape (n_samples,), integer labels
        n_trees : int, number of trees to build
        max_depth: int, maximum depth for each tree
        seed    : int, random seed for reproducibility

    Returns:
        trees          : list of tree dicts
        oob_vote_array : np.ndarray of shape (n_samples, n_classes)
    """
    pass   # TODO: loop n_trees times, bootstrap + build + track OOB votes


# Test Exercise 4
print("--- Exercise 4: Train Random Forest ---")
if (bootstrap_sample(X_all, y_all, np.random.default_rng(42)) is not None
        and majority_vote(np.array([0, 1])) is not None
        and oob_indices(10, np.array([0, 1, 2])) is not None):
    rf_result = random_forest_train(X_all, y_all, n_trees=30, max_depth=4, seed=42)
    if rf_result is not None:
        rf_trees, oob_votes = rf_result
        print(f"Trained {len(rf_trees)} trees successfully.")
        # Check OOB coverage
        has_oob = oob_votes.sum(axis=1) > 0
        print(f"Samples with at least 1 OOB vote: {has_oob.sum()} / {len(y_all)}")
        # EXPECTED: most or all samples have at least 1 OOB vote after 30 trees
    else:
        print("random_forest_train returned None -- check implementation")
else:
    print("Complete exercises 1-3 first")
print()

# ============================================================
#  EXERCISE 5
#  Topic: Predict with the Forest and Compute OOB Accuracy
#
#  Background:
#    Forest prediction: for each sample, collect votes from all trees,
#    then call majority_vote to get the final prediction.
#
#    OOB accuracy: for each sample i, look at oob_vote_array[i].
#    If at least one tree voted (sum > 0), predict using those OOB votes.
#    Compare to true label y[i] and compute accuracy.
#
#  Your Task:
#    Implement: forest_predict(trees, X) -> np.ndarray
#    Implement: oob_accuracy(oob_votes, y) -> float
#
#  C# Analogy:
#    int[] ForestPredict(List<TreeNode> trees, double[,] X) =>
#        Enumerable.Range(0, X.GetLength(0))
#                  .Select(i => MajorityVote(trees.Select(t => Predict(t, X.Row(i))).ToArray()))
#                  .ToArray();
# ============================================================

def forest_predict(trees, X):
    """
    Predict classes for all rows of X using majority vote.

    Parameters:
        trees : list of tree dicts
        X     : np.ndarray of shape (n_samples, n_features)

    Returns:
        np.ndarray of shape (n_samples,), predicted class labels
    """
    pass   # TODO: get all tree predictions, then majority vote per sample


def oob_accuracy(oob_votes, y):
    """
    Compute OOB (out-of-bag) accuracy.

    For each sample with at least 1 OOB vote, predict using argmax of votes.
    Return fraction of such samples where the OOB prediction is correct.

    Parameters:
        oob_votes : np.ndarray of shape (n_samples, n_classes)
        y         : np.ndarray of shape (n_samples,), true labels

    Returns:
        float: OOB accuracy between 0 and 1
    """
    pass   # TODO: mask samples with votes, compute accuracy


# Test Exercise 5
print("--- Exercise 5: Forest Predict + OOB Accuracy ---")
if rf_result is not None and majority_vote(np.array([0, 1])) is not None:
    rf_trees, oob_votes = rf_result

    train_preds = forest_predict(rf_trees, X_all)
    if train_preds is not None:
        train_acc = np.mean(train_preds == y_all)
        print(f"Training accuracy: {train_acc*100:.1f}%")

    oob_acc = oob_accuracy(oob_votes, y_all)
    if oob_acc is not None:
        print(f"OOB accuracy:      {oob_acc*100:.1f}%  <- honest estimate (no test set needed!)")
        # EXPECTED: OOB accuracy lower than training accuracy (which is optimistic)
        # EXPECTED: OOB accuracy should still be high (>= 85% for this well-separated dataset)
else:
    print("Complete exercises 1-4 first")
print()

print("=" * 60)
print("All exercises complete!")
print("Key takeaways:")
print("  Bootstrap: each tree sees ~63% of data (different subset)")
print("  OOB: free validation using the ~37% not seen by each tree")
print("  Majority vote: class with most tree votes wins")
print("  More trees = lower variance, better generalization")
print("  Random features at splits = less correlated trees = better ensemble")
print("=" * 60)
