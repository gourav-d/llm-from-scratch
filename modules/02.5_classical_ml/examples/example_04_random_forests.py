"""
Module 02.5 - Classical Machine Learning
Example 04: Random Forest from Scratch

GLOSSARY
--------
Ensemble     : A group of models whose predictions are combined.
               Combining weak learners produces a strong learner.
               C# analogy: running N parallel worker tasks and aggregating results.

Bootstrap    : Sampling WITH replacement from the training data.
               Each tree sees a different random subset of examples.
               About 63% unique samples appear in each bootstrap sample.

Bagging      : Bootstrap + Aggregating. Each model is trained on a bootstrap sample,
               then predictions are averaged (regression) or majority-voted (classification).

OOB Error    : Out-Of-Bag error. The ~37% of samples NOT in a bootstrap sample
               can be used to evaluate that tree for free, without a separate val set.

Feature      : At each split, only sqrt(n_features) randomly chosen features are
Subsampling    considered. Forces trees to be different from each other (less correlated).

Majority Vote: Classification prediction: whichever class gets the most votes wins.
               C# analogy: LINQ's GroupBy().MaxBy(g => g.Count()).Key.

Random Seed  : A fixed starting point for the random number generator.
               Using a seed makes random results reproducible.
               C# analogy: new Random(42).
"""

import numpy as np   # NumPy: all numerical operations

print("=" * 60)
print("Example 04: Random Forest from Scratch")
print("=" * 60)
print()

# ----------------------------------------------------------------
# HELPER: Minimal decision tree (reused from example 03)
# ----------------------------------------------------------------
# We define a compact version here. The key operations are:
# gini_impurity, find best split, build recursively, predict.

def gini_impurity(y):
    """Gini = 1 - sum(p_i^2) for each class proportion p_i."""
    n = len(y)
    if n == 0:
        return 0.0
    classes = np.unique(y)
    return 1.0 - sum((np.sum(y == c) / n) ** 2 for c in classes)

def best_split(X, y, feature_indices):
    """
    Find best split considering ONLY features in feature_indices.
    This is the Random Forest modification: random feature subsets.

    Instead of all features, we only check a random subset at each node.
    This forces different trees to develop different structure.

    C# analogy:
        var featureSubset = allFeatures.OrderBy(_ => rand.Next())
                                       .Take((int)Math.Sqrt(allFeatures.Length))
                                       .ToArray();
    """
    best_gain = -1.0
    best_feature = None
    best_threshold = None
    parent_gini = gini_impurity(y)

    for feat in feature_indices:               # Only consider the given features
        thresholds = np.unique(X[:, feat])     # All unique values for this feature
        for t in thresholds:
            left  = y[X[:, feat] <= t]
            right = y[X[:, feat] >  t]
            if len(left) == 0 or len(right) == 0:
                continue
            n = len(y)
            weighted = gini_impurity(left)*len(left)/n + gini_impurity(right)*len(right)/n
            gain = parent_gini - weighted
            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, feat, t

    return best_feature, best_threshold

def build_tree(X, y, max_depth, max_features, rng, depth=0):
    """
    Build one decision tree with random feature subsampling.

    rng: a numpy random number generator instance (for reproducibility).
    max_features: how many features to consider at each split.
    """
    n_features = X.shape[1]                    # Total number of features

    # Base cases: stop splitting
    if len(np.unique(y)) == 1:                 # All labels same -> leaf
        return {'leaf': True, 'prediction': y[0]}
    if depth >= max_depth or len(y) < 2:       # Too deep or too few samples
        pred = np.bincount(y.astype(int)).argmax()
        return {'leaf': True, 'prediction': pred}

    # Randomly choose which features to consider at this split
    n_consider = max(1, int(np.sqrt(n_features)))   # sqrt(n_features) is standard
    feature_subset = rng.choice(n_features, size=n_consider, replace=False)
    # replace=False means each feature can appear at most once in the subset

    feat, thresh = best_split(X, y, feature_subset)

    if feat is None:                           # No useful split found
        pred = np.bincount(y.astype(int)).argmax()
        return {'leaf': True, 'prediction': pred}

    left_mask  = X[:, feat] <= thresh
    right_mask = X[:, feat] >  thresh

    return {
        'leaf':      False,
        'feature':   feat,
        'threshold': thresh,
        'left':  build_tree(X[left_mask],  y[left_mask],  max_depth, max_features, rng, depth+1),
        'right': build_tree(X[right_mask], y[right_mask], max_depth, max_features, rng, depth+1),
    }

def predict_one(node, x):
    """Walk one sample down the tree to a leaf."""
    if node['leaf']:
        return node['prediction']
    if x[node['feature']] <= node['threshold']:
        return predict_one(node['left'], x)
    return predict_one(node['right'], x)

def tree_predict(tree, X):
    """Predict class for all rows of X using one tree."""
    return np.array([predict_one(tree, x) for x in X])

# ----------------------------------------------------------------
# STEP 1: Create a larger classification dataset
# ----------------------------------------------------------------

print("--- Step 1: Create Dataset ---")
print()

np.random.seed(42)                             # Fixed seed for reproducibility
n_samples = 80                                 # 80 training examples

# Generate two overlapping clusters (classification)
n_class0 = n_class1 = n_samples // 2          # 40 samples each

# Class 0: centered around (2, 2)
X0 = np.random.randn(n_class0, 2) * 1.5 + np.array([2, 2])
# Class 1: centered around (5, 5)
X1 = np.random.randn(n_class1, 2) * 1.5 + np.array([5, 5])

X_all = np.vstack([X0, X1])                   # Stack vertically: (80, 2)
y_all = np.hstack([np.zeros(n_class0, dtype=int),   # Labels for class 0
                   np.ones(n_class1,  dtype=int)])   # Labels for class 1

print(f"Dataset: {X_all.shape[0]} samples, {X_all.shape[1]} features")
print(f"Class 0: {np.sum(y_all==0)} samples | Class 1: {np.sum(y_all==1)} samples")
print()

# ----------------------------------------------------------------
# STEP 2: Bootstrap sampling
# ----------------------------------------------------------------

print("--- Step 2: Bootstrap Sampling Demo ---")
print()

def bootstrap_sample(X, y, rng):
    """
    Sample n rows WITH replacement from (X, y).

    With replacement means the same row can be chosen multiple times.
    On average, ~63.2% of unique rows appear at least once.
    The other ~36.8% are "out-of-bag" samples for free evaluation.

    C# analogy:
        var indices = Enumerable.Range(0, n)
                               .Select(_ => rand.Next(n))
                               .ToArray();
        // indices can have duplicates
    """
    n = len(y)                                 # Number of training samples
    indices = rng.choice(n, size=n, replace=True)  # Sample WITH replacement
    return X[indices], y[indices], indices      # Return sampled X, y, and which indices

# Demo: show how bootstrap creates different subsets
rng_demo = np.random.default_rng(seed=99)
_, _, idx1 = bootstrap_sample(X_all[:10], y_all[:10], rng_demo)
_, _, idx2 = bootstrap_sample(X_all[:10], y_all[:10], rng_demo)
_, _, idx3 = bootstrap_sample(X_all[:10], y_all[:10], rng_demo)

print("Bootstrap demo (first 10 samples only):")
print(f"  Tree 1 indices: {sorted(idx1)}  unique={len(set(idx1))}/10")
print(f"  Tree 2 indices: {sorted(idx2)}  unique={len(set(idx2))}/10")
print(f"  Tree 3 indices: {sorted(idx3)}  unique={len(set(idx3))}/10")
print("  Note: each tree sees different (possibly repeated) samples!")
print()

# ----------------------------------------------------------------
# STEP 3: Build a Random Forest
# ----------------------------------------------------------------

print("--- Step 3: Build Random Forest ---")
print()

def random_forest_fit(X, y, n_trees=50, max_depth=5, random_seed=42):
    """
    Train a random forest: n_trees independent decision trees,
    each on a bootstrap sample, with random feature selection.

    Returns:
        trees:      list of trained tree dicts
        oob_votes:  out-of-bag predictions for each sample (dict)
        oob_counts: how many trees contributed an OOB vote per sample
    """
    rng = np.random.default_rng(seed=random_seed)  # Seeded RNG for reproducibility
    n_samples = len(y)
    n_classes  = len(np.unique(y))

    trees = []                                 # Will store all trained trees
    oob_vote_matrix = np.zeros((n_samples, n_classes), dtype=int)
    # oob_vote_matrix[i, c] = number of trees that predicted class c for sample i
    # (only counting when sample i was out-of-bag for that tree)

    for tree_idx in range(n_trees):            # Build each tree
        X_boot, y_boot, boot_indices = bootstrap_sample(X, y, rng)
        # X_boot, y_boot are bootstrap copies (may have duplicates)

        tree = build_tree(X_boot, y_boot,
                          max_depth=max_depth,
                          max_features=int(np.sqrt(X.shape[1])),
                          rng=rng)
        trees.append(tree)                     # Store the trained tree

        # Compute out-of-bag (OOB) predictions
        all_indices = set(range(n_samples))    # All sample indices
        in_bag      = set(boot_indices)        # Indices that appear in bootstrap
        oob_indices = list(all_indices - in_bag)  # Indices NOT in this tree's training

        for idx in oob_indices:                # For each OOB sample...
            pred = predict_one(tree, X[idx])   # Predict using this tree
            oob_vote_matrix[idx, int(pred)] += 1  # Accumulate the vote

    return trees, oob_vote_matrix

def random_forest_predict(trees, X):
    """
    Predict by majority vote of all trees.

    For each sample, count votes from all trees.
    The class with the most votes wins.

    C# analogy:
        var votes = trees.Select(t => t.Predict(x)).ToList();
        return votes.GroupBy(v => v).MaxBy(g => g.Count()).Key;
    """
    # Collect predictions from all trees: shape (n_trees, n_samples)
    all_preds = np.array([tree_predict(t, X) for t in trees])

    # Majority vote: pick the class with most votes for each sample
    final_preds = []
    for sample_idx in range(X.shape[0]):      # For each sample...
        votes = all_preds[:, sample_idx]       # All tree predictions for this sample
        majority = np.bincount(votes.astype(int)).argmax()  # Most voted class
        final_preds.append(majority)

    return np.array(final_preds)

# Train with different numbers of trees
print(f"{'N Trees':>8}  {'Train Acc':>10}  {'OOB Acc':>9}  Note")
print("-" * 50)

for n_trees in [1, 5, 20, 50]:
    trees, oob_votes = random_forest_fit(X_all, y_all,
                                         n_trees=n_trees,
                                         max_depth=5,
                                         random_seed=42)

    # Training accuracy
    y_pred_train = random_forest_predict(trees, X_all)
    train_acc = np.mean(y_pred_train == y_all)

    # OOB accuracy: for samples with at least one OOB vote
    oob_pred = np.argmax(oob_votes, axis=1)     # Best class from OOB votes
    has_oob  = oob_votes.sum(axis=1) > 0        # Samples that got at least 1 OOB vote
    oob_acc  = np.mean(oob_pred[has_oob] == y_all[has_oob])

    note = "<- single tree (high variance)" if n_trees == 1 else ""
    print(f"{n_trees:>8}  {train_acc*100:>9.1f}%  {oob_acc*100:>8.1f}%  {note}")

print()
print("More trees -> OOB error stabilizes. Training acc stays high.")
print("OOB accuracy is a FREE estimate of generalization (no test set needed).")
print()

# ----------------------------------------------------------------
# STEP 4: Compare single tree vs forest
# ----------------------------------------------------------------

print("--- Step 4: Single Tree vs. Random Forest ---")
print()

# Build a single deep tree (likely to overfit)
rng_single = np.random.default_rng(seed=42)
single_tree = build_tree(X_all, y_all,
                          max_depth=10,        # Very deep: likely overfits
                          max_features=None,
                          rng=rng_single)
single_preds = tree_predict(single_tree, X_all)
single_acc   = np.mean(single_preds == y_all)

# Build a 50-tree forest
rf_trees, _ = random_forest_fit(X_all, y_all, n_trees=50, max_depth=5)
rf_preds     = random_forest_predict(rf_trees, X_all)
rf_acc       = np.mean(rf_preds == y_all)

print(f"Single deep tree (depth=10): training accuracy = {single_acc*100:.1f}%")
print(f"Random Forest   (50 trees) : training accuracy = {rf_acc*100:.1f}%")
print()
print("Both have high training accuracy. The difference shows on NEW data.")
print("The forest's OOB error is a better indicator of real-world performance.")
print()

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print("Random Forest = many decision trees trained on bootstrap samples")
print("with random feature subsets at each split.")
print()
print("Key ideas:")
print("  1. Bootstrap: each tree sees ~63% of data (different subset)")
print("  2. Random features: sqrt(n_features) considered at each split")
print("  3. Prediction: majority vote (classification) or average (regression)")
print("  4. OOB error: free validation estimate using the ~37% not sampled")
print()
print("Why it works:")
print("  Individual trees overfit. Averaging many uncorrelated trees")
print("  cancels out individual errors -> much better generalization.")
print()
print("Connection to LLMs:")
print("  Dropout in neural nets is similar: random sub-networks are")
print("  trained at each step, averaged at inference time.")
