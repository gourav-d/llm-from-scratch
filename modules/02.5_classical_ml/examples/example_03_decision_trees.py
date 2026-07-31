"""
Module 02.5 - Classical Machine Learning
Example 03: Decision Tree from Scratch

GLOSSARY
--------
Node         : One decision point in the tree. Contains a question (feature + threshold).
               C# analogy: an if-statement in a nested if-else chain.

Leaf Node    : A terminal node with no children. Returns a prediction.
               C# analogy: the return statement at the end of an if-else branch.

Gini Impurity: Measure of how mixed the classes are at a node.
               0 = perfectly pure (all one class). 0.5 = maximally mixed (50/50).
               Formula: 1 - sum(p_i^2) for each class proportion p_i.

Information  : How much a split reduces uncertainty.
Gain           Higher = better split. We pick the split with highest information gain.

Threshold    : The cutoff value for a feature split.
               Example: "age > 30" -- 30 is the threshold.

Recursive    : A function that calls itself. Decision trees are built recursively:
               each sub-tree is built by calling build_tree on a subset.
               C# analogy: a recursive method that processes sub-problems.

max_depth    : Hyperparameter. Limits tree depth to prevent overfitting.
               Like a recursion depth limit in C#.
"""

import numpy as np   # NumPy: all array operations

print("=" * 60)
print("Example 03: Decision Tree Classifier from Scratch")
print("=" * 60)
print()

# ----------------------------------------------------------------
# STEP 1: Create a toy classification dataset
# ----------------------------------------------------------------
# We will classify emails as spam (1) or not spam (0)
# Features: [spam_score, link_count]
# Label: 0 = legit, 1 = spam

print("--- Step 1: Dataset ---")
print()

# Each row: [spam_score_0_to_1, link_count]
X = np.array([
    [0.2,  1],   # Low spam score, 1 link  -> legit
    [0.9,  8],   # High spam score, 8 links -> spam
    [0.1,  0],   # Very low score, 0 links  -> legit
    [0.85, 10],  # High score, 10 links     -> spam
    [0.3,  2],   # Moderate score, 2 links  -> legit
    [0.7,  6],   # High score, 6 links      -> spam
    [0.15, 1],   # Low score, 1 link        -> legit
    [0.95, 12],  # Very high score, 12 links-> spam
    [0.4,  3],   # Moderate score, 3 links  -> legit
    [0.6,  7],   # Borderline, 7 links      -> spam
], dtype=float)

y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 0=legit, 1=spam

print(f"X shape: {X.shape}  (10 emails, 2 features)")
print(f"y shape: {y.shape}  (10 labels: 0=legit 1=spam)")
print(f"Class distribution: {np.sum(y==0)} legit, {np.sum(y==1)} spam")
print()

# ----------------------------------------------------------------
# STEP 2: Core functions - Gini impurity and best split
# ----------------------------------------------------------------

def gini_impurity(y):
    """
    Compute Gini impurity of a set of labels.

    Formula: Gini = 1 - sum(p_i^2)
    where p_i = proportion of class i.

    Pure node (all same class): Gini = 0
    50/50 mixed:                Gini = 0.5 (for binary classification)

    C# analogy:
        double GiniImpurity(int[] labels) {
            var counts = labels.GroupBy(x => x)
                              .ToDictionary(g => g.Key, g => g.Count());
            double n = labels.Length;
            return 1 - counts.Values.Sum(c => Math.Pow(c / n, 2));
        }
    """
    n = len(y)                                  # Total number of samples in this node
    if n == 0:                                  # Edge case: empty node
        return 0.0

    classes = np.unique(y)                      # Get unique class labels [0, 1]
    gini = 1.0                                  # Start at 1, subtract squared proportions

    for cls in classes:                         # For each unique class...
        p = np.sum(y == cls) / n               # Proportion of this class
        gini -= p ** 2                          # Subtract p^2 from running total

    return gini                                 # Lower = more pure

def best_split(X, y):
    """
    Find the best feature and threshold to split the data.

    Strategy: try EVERY feature and EVERY unique value as a threshold.
    Pick the split that gives the highest information gain.

    Information Gain = Gini_parent - weighted_avg(Gini_left, Gini_right)

    C# analogy:
        // Loop over all features and all unique thresholds
        // Track the split that gives minimum weighted impurity
    """
    n_samples, n_features = X.shape            # Shape of the current data chunk
    best_gain = -1.0                           # Start with worst possible gain
    best_feature = None                        # Which feature to split on
    best_threshold = None                      # The cutoff value

    parent_gini = gini_impurity(y)             # Impurity BEFORE splitting

    for feature_idx in range(n_features):      # Try each feature
        thresholds = np.unique(X[:, feature_idx])   # Unique values of this feature
        # Trying each unique value as a potential cutoff

        for threshold in thresholds:           # Try each unique value as cutoff
            # Split samples into left (<= threshold) and right (> threshold)
            left_mask  = X[:, feature_idx] <= threshold
            right_mask = X[:, feature_idx] >  threshold

            y_left  = y[left_mask]             # Labels for left group
            y_right = y[right_mask]            # Labels for right group

            if len(y_left) == 0 or len(y_right) == 0:
                continue                        # Skip trivial splits (one side empty)

            # Weighted average Gini after the split
            n = len(y)                         # Total samples in this node
            gini_left  = gini_impurity(y_left)  * len(y_left)  / n
            gini_right = gini_impurity(y_right) * len(y_right) / n
            weighted_gini = gini_left + gini_right

            gain = parent_gini - weighted_gini  # How much did impurity decrease?

            if gain > best_gain:               # New best split found
                best_gain      = gain
                best_feature   = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

# ----------------------------------------------------------------
# STEP 3: Build the tree (recursive)
# ----------------------------------------------------------------
# The tree is a nested Python dict (like a JSON object).
# C# analogy: a recursive TreeNode<T> class with Left and Right children.

def build_tree(X, y, depth=0, max_depth=3):
    """
    Recursively build a decision tree.

    Base cases (stop splitting):
    1. Reached max_depth
    2. All samples have the same label (pure node)
    3. No good split exists

    Returns a dict representing a tree node:
      {'leaf': True,  'prediction': 0 or 1}              <- leaf node
      {'leaf': False, 'feature': j, 'threshold': t,      <- decision node
       'left': subtree, 'right': subtree}
    """
    n_samples = len(y)

    # Check if all labels are the same (pure node)
    if len(np.unique(y)) == 1:                  # Only one class left
        return {'leaf': True, 'prediction': y[0]}

    # Check if we've hit maximum depth
    if depth >= max_depth:
        most_common = np.bincount(y.astype(int)).argmax()  # Most frequent class
        return {'leaf': True, 'prediction': most_common}

    # Find the best feature and threshold to split on
    feature, threshold, gain = best_split(X, y)

    if feature is None or gain <= 0:           # No beneficial split found
        most_common = np.bincount(y.astype(int)).argmax()
        return {'leaf': True, 'prediction': most_common}

    # Split the data
    left_mask  = X[:, feature] <= threshold
    right_mask = X[:, feature] >  threshold

    # Recursively build left and right subtrees
    left_subtree  = build_tree(X[left_mask],  y[left_mask],  depth + 1, max_depth)
    right_subtree = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)

    return {
        'leaf':      False,
        'feature':   feature,                  # Which feature column to check
        'threshold': threshold,                # The cutoff value
        'left':      left_subtree,             # Subtree for samples <= threshold
        'right':     right_subtree,            # Subtree for samples > threshold
    }

def predict_one(node, x):
    """
    Predict the class for a SINGLE sample x by walking down the tree.

    C# analogy: recursively traverse a binary tree:
        int Predict(Node node, double[] x) {
            if (node.IsLeaf) return node.Prediction;
            if (x[node.Feature] <= node.Threshold)
                return Predict(node.Left, x);
            else
                return Predict(node.Right, x);
        }
    """
    if node['leaf']:                           # Hit a leaf: return prediction
        return node['prediction']

    feature   = node['feature']               # Which feature to check
    threshold = node['threshold']             # What threshold to compare against

    if x[feature] <= threshold:               # Go left if below threshold
        return predict_one(node['left'], x)
    else:                                     # Go right if above threshold
        return predict_one(node['right'], x)

def predict(tree, X):
    """Predict classes for all rows of X."""
    return np.array([predict_one(tree, x) for x in X])  # Predict each row

# ----------------------------------------------------------------
# STEP 4: Train the tree and evaluate
# ----------------------------------------------------------------

print("--- Step 2-4: Train and Evaluate ---")
print()

# Show Gini impurity at root before any split
root_gini = gini_impurity(y)
print(f"Root Gini impurity (before split): {root_gini:.4f}")
print()

# Find and show the best first split
feat, thresh, gain = best_split(X, y)
feature_names = ['spam_score', 'link_count']
print(f"Best first split: {feature_names[feat]} <= {thresh:.2f}")
print(f"Information gain: {gain:.4f}")
print()

# Build the tree
tree = build_tree(X, y, max_depth=3)
print("Tree built successfully.")
print()

# Make predictions
y_pred = predict(tree, X)
accuracy = np.mean(y_pred == y)               # Fraction of correct predictions

print(f"{'Email':>7}  {'Actual':>8}  {'Predicted':>10}  {'Correct':>8}")
print("-" * 40)
labels = {0: 'legit', 1: 'spam'}
for i in range(len(y)):
    correct = "YES" if y_pred[i] == y[i] else "WRONG"
    print(f"{i+1:>7}  {labels[y[i]]:>8}  {labels[int(y_pred[i])]:>10}  {correct:>8}")

print()
print(f"Training accuracy: {accuracy*100:.1f}%")
print()

# ----------------------------------------------------------------
# STEP 5: Print the tree structure
# ----------------------------------------------------------------

def print_tree(node, depth=0, prefix="Root"):
    """Print a human-readable tree structure."""
    indent = "  " * depth                     # Indentation based on depth
    feature_names = ['spam_score', 'link_count']

    if node['leaf']:
        label = {0: 'LEGIT', 1: 'SPAM'}[node['prediction']]
        print(f"{indent}[{prefix}] PREDICT: {label}")
    else:
        feat_name = feature_names[node['feature']]
        print(f"{indent}[{prefix}] IF {feat_name} <= {node['threshold']:.2f}:")
        print_tree(node['left'],  depth + 1, "LEFT  (YES)")
        print_tree(node['right'], depth + 1, "RIGHT (NO) ")

print("--- Step 5: Tree Structure ---")
print()
print_tree(tree)
print()

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print("A decision tree is a hierarchy of if-else decisions.")
print("At each node, we pick the feature and threshold that")
print("reduces impurity (Gini) the most.")
print()
print("Training is recursive: build left subtree, then right.")
print("Prediction: walk from root to a leaf, follow the splits.")
print()
print("Key hyperparameter: max_depth")
print("  Too small -> underfits (missing real patterns)")
print("  Too large -> overfits (memorizes noise)")
print()
print("Next up: Random Forests combine many trees to fix overfitting!")
