"""
Module 02.5 - Classical Machine Learning
Exercise 03: Decision Tree from Scratch

GLOSSARY
--------
Gini Impurity: Measure of how mixed the classes are at a node.
               Formula: 1 - sum(p_i^2) for each class proportion p_i.
               0 = perfectly pure. 0.5 = maximally mixed (binary case).
               C# analogy: inverse of "how confident are we in one class?"

Information  : Parent impurity minus weighted child impurity after a split.
Gain           Higher = better split. We always pick the split with highest gain.

Threshold    : The cutoff value for a split. e.g., "age > 30" uses 30.

Node Dict    : We represent each tree node as a Python dictionary (like a
               C# anonymous object or Dictionary<string, object>).
               {'leaf': True, 'prediction': 1}           <- leaf node
               {'leaf': False, 'feature': 0, 'threshold': 0.5,
                'left': ..., 'right': ...}                <- split node

Recursive    : A function that calls itself. The tree-building algorithm is
               recursive: build the tree -> build the left subtree -> etc.
               C# analogy: a recursive TreeNode<T>.Build(data) method.

HOW TO USE THIS FILE
--------------------
1. Read each EXERCISE section.
2. Replace `pass` with your implementation.
3. Run and check your output against the EXPECTED comments.
"""

import numpy as np   # NumPy: array operations

print("=" * 60)
print("Exercise 03: Decision Tree Classifier")
print("=" * 60)
print()

# ----------------------------------------------------------------
# Dataset (do not modify)
# ----------------------------------------------------------------
# Classify emails: spam (1) or not spam (0)
# Features: [spam_score_0_to_1, link_count, has_money_word_0_or_1]
X = np.array([
    [0.1,  0, 0],   # legit
    [0.9,  8, 1],   # spam
    [0.2,  1, 0],   # legit
    [0.85, 9, 1],   # spam
    [0.3,  2, 0],   # legit
    [0.7,  6, 1],   # spam
    [0.15, 1, 0],   # legit
    [0.95,12, 1],   # spam
    [0.4,  3, 0],   # legit
    [0.65, 7, 1],   # spam
], dtype=float)

y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])   # 0=legit, 1=spam

feature_names = ['spam_score', 'link_count', 'has_money_word']

# ============================================================
#  EXERCISE 1
#  Topic: Gini Impurity
#
#  Background:
#    Gini impurity measures how mixed the class labels are.
#
#    Formula: Gini = 1 - sum(p_i^2)
#    where p_i = proportion of class i in the node.
#
#    Examples:
#      All spam (10/10):         Gini = 1 - 1^2        = 0.0    (pure!)
#      50% spam, 50% legit:      Gini = 1 - (0.5^2 + 0.5^2) = 0.5  (worst)
#      70% spam, 30% legit:      Gini = 1 - (0.7^2 + 0.3^2) = 0.42
#
#  Your Task:
#    Implement: gini_impurity(y) -> float
#
#  C# Analogy:
#    double GiniImpurity(int[] labels) {
#        double n = labels.Length;
#        return 1.0 - labels.Distinct()
#                           .Sum(c => Math.Pow(labels.Count(l=>l==c) / n, 2));
#    }
#
#  Hint:
#    np.unique(y)      -> unique class values
#    np.sum(y == cls)  -> count of class cls in y
# ============================================================

def gini_impurity(y):
    """
    Compute Gini impurity for an array of class labels.

    Parameters:
        y : np.ndarray of shape (n,) with integer class labels

    Returns:
        float: Gini impurity in range [0, 0.5] for binary classification
    """
    pass   # TODO: implement Gini = 1 - sum(p_i^2)


# Test Exercise 1
print("--- Exercise 1: Gini Impurity ---")
test_gini_cases = [
    (np.array([1, 1, 1, 1, 1]), "all spam    -> expected: 0.0"),
    (np.array([0, 0, 0, 0, 0]), "all legit   -> expected: 0.0"),
    (np.array([1, 0, 1, 0, 1, 0]), "50/50 split -> expected: 0.5"),
    (np.array([1, 1, 1, 0]),    "75/25 split -> expected: 0.375"),
    (y,                          "full dataset-> expected: 0.5 (balanced)"),
]
for y_test, desc in test_gini_cases:
    result = gini_impurity(y_test)
    result_str = f"{result:.4f}" if result is not None else "None"
    print(f"  {desc}: got {result_str}")
print()

# ============================================================
#  EXERCISE 2
#  Topic: Weighted Gini After a Split
#
#  Background:
#    When we split a node into left and right children, we measure the
#    improvement using Information Gain:
#
#      Gain = Gini(parent) - (n_left/n * Gini(left) + n_right/n * Gini(right))
#
#    The weighted average accounts for the sizes of each child:
#    a large impure child is worse than a small impure child.
#
#  Your Task:
#    Implement: information_gain(y, left_mask, right_mask) -> float
#
#    Parameters:
#      y          : full label array for the current node
#      left_mask  : boolean array (True where sample goes left)
#      right_mask : boolean array (True where sample goes right)
#
#  C# Analogy:
#    double InfoGain(int[] y, bool[] leftMask) {
#        var left  = y.Where((_, i) => leftMask[i]).ToArray();
#        var right = y.Where((_, i) => !leftMask[i]).ToArray();
#        double n  = y.Length;
#        return GiniImpurity(y)
#             - (left.Length/n  * GiniImpurity(left)
#             +  right.Length/n * GiniImpurity(right));
#    }
# ============================================================

def information_gain(y, left_mask, right_mask):
    """
    Compute information gain for one split.

    Parameters:
        y          : np.ndarray of shape (n,), full label array for current node
        left_mask  : np.ndarray of bool, True for samples going left
        right_mask : np.ndarray of bool, True for samples going right

    Returns:
        float: information gain (>0 = useful split, 0 = no improvement)
    """
    pass   # TODO: return Gini(parent) - weighted_avg(Gini(left), Gini(right))


# Test Exercise 2
print("--- Exercise 2: Information Gain ---")
if gini_impurity(y) is not None:
    # Try splitting on spam_score <= 0.5
    left_mask  = X[:, 0] <= 0.5      # spam_score <= 0.5
    right_mask = X[:, 0] >  0.5      # spam_score >  0.5

    gain = information_gain(y, left_mask, right_mask)
    print(f"Split on spam_score <= 0.5:")
    print(f"  Left  ({left_mask.sum()} samples, labels {y[left_mask]})")
    print(f"  Right ({right_mask.sum()} samples, labels {y[right_mask]})")
    print(f"  Information gain: {gain if gain is not None else 'None'}")
    # EXPECTED: high gain because the split perfectly separates the classes
print()

# ============================================================
#  EXERCISE 3
#  Topic: Find the Best Split
#
#  Background:
#    To find the best split, try EVERY feature and EVERY unique threshold.
#    For each candidate split, compute information gain.
#    Return the feature and threshold that give the HIGHEST information gain.
#
#    Algorithm:
#      best_gain = -1
#      for each feature j:
#          for each unique value t in X[:, j]:
#              left_mask  = X[:, j] <= t
#              right_mask = X[:, j] >  t
#              if either side is empty: skip
#              gain = information_gain(y, left_mask, right_mask)
#              if gain > best_gain: update best
#
#  Your Task:
#    Implement: best_split(X, y) -> (feature_idx, threshold, gain)
#
#  C# Analogy:
#    (int feat, double thresh, double gain) BestSplit(double[,] X, int[] y) {
#        double bestGain = -1; int bestFeat = 0; double bestThresh = 0;
#        for (int j = 0; j < X.GetLength(1); j++) {
#            foreach (double t in X.Col(j).Distinct()) {
#                // compute gain and update best if better
#            }
#        }
#        return (bestFeat, bestThresh, bestGain);
#    }
# ============================================================

def best_split(X, y):
    """
    Find the best feature and threshold to split on.

    Parameters:
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

    Returns:
        (best_feature_idx, best_threshold, best_gain)
        Returns (None, None, -1) if no useful split is found.
    """
    pass   # TODO: loop over features and thresholds, track best gain


# Test Exercise 3
print("--- Exercise 3: Best Split ---")
if gini_impurity(y) is not None and information_gain(y, X[:,0]<=0.5, X[:,0]>0.5) is not None:
    feat, thresh, gain = best_split(X, y)
    if feat is not None:
        print(f"Best split: feature '{feature_names[feat]}' <= {thresh}")
        print(f"Information gain: {gain:.4f}")
        # EXPECTED: spam_score or has_money_word likely gives perfect split
        # EXPECTED: gain should be close to 0.5 (splitting 50/50 dataset perfectly)
    else:
        print("best_split returned None -- check implementation")
print()

# ============================================================
#  EXERCISE 4
#  Topic: Predict Using a Trained Tree
#
#  Background:
#    After a tree is built (we provide a pre-built tree for you),
#    prediction is simple: walk from the root to a leaf.
#
#    At each node:
#      - If it's a leaf: return the prediction
#      - Otherwise: check if x[feature] <= threshold
#          If yes: go left; if no: go right
#
#    This is recursive: predict_one calls itself on the left or right subtree.
#
#  Your Task:
#    Implement: predict_one(node, x) -> int
#    Predict the class for a single sample x by walking the tree.
#
#  C# Analogy:
#    int PredictOne(TreeNode node, double[] x) {
#        if (node.IsLeaf) return node.Prediction;
#        return x[node.Feature] <= node.Threshold
#            ? PredictOne(node.Left, x)
#            : PredictOne(node.Right, x);
#    }
# ============================================================

# Pre-built tree for testing Exercise 4
# (You do not need to build this yourself -- it is provided)
pretrained_tree = {
    'leaf': False, 'feature': 0, 'threshold': 0.5,  # spam_score <= 0.5?
    'left':  {'leaf': True, 'prediction': 0},        # Yes -> LEGIT
    'right': {'leaf': True, 'prediction': 1},        # No  -> SPAM
}

def predict_one(node, x):
    """
    Predict the class for one sample by walking the tree.

    Parameters:
        node : dict representing a tree node
        x    : np.ndarray of shape (n_features,), one sample's features

    Returns:
        int: predicted class label
    """
    pass   # TODO: base case (leaf) + recursive case (go left or right)


# Test Exercise 4
print("--- Exercise 4: Tree Prediction ---")
test_samples = [
    (np.array([0.2, 1.0, 0.0]), 0, "low score -> should predict LEGIT (0)"),
    (np.array([0.9, 8.0, 1.0]), 1, "high score -> should predict SPAM (1)"),
    (np.array([0.5, 3.0, 0.0]), 0, "exactly 0.5 -> goes LEFT -> LEGIT (0)"),
    (np.array([0.51, 4.0, 1.0]), 1, "just above 0.5 -> RIGHT -> SPAM (1)"),
]
for x_test, expected, desc in test_samples:
    result = predict_one(pretrained_tree, x_test)
    correct = "OK" if result == expected else "WRONG"
    print(f"  {desc}: predicted {result}  [{correct}]")
print()

# ============================================================
#  EXERCISE 5
#  Topic: Compute Tree Accuracy
#
#  Background:
#    Accuracy = number of correct predictions / total predictions.
#    For each sample in X, call predict_one to get the prediction.
#    Count how many match the true label in y.
#
#  Your Task:
#    Implement: tree_accuracy(tree, X, y) -> float
#    Returns accuracy as a fraction between 0.0 and 1.0.
#
#  C# Analogy:
#    double Accuracy(TreeNode tree, double[,] X, int[] y) {
#        int correct = Enumerable.Range(0, y.Length)
#                                .Count(i => PredictOne(tree, X.Row(i)) == y[i]);
#        return (double)correct / y.Length;
#    }
# ============================================================

def tree_accuracy(tree, X, y):
    """
    Compute classification accuracy on dataset (X, y).

    Parameters:
        tree : dict representing the root of a decision tree
        X    : np.ndarray of shape (n_samples, n_features)
        y    : np.ndarray of shape (n_samples,), true labels

    Returns:
        float: accuracy between 0.0 and 1.0
    """
    pass   # TODO: predict each row of X, compare to y, return fraction correct


# Test Exercise 5
print("--- Exercise 5: Tree Accuracy ---")
if predict_one(pretrained_tree, X[0]) is not None:
    acc = tree_accuracy(pretrained_tree, X, y)
    if acc is not None:
        print(f"Accuracy of pretrained tree on dataset: {acc*100:.1f}%")
        # EXPECTED: 100% because this dataset is perfectly separable at threshold 0.5
        print()
        print("Individual predictions:")
        print(f"{'Email':>7}  {'True':>6}  {'Pred':>6}  {'OK?':>5}")
        print("-" * 30)
        for i in range(len(y)):
            pred = predict_one(pretrained_tree, X[i])
            ok = "YES" if pred == y[i] else "NO"
            print(f"{i+1:>7}  {y[i]:>6}  {pred:>6}  {ok:>5}")
print()

print("=" * 60)
print("All exercises complete!")
print("Key takeaways:")
print("  Gini measures how mixed the classes are (0=pure, 0.5=worst)")
print("  Best split = highest information gain across all features/thresholds")
print("  Prediction = walk from root to leaf following the if-else splits")
print("  Deep trees overfit; always set max_depth in practice")
print("=" * 60)
