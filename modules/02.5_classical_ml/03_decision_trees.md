# Lesson 03 - Decision Trees

## What Is a Decision Tree?

A **decision tree** is a model that makes predictions by asking a series of
yes/no questions about the input features.

Think of it like the game "20 Questions". You start with a broad question
("Is it bigger than a breadbox?") and narrow down based on the answer.

A decision tree automatically learns which questions to ask and in what order
by analyzing your training data.

---

## C# Analogy: Auto-Generated if-else Chains

In C#, you might write classification logic like this:

```csharp
string ClassifyEmail(double spamScore, int linkCount, bool hasCapsLock)
{
    if (spamScore > 0.8)
    {
        return "SPAM";
    }
    else if (linkCount > 5)
    {
        if (hasCapsLock)
            return "SPAM";
        else
            return "REVIEW";
    }
    else
    {
        return "LEGITIMATE";
    }
}
```

A decision tree is an algorithm that AUTOMATICALLY generates this kind of
nested if-else logic from training data. You do not write the conditions
yourself; the algorithm finds them.

---

## Anatomy of a Decision Tree

```
                    +---------------------+
                    | spam_score > 0.8 ?  |  <- ROOT NODE (first question)
                    +---------------------+
                      YES /         \ NO
                         /           \
              +----------+         +------------------+
              |   SPAM   |         | link_count > 5 ? |  <- INTERNAL NODE
              +----------+         +------------------+
              (leaf node)            YES /     \ NO
                                       /       \
                             +-----------+  +-----------+
                             | has_caps? |  | LEGIT     |  <- LEAF NODE
                             +-----------+  +-----------+
                              YES / \ NO
                                 /   \
                          +------+  +--------+
                          | SPAM |  | REVIEW |
                          +------+  +--------+
```

**Terms:**
- **Root node**: the very first split (top of the tree)
- **Internal node**: a decision point (has children below it)
- **Leaf node**: a final answer, no more splits
- **Depth**: how many levels deep the tree goes

---

## How Does the Tree Decide Where to Split?

The key question is: **which feature and threshold gives the best split?**

We want splits that produce groups that are as "pure" as possible.
A **pure** group contains mostly one class (e.g., 90% spam or 90% legit).

Two common measures of impurity:

### Gini Impurity

```
Gini = 1 - sum(p_i^2)

Where p_i is the proportion of class i in the current node.

Example: a node has 70% spam (p_spam = 0.7) and 30% legit (p_legit = 0.3)
Gini = 1 - (0.7^2 + 0.3^2)
     = 1 - (0.49 + 0.09)
     = 1 - 0.58
     = 0.42

A perfectly pure node (all one class): Gini = 1 - 1.0^2 = 0 (perfect!)
A completely mixed node (50/50):        Gini = 1 - (0.5^2 + 0.5^2) = 0.5 (worst)
```

Lower Gini = more pure = better.

### Entropy (Information Gain)

```
Entropy = -sum( p_i * log2(p_i) )

Same example: 70% spam, 30% legit
Entropy = -(0.7 * log2(0.7) + 0.3 * log2(0.3))
        = -(0.7 * (-0.514) + 0.3 * (-1.737))
        = -(- 0.360   -   0.521)
        = 0.881

Pure node:   Entropy = -(1.0 * log2(1.0)) = -(1.0 * 0) = 0  (perfect!)
50/50 node:  Entropy = -(0.5 * log2(0.5) * 2) = 1.0   (worst)
```

Lower entropy = more information gained = better split.

**Information Gain** = entropy before split - weighted average entropy after split.
The tree picks the split with the highest information gain.

---

## Diagram: Information Gain

```
Before split: 10 spam, 10 legit (50/50 mixed, high entropy = 1.0)

Try splitting on spam_score > 0.8:
  Left branch (score > 0.8):  9 spam, 1 legit  -> entropy = 0.47 (fairly pure)
  Right branch (score <= 0.8): 1 spam, 9 legit -> entropy = 0.47 (fairly pure)

Weighted entropy after = (10/20)*0.47 + (10/20)*0.47 = 0.47

Information Gain = 1.0 - 0.47 = 0.53  <- big gain, good split!

Try splitting on has_capslock:
  Left branch (capslock=True):  6 spam, 4 legit -> entropy = 0.97 (still messy)
  Right branch (capslock=False): 4 spam, 6 legit -> entropy = 0.97

Information Gain = 1.0 - 0.97 = 0.03  <- tiny gain, bad split

Decision: Split on spam_score > 0.8 first (higher information gain).
```

The algorithm tries ALL features and ALL possible thresholds, picks the best one,
then repeats recursively on each sub-group.

---

## The Training Algorithm (Pseudocode)

```
function build_tree(X, y, depth):
    if depth == max_depth OR all samples have same label:
        return LeafNode(most_common_class(y))

    best_feature, best_threshold = find_best_split(X, y)

    left_mask  = X[:, best_feature] <= best_threshold
    right_mask = X[:, best_feature] >  best_threshold

    left_subtree  = build_tree(X[left_mask],  y[left_mask],  depth+1)
    right_subtree = build_tree(X[right_mask], y[right_mask], depth+1)

    return InternalNode(best_feature, best_threshold,
                        left_subtree, right_subtree)
```

This is a **recursive** algorithm. Each node asks: "Should I split further
or return a prediction?" It stops when it reaches the max depth or when
a node is already pure.

---

## Hyperparameters of Decision Trees

| Hyperparameter  | What It Controls             | If Too Small  | If Too Large |
|-----------------|------------------------------|---------------|--------------|
| max_depth       | How deep the tree can grow   | Underfits     | Overfits     |
| min_samples_leaf| Min samples needed at a leaf | Overfits      | Underfits    |
| criterion       | Gini or entropy (split rule) | -             | -            |

**The biggest risk with decision trees is overfitting.** A tree with unlimited
depth will perfectly memorize the training data (every leaf has one sample).
Always set max_depth or min_samples_leaf.

---

## Decision Trees for Regression

Decision trees also work for regression (predicting continuous values):
- Instead of a class label, each leaf returns the **mean of training targets**
  that fell into that leaf
- Instead of Gini/entropy, we use variance reduction as the split criterion

---

## Connection to Neural Networks

Decision trees are NOT used inside neural networks directly, but:
- The concept of **hierarchical feature splitting** maps to layers in a network
- Random Forests (next lesson) are the closest classical ML equivalent to
  ensemble/mixture-of-experts architectures
- The concept of **depth** and **width** (tree depth vs. tree count) is
  directly analogous to **layers** and **width** in a neural network

---

## Quiz

**Question 1:**
A leaf node in a classification tree contains 8 samples of class A and
2 samples of class B. What is the Gini impurity of this node?

a) 0
b) 0.32
c) 0.5
d) 1.0

**Answer: b) 0.32**
Explanation:
  p_A = 8/10 = 0.8
  p_B = 2/10 = 0.2
  Gini = 1 - (0.8^2 + 0.2^2)
       = 1 - (0.64 + 0.04)
       = 1 - 0.68
       = 0.32

---

**Question 2:**
You train a decision tree with max_depth=100 on 200 training samples.
What will likely happen?

a) The tree will underfit and have high training error
b) The tree will be perfectly balanced
c) The tree will overfit: near-zero training error but high test error
d) The tree cannot be built with depth > n_samples

**Answer: c) The tree will overfit: near-zero training error but high test error**
Explanation: A very deep tree will create a leaf for nearly every training sample,
memorizing the data perfectly. It will fail on new data because it learned noise
instead of general patterns.

---

**Question 3:**
The algorithm tries splitting on feature "age" at threshold 30, producing:
  Left group (age <= 30): 5 class A, 5 class B
  Right group (age > 30): 9 class A, 1 class B

It also tries feature "income" at threshold 50000, producing:
  Left group (income <= 50000): 8 class A, 2 class B
  Right group (income > 50000): 6 class A, 4 class B

Which split should the tree choose and why?

a) Age split: it splits the data into equal halves
b) Income split: the groups are more balanced in size
c) Age split: the right group is much purer, giving higher information gain
d) Neither: the tree should not split at all

**Answer: c) Age split: the right group is much purer, giving higher information gain**
Explanation: The "age > 30" group is 90% class A (very pure, Gini = 0.18).
Even though the left group is 50/50, the overall impurity reduction is higher.
Income split produces two moderately impure groups, giving less information gain.
