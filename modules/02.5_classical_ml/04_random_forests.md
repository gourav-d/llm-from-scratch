# Lesson 04 - Random Forests

## The Problem with a Single Decision Tree

From Lesson 03 you learned that decision trees can overfit badly if grown too deep.

Even with max_depth set carefully, a single tree is **high variance**:
- Small changes in training data lead to very different trees
- One noisy training sample can completely change the structure of the tree
- The tree is fragile

**Solution: don't rely on one tree. Build many trees and average their votes.**

This is a **Random Forest**.

---

## What Is a Random Forest?

A Random Forest is an **ensemble** of decision trees. Each tree is trained
independently on a random subset of the training data, using a random subset
of features at each split.

At prediction time, all trees vote (classification) or their predictions are
averaged (regression).

```
Training data --> [Tree 1, Tree 2, ..., Tree N] --> Vote/Average --> Final answer
```

The randomness in the name "Random Forest" comes from two sources:
1. **Random data**: each tree is trained on a different bootstrap sample
2. **Random features**: at each split, only a random subset of features is considered

---

## C# Analogy: Parallel Simulations with Different Scenarios

Imagine you are forecasting whether a software project will be on time.

Instead of one estimate, you run 100 simulations:
- Each simulation uses slightly different assumptions about team velocity
- Each simulation uses a different historical project as a baseline
- You run all 100 in parallel (like `Parallel.For` in C#)
- The final answer is the majority outcome: "60 simulations say ON TIME,
  40 say DELAYED, so we predict ON TIME with 60% confidence"

This is exactly how a Random Forest works. Individual "simulations" (trees)
may be wrong, but collectively they average out to a much more reliable answer.

---

## Step 1: Bootstrap Sampling (Bagging)

**Bagging** = Bootstrap Aggregating.

For each tree, you create a new training set by sampling your original data
**with replacement**. "With replacement" means the same sample can appear more
than once in the new set.

```
Original data (5 samples): [A, B, C, D, E]

Tree 1 gets: [A, C, C, D, A]  <- C and A appear twice, B and E are missing
Tree 2 gets: [B, B, D, E, C]  <- B appears twice, A is missing
Tree 3 gets: [E, A, B, B, D]  <- B appears twice, C is missing
...
```

On average, each bootstrap sample includes about **63%** of the original data.
The remaining ~37% (called "out-of-bag" samples) can be used to evaluate the
tree without needing a separate validation set.

---

## Step 2: Random Feature Selection at Each Split

When building each tree, at every internal node, instead of considering ALL
features for the split, the algorithm randomly selects a small subset.

```
Suppose you have 100 features total.
At each split, randomly pick sqrt(100) = 10 features to consider.
Choose the best split from only those 10 features.

This forces different trees to use different features,
making them less correlated with each other.
```

Why does lower correlation between trees help?
If all trees made the same mistakes, averaging them would not help.
By forcing each tree to use different features, their errors are different.
Averaging independent errors cancels them out.

---

## Diagram: From Many Trees to One Answer

```
Training Data
     |
     +---bootstrap--->  Sample 1 --[Tree 1 built with random features]--> pred_1
     |
     +---bootstrap--->  Sample 2 --[Tree 2 built with random features]--> pred_2
     |
     +---bootstrap--->  Sample 3 --[Tree 3 built with random features]--> pred_3
     |
     ...
     |
     +---bootstrap--->  Sample N --[Tree N built with random features]--> pred_N

Classification: Final = majority_vote(pred_1, pred_2, ..., pred_N)
Regression:     Final = mean(pred_1, pred_2, ..., pred_N)
```

---

## Why Does Averaging Work? The Math Intuition

Suppose each tree makes a prediction with some random error:
```
pred_i = true_value + error_i
```

If errors are independent and have mean 0 (some positive, some negative):
```
mean(pred_i) = true_value + mean(error_i) = true_value + ~0 = true_value
```

Averaging cancels out individual errors! This is the same reason polls of many
people are more reliable than asking one person: random errors wash out.

The key requirement is that errors must be somewhat **independent**.
Bootstrap sampling and random feature selection achieve this.

---

## Feature Importance

A useful byproduct of Random Forests is **feature importance**:
for each feature, you measure how much it contributes to reducing impurity
across all splits in all trees. Features that appear in many important splits
get high importance scores.

This tells you which features are most predictive.
In practice, Random Forest feature importance is used to eliminate useless
columns before training more expensive models.

---

## Hyperparameters of Random Forests

| Parameter         | What It Controls                     | Typical Value       |
|-------------------|--------------------------------------|---------------------|
| n_estimators      | Number of trees                      | 100 to 1000         |
| max_depth         | Max depth of each tree               | None (unlimited) or 10-20 |
| max_features      | Features considered at each split    | sqrt(n_features)    |
| min_samples_leaf  | Min samples in a leaf node           | 1 to 5              |

More trees almost never hurts (only costs more compute time).
Typical advice: use n_estimators = 100 as a starting point.

---

## Out-of-Bag Error: Free Validation

Because each tree sees only ~63% of the data, the remaining ~37% can be used
to evaluate that tree without touching the test set.

```
Tree 1 was trained on samples {A, C, D}
Out-of-bag for Tree 1 = samples {B, E} (not seen during training)

We predict B and E using Tree 1 and measure the error.
Repeat for all trees, average the out-of-bag errors.
This gives us the Out-of-Bag (OOB) error estimate.
```

OOB error is a free, honest estimate of generalization error.
If OOB error is low, you likely do not need separate cross-validation.

---

## Comparison: Single Tree vs. Random Forest

```
+-------------------+------------------+--------------------+
| Property          | Decision Tree    | Random Forest      |
+-------------------+------------------+--------------------+
| Training speed    | Fast             | Slower (N trees)   |
| Prediction speed  | Very fast        | Slower (N trees)   |
| Interpretability  | Easy (draw it!)  | Hard (100 trees)   |
| Overfitting risk  | High             | Much lower         |
| Accuracy          | Moderate         | High               |
| Feature importance| Not reliable     | Reliable           |
+-------------------+------------------+--------------------+
```

---

## Connection to Neural Networks and LLMs

Random Forests show that combining many weak learners produces a strong learner.
This "ensemble thinking" appears throughout deep learning:

- **Dropout**: training with random neurons zeroed out is similar to training
  many slightly different sub-networks, then averaging at inference
- **Multi-head attention**: running many "attention views" in parallel and
  combining them is an ensemble within one layer
- **Mixture of Experts (MoE)**: used in large LLMs like GPT-4 -- different
  expert sub-networks handle different inputs, results are combined

Understanding Random Forests gives you the intuition for ALL of these.

---

## Quiz

**Question 1:**
You have a training set with 1000 samples. You build a Random Forest where
each tree uses bootstrap sampling. Approximately how many unique samples will
each tree see?

a) 1000 (all of them)
b) 500 (half)
c) 630 (about 63%)
d) 370 (about 37%)

**Answer: c) 630 (about 63%)**
Explanation: Bootstrap sampling (sampling with replacement) includes each unique
sample with probability 1 - (1 - 1/n)^n which converges to 1 - 1/e = ~0.632
as n grows. About 63% of samples appear in each bootstrap sample.

---

**Question 2:**
Your Random Forest has 200 features. At each split in each tree, how many
features are typically considered (default)?

a) All 200 features
b) sqrt(200) = ~14 features
c) 200 / 2 = 100 features
d) Just 1 random feature

**Answer: b) sqrt(200) = ~14 features**
Explanation: The standard default for classification random forests is to
consider sqrt(n_features) features at each split. For regression, the typical
default is n_features / 3. Both values force trees to differ from each other,
reducing correlation between trees and improving ensemble performance.

---

**Question 3:**
You increase the number of trees in a Random Forest from 100 to 1000.
What is the likely outcome?

a) The model will severely overfit
b) Training accuracy will drop significantly
c) Generalization accuracy will improve slightly or stay the same, but
   training will take longer
d) The model cannot function with more than 500 trees

**Answer: c) Generalization accuracy will improve slightly or stay the same,
but training will take longer**
Explanation: More trees in a Random Forest virtually never cause overfitting.
The ensemble error generally decreases (or plateaus) as you add more trees.
The only cost is computation time and memory. In practice, 100-500 trees
is usually sufficient for the accuracy gains to plateau.
