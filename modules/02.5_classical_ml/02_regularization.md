# Lesson 02 - Regularization

## What Is Overfitting?

Imagine you are studying for an exam. Instead of learning the concepts, you
memorize every question and answer from last year's exam paper word-for-word.

On the actual exam (with new questions), you fail completely.

This is **overfitting**: the model memorizes the training data perfectly but
cannot generalize to new, unseen data.

In regression, overfitting happens when:
- You have too many features relative to training samples
- You use polynomial features of high degree (x^10, x^20, ...)
- The model learns the noise in the data, not the true pattern

**Underfitting** is the opposite: the model is too simple to capture the real
pattern, even on training data.

---

## Diagram: Underfitting vs. Good Fit vs. Overfitting

```
True pattern: a gentle curve (like y = x^2 + noise)

UNDERFITTING (too simple - straight line):
   y |        /
     |      /
     |    /
     |  /
     +---------> x
   Misses the curve entirely. High error everywhere.

GOOD FIT (polynomial degree 2 or 3):
   y |     __
     |   /    \
     |  /      \
     | /        \
     +---------> x
   Captures the real shape. Low error on new data.

OVERFITTING (polynomial degree 10+):
   y |  /\/\/\/\
     | /        \
     |/
     +---------> x
   Perfectly fits training points but wiggles wildly between them.
   High error on new data.
```

---

## C# Analogy

Think of model weights as **developer salaries in a budget**.

You have $1 million to allocate to 100 developers. With no constraint, you
might give $900,000 to one developer and pennies to the rest (an extreme,
unstable allocation that depends too heavily on one person).

**Regularization is like adding an HR policy**: every developer must earn
a reasonable salary. No single developer can consume most of the budget.
This forces the budget to be distributed more evenly (weights stay small).

Another analogy: in C# code review, a rule says "no method should be longer
than 50 lines." This is regularization on code complexity. It forces simpler,
more general solutions that work in more situations.

---

## Ridge Regression (L2 Regularization)

Ridge adds a penalty to the loss function based on the **square** of each weight:

```
Ridge Loss = MSE + lambda * sum(W^2)

Where:
  MSE    = mean( (y_pred - y_true)^2 )  <- standard regression loss
  lambda = regularization strength (you choose this, called a hyperparameter)
  sum(W^2) = w1^2 + w2^2 + w3^2 + ...  <- penalty for large weights
```

Effect:
- When lambda = 0: pure regression, no penalty, can overfit
- When lambda = very large: all weights forced near zero, model underfits
- You tune lambda to find the sweet spot

**Key property of Ridge**: it **shrinks all weights toward zero** but rarely
makes them exactly zero. Every feature still contributes a little.

The modified Normal Equation for Ridge is:

```
W = inv(X.T @ X + lambda * I) @ X.T @ y

Where I is the identity matrix (diagonal of 1s, zeros elsewhere).
Adding lambda * I ensures the matrix is always invertible!
```

This is a nice bonus: Ridge regression solves the singular matrix problem.

---

## Lasso Regression (L1 Regularization)

Lasso adds a penalty based on the **absolute value** of each weight:

```
Lasso Loss = MSE + lambda * sum(|W|)

Where:
  sum(|W|) = |w1| + |w2| + |w3| + ...  <- penalty for weight magnitude
```

**Key property of Lasso**: it can drive weights to **exactly zero**.
This means Lasso performs automatic **feature selection** -- it discards
features that do not contribute meaningfully to predictions.

Example: if you have 100 features but only 5 actually matter, Lasso will
set 95 weights to exactly zero. Ridge will shrink them all but keep them nonzero.

---

## Comparison: Ridge vs. Lasso

```
+------------------+---------------------------+---------------------------+
| Property         | Ridge (L2)                | Lasso (L1)                |
+------------------+---------------------------+---------------------------+
| Penalty term     | lambda * sum(W^2)         | lambda * sum(|W|)         |
| Effect on weights| Shrinks toward zero       | Can set to exactly zero   |
| Feature selection| No (keeps all features)   | Yes (eliminates features) |
| Best when        | All features are useful   | Only a few features matter|
| Closed-form soln | Yes (modified Normal Eq.) | No (need iterative solver) |
+------------------+---------------------------+---------------------------+
```

---

## Polynomial Features

What if the true relationship is curved, not linear?
For example: price grows quadratically with size (bigger houses are
disproportionately expensive).

You can handle this with **polynomial features**: add columns like x^2, x^3.

```
Original X (1 feature):
  [1, 2, 3, 4, 5]

After adding polynomial features (degree 2):
  Original:  [1, 2, 3, 4, 5]
  Squared:   [1, 4, 9, 16, 25]

New X matrix:
  [[1, 1],
   [2, 4],
   [3, 9],
   [4, 16],
   [5, 25]]

Now run linear regression on this expanded X.
The model can fit curves even though it uses linear algebra!
```

This is powerful but dangerous: high-degree polynomials cause overfitting.
You must combine polynomial features with regularization.

---

## Diagram: How Lambda Controls Complexity

```
lambda = 0          lambda = 1          lambda = 100
(no regularization) (moderate)          (heavy)

   y | /\/\/\/\         y |  __            y | ----
     |/        \          | /  \             |
     |          \         |/    \            |
     +---------> x        +-------> x        +-------> x
   Overfits badly       Good fit!         Underfits (too flat)
```

Choosing lambda is a **hyperparameter tuning** problem.
You try many values of lambda and pick the one with the best validation error.

---

## Connection to Neural Networks

In neural networks:
- **Weight decay** is Ridge regularization applied to all weight matrices
- **Dropout** is a different kind of regularization (randomly zero out neurons)
- L1 regularization is used in sparse models and attention mechanisms

Understanding Ridge/Lasso here makes it trivial to understand
"weight decay = 0.01" in a neural network config file.

---

## Quiz

**Question 1:**
You are training a model with 200 features and only 50 training samples.
Which regularization technique should you choose first?

a) No regularization
b) Ridge (L2)
c) Polynomial features
d) Increase polynomial degree

**Answer: b) Ridge (L2)**
Explanation: With far more features than samples, the model will almost certainly
overfit without regularization. Ridge is a safe default. Lasso is also valid if
you suspect only a few features matter.

---

**Question 2:**
After applying Lasso with lambda = 0.5, you find that 90 out of 100 feature
weights are exactly 0. What has Lasso done?

a) Made the model worse
b) Performed feature selection (eliminated 90 useless features)
c) Regularized using L2 penalty
d) Increased the model complexity

**Answer: b) Performed feature selection (eliminated 90 useless features)**
Explanation: Lasso's L1 penalty drives irrelevant weights to exactly zero.
This is Lasso's key advantage over Ridge: it tells you which features matter.

---

**Question 3:**
What happens to the Ridge loss when lambda = 0?

a) The model underfits
b) The penalty term disappears and Ridge becomes plain linear regression
c) All weights are forced to zero
d) The loss becomes negative

**Answer: b) The penalty term disappears and Ridge becomes plain linear regression**
Explanation: Ridge Loss = MSE + lambda * sum(W^2). When lambda = 0, the penalty term
is 0 * sum(W^2) = 0, leaving just the MSE. Ridge with lambda = 0 is identical
to ordinary least squares linear regression.
