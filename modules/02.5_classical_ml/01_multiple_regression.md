# Lesson 01 - Multiple Regression

## What Is Multiple Regression?

Imagine you are trying to predict the **price of a house**.

You know three things about each house:
- Size in square feet (e.g., 1500)
- Number of bedrooms (e.g., 3)
- Distance from city center in miles (e.g., 5)

You want a formula like this:

```
price = w1 * size + w2 * bedrooms + w3 * distance + bias
```

Where w1, w2, w3 are **weights** (how much each feature matters) and
**bias** is a constant offset (like the base price even for a tiny house
with zero bedrooms at infinite distance).

This is **multiple regression**: predicting one output from multiple inputs
using a weighted sum plus a bias term.

---

## C# Analogy

In C# you might write a formula function like this:

```csharp
double PredictPrice(double size, double bedrooms, double distance)
{
    // Hard-coded coefficients -- you would get these by running LINEST in Excel
    double w1 = 150.0;   // $150 per square foot
    double w2 = 5000.0;  // $5000 per bedroom
    double w3 = -2000.0; // -$2000 per mile from city
    double bias = 10000.0;

    return w1 * size + w2 * bedrooms + w3 * distance + bias;
}
```

**Multiple regression** is the algorithm that AUTOMATICALLY finds w1, w2, w3,
and bias from a set of training examples. You do not hard-code them.
It is like running Excel's LINEST() function but understanding the math behind it.

---

## The Design Matrix

When you have many training samples, you organize them into a matrix.

Each **row** is one training sample (one house).
Each **column** is one feature.

```
            size  bedrooms  distance
House 1  [  1500,    3,       5   ]
House 2  [  2000,    4,       2   ]
House 3  [  1200,    2,       8   ]
House 4  [  1800,    3,       3   ]
```

But we also need to handle the **bias** term. The trick is to add a column of
1s as the first column. Then the bias becomes just another weight (w0).

```
         bias  size  bedrooms  distance
House 1  [ 1,  1500,    3,       5   ]
House 2  [ 1,  2000,    4,       2   ]
House 3  [ 1,  1200,    2,       8   ]
House 4  [ 1,  1800,    3,       3   ]
```

This matrix is called the **Design Matrix**, written as **X**.
Shape: (n_samples, n_features + 1) because we added the bias column.

---

## The Prediction Formula

If we stack the weights into a column vector W:

```
W = [ w0 (bias weight),
      w1 (size weight),
      w2 (bedrooms weight),
      w3 (distance weight) ]
```

Then the predictions for ALL houses at once is just:

```
y_pred = X @ W
```

Where @ is matrix multiplication (dot product). This gives us a vector of
predictions, one per house.

```
Shape check:
  X     has shape (4, 4)    <- 4 houses, 4 columns (bias + 3 features)
  W     has shape (4,)      <- 4 weights
  y_pred = X @ W has shape (4,)  <- 4 predictions, one per house
```

---

## Diagram: How Matrix Multiplication Makes Predictions

```
X (design matrix)                  W (weights)        y_pred
+----+------+----+------+         +---------+        +---------+
| 1  | 1500 |  3 |    5 |    @    |   w0    |   =    | pred_1  |
| 1  | 2000 |  4 |    2 |         |   w1    |        | pred_2  |
| 1  | 1200 |  2 |    8 |         |   w2    |        | pred_3  |
| 1  | 1800 |  3 |    3 |         |   w3    |        | pred_4  |
+----+------+----+------+         +---------+        +---------+

Shape: (4, 4)                      (4,)               (4,)

Each prediction = 1*w0 + size*w1 + bedrooms*w2 + distance*w3
```

---

## The Normal Equation: Finding Optimal Weights in One Step

How do we find the BEST weights? The ones that minimize the total prediction error?

For linear regression, there is a closed-form (exact algebraic) solution
called the **Normal Equation**:

```
W = inv(X.T @ X) @ X.T @ y
```

Where:
- `X.T` is the transpose of X (rows and columns flipped)
- `inv(...)` is the matrix inverse
- `y` is the vector of true target values (actual house prices)

This formula looks scary but it is just algebra. It finds the W that minimizes
the sum of squared errors in one shot -- no iteration needed.

**When does it fail?**
If two features are perfectly correlated (e.g., you include both
"size in sqft" and "size in m2"), the matrix X.T @ X cannot be inverted.
This is called a **singular matrix**. In practice, we use slightly modified
versions (Ridge regression) to handle this.

---

## The Loss Function: Mean Squared Error

To measure how well our weights are doing, we use **Mean Squared Error (MSE)**:

```
MSE = mean( (y_pred - y_true)^2 )
    = (1/n) * sum( (y_pred_i - y_true_i)^2 )
```

- Squaring makes all errors positive (no cancellation)
- Squaring also penalizes large errors more than small ones
- Lower MSE = better model

---

## R-Squared: How Good Is the Fit?

MSE is hard to interpret because it depends on the units of y.
**R-squared (R2)** is a unit-free metric between 0 and 1:

```
SS_residual = sum( (y_pred - y_true)^2 )    <- error of your model
SS_total    = sum( (y_true - mean(y_true))^2 ) <- error of a "predict the mean" baseline

R2 = 1 - SS_residual / SS_total
```

- R2 = 1.0 means your model is perfect
- R2 = 0.0 means your model is no better than predicting the mean every time
- R2 < 0   means your model is worse than predicting the mean (something is wrong)

---

## Step-by-Step Example

```
Training data (3 houses):
  Size  Bedrooms  Price
  1000     2      200000
  1500     3      280000
  2000     4      360000

Step 1: Build design matrix X
  [[1, 1000, 2],
   [1, 1500, 3],
   [1, 2000, 4]]

Step 2: Build target vector y
  [200000, 280000, 360000]

Step 3: Apply Normal Equation
  W = inv(X.T @ X) @ X.T @ y

Step 4: Predict on new house (size=1800, bedrooms=3)
  x_new = [1, 1800, 3]
  price  = x_new @ W
```

---

## Connection to Neural Networks

In a neural network:
- The first layer does the same thing: output = X @ W + bias
- But a neural net stacks MANY such layers with non-linear functions between them
- Training neural nets finds W by **gradient descent** (iterative), not the Normal Equation
- The Normal Equation is only feasible for small feature sets; gradient descent scales to millions of features

So multiple regression is literally the first layer of every neural network. Once you understand it, you understand the core operation of deep learning.

---

## Quiz

**Question 1:**
You have a dataset with 500 training samples and 10 features.
After adding the bias column, what is the shape of the design matrix X?

a) (10, 500)
b) (500, 10)
c) (500, 11)
d) (11, 500)

**Answer: c) (500, 11)**
Explanation: 500 rows (one per sample) and 11 columns (10 features + 1 bias column).

---

**Question 2:**
Your model predicts house prices. y_true = [200000, 300000] and
y_pred = [210000, 290000]. What is the MSE?

a) 5000
b) 50000000
c) 100000000
d) 10000

**Answer: c) 100000000**
Explanation:
  errors = [210000-200000, 290000-300000] = [10000, -10000]
  squared = [100000000, 100000000]
  MSE = mean([100000000, 100000000]) = 100000000

---

**Question 3:**
Your model has R2 = 0.85. What does this mean?

a) Your model is wrong 85% of the time
b) Your model explains 85% of the variation in the target
c) Your model has 85% accuracy
d) Your model's MSE is 0.15

**Answer: b) Your model explains 85% of the variation in the target**
Explanation: R2 measures how much of the variance in y is captured by the model.
R2 = 0.85 means 85% of the variation is explained; 15% is unexplained noise or missing features.
