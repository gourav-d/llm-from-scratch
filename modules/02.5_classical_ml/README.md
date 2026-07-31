# Module 02.5 - Classical Machine Learning

## What Is This Module?

You have just finished Module 02 (NumPy). You know how to multiply matrices,
compute dot products, and think in terms of arrays and shapes.

This module is the **bridge** between raw NumPy math and neural networks.

Before deep learning, engineers used **classical machine learning** algorithms
to solve prediction problems. Understanding these algorithms will give you:

- A clear mental model of what "training a model" means
- Intuition for why neural networks work the way they do
- Hands-on experience implementing ML math using only NumPy

You will NOT use scikit-learn or pandas here. Every algorithm is built from
scratch so you can see exactly what is happening.

---

## Prerequisites

- Module 01: Python Basics (functions, loops, lists, classes)
- Module 02: NumPy (arrays, matrix multiplication, dot products, shapes)
- Basic algebra (you do not need calculus yet)

---

## What You Will Learn

- How to predict a continuous value (price, score, temperature) from many
  input features using **Multiple Regression**
- How to prevent your model from memorizing noise using **Regularization**
- How to split data into regions using **Decision Trees**
- How to combine many imperfect models into one strong model using
  **Random Forests**
- How to evaluate models fairly and choose the best one using
  **Model Selection and Cross-Validation**

---

## Why This Matters for LLMs

Neural networks and LLMs are extensions of these ideas:

| Classical ML Concept  | Neural Network Equivalent          |
|-----------------------|------------------------------------|
| Weights W in regression | Weight matrices in each layer    |
| MSE loss function     | Cross-entropy loss                 |
| Regularization        | Dropout, weight decay              |
| Model selection       | Hyperparameter search              |
| Overfitting           | Memorizing training prompts        |

---

## Files in This Module

```
02.5_classical_ml/
|
+-- README.md                      <- You are here
+-- 01_multiple_regression.md      <- Lesson 1: Predict with many features
+-- 02_regularization.md           <- Lesson 2: Prevent overfitting
+-- 03_decision_trees.md           <- Lesson 3: Split data into regions
+-- 04_random_forests.md           <- Lesson 4: Combine many trees
+-- 05_model_selection.md          <- Lesson 5: Evaluate and choose models
|
+-- examples/
|   +-- example_01_multiple_regression.py
|   +-- example_02_regularization.py
|   +-- example_03_decision_trees.py
|   +-- example_04_random_forests.py
|   +-- example_05_model_selection.py
|
+-- exercises/
    +-- exercise_01_multiple_regression.py
    +-- exercise_02_regularization.py
    +-- exercise_03_decision_trees.py
    +-- exercise_04_random_forests.py
    +-- exercise_05_model_selection.py
```

---

## How to Run Examples

```bash
# Make sure your virtual environment is active
# Windows:
venv\Scripts\activate

# Run any example directly:
python modules/02.5_classical_ml/examples/example_01_multiple_regression.py
python modules/02.5_classical_ml/examples/example_02_regularization.py
python modules/02.5_classical_ml/examples/example_03_decision_trees.py
python modules/02.5_classical_ml/examples/example_04_random_forests.py
python modules/02.5_classical_ml/examples/example_05_model_selection.py
```

---

## How to Do Exercises

1. Open the exercise file in your editor.
2. Read the GLOSSARY at the top.
3. Read each EXERCISE section: Background, Your Task, C# Analogy.
4. Replace the `pass` placeholder with your implementation.
5. Run the file. If the output matches the EXPECTED OUTPUT comment, you got it!

```bash
python modules/02.5_classical_ml/exercises/exercise_01_multiple_regression.py
```

---

## Recommended Order

1. Read lesson: 01_multiple_regression.md
2. Run example: example_01_multiple_regression.py
3. Try exercise: exercise_01_multiple_regression.py
4. Repeat for lessons 02 through 05

---

## Key Vocabulary (Plain English)

| Term            | Plain English                                               |
|-----------------|-------------------------------------------------------------|
| Feature         | One input column. House size, number of bedrooms, etc.     |
| Target          | The value to predict. House price, pass/fail, etc.         |
| Training        | Showing the algorithm many examples so it learns patterns  |
| Overfitting     | The model memorizes training data but fails on new data    |
| Underfitting    | The model is too simple to capture the real pattern        |
| Hyperparameter  | A setting you choose before training (like max tree depth) |
| Regularization  | A penalty that stops the model from getting too complex    |

---

## Quick Reference: Key Formulas

```
Multiple Regression prediction:
    y_pred = X @ W         (matrix multiplication)

Normal Equation (find optimal W in one step):
    W = inv(X.T @ X) @ X.T @ y

Mean Squared Error:
    MSE = mean( (y_pred - y_true)^2 )

R-squared (how good is the fit?):
    R2 = 1 - SS_residual / SS_total
       where SS_residual = sum( (y_pred - y_true)^2 )
             SS_total    = sum( (y_true - mean(y_true))^2 )

Ridge Loss (L2 regularization):
    Loss = MSE + lambda * sum(W^2)

Lasso Loss (L1 regularization):
    Loss = MSE + lambda * sum(|W|)

Gini Impurity (decision tree split quality):
    Gini = 1 - sum(p_i^2)   for each class probability p_i
```
