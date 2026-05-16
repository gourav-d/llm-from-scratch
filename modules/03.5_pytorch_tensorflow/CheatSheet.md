Simplest possible neural network.
    No hidden layer — just one direct connection from input to output.

    Architecture:
        Input (2) ──→ Output (1)
                       Sigmoid

    WHY this works for AND:
      AND data is linearly separable.
      One straight line can divide 0s from 1s on a 2D grid.
      A single layer = one straight line decision boundary.


WHY single layer fails:
  Plot these 4 points on a grid:
  
    y=1 |  1(class 1)   0(class 0)
        |
    y=0 |  0(class 0)   1(class 1)
        +────────────────────────
           x=0          x=1

  Class 0 = top-right and bottom-left (diagonal)
  Class 1 = top-left and bottom-right (other diagonal)
  No single straight line can separate them!
  This is the famous XOR problem that killed neural networks in 1960s.


FIX: Add a hidden layer.

Hidden layer learns intermediate features:
  Neuron 1 might learn: "are inputs different?"
  Neuron 2 might learn: "are both inputs 1?"
  Output combines these to get the right answer.

Architecture:
  Input (2) → Hidden (4, ReLU) → Output (1)
                 ↑
         This is the key difference!
         Hidden layer creates new representation
         that IS linearly separable.

WHY THESE TWO TOGETHER?
  AND is easy — a single layer solves it.
  XOR is impossible for a single layer — needs a hidden layer.
  This difference teaches the most important concept in deep learning:
  "Why do we need multiple layers?"


Two-layer network with hidden neurons.

    Architecture:
        Input (2) → Hidden (4, ReLU) → Output (1)

    WHY 4 hidden neurons?
      XOR only really needs 2, but we use 4 to give
      the model more "room to think" and train faster.