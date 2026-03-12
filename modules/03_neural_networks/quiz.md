# Module 3: Neural Networks - Quiz

Test your understanding of neural networks! This quiz covers all 6 lessons.

**Instructions:**
- 50 questions total (mixing multiple choice and true/false)
- Try to answer without looking at the lessons first
- Check your answers at the bottom
- Score: 40+ Excellent, 30-39 Good, 20-29 Review needed, <20 Re-study the lessons

---

## Section 1: Perceptrons (Lesson 1)

### Question 1
What is a perceptron?
- A) A complex deep learning model
- B) The simplest type of artificial neuron
- C) A type of activation function
- D) A training algorithm

### Question 2
What is the formula for a perceptron's output (before activation)?
- A) z = x + w + b
- B) z = w × x + b
- C) z = w · x + b (dot product)
- D) z = x / w - b

### Question 3
True or False: A single perceptron can learn the XOR function.
- A) True
- B) False

### Question 4
What does the bias term (b) in a perceptron do?
- A) Makes the neuron faster
- B) Shifts the decision boundary
- C) Prevents overfitting
- D) Adds more weights

### Question 5
In layman terms, what does a perceptron do?
- A) Takes inputs, multiplies by weights, adds bias, makes a decision
- B) Stores data in memory
- C) Generates random numbers
- D) Compresses information

---

## Section 2: Activation Functions (Lesson 2)

### Question 6
Why do we need activation functions in neural networks?
- A) To make them faster
- B) To add non-linearity (allow learning complex patterns)
- C) To save memory
- D) To prevent bugs

### Question 7
What is the formula for ReLU activation?
- A) ReLU(x) = 1/(1+e^-x)
- B) ReLU(x) = max(0, x)
- C) ReLU(x) = x^2
- D) ReLU(x) = tanh(x)

### Question 8
Which activation function is used in GPT models?
- A) Sigmoid
- B) Tanh
- C) ReLU
- D) GELU

### Question 9
What range of values does the Sigmoid function output?
- A) -∞ to +∞
- B) -1 to +1
- C) 0 to 1
- D) 0 to 100

### Question 10
True or False: Without activation functions, a deep neural network acts like a single layer.
- A) True
- B) False

### Question 11
When should you use Softmax activation?
- A) Hidden layers
- B) Binary classification output
- C) Multi-class classification output
- D) Regression output

### Question 12
What problem does ReLU solve that Sigmoid/Tanh have?
- A) Speed
- B) Gradient vanishing (gradients become too small)
- C) Memory usage
- D) Accuracy

---

## Section 3: Multi-Layer Networks (Lesson 3)

### Question 13
What does "deep" mean in "deep learning"?
- A) Complex math
- B) Many layers stacked together
- C) Large datasets
- D) Difficult to understand

### Question 14
What is forward propagation?
- A) Training the network
- B) Passing data forward through layers to get predictions
- C) Computing gradients
- D) Updating weights

### Question 15
In a network with layers: Input(3) → Hidden(5) → Output(2), how many weights connect Input to Hidden?
- A) 3
- B) 5
- C) 8
- D) 15 (3 × 5)

### Question 16
True or False: In a fully connected layer, every neuron connects to every neuron in the next layer.
- A) True
- B) False

### Question 17
Why can multi-layer networks solve XOR but single neurons cannot?
- A) More memory
- B) Layers can learn intermediate features (non-linear combinations)
- C) Faster computation
- D) Better random initialization

---

## Section 4: Backpropagation (Lesson 4)

### Question 18
What is backpropagation?
- A) A new type of neural network
- B) An algorithm to compute gradients (how to adjust weights)
- C) A way to visualize networks
- D) A programming language

### Question 19
In simple terms, what does backpropagation do?
- A) Makes predictions
- B) Traces errors backward through the network to find who's responsible
- C) Saves the model
- D) Initializes weights

### Question 20
What is the chain rule used for in backpropagation?
- A) To link layers together
- B) To compute how output error affects earlier layers' weights
- C) To create new neurons
- D) To visualize the network

### Question 21
True or False: Backpropagation is the same as gradient descent.
- A) True
- B) False (Backprop computes gradients, gradient descent uses them to update weights)

### Question 22
What happens if gradients become very small (gradient vanishing)?
- A) Training speeds up
- B) Network learns better
- C) Early layers stop learning (weights don't update)
- D) Nothing changes

### Question 23
Why is backpropagation called "back"propagation?
- A) It's outdated
- B) Errors flow backward from output to input
- C) It runs backwards in time
- D) It uses reverse Python code

---

## Section 5: Training Loop (Lesson 5)

### Question 24
What is an epoch in training?
- A) One weight update
- B) One complete pass through the entire training dataset
- C) One batch of data
- D) One second of training

### Question 25
What is a mini-batch?
- A) A small neural network
- B) A subset of training data processed together
- C) A single training example
- D) A small learning rate

### Question 26
Why do we split data into train/validation/test sets?
- A) To make training faster
- B) To check if model generalizes (doesn't just memorize)
- C) To use less memory
- D) To confuse the network

### Question 27
What is overfitting?
- A) Network is too small
- B) Network memorizes training data but fails on new data
- C) Training too slowly
- D) Using too much memory

### Question 28
True or False: We update weights using the validation set.
- A) True
- B) False (only use training set for updates!)

### Question 29
What is early stopping?
- A) Stopping training if validation loss stops improving
- B) Training for fewer epochs
- C) Using smaller batches
- D) Reducing learning rate

### Question 30
What's a typical train/validation/test split?
- A) 33% / 33% / 33%
- B) 80% / 10% / 10%
- C) 50% / 50% / 0%
- D) 100% / 0% / 0%

---

## Section 6: Optimizers (Lesson 6)

### Question 31
What is gradient descent?
- A) A type of neural network
- B) An algorithm that adjusts weights to minimize loss
- C) A way to visualize data
- D) A programming technique

### Question 32
What does the learning rate control?
- A) Network size
- B) How big each weight update step is
- C) Number of layers
- D) Batch size

### Question 33
Which optimizer is most commonly used in modern deep learning (including GPT)?
- A) SGD (Stochastic Gradient Descent)
- B) Momentum
- C) RMSProp
- D) Adam

### Question 34
What problem does Momentum solve?
- A) Makes training faster and helps escape local minima
- B) Prevents overfitting
- C) Saves memory
- D) Increases accuracy

### Question 35
True or False: A learning rate that's too high can cause training to diverge (get worse).
- A) True
- B) False

### Question 36
What does Adam optimizer do?
- A) Combines benefits of Momentum and RMSProp (adaptive learning rates)
- B) Makes networks deeper
- C) Removes activation functions
- D) Speeds up GPUs

### Question 37
What's a good starting learning rate for Adam?
- A) 10.0
- B) 1.0
- C) 0.001 (1e-3)
- D) 0.0000001

---

## Section 7: Practical Application

### Question 38
For image classification (cat vs dog), which output activation should you use?
- A) ReLU
- B) Sigmoid
- C) Linear
- D) GELU

### Question 39
For predicting house prices (regression), which output activation?
- A) Softmax
- B) Sigmoid
- C) Linear (no activation)
- D) ReLU

### Question 40
True or False: GPT uses the exact same backpropagation algorithm you learned.
- A) True (same algorithm, just bigger network!)
- B) False

### Question 41
What loss function should you use for binary classification?
- A) Mean Squared Error (MSE)
- B) Binary Cross-Entropy
- C) Categorical Cross-Entropy
- D) Absolute Error

### Question 42
What loss function for multi-class classification (10 classes)?
- A) Mean Squared Error (MSE)
- B) Binary Cross-Entropy
- C) Categorical Cross-Entropy
- D) Hinge Loss

---

## Section 8: Understanding Concepts

### Question 43
In simple terms, what is a neural network?
- A) A program that copies how brain neurons work to learn patterns
- B) A complicated algorithm no one understands
- C) A database
- D) A search engine

### Question 44
What's the difference between a parameter and a hyperparameter?
- A) No difference
- B) Parameters are learned (weights), hyperparameters are set by you (learning rate, epochs)
- C) Parameters are bigger
- D) Hyperparameters are faster

### Question 45
True or False: More layers always means better performance.
- A) True
- B) False (can lead to overfitting, vanishing gradients, longer training)

### Question 46
Why do we initialize weights randomly (not all zeros)?
- A) For security
- B) To break symmetry (otherwise all neurons learn the same thing)
- C) Makes it faster
- D) Tradition

### Question 47
What happens during one training iteration?
- A) Forward pass → calculate loss → backward pass → update weights
- B) Just forward pass
- C) Just update weights
- D) Save the model

---

## Section 9: Mathematical Understanding (Simple)

### Question 48
What does "gradient" mean in simple terms?
- A) The slope/direction of steepest increase
- B) A random number
- C) The final output
- D) Number of layers

### Question 49
Why do we want to minimize loss?
- A) Makes training faster
- B) Lower loss = more accurate predictions
- C) Saves memory
- D) Prevents crashes

### Question 50
True or False: You need to understand all the calculus to use neural networks.
- A) True
- B) False (understanding concepts is enough for most applications!)

---

# Answer Key

## Section 1: Perceptrons
1. **B** - The simplest type of artificial neuron
2. **C** - z = w · x + b (dot product)
3. **B** - False (single perceptron can only learn linearly separable patterns)
4. **B** - Shifts the decision boundary
5. **A** - Takes inputs, multiplies by weights, adds bias, makes a decision

## Section 2: Activation Functions
6. **B** - To add non-linearity (allow learning complex patterns)
7. **B** - ReLU(x) = max(0, x)
8. **D** - GELU
9. **C** - 0 to 1
10. **A** - True
11. **C** - Multi-class classification output
12. **B** - Gradient vanishing

## Section 3: Multi-Layer Networks
13. **B** - Many layers stacked together
14. **B** - Passing data forward through layers to get predictions
15. **D** - 15 (3 × 5)
16. **A** - True
17. **B** - Layers can learn intermediate features

## Section 4: Backpropagation
18. **B** - An algorithm to compute gradients
19. **B** - Traces errors backward through the network
20. **B** - To compute how output error affects earlier layers' weights
21. **B** - False (different but related)
22. **C** - Early layers stop learning
23. **B** - Errors flow backward from output to input

## Section 5: Training Loop
24. **B** - One complete pass through the entire training dataset
25. **B** - A subset of training data processed together
26. **B** - To check if model generalizes
27. **B** - Network memorizes training data but fails on new data
28. **B** - False
29. **A** - Stopping training if validation loss stops improving
30. **B** - 80% / 10% / 10%

## Section 6: Optimizers
31. **B** - An algorithm that adjusts weights to minimize loss
32. **B** - How big each weight update step is
33. **D** - Adam
34. **A** - Makes training faster and helps escape local minima
35. **A** - True
36. **A** - Combines benefits of Momentum and RMSProp
37. **C** - 0.001 (1e-3)

## Section 7: Practical Application
38. **B** - Sigmoid
39. **C** - Linear (no activation)
40. **A** - True
41. **B** - Binary Cross-Entropy
42. **C** - Categorical Cross-Entropy

## Section 8: Understanding Concepts
43. **A** - A program that copies how brain neurons work to learn patterns
44. **B** - Parameters are learned, hyperparameters are set by you
45. **B** - False
46. **B** - To break symmetry
47. **A** - Forward pass → calculate loss → backward pass → update weights

## Section 9: Mathematical Understanding
48. **A** - The slope/direction of steepest increase
49. **B** - Lower loss = more accurate predictions
50. **B** - False

---

# Scoring Guide

**45-50 correct:** Excellent! You've mastered neural networks! 🎉
- You understand all core concepts
- Ready for Module 4: Transformers
- Can start building real projects

**38-44 correct:** Very Good! 👍
- Strong understanding of most concepts
- Review questions you missed
- Practice with the examples
- Ready to move forward

**30-37 correct:** Good Progress 📚
- You understand the basics
- Need to review some lessons
- Focus on sections where you scored lowest
- Run the examples and exercises

**20-29 correct:** Needs Review ⚠️
- Go back and re-read the lessons
- Run all the example code
- Try the exercises
- Take notes on key concepts
- Retake quiz after review

**< 20 correct:** Re-study Required 📖
- Start from Lesson 1
- Read carefully and take notes
- Run examples line by line
- Ask questions about confusing parts
- Don't rush - understanding is more important than speed

---

# Key Concepts to Remember

## The Big Picture
1. **Neurons** combine inputs with weights, add bias, apply activation
2. **Forward propagation** makes predictions
3. **Loss** measures how wrong predictions are
4. **Backpropagation** computes how to adjust weights
5. **Gradient descent** updates weights to reduce loss
6. **Repeat** until accurate!

## For Each Layer
- **Hidden layers:** Usually ReLU (or GELU for transformers)
- **Output layer:** Depends on task
  - Binary classification: Sigmoid
  - Multi-class: Softmax
  - Regression: Linear (no activation)

## Training Best Practices
- **Split data:** 80% train, 10% val, 10% test
- **Use batches:** 16-128 samples typical
- **Monitor:** Watch train vs validation loss
- **Early stopping:** Prevent overfitting
- **Learning rate:** Start with 0.001 for Adam
- **Optimizer:** Adam is usually best choice

## Connection to GPT
GPT uses:
- **Feed-forward layers** (what you learned!)
- **GELU activation** (smoother than ReLU)
- **Backpropagation** (same algorithm!)
- **Adam optimizer** (same optimizer!)
- **Mini-batch training** (same approach!)

The only new concept in GPT is **attention mechanism** (Module 4)!

---

# Next Steps

### If you scored well (35+):
✅ Move to **Module 4: Transformers**
✅ Start building practice projects
✅ Experiment with the code examples

### If you need review (< 35):
📚 Re-read lessons where you struggled
🔧 Run the examples again
💻 Complete the exercises
📝 Make flashcards for formulas
🔄 Retake this quiz

---

**Remember:** Understanding neural networks takes time! Don't get discouraged if you need to review. Even experienced ML engineers forget details and look things up. The important thing is understanding the concepts, not memorizing formulas!

**Good luck! You've got this!** 🚀
