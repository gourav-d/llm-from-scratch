"""
Exercise 6: Optimizers - Practice Problems

This exercise helps you master different optimization algorithms used to train neural networks.

DIFFICULTY LEVELS:
- Exercises 1-3: Beginner (understand basic optimizers)
- Exercises 4-6: Intermediate (implement and compare)
- Exercises 7-10: Advanced (hyperparameter tuning, advanced techniques)

For .NET developers: Think of optimizers like different sorting algorithms -
all achieve the same goal but with different trade-offs!
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


# ============================================================================
# EXERCISE 1: Vanilla Gradient Descent (Beginner)
# ============================================================================

def exercise_1_vanilla_gd():
    """
    TASK: Implement and understand vanilla (basic) gradient descent

    What is gradient descent?
    - An algorithm to find the minimum of a function
    - Like rolling a ball downhill - it naturally goes to the lowest point

    How it works:
    - Calculate gradient (slope/direction)
    - Move in opposite direction (downhill)
    - Repeat until you reach the bottom (minimum loss)

    For .NET devs: Like a foreach loop that keeps improving the solution
    """
    print("\n" + "="*70)
    print("EXERCISE 1: Vanilla Gradient Descent")
    print("="*70)

    # Simple optimization problem: minimize f(x) = x²
    # Minimum is at x = 0

    print("\nProblem: Minimize f(x) = x²")
    print("Expected solution: x = 0")

    # TODO: Implement gradient descent
    def gradient_descent(start_x, learning_rate, iterations):
        """
        Vanilla gradient descent to minimize f(x) = x²

        Steps in each iteration:
        1. Calculate gradient: df/dx = 2x
        2. Update: x = x - learning_rate * gradient
        3. Repeat

        Args:
            start_x: Initial value of x
            learning_rate: Step size
            iterations: Number of steps

        Returns:
            history: List of x values
        """
        x = start_x
        history = [x]

        for i in range(iterations):
            # TODO: Calculate gradient of f(x) = x²
            # Gradient df/dx = 2x
            gradient = 2 * x

            # TODO: Update x
            x = x - learning_rate * gradient

            history.append(x)

        return history

    # Test with different learning rates
    start_x = 5.0  # Start far from solution
    iterations = 20

    learning_rates = [0.1, 0.3, 0.5, 0.9]
    results = {}

    print(f"\nStarting at x = {start_x}")
    print(f"Iterations: {iterations}\n")

    for lr in learning_rates:
        history = gradient_descent(start_x, lr, iterations)
        results[lr] = history

        print(f"Learning rate = {lr:.1f}:")
        print(f"  Final x = {history[-1]:.6f}")
        print(f"  Final f(x) = {history[-1]**2:.6f}")

    # Plot convergence
    plt.figure(figsize=(12, 5))

    # Plot 1: Value of x over iterations
    plt.subplot(1, 2, 1)
    for lr, history in results.items():
        plt.plot(history, 'o-', label=f'LR={lr}', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', label='True minimum')
    plt.xlabel('Iteration')
    plt.ylabel('x')
    plt.title('Convergence to Minimum')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Loss (f(x) = x²) over iterations
    plt.subplot(1, 2, 2)
    for lr, history in results.items():
        loss_history = [x**2 for x in history]
        plt.semilogy(loss_history, 'o-', label=f'LR={lr}', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss f(x) = x² (log scale)')
    plt.title('Loss Reduction')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_06_vanilla_gd.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_06_vanilla_gd.png")
    plt.show()

    print("\nKey observations:")
    print("- Too small LR (0.1): Slow but steady convergence")
    print("- Good LR (0.3, 0.5): Fast convergence")
    print("- Too large LR (0.9): Oscillations, slower convergence")


# ============================================================================
# EXERCISE 2: Momentum (Beginner)
# ============================================================================

def exercise_2_momentum():
    """
    TASK: Implement Momentum optimizer

    What is Momentum?
    - Adds "velocity" to gradient descent
    - Like a ball rolling downhill - builds up speed
    - Helps escape local minima and speeds up convergence

    For .NET devs: Like adding acceleration to movement -
    keeps moving in same direction with momentum

    Formula:
    - velocity = beta * velocity + gradient
    - x = x - learning_rate * velocity
    """
    print("\n" + "="*70)
    print("EXERCISE 2: Momentum Optimizer")
    print("="*70)

    # Test on a valley-shaped function where momentum helps
    # f(x, y) = x²/100 + y²  (narrow valley along x-axis)

    print("\nProblem: Minimize f(x,y) = x²/100 + y²")
    print("This has a narrow valley - perfect for testing momentum!")

    # TODO: Implement vanilla gradient descent
    def vanilla_gd_2d(start_pos, learning_rate, iterations):
        """Vanilla GD on 2D function"""
        x, y = start_pos
        history = [[x, y]]

        for _ in range(iterations):
            # Gradients: df/dx = x/50, df/dy = 2y
            grad_x = x / 50
            grad_y = 2 * y

            x = x - learning_rate * grad_x
            y = y - learning_rate * grad_y

            history.append([x, y])

        return np.array(history)

    # TODO: Implement momentum
    def momentum_gd_2d(start_pos, learning_rate, iterations, beta=0.9):
        """
        Gradient descent with momentum

        Args:
            beta: Momentum coefficient (typically 0.9)
                 Higher = more momentum

        Formula:
        - v = beta * v + gradient
        - x = x - learning_rate * v
        """
        x, y = start_pos
        vx, vy = 0, 0  # Initialize velocity
        history = [[x, y]]

        for _ in range(iterations):
            # Gradients
            grad_x = x / 50
            grad_y = 2 * y

            # TODO: Update velocity
            vx = beta * vx + grad_x
            vy = beta * vy + grad_y

            # TODO: Update position
            x = x - learning_rate * vx
            y = y - learning_rate * vy

            history.append([x, y])

        return np.array(history)

    # Test both optimizers
    start_pos = [10.0, 10.0]
    learning_rate = 0.1
    iterations = 50

    history_vanilla = vanilla_gd_2d(start_pos, learning_rate, iterations)
    history_momentum = momentum_gd_2d(start_pos, learning_rate, iterations, beta=0.9)

    print(f"\nStarting at: x={start_pos[0]}, y={start_pos[1]}")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {iterations}\n")

    # Calculate final loss
    final_vanilla = history_vanilla[-1]
    final_momentum = history_momentum[-1]

    loss_vanilla = final_vanilla[0]**2/100 + final_vanilla[1]**2
    loss_momentum = final_momentum[0]**2/100 + final_momentum[1]**2

    print("Vanilla GD:")
    print(f"  Final position: x={final_vanilla[0]:.4f}, y={final_vanilla[1]:.4f}")
    print(f"  Final loss: {loss_vanilla:.6f}")

    print("\nMomentum GD:")
    print(f"  Final position: x={final_momentum[0]:.4f}, y={final_momentum[1]:.4f}")
    print(f"  Final loss: {loss_momentum:.6f}")

    # Plot paths
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_vanilla[:, 0], history_vanilla[:, 1],
             'o-', label='Vanilla GD', linewidth=2, markersize=4)
    plt.plot(history_momentum[:, 0], history_momentum[:, 1],
             's-', label='Momentum', linewidth=2, markersize=4)
    plt.plot(0, 0, 'r*', markersize=20, label='True minimum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Paths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Plot loss over iterations
    plt.subplot(1, 2, 2)
    loss_vanilla_history = [x**2/100 + y**2 for x, y in history_vanilla]
    loss_momentum_history = [x**2/100 + y**2 for x, y in history_momentum]

    plt.semilogy(loss_vanilla_history, 'o-', label='Vanilla GD', linewidth=2)
    plt.semilogy(loss_momentum_history, 's-', label='Momentum', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Reduction')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_06_momentum.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_06_momentum.png")
    plt.show()

    print("\nKey insight:")
    print("- Vanilla GD: Zigzags down the valley (slow)")
    print("- Momentum: Builds speed along the valley (fast!)")


# ============================================================================
# EXERCISE 3: RMSProp (Intermediate)
# ============================================================================

def exercise_3_rmsprop():
    """
    TASK: Implement RMSProp optimizer

    What is RMSProp?
    - Adapts learning rate for each parameter
    - Divides learning rate by running average of gradient magnitudes
    - Good for non-stationary problems (like RNNs)

    For .NET devs: Like auto-adjusting step size based on terrain steepness

    Formula:
    - v = beta * v + (1-beta) * gradient²
    - x = x - learning_rate * gradient / sqrt(v + epsilon)
    """
    print("\n" + "="*70)
    print("EXERCISE 3: RMSProp Optimizer")
    print("="*70)

    print("\nRMSProp = Root Mean Square Propagation")
    print("Adapts learning rate automatically for each parameter!\n")

    # TODO: Implement RMSProp
    def rmsprop_1d(start_x, learning_rate, iterations, beta=0.9, epsilon=1e-8):
        """
        RMSProp for minimizing f(x) = x²

        Args:
            beta: Decay rate for running average (typically 0.9)
            epsilon: Small constant to prevent division by zero
        """
        x = start_x
        v = 0  # Running average of squared gradients
        history = [x]

        for _ in range(iterations):
            # Gradient of f(x) = x²
            gradient = 2 * x

            # TODO: Update running average of squared gradients
            v = beta * v + (1 - beta) * (gradient ** 2)

            # TODO: Adaptive update
            x = x - learning_rate * gradient / (np.sqrt(v) + epsilon)

            history.append(x)

        return history

    # Compare with vanilla GD
    start_x = 10.0
    learning_rate = 0.1
    iterations = 20

    history_vanilla = []
    x = start_x
    for _ in range(iterations + 1):
        history_vanilla.append(x)
        if _ < iterations:
            x = x - learning_rate * (2 * x)

    history_rmsprop = rmsprop_1d(start_x, learning_rate, iterations)

    print(f"Starting at x = {start_x}")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {iterations}\n")

    print("Vanilla GD:")
    print(f"  Final x: {history_vanilla[-1]:.6f}")

    print("\nRMSProp:")
    print(f"  Final x: {history_rmsprop[-1]:.6f}")

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_vanilla, 'o-', label='Vanilla GD', linewidth=2)
    plt.plot(history_rmsprop, 's-', label='RMSProp', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', label='True minimum')
    plt.xlabel('Iteration')
    plt.ylabel('x')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    loss_vanilla = [x**2 for x in history_vanilla]
    loss_rmsprop = [x**2 for x in history_rmsprop]
    plt.semilogy(loss_vanilla, 'o-', label='Vanilla GD', linewidth=2)
    plt.semilogy(loss_rmsprop, 's-', label='RMSProp', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Reduction')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_06_rmsprop.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_06_rmsprop.png")
    plt.show()

    print("\nKey insight:")
    print("- RMSProp adapts the learning rate automatically")
    print("- Often converges faster and more smoothly")


# ============================================================================
# EXERCISE 4: Adam (THE Optimizer!) (Intermediate)
# ============================================================================

def exercise_4_adam():
    """
    TASK: Implement Adam optimizer

    What is Adam?
    - Adaptive Moment Estimation
    - Combines best of Momentum + RMSProp
    - THE most popular optimizer (used in GPT-3!)

    For .NET devs: Like having both cruise control AND automatic transmission

    Formula:
    - m = beta1 * m + (1-beta1) * gradient          (momentum)
    - v = beta2 * v + (1-beta2) * gradient²         (RMSProp)
    - m_hat = m / (1 - beta1^t)                     (bias correction)
    - v_hat = v / (1 - beta2^t)                     (bias correction)
    - x = x - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    """
    print("\n" + "="*70)
    print("EXERCISE 4: Adam Optimizer (GPT's Choice!)")
    print("="*70)

    print("\nAdam = Adaptive Moment Estimation")
    print("Best of both worlds: Momentum + RMSProp!")
    print("This is what trains GPT-3!\n")

    # TODO: Implement Adam
    def adam(start_x, learning_rate=0.1, iterations=20,
             beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam optimizer for f(x) = x²

        Args:
            beta1: Momentum decay (typically 0.9)
            beta2: RMSProp decay (typically 0.999)
            epsilon: Small constant
        """
        x = start_x
        m = 0  # First moment (momentum)
        v = 0  # Second moment (RMSProp)
        history = [x]

        for t in range(1, iterations + 1):
            # Gradient
            gradient = 2 * x

            # TODO: Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * gradient

            # TODO: Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * (gradient ** 2)

            # TODO: Compute bias-corrected moments
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # TODO: Update parameters
            x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            history.append(x)

        return history

    # Compare all optimizers
    start_x = 10.0
    learning_rate = 0.1
    iterations = 20

    # Vanilla GD
    history_vanilla = []
    x = start_x
    for _ in range(iterations + 1):
        history_vanilla.append(x)
        if _ < iterations:
            x = x - learning_rate * (2 * x)

    # Momentum
    history_momentum = []
    x = start_x
    v = 0
    for _ in range(iterations + 1):
        history_momentum.append(x)
        if _ < iterations:
            gradient = 2 * x
            v = 0.9 * v + gradient
            x = x - learning_rate * v

    # Adam
    history_adam = adam(start_x, learning_rate, iterations)

    print(f"Starting at x = {start_x}")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {iterations}\n")

    optimizers = {
        'Vanilla GD': history_vanilla,
        'Momentum': history_momentum,
        'Adam': history_adam
    }

    for name, history in optimizers.items():
        print(f"{name:12} → Final x: {history[-1]:.6f}, Loss: {history[-1]**2:.8f}")

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, history in optimizers.items():
        plt.plot(history, 'o-', label=name, linewidth=2, markersize=5)
    plt.axhline(y=0, color='r', linestyle='--', label='True minimum')
    plt.xlabel('Iteration')
    plt.ylabel('x')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for name, history in optimizers.items():
        loss = [x**2 for x in history]
        plt.semilogy(loss, 'o-', label=name, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Reduction')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_06_adam.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_06_adam.png")
    plt.show()

    print("\n" + "="*60)
    print("Why Adam is THE default choice:")
    print("="*60)
    print("✓ Combines momentum (speeds up) and RMSProp (adapts LR)")
    print("✓ Works well with default parameters")
    print("✓ Robust across different problems")
    print("✓ Used to train GPT-2, GPT-3, BERT, and most modern models")
    print("✓ Usually no tuning needed - just use it!")


# ============================================================================
# EXERCISE 5: Comparing All Optimizers (Advanced)
# ============================================================================

def exercise_5_comprehensive_comparison():
    """
    TASK: Compare all optimizers on a more complex optimization problem

    This shows when each optimizer shines!
    """
    print("\n" + "="*70)
    print("EXERCISE 5: Comprehensive Optimizer Comparison")
    print("="*70)

    # Complex 2D function: Rosenbrock (banana function)
    # f(x, y) = (1-x)² + 100(y-x²)²
    # Minimum at (1, 1)
    # This is HARD to optimize!

    print("\nOptimizing Rosenbrock function (the 'banana' function)")
    print("This is a challenging problem - perfect for testing optimizers!")
    print("Minimum at (1, 1)\n")

    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2

    def rosenbrock_grad(x, y):
        dx = -2*(1-x) - 400*x*(y-x**2)
        dy = 200*(y-x**2)
        return dx, dy

    # TODO: Implement all optimizers
    class Optimizers:
        @staticmethod
        def sgd(start_pos, lr, iterations):
            x, y = start_pos
            history = [[x, y]]

            for _ in range(iterations):
                dx, dy = rosenbrock_grad(x, y)
                x -= lr * dx
                y -= lr * dy
                history.append([x, y])

            return np.array(history)

        @staticmethod
        def momentum(start_pos, lr, iterations, beta=0.9):
            x, y = start_pos
            vx, vy = 0, 0
            history = [[x, y]]

            for _ in range(iterations):
                dx, dy = rosenbrock_grad(x, y)
                vx = beta * vx + dx
                vy = beta * vy + dy
                x -= lr * vx
                y -= lr * vy
                history.append([x, y])

            return np.array(history)

        @staticmethod
        def rmsprop(start_pos, lr, iterations, beta=0.9, eps=1e-8):
            x, y = start_pos
            vx, vy = 0, 0
            history = [[x, y]]

            for _ in range(iterations):
                dx, dy = rosenbrock_grad(x, y)
                vx = beta * vx + (1-beta) * dx**2
                vy = beta * vy + (1-beta) * dy**2
                x -= lr * dx / (np.sqrt(vx) + eps)
                y -= lr * dy / (np.sqrt(vy) + eps)
                history.append([x, y])

            return np.array(history)

        @staticmethod
        def adam(start_pos, lr, iterations, beta1=0.9, beta2=0.999, eps=1e-8):
            x, y = start_pos
            mx, my = 0, 0
            vx, vy = 0, 0
            history = [[x, y]]

            for t in range(1, iterations + 1):
                dx, dy = rosenbrock_grad(x, y)

                mx = beta1 * mx + (1-beta1) * dx
                my = beta1 * my + (1-beta1) * dy

                vx = beta2 * vx + (1-beta2) * dx**2
                vy = beta2 * vy + (1-beta2) * dy**2

                mx_hat = mx / (1 - beta1**t)
                my_hat = my / (1 - beta1**t)
                vx_hat = vx / (1 - beta2**t)
                vy_hat = vy / (1 - beta2**t)

                x -= lr * mx_hat / (np.sqrt(vx_hat) + eps)
                y -= lr * my_hat / (np.sqrt(vy_hat) + eps)

                history.append([x, y])

            return np.array(history)

    # Run all optimizers
    start_pos = [-1.0, -1.0]
    learning_rate = 0.001
    iterations = 1000

    results = {
        'SGD': Optimizers.sgd(start_pos, learning_rate, iterations),
        'Momentum': Optimizers.momentum(start_pos, learning_rate, iterations),
        'RMSProp': Optimizers.rmsprop(start_pos, learning_rate, iterations),
        'Adam': Optimizers.adam(start_pos, learning_rate, iterations)
    }

    print(f"Starting position: {start_pos}")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {iterations}\n")

    print("Final results:")
    print("-" * 60)
    for name, history in results.items():
        final_x, final_y = history[-1]
        final_loss = rosenbrock(final_x, final_y)
        print(f"{name:10} → x={final_x:.4f}, y={final_y:.4f}, loss={final_loss:.6f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Create contour plot
    x_range = np.linspace(-2, 2, 200)
    y_range = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = rosenbrock(X, Y)

    # Plot each optimizer's path
    for idx, (name, history) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]

        # Contour plot
        contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
        ax.clabel(contour, inline=True, fontsize=8)

        # Optimizer path
        ax.plot(history[:, 0], history[:, 1], 'r-', linewidth=2, alpha=0.7)
        ax.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Start')
        ax.plot(1, 1, 'r*', markersize=20, label='True minimum')
        ax.plot(history[-1, 0], history[-1, 1], 'bs', markersize=10, label='Final')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name} Optimizer')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_06_comprehensive.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_06_comprehensive.png")
    plt.show()

    # Plot loss curves
    plt.figure(figsize=(10, 6))

    for name, history in results.items():
        losses = [rosenbrock(x, y) for x, y in history]
        plt.semilogy(losses, label=name, linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Reduction Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig('exercise_06_loss_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved as: exercise_06_loss_comparison.png")
    plt.show()

    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("- SGD: Slowest, but steady")
    print("- Momentum: Faster, but can overshoot")
    print("- RMSProp: Adaptive, handles different scales well")
    print("- Adam: Best overall (combines momentum + adaptation)")
    print("\nFor most problems: Just use Adam!")


# ============================================================================
# EXERCISE 6: Hyperparameter Tuning Guide (Advanced)
# ============================================================================

def exercise_6_hyperparameter_guide():
    """
    TASK: Learn how to choose hyperparameters for each optimizer

    This is practical knowledge you'll use in real projects!
    """
    print("\n" + "="*70)
    print("EXERCISE 6: Optimizer Hyperparameter Guide")
    print("="*70)

    guide = """
╔══════════════════════════════════════════════════════════════════════╗
║              OPTIMIZER HYPERPARAMETER GUIDE                          ║
╚══════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────┐
│ 1. SGD (Stochastic Gradient Descent)                                │
└──────────────────────────────────────────────────────────────────────┘

Parameters:
  - learning_rate: 0.01 to 0.1 (start high, decay over time)

Pros:
  ✓ Simple and interpretable
  ✓ Good generalization (test performance)
  ✓ Works well with learning rate schedule

Cons:
  ✗ Slow convergence
  ✗ Sensitive to learning rate
  ✗ Can get stuck in local minima

When to use:
  - When you have time to tune learning rate
  - When generalization is critical
  - Classic computer vision tasks

Typical setup:
  learning_rate = 0.1
  Use learning rate decay: multiply by 0.1 every N epochs

┌──────────────────────────────────────────────────────────────────────┐
│ 2. Momentum                                                          │
└──────────────────────────────────────────────────────────────────────┘

Parameters:
  - learning_rate: 0.01 to 0.1
  - momentum (beta): 0.9 (most common)

Pros:
  ✓ Faster than SGD
  ✓ Smooths out noise
  ✓ Helps escape local minima

Cons:
  ✗ Can overshoot minimum
  ✗ Still needs LR tuning

When to use:
  - Classic deep learning (CNNs, MLPs)
  - When you want speed + good generalization
  - Standard computer vision

Typical setup:
  learning_rate = 0.01
  momentum = 0.9
  Use with learning rate decay

┌──────────────────────────────────────────────────────────────────────┐
│ 3. RMSProp                                                           │
└──────────────────────────────────────────────────────────────────────┘

Parameters:
  - learning_rate: 0.001 to 0.01
  - beta (decay): 0.9 or 0.99
  - epsilon: 1e-8 (prevent division by zero)

Pros:
  ✓ Adapts learning rate per parameter
  ✓ Good for non-stationary problems
  ✓ Works well for RNNs

Cons:
  ✗ Aggressive learning rate reduction (can be too slow)
  ✗ Less popular now (Adam preferred)

When to use:
  - Recurrent neural networks (RNNs)
  - Non-stationary problems
  - When features have very different scales

Typical setup:
  learning_rate = 0.001
  beta = 0.9
  epsilon = 1e-8

┌──────────────────────────────────────────────────────────────────────┐
│ 4. Adam (Adaptive Moment Estimation) ⭐ RECOMMENDED                  │
└──────────────────────────────────────────────────────────────────────┘

Parameters:
  - learning_rate: 0.001 (1e-3) [DEFAULT - usually works!]
  - beta1 (momentum): 0.9
  - beta2 (RMSProp): 0.999
  - epsilon: 1e-8

Pros:
  ✓ Combines best of momentum + RMSProp
  ✓ Works well with default parameters
  ✓ Adaptive learning rate per parameter
  ✓ Most widely used (GPT-3, BERT, etc.)
  ✓ Usually no tuning needed!

Cons:
  ✗ Can sometimes generalize worse than SGD+Momentum
  ✗ Uses more memory (stores m and v)

When to use:
  - DEFAULT CHOICE for most problems!
  - Transformers (GPT, BERT)
  - General deep learning
  - When you want "just works" solution

Typical setup:
  learning_rate = 0.001  # Start here, rarely need to change!
  beta1 = 0.9
  beta2 = 0.999
  epsilon = 1e-8

┌──────────────────────────────────────────────────────────────────────┐
│ QUICK DECISION GUIDE                                                 │
└──────────────────────────────────────────────────────────────────────┘

"Which optimizer should I use?"

├─ Not sure? → Use Adam (lr=0.001)
│
├─ Training Transformer/GPT? → Adam (lr=0.001 to 0.0003)
│
├─ Training CNN for images? → Adam (lr=0.001) or SGD+Momentum (lr=0.01)
│
├─ Training RNN/LSTM? → Adam (lr=0.001) or RMSProp (lr=0.001)
│
├─ Need best generalization? → SGD+Momentum with LR decay
│
└─ Production system? → Adam (most reliable)

┌──────────────────────────────────────────────────────────────────────┐
│ LEARNING RATE GUIDELINES                                             │
└──────────────────────────────────────────────────────────────────────┘

Too high:
  - Loss diverges (increases)
  - NaN values
  - Oscillations

Too low:
  - Very slow progress
  - Gets stuck in local minima
  - Hours/days to train

Good learning rate:
  - Loss decreases steadily
  - Some oscillation is OK
  - Converges in reasonable time

Starting points:
  - Adam: 0.001 (1e-3)        ← START HERE
  - SGD: 0.01 to 0.1
  - Momentum: 0.01
  - RMSProp: 0.001

Tuning strategy:
  1. Start with default
  2. If training is unstable: decrease by 10x (0.001 → 0.0001)
  3. If training is too slow: increase by 3x (0.001 → 0.003)
  4. Use learning rate schedulers (decay over time)

┌──────────────────────────────────────────────────────────────────────┐
│ REAL-WORLD EXAMPLES                                                  │
└──────────────────────────────────────────────────────────────────────┘

GPT-3 Training (OpenAI):
  Optimizer: Adam
  Learning rate: 0.0006 (6e-4)
  beta1: 0.9
  beta2: 0.95
  Batch size: 3.2M tokens

BERT Training (Google):
  Optimizer: Adam
  Learning rate: 1e-4
  beta1: 0.9
  beta2: 0.999
  Warmup: 10,000 steps

ResNet Image Classification:
  Optimizer: SGD + Momentum
  Learning rate: 0.1 (with decay)
  Momentum: 0.9
  LR schedule: Divide by 10 at epochs 30, 60, 90

┌──────────────────────────────────────────────────────────────────────┐
│ BEST PRACTICES                                                       │
└──────────────────────────────────────────────────────────────────────┘

✓ Start with Adam (lr=0.001) - works 90% of the time
✓ Monitor training/validation loss curves
✓ Use learning rate warmup (gradual increase at start)
✓ Apply learning rate decay (reduce over time)
✓ Clip gradients if you see NaN (max norm = 1.0)
✓ Save best model (lowest validation loss)

✗ Don't change all hyperparameters at once
✗ Don't use too large batches (reduces generalization)
✗ Don't forget to normalize/standardize inputs
✗ Don't train without validation set

┌──────────────────────────────────────────────────────────────────────┐
│ SUMMARY TABLE                                                        │
└──────────────────────────────────────────────────────────────────────┘

Optimizer    | LR      | Best For                | Generalization
─────────────┼─────────┼─────────────────────────┼───────────────
SGD          | 0.01    | Classic vision tasks    | ★★★★★
Momentum     | 0.01    | CNNs, general deep learning | ★★★★☆
RMSProp      | 0.001   | RNNs, non-stationary    | ★★★☆☆
Adam         | 0.001   | Transformers, general   | ★★★★☆ (default!)

Speed: Adam ≈ RMSProp > Momentum > SGD
Ease of use: Adam > RMSProp > Momentum > SGD
Generalization: SGD ≥ Momentum ≥ Adam ≥ RMSProp

┌──────────────────────────────────────────────────────────────────────┐
│ THE BOTTOM LINE                                                      │
└──────────────────────────────────────────────────────────────────────┘

🎯 For 95% of projects: Use Adam with lr=0.001

💡 Only if you have specific needs or lots of time for tuning,
   consider other optimizers.

🚀 Modern AI (GPT, BERT, etc.) almost exclusively uses Adam!
"""

    print(guide)

    print("\n" + "="*70)
    print("CONGRATULATIONS!")
    print("="*70)
    print("\nYou now understand:")
    print("✓ All major optimizers (SGD, Momentum, RMSProp, Adam)")
    print("✓ When to use each optimizer")
    print("✓ How to choose hyperparameters")
    print("✓ What GPT-3 uses (Adam!)")
    print("✓ Best practices for production")
    print("\nYou're ready to train any neural network!")


# ============================================================================
# MAIN: Run All Exercises
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXERCISE 6: OPTIMIZERS - PRACTICE PROBLEMS")
    print("="*70)
    print("\nThis exercise covers:")
    print("1. Vanilla Gradient Descent")
    print("2. Momentum (adds velocity)")
    print("3. RMSProp (adaptive learning rate)")
    print("4. Adam (GPT's optimizer!)")
    print("5. Comprehensive comparison")
    print("6. Hyperparameter tuning guide")
    print("\n" + "="*70)

    # Run exercises (uncomment the ones you want to run)

    # Beginner
    exercise_1_vanilla_gd()
    exercise_2_momentum()

    # Intermediate
    exercise_3_rmsprop()
    exercise_4_adam()

    # Advanced
    exercise_5_comprehensive_comparison()
    exercise_6_hyperparameter_guide()

    print("\n" + "="*70)
    print("ALL EXERCISES COMPLETE!")
    print("="*70)
    print("\nYou now know:")
    print("✓ How gradient descent works")
    print("✓ Why Momentum helps")
    print("✓ How RMSProp adapts learning rate")
    print("✓ Why Adam is the default choice")
    print("✓ How to choose hyperparameters")
    print("\nFor your projects: Just use Adam with lr=0.001!")
    print("(That's what GPT-3, BERT, and most modern models use)")
