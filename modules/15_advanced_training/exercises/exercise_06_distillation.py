"""
Exercise 06: Knowledge Distillation — Train a Student Model
Module 15: Advanced LLM Training

TASKS:
  1. Implement temperature_softmax() — softmax with temperature scaling
  2. Implement kl_divergence() — measure distribution distance
  3. Implement distillation_loss() — combined KL + cross-entropy loss
  4. Implement distillation_gradient() — gradient for the student
  5. Run distillation training and compare to CE-only baseline

Run:  python exercise_06_distillation.py
Deps: numpy
"""

import numpy as np


# ─────────────────────────────────────────────────────────
# TASK 1: Temperature Softmax
# ─────────────────────────────────────────────────────────

def temperature_softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """
    Compute softmax with temperature scaling.

    Steps:
      1. Divide logits by T
      2. Subtract row max (for numerical stability)
      3. Compute exp
      4. Divide by row sum

    Args:
        logits: 2D array [batch, n_classes]
        T:      temperature (>1 = softer, <1 = sharper)

    Returns:
        Probability distribution, shape [batch, n_classes]

    HINT:
        scaled = logits / T
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_s = np.exp(shifted)
        return exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-10)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: KL Divergence
# ─────────────────────────────────────────────────────────

def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute KL divergence KL(P || Q) averaged over batch.

    KL(P || Q) = sum over classes: P × log(P / Q)
               = sum: P × (log P - log Q)

    Returns 0 when P == Q. Higher when distributions differ more.

    Args:
        P: target (teacher) probabilities [batch, n_classes]
        Q: approximate (student) probabilities [batch, n_classes]

    Returns:
        Scalar: mean KL divergence over batch

    HINT:
        kl_per_sample = np.sum(P * (np.log(P + 1e-10) - np.log(Q + 1e-10)), axis=-1)
        return float(np.mean(kl_per_sample))
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Distillation Loss
# ─────────────────────────────────────────────────────────

def distillation_loss(student_logits: np.ndarray,
                       teacher_logits: np.ndarray,
                       hard_targets: np.ndarray,
                       T: float = 4.0,
                       alpha: float = 0.7) -> tuple[float, float, float]:
    """
    Combined knowledge distillation loss.

    Formula:
        loss_kl = KL(teacher_soft(T) || student_soft(T)) × T²
        loss_ce = CrossEntropy(student_logits, hard_targets)
        total   = alpha × loss_kl + (1 - alpha) × loss_ce

    The T² factor keeps gradient magnitude consistent across temperatures.

    Args:
        student_logits: [batch, n_classes]
        teacher_logits: [batch, n_classes]
        hard_targets:   [batch] integer class labels
        T:              temperature
        alpha:          weight for distillation vs CE loss

    Returns:
        (total_loss, loss_kl, loss_ce)

    HINT:
        P = temperature_softmax(teacher_logits, T)
        Q = temperature_softmax(student_logits, T)
        loss_kl = kl_divergence(P, Q) * (T ** 2)

        probs_ce = temperature_softmax(student_logits, T=1.0)
        batch = student_logits.shape[0]
        correct_log_probs = np.log(probs_ce[np.arange(batch), hard_targets] + 1e-10)
        loss_ce = -np.mean(correct_log_probs)

        total = alpha * loss_kl + (1 - alpha) * loss_ce
        return total, loss_kl, loss_ce
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Distillation Gradient
# ─────────────────────────────────────────────────────────

def distillation_gradient(student_logits: np.ndarray,
                           teacher_logits: np.ndarray,
                           hard_targets: np.ndarray,
                           T: float = 4.0,
                           alpha: float = 0.7) -> np.ndarray:
    """
    Compute gradient of distillation loss w.r.t. student_logits.

    Combined gradient:
        grad_total = alpha × grad_kl + (1 - alpha) × grad_ce

    Where:
        grad_kl = (Q - P) / T × T²  = T × (Q - P)
                  Q = student soft probs at temperature T
                  P = teacher soft probs at temperature T

        grad_ce = probs_ce - one_hot(hard_targets)
                  probs_ce = student probs at T=1

    Args:
        student_logits: [batch, n_classes]
        teacher_logits: [batch, n_classes]
        hard_targets:   [batch] integer class labels
        T:              temperature
        alpha:          weight for distillation vs CE

    Returns:
        Gradient w.r.t. student_logits, shape [batch, n_classes]

    HINT:
        batch, n_classes = student_logits.shape

        # KL gradient
        P = temperature_softmax(teacher_logits, T)
        Q = temperature_softmax(student_logits, T)
        grad_kl = (Q - P) * T          # T² / T = T (one T cancels from chain rule)

        # CE gradient
        probs_ce = temperature_softmax(student_logits, T=1.0)
        grad_ce = probs_ce.copy()
        grad_ce[np.arange(batch), hard_targets] -= 1.0
        grad_ce /= batch

        return alpha * grad_kl + (1 - alpha) * grad_ce
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# SIMPLE MODEL FOR TESTING
# ─────────────────────────────────────────────────────────

class TinyLinearModel:
    """Single linear layer: y = xW^T + b."""

    def __init__(self, input_dim: int, n_classes: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (input_dim + n_classes))
        self.W = rng.randn(n_classes, input_dim).astype(np.float32) * scale
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.last_x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        return x @ self.W.T + self.b

    def backward(self, grad_logits: np.ndarray):
        batch = self.last_x.shape[0]
        self.dW = grad_logits.T @ self.last_x / batch
        self.db = np.mean(grad_logits, axis=0)

    def step(self, lr: float):
        self.W -= lr * self.dW
        self.b -= lr * self.db


def accuracy(model: TinyLinearModel, X: np.ndarray, y: np.ndarray) -> float:
    logits = model.forward(X)
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 60)
    print("  Exercise 06: Knowledge Distillation")
    print("=" * 60)

    np.random.seed(42)

    # Test 1: temperature_softmax
    print("\n--- Test 1: temperature_softmax ---")
    logits = np.array([[2.0, 1.0, 0.0, -1.0]])
    result_T1 = temperature_softmax(logits, T=1.0)
    result_T4 = temperature_softmax(logits, T=4.0)

    if result_T1 is None:
        print("  NOT IMPLEMENTED YET")
    else:
        # Check: row sums = 1
        if abs(np.sum(result_T1) - 1.0) < 1e-5:
            print(f"  PASS  T=1 row sum = 1.0")
        else:
            print(f"  FAIL  T=1 row sum = {np.sum(result_T1):.4f} (should be 1)")

        # Check: T=4 should be more uniform than T=1
        max_T1 = np.max(result_T1)
        max_T4 = np.max(result_T4)
        if max_T4 < max_T1:
            print(f"  PASS  T=4 more uniform: max_prob {max_T4:.3f} < T=1 max_prob {max_T1:.3f}")
        else:
            print(f"  FAIL  expected T=4 to be more uniform")
        print(f"  T=1: {result_T1[0]}")
        print(f"  T=4: {result_T4[0]}")

    # Test 2: kl_divergence
    print("\n--- Test 2: kl_divergence ---")
    P = np.array([[0.6, 0.3, 0.1]])
    Q_same = np.array([[0.6, 0.3, 0.1]])
    Q_diff = np.array([[0.1, 0.3, 0.6]])

    if temperature_softmax(logits) is not None:
        kl_zero = kl_divergence(P, Q_same)
        kl_diff = kl_divergence(P, Q_diff)
        if kl_zero is None:
            print("  NOT IMPLEMENTED YET")
        else:
            if abs(kl_zero) < 1e-5:
                print(f"  PASS  KL(P || P) = {kl_zero:.6f} (should be ~0)")
            else:
                print(f"  FAIL  KL(P || P) = {kl_zero:.6f} (should be 0)")
            if kl_diff > 0.1:
                print(f"  PASS  KL(P || Q_different) = {kl_diff:.4f} (should be > 0)")
            else:
                print(f"  FAIL  KL(P || Q_different) = {kl_diff:.4f} (should be > 0)")

    # Test 3: distillation_loss
    print("\n--- Test 3: distillation_loss ---")
    student_logits = np.random.randn(4, 5)
    teacher_logits = np.random.randn(4, 5)
    targets = np.array([0, 2, 4, 1])

    result = distillation_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.7)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        total, kl, ce = result
        if total > 0 and kl > 0 and ce > 0:
            print(f"  PASS  total={total:.4f}, kl={kl:.4f}, ce={ce:.4f}")
            # Verify: total ≈ 0.7 × kl + 0.3 × ce
            expected_total = 0.7 * kl + 0.3 * ce
            if abs(total - expected_total) < 1e-5:
                print(f"  PASS  total = 0.7×kl + 0.3×ce (verified)")
            else:
                print(f"  FAIL  total mismatch: {total:.6f} vs {expected_total:.6f}")
        else:
            print(f"  FAIL  got ({total}, {kl}, {ce}) — all should be > 0")

    # Test 4: Full Distillation Training
    print("\n--- Test 4: Distillation Training ---")

    grad_fn = distillation_gradient
    if grad_fn(student_logits, teacher_logits, targets) is None:
        print("  NOT IMPLEMENTED YET")
    else:
        # Setup toy classification problem
        N_CLASSES = 6
        INPUT_DIM = 16
        N_TRAIN   = 150
        EPOCHS    = 25
        LR        = 0.02
        T         = 4.0
        ALPHA     = 0.7

        np.random.seed(0)
        X = np.random.randn(N_TRAIN, INPUT_DIM).astype(np.float32)
        y = np.random.randint(0, N_CLASSES, N_TRAIN)

        # Pretrained teacher (given good weights via manual adjustment)
        teacher = TinyLinearModel(INPUT_DIM, N_CLASSES, seed=0)
        for i in range(N_TRAIN):
            teacher.W[y[i]] += X[i] * 0.08

        # Train two students: CE-only and distillation
        student_ce = TinyLinearModel(INPUT_DIM, N_CLASSES, seed=1)
        student_kd = TinyLinearModel(INPUT_DIM, N_CLASSES, seed=1)

        for epoch in range(EPOCHS):
            # Shuffle
            idx = np.random.permutation(N_TRAIN)
            for start in range(0, N_TRAIN, 16):
                batch_idx = idx[start:start+16]
                xb = X[batch_idx]
                yb = y[batch_idx]

                # CE-only student
                s_logits = student_ce.forward(xb)
                probs = temperature_softmax(s_logits, 1.0)
                grad = probs.copy()
                grad[np.arange(len(yb)), yb] -= 1.0
                grad /= len(yb)
                student_ce.backward(grad)
                student_ce.step(LR)

                # Distillation student
                t_logits = teacher.forward(xb)
                s_logits = student_kd.forward(xb)
                grad = distillation_gradient(s_logits, t_logits, yb, T, ALPHA)
                if grad is not None:
                    student_kd.backward(grad)
                    student_kd.step(LR)

        acc_teacher = accuracy(teacher, X, y)
        acc_ce = accuracy(student_ce, X, y)
        acc_kd = accuracy(student_kd, X, y)

        print(f"\n  Teacher accuracy:          {acc_teacher:.3f}")
        print(f"  Student (CE only):         {acc_ce:.3f}")
        print(f"  Student (distillation):    {acc_kd:.3f}")

        if acc_kd >= acc_ce - 0.05:
            print(f"\n  PASS  distillation student ≥ CE student (within 5%)")
        else:
            print(f"\n  FAIL  distillation student worse by more than 5%")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def temperature_softmax(logits, T=1.0):
#     scaled = logits / T
#     shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
#     exp_s = np.exp(shifted)
#     return exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-10)
#
# def kl_divergence(P, Q):
#     kl = np.sum(P * (np.log(P + 1e-10) - np.log(Q + 1e-10)), axis=-1)
#     return float(np.mean(kl))
#
# def distillation_loss(student_logits, teacher_logits, hard_targets, T=4.0, alpha=0.7):
#     P = temperature_softmax(teacher_logits, T)
#     Q = temperature_softmax(student_logits, T)
#     loss_kl = kl_divergence(P, Q) * (T ** 2)
#     probs_ce = temperature_softmax(student_logits, T=1.0)
#     batch = student_logits.shape[0]
#     loss_ce = -np.mean(np.log(probs_ce[np.arange(batch), hard_targets] + 1e-10))
#     total = alpha * loss_kl + (1 - alpha) * loss_ce
#     return total, loss_kl, loss_ce
#
# def distillation_gradient(student_logits, teacher_logits, hard_targets, T=4.0, alpha=0.7):
#     batch, n_classes = student_logits.shape
#     P = temperature_softmax(teacher_logits, T)
#     Q = temperature_softmax(student_logits, T)
#     grad_kl = (Q - P) * T
#     probs_ce = temperature_softmax(student_logits, T=1.0)
#     grad_ce = probs_ce.copy()
#     grad_ce[np.arange(batch), hard_targets] -= 1.0
#     grad_ce /= batch
#     return alpha * grad_kl + (1 - alpha) * grad_ce
