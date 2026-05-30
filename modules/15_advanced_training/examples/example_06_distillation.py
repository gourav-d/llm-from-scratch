"""
Example 06: Knowledge Distillation — Teacher-Student Training
Module 15: Advanced LLM Training

Implements knowledge distillation from scratch:
  - Teacher produces soft labels (probability distributions)
  - Student learns from both soft labels AND hard labels
  - Temperature scaling to soften teacher distributions
  - KL divergence loss for matching distributions

Run:  python example_06_distillation.py
Deps: numpy
"""

import numpy as np


def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


# ─────────────────────────────────────────────────────────
# MATH: SOFTMAX AND LOSSES
# ─────────────────────────────────────────────────────────

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Temperature-scaled softmax.
    Higher T → softer (more uniform) distribution.
    T=1 → standard softmax.
    """
    x_scaled = x / temperature
    x_shifted = x_scaled - np.max(x_scaled, axis=-1, keepdims=True)  # numerical stability
    exp_x = np.exp(x_shifted)
    return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Standard cross-entropy loss for hard labels.
    targets: integer class indices (hard labels).
    """
    probs = softmax(logits, temperature=1.0)
    batch_size = logits.shape[0]
    # Log-probability of the correct class
    correct_log_probs = np.log(probs[np.arange(batch_size), targets] + 1e-10)
    return -np.mean(correct_log_probs)


def kl_divergence_loss(student_logits: np.ndarray,
                        teacher_logits: np.ndarray,
                        temperature: float) -> float:
    """
    KL divergence loss between teacher's soft distribution and student's.

    KL(P || Q) = sum(P × log(P / Q))
    P = teacher soft distribution
    Q = student soft distribution

    Scaled by T^2 to keep gradient magnitude consistent.
    """
    P = softmax(teacher_logits, temperature=temperature)   # teacher soft probs
    Q = softmax(student_logits, temperature=temperature)   # student soft probs

    # KL divergence per sample: sum over classes
    kl_per_sample = np.sum(P * (np.log(P + 1e-10) - np.log(Q + 1e-10)), axis=-1)

    return np.mean(kl_per_sample) * (temperature ** 2)


def distillation_loss(student_logits: np.ndarray,
                       teacher_logits: np.ndarray,
                       hard_targets: np.ndarray,
                       temperature: float = 4.0,
                       alpha: float = 0.7) -> tuple[float, float, float]:
    """
    Combined distillation loss.

    total = alpha × KL(teacher || student) + (1-alpha) × CrossEntropy(student, targets)

    Returns: (total_loss, kl_loss, ce_loss)
    """
    loss_kl = kl_divergence_loss(student_logits, teacher_logits, temperature)
    loss_ce = cross_entropy_loss(student_logits, hard_targets)
    total = alpha * loss_kl + (1 - alpha) * loss_ce
    return total, loss_kl, loss_ce


# ─────────────────────────────────────────────────────────
# SIMPLE LINEAR MODELS (teacher and student)
# ─────────────────────────────────────────────────────────

class SimpleClassifier:
    """
    Single-layer linear classifier.
    Represents a (very simplified) language model head.
    Teacher: larger hidden dim.
    Student: smaller hidden dim.
    """

    def __init__(self, input_dim: int, n_classes: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + n_classes))
        self.W = rng.randn(n_classes, input_dim).astype(np.float32) * scale
        self.b = np.zeros(n_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x shape: [batch, input_dim] → logits shape: [batch, n_classes]"""
        return x @ self.W.T + self.b

    def backward(self, x: np.ndarray, grad_logits: np.ndarray) -> np.ndarray:
        """Compute gradients for W and b. Returns grad for x."""
        batch = x.shape[0]
        self.dW = grad_logits.T @ x / batch
        self.db = np.mean(grad_logits, axis=0)
        return grad_logits @ self.W

    def step(self, lr: float):
        """SGD update."""
        self.W -= lr * self.dW
        self.b -= lr * self.db


def softmax_grad(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Gradient of cross-entropy loss w.r.t. logits = (probs - one_hot)."""
    batch = probs.shape[0]
    grad = probs.copy()
    grad[np.arange(batch), targets] -= 1.0
    return grad / batch


def kl_grad_student(student_logits: np.ndarray,
                     teacher_logits: np.ndarray,
                     temperature: float) -> np.ndarray:
    """
    Gradient of KL(teacher || student) w.r.t. student_logits.
    = T^2 × (Q - P) / temperature  (where Q = student probs, P = teacher probs)
    """
    P = softmax(teacher_logits, temperature)
    Q = softmax(student_logits, temperature)
    # Gradient of KL w.r.t. student softmax input (before softmax)
    grad = (Q - P) / temperature
    return grad * (temperature ** 2)


# ─────────────────────────────────────────────────────────
# DEMO 1: Temperature Effect on Soft Labels
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: Temperature Scaling — Soft Label Examples")

# Simulated teacher logits for vocabulary of 10 tokens
teacher_logits = np.array([[5.0, 1.0, 0.5, 0.2, -1.0, -2.0, -0.5, 0.3, -0.3, 0.1]])
labels = ["cat", "feline", "pet", "animal", "dog", "bird", "kitten", "creature", "paw", "fur"]

print("\nTeacher logits (10-token vocab): [5.0, 1.0, 0.5, 0.2, -1.0, ...]")
print("\nSoftmax distributions at different temperatures:")
print(f"\n{'Token':<12}", end="")

for T in [1.0, 2.0, 4.0, 8.0]:
    print(f"T={T:>3.1f}  ", end="")
print()
print("-" * 62)

probs_by_T = {}
for i, label in enumerate(labels):
    print(f"{label:<12}", end="")
    for T in [1.0, 2.0, 4.0, 8.0]:
        probs = softmax(teacher_logits, temperature=T)[0]
        if T not in probs_by_T:
            probs_by_T[T] = probs
        print(f"{probs[i]:.4f}  ", end="")
    print()

print("\nEntropy of teacher distribution (higher = more spread out):")
for T in [1.0, 2.0, 4.0, 8.0]:
    probs = probs_by_T[T]
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    print(f"  T={T}: entropy = {entropy:.4f}")

print("""
LESSON: Higher temperature → more uniform distribution.
        At T=1: 'cat' gets 96% probability — not much information.
        At T=4: 'cat' gets 50%, 'feline' 18%, 'pet' 13% — richer signal.
        Student learns relationships between tokens, not just "cat is correct".
""")


# ─────────────────────────────────────────────────────────
# DEMO 2: KL Divergence
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: KL Divergence — Measuring Distribution Distance")

np.random.seed(42)
n_classes = 8

P = np.array([[0.5, 0.2, 0.1, 0.08, 0.05, 0.04, 0.02, 0.01]])  # teacher
Q_good = np.array([[0.48, 0.21, 0.11, 0.07, 0.06, 0.04, 0.02, 0.01]])  # good student
Q_bad  = np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.93]])  # bad student
Q_equal = np.ones((1, n_classes)) / n_classes                             # uniform student

print("\nTeacher P:       ", [f"{x:.3f}" for x in P[0]])
print("Good student Q1: ", [f"{x:.3f}" for x in Q_good[0]])
print("Bad student Q2:  ", [f"{x:.3f}" for x in Q_bad[0]])
print("Uniform Q3:      ", [f"{x:.3f}" for x in Q_equal[0]])

def kl(P, Q):
    return float(np.sum(P * (np.log(P + 1e-10) - np.log(Q + 1e-10))))

print(f"\n  KL(P || P)       = {kl(P[0], P[0]):.6f}   (identical = 0)")
print(f"  KL(P || Q_good)  = {kl(P[0], Q_good[0]):.6f}   (close)")
print(f"  KL(P || Q_bad)   = {kl(P[0], Q_bad[0]):.6f}   (very different)")
print(f"  KL(P || Q_equal) = {kl(P[0], Q_equal[0]):.6f}   (uniform)")

print("""
LESSON: KL(P||Q) = 0 when student = teacher (goal of distillation).
        Higher KL = student more wrong.
        We minimize KL during training to make student match teacher.
""")


# ─────────────────────────────────────────────────────────
# DEMO 3: Distillation Training Loop
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: Distillation Training Loop")

np.random.seed(42)

N_CLASSES = 10   # vocabulary size (toy)
N_TRAIN   = 200  # training examples
INPUT_DIM = 32   # input feature size

# Generate synthetic classification task
X_train = np.random.randn(N_TRAIN, INPUT_DIM).astype(np.float32)
y_train = np.random.randint(0, N_CLASSES, size=N_TRAIN)

# Teacher: already trained (good accuracy)
teacher = SimpleClassifier(INPUT_DIM, N_CLASSES, seed=0)
# "Train" teacher: give it good weights by biasing toward correct class
for i in range(N_TRAIN):
    teacher.W[y_train[i]] += X_train[i] * 0.05   # simple pre-training simulation

# Student: untrained, smaller (pretend it's smaller by using a different seed)
student_ce  = SimpleClassifier(INPUT_DIM, N_CLASSES, seed=10)   # trained with CE only
student_kd  = SimpleClassifier(INPUT_DIM, N_CLASSES, seed=10)   # trained with distillation

EPOCHS = 30
LR = 0.01
TEMP = 4.0
ALPHA = 0.7
BATCH = 32

def accuracy(model, X, y):
    logits = model.forward(X)
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

def train_epoch_ce(model, X, y, lr, batch_size):
    indices = np.random.permutation(len(X))
    total_loss = 0
    for start in range(0, len(X), batch_size):
        idx = indices[start:start+batch_size]
        xb, yb = X[idx], y[idx]
        logits = model.forward(xb)
        loss = cross_entropy_loss(logits, yb)
        total_loss += loss
        # Backward
        probs = softmax(logits)
        grad = softmax_grad(probs, yb)
        model.backward(xb, grad)
        model.step(lr)
    return total_loss / (len(X) // batch_size)

def train_epoch_kd(student, teacher, X, y, lr, batch_size, temp, alpha):
    indices = np.random.permutation(len(X))
    total_loss = 0
    for start in range(0, len(X), batch_size):
        idx = indices[start:start+batch_size]
        xb, yb = X[idx], y[idx]
        t_logits = teacher.forward(xb)    # teacher (frozen)
        s_logits = student.forward(xb)    # student

        loss, loss_kl, loss_ce = distillation_loss(s_logits, t_logits, yb, temp, alpha)
        total_loss += loss

        # Combined gradient: alpha × KL_grad + (1-alpha) × CE_grad
        grad_kl = kl_grad_student(s_logits, t_logits, temp)
        probs_s = softmax(s_logits)
        grad_ce = softmax_grad(probs_s, yb)
        grad = alpha * grad_kl + (1 - alpha) * grad_ce

        student.backward(xb, grad)
        student.step(lr)
    return total_loss / (len(X) // batch_size)

print(f"\nTraining {EPOCHS} epochs | LR={LR} | T={TEMP} | alpha={ALPHA}")
print(f"\n{'Epoch':>6} {'CE-only acc':>12} {'Distill acc':>12} {'Teacher acc':>12}")
print("-" * 50)

for epoch in range(EPOCHS):
    train_epoch_ce(student_ce, X_train, y_train, LR, BATCH)
    train_epoch_kd(student_kd, teacher, X_train, y_train, LR, BATCH, TEMP, ALPHA)

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        acc_ce = accuracy(student_ce, X_train, y_train)
        acc_kd = accuracy(student_kd, X_train, y_train)
        acc_t  = accuracy(teacher, X_train, y_train)
        print(f"{epoch:>6} {acc_ce:>12.3f} {acc_kd:>12.3f} {acc_t:>12.3f}")

print(f"\nFinal teacher accuracy:         {accuracy(teacher, X_train, y_train):.3f}")
print(f"Final student (CE only):        {accuracy(student_ce, X_train, y_train):.3f}")
print(f"Final student (distillation):   {accuracy(student_kd, X_train, y_train):.3f}")


# ─────────────────────────────────────────────────────────
# DEMO 4: Effect of Alpha (distillation weight)
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: Effect of Alpha on Distillation Quality")

print(f"\nAlpha controls: total = alpha×KL + (1-alpha)×CrossEntropy")
print(f"alpha=0.0 → pure CE, no distillation")
print(f"alpha=1.0 → pure KL, no hard labels\n")

alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

print(f"{'Alpha':>8} {'Final Accuracy':>16} {'vs CE-only':>12}")
print("-" * 40)

ce_acc = accuracy(student_ce, X_train, y_train)

for alpha in alphas:
    student_test = SimpleClassifier(INPUT_DIM, N_CLASSES, seed=10)
    for _ in range(EPOCHS):
        train_epoch_kd(student_test, teacher, X_train, y_train,
                       LR, BATCH, TEMP, alpha)
    acc = accuracy(student_test, X_train, y_train)
    diff = acc - ce_acc
    marker = " <-- CE baseline" if alpha == 0.0 else ""
    print(f"{alpha:>8.1f} {acc:>16.3f} {diff:>+12.3f}{marker}")

print("""
LESSON: Pure CE (alpha=0) is the baseline.
        Adding distillation (alpha > 0) often improves accuracy.
        Pure KL (alpha=1) can underperform if teacher is imperfect.
        Typical best: alpha = 0.5 to 0.9 (favor distillation but keep hard labels).
""")
