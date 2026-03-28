"""
example_03_visualization.py
============================
Visualizes the MLP from example_03_simple_multilayer_networks_gd.py

What this file shows:
  1. Network Architecture  — nodes and connections diagram
  2. Activation Heatmap    — values flowing through each layer
  3. Weight Heatmap        — weight matrix of each layer

Run:
    python modules/03_neural_networks/examples/example_03_visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────
# 1. Re-build the same MLP as example_03
#    (We use a fixed random seed so weights are always the same)
# ─────────────────────────────────────────────────────────────────

def sigmoid(x):
    """Sigmoid activation: squishes any number into the range (0, 1)."""
    return 1 / (1 + np.exp(-x))


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs  = n_inputs
        self.n_neurons = n_neurons
        self.weights   = np.random.rand(n_inputs, n_neurons) * 0.1
        self.biases    = np.zeros((1, n_neurons))
        self.output    = None

    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.biases
        self.output  = sigmoid(weighted_sum)
        return self.output


np.random.seed(42)   # Fixed seed → reproducible weights every run

layer1 = DenseLayer(n_inputs=4, n_neurons=5)   # Input  → Hidden 1
layer2 = DenseLayer(n_inputs=5, n_neurons=3)   # Hidden 1 → Hidden 2
layer3 = DenseLayer(n_inputs=3, n_neurons=1)   # Hidden 2 → Output

layers      = [layer1, layer2, layer3]
layer_sizes = [4, 5, 3, 1]
layer_names = ["Input", "Hidden 1", "Hidden 2", "Output"]

# Input sample (same as example_03)
X = np.array([[1.0, 0.5, -1.0, 2.0]])   # Shape (1, 4)

# Forward pass — collect output of every layer
all_activations = [X]          # index 0 = raw input
current = X
for layer in layers:
    current = layer.forward(current)
    all_activations.append(current)


# ─────────────────────────────────────────────────────────────────
# 2. VISUALIZATION A — Network Architecture Diagram
# ─────────────────────────────────────────────────────────────────

def draw_architecture(layer_sizes, layer_names):
    """Draw circles for neurons and lines for connections."""

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#f7f9fc')
    ax.set_facecolor('#f7f9fc')
    ax.axis('off')
    ax.set_title("MLP Architecture  (4 → 5 → 3 → 1)",
                 fontsize=15, fontweight='bold', pad=20)

    n_layers    = len(layer_sizes)
    x_positions = np.linspace(0.1, 0.9, n_layers)  # Evenly spaced columns

    # Build a dict: positions[layer_idx] = list of (x, y) for each neuron
    positions = {}
    for li, (x, n) in enumerate(zip(x_positions, layer_sizes)):
        ys = np.linspace(0.1, 0.9, n)
        positions[li] = [(x, y) for y in ys]

    # Draw connection lines first (so circles appear on top)
    for li in range(n_layers - 1):
        for (x1, y1) in positions[li]:
            for (x2, y2) in positions[li + 1]:
                ax.plot([x1, x2], [y1, y2],
                        color='#aac4e0', linewidth=0.7, alpha=0.6, zorder=1)

    # Draw neuron circles
    COLORS = ['#4CAF50', '#2196F3', '#2196F3', '#FF5722']  # Input, Hidden, Hidden, Output
    for li, nodes in positions.items():
        for ni, (x, y) in enumerate(nodes):
            circle = plt.Circle((x, y), 0.028,
                                 color=COLORS[li], ec='white', linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            # Neuron label inside circle
            ax.text(x, y, str(ni + 1),
                    ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold', zorder=4)

    # Layer labels below each column
    for li, (x, name, n) in enumerate(zip(x_positions, layer_names, layer_sizes)):
        ax.text(x, 0.02, f"{name}\n({n} neurons)",
                ha='center', va='bottom', fontsize=9,
                color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.8))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Input layer'),
        Patch(facecolor='#2196F3', label='Hidden layers'),
        Patch(facecolor='#FF5722', label='Output layer'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              framealpha=0.9, fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("mlp_architecture.png", dpi=150, bbox_inches='tight')
    print("Saved: mlp_architecture.png")
    return fig


# ─────────────────────────────────────────────────────────────────
# 3. VISUALIZATION B — Activation Heatmap (signal through layers)
# ─────────────────────────────────────────────────────────────────

def draw_activations(all_activations, layer_names):
    """
    Show the output value of every neuron at every stage.
    Green = high activation, Red = low activation.
    """

    n_stages = len(all_activations)           # 4 stages: Input + 3 layers
    fig, axes = plt.subplots(1, n_stages, figsize=(14, 4))
    fig.patch.set_facecolor('#f7f9fc')
    fig.suptitle("Activation Values Flowing Through the MLP\n"
                 "(each cell = one neuron's output value)",
                 fontsize=13, fontweight='bold')

    for i, (values, name) in enumerate(zip(all_activations, layer_names)):
        ax  = axes[i]
        col = values.T                         # Shape: (n_neurons, 1) for imshow

        # Color range: Input can be negative; activations are 0-1
        vmin, vmax = (-2, 2) if i == 0 else (0, 1)

        im = ax.imshow(col, cmap='RdYlGn', vmin=vmin, vmax=vmax,
                       aspect='auto', interpolation='nearest')

        ax.set_title(f"{name}\n({values.shape[1]} neurons)",
                     fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks(range(values.shape[1]))
        ax.set_yticklabels([f'n{j + 1}' for j in range(values.shape[1])],
                           fontsize=8)

        # Print the actual number inside each cell
        for j in range(values.shape[1]):
            ax.text(0, j, f'{values[0, j]:.3f}',
                    ha='center', va='center',
                    fontsize=9, color='black', fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.8)

        # Draw arrow between stages
        if i < n_stages - 1:
            fig.text((i + 1) / n_stages - 0.01, 0.52, '→',
                     ha='center', fontsize=18, color='#555555')

    plt.tight_layout()
    plt.savefig("mlp_activations.png", dpi=150, bbox_inches='tight')
    print("Saved: mlp_activations.png")
    return fig


# ─────────────────────────────────────────────────────────────────
# 4. VISUALIZATION C — Weight Heatmaps for every layer
# ─────────────────────────────────────────────────────────────────

def draw_weights(layers):
    """
    Show the weight matrix of each layer.
    Each cell = the weight connecting input_i → neuron_j.
    Blue = positive weight, Red = negative weight.
    """

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(14, 4))
    fig.patch.set_facecolor('#f7f9fc')
    fig.suptitle("Weight Matrices for Each Layer\n"
                 "(rows = inputs coming IN, columns = neurons going OUT)",
                 fontsize=13, fontweight='bold')

    for i, layer in enumerate(layers):
        ax = axes[i]
        W  = layer.weights                     # Shape: (n_inputs, n_neurons)

        im = ax.imshow(W, cmap='coolwarm',
                       vmin=-W.max(), vmax=W.max(),
                       aspect='auto', interpolation='nearest')

        ax.set_title(f"Layer {i + 1} Weights\n"
                     f"({layer.n_inputs} inputs → {layer.n_neurons} neurons)",
                     fontsize=9, fontweight='bold')

        ax.set_xlabel("Neuron (output side)", fontsize=8)
        ax.set_ylabel("Input (input side)", fontsize=8)

        ax.set_xticks(range(layer.n_neurons))
        ax.set_xticklabels([f'N{j + 1}' for j in range(layer.n_neurons)], fontsize=8)
        ax.set_yticks(range(layer.n_inputs))
        ax.set_yticklabels([f'X{j + 1}' for j in range(layer.n_inputs)], fontsize=8)

        # Print the actual weight value inside each cell
        for r in range(layer.n_inputs):
            for c in range(layer.n_neurons):
                ax.text(c, r, f'{W[r, c]:.3f}',
                        ha='center', va='center',
                        fontsize=8, color='black')

        plt.colorbar(im, ax=ax, shrink=0.8, label='Weight value')

    plt.tight_layout()
    plt.savefig("mlp_weights.png", dpi=150, bbox_inches='tight')
    print("Saved: mlp_weights.png")
    return fig


# ─────────────────────────────────────────────────────────────────
# 5. RUN ALL THREE VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────

print("=" * 55)
print("  MLP Visualization")
print("=" * 55)
print(f"\nInput  X : {X}")
print(f"\nForward pass outputs:")
for i, (act, name) in enumerate(zip(all_activations[1:], layer_names[1:])):
    print(f"  {name}: {act}")

print("\nGenerating plots...")

fig1 = draw_architecture(layer_sizes, layer_names)
fig2 = draw_activations(all_activations, layer_names)
fig3 = draw_weights(layers)

print("\nAll 3 plots ready. Close each window to continue.\n")
plt.show()   # Shows all 3 windows together
