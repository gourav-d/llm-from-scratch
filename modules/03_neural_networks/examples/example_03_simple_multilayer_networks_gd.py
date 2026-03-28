import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # Weights: (n_inputs, n_neurons), Biases: (1, n_neurons)
        self.weights = np.random.rand(n_inputs, n_neurons) * 0.1 
        self.biases = np.zeros((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.output = None 
        # For now, DenseLayer defaults to Sigmoid activation
        self.activation_fn = sigmoid
        self.activation_fn_name = "sigmoid" # For informational purposes

    def forward(self, inputs):
        """Perform a forward pass through the dense layer."""
        weighted_sum = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation_fn(weighted_sum)
        return self.output
    

class MLP:
    def __init__(self):
        """Initialize a Multi-Layer Perceptron."""
        self.layers = []
    
    def add_layer(self, layer):
        """Add a layer to the MLP."""
        self.layers.append(layer)

    def forward(self, inputs):
        """
        Perform a forward pass through all layers in the MLP.
        inputs: Input data for the first layer.
        """
        current_input = inputs
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    


# Create a sample input
X_sample = np.array([[1.0, 0.5, -1.0, 2.0]])  # Shape (1, 4)
print(f"Input X (shape {X_sample.shape}):\n{X_sample}")

# Create the MLP
mlp = MLP()
mlp.add_layer(DenseLayer(n_inputs=4, n_neurons=5))  # First layer
mlp.add_layer(DenseLayer(n_inputs=5, n_neurons=3))  # Hidden layer
mlp.add_layer(DenseLayer(n_inputs=3, n_neurons=1))  # Output layer

# Print information about the MLP
print(f"\nMLP created with {len(mlp.layers)} layers.")
for i, layer in enumerate(mlp.layers):
    print(f"  Layer {i+1}: {layer.n_inputs} inputs, {layer.n_neurons} neurons, Activation: {layer.activation_fn_name}")



# Processing Data Through the MLP

# Perform forward pass through all layers
output = mlp.forward(X_sample)
print(f"\nOutput of the MLP (shape {output.shape}):\n{output}")

# Create a batch of inputs
X_batch = np.array([
    [1.0, 0.5, -1.0, 2.0],  # First sample
    [0.1, -0.2, 0.3, -0.4]  # Second sample
])  # Shape (2, 4)
print(f"\nInput Batch X (shape {X_batch.shape}):\n{X_batch}")

# Process the batch
output_batch = mlp.forward(X_batch)
print(f"\nOutput of the MLP for batch (shape {output_batch.shape}):\n{output_batch}")