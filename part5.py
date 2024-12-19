
# Import necessary libraries
import sys  # For system-specific parameters and functions
import numpy as np  # For numerical computations
import matplotlib  # For data visualization (not used directly here)
from nnfs.datasets import spiral_data  # To generate a dataset for testing
import nnfs  # Helper library for neural networks

# Initialize nnfs library (sets random seeds and configures the environment for reproducibility)
nnfs.init()

# Set a random seed for numpy to ensure consistent results
np.random.seed(0)

# Generate a dataset: 100 samples per class, 3 classes
# X: The input features (coordinates)
# y: The class labels (not used in this example)
X, y = spiral_data(100, 3)

# Define a dense (fully connected) layer for a neural network
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """
        Initialize the layer with random weights and biases.
        - n_inputs: Number of input features (dimensions of the input data).
        - n_neurons: Number of neurons in the layer.
        """
        # Initialize weights with a small random value scaled by 0.10
        # Shape of weights: (n_inputs, n_neurons)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases to zero
        # Shape of biases: (1, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        """
        Perform a forward pass through the layer:
        - inputs: The input data to the layer (e.g., from the previous layer or the dataset).
        """
        # Calculate the output of the layer using the formula:
        # output = dot(inputs, weights) + biases
        self.output = np.dot(inputs, self.weights) + self.biases

# Define the ReLU (Rectified Linear Unit) activation function
class Activation_ReLU:
    def forward(self, inputs):
        """
        Perform the ReLU activation function:
        - inputs: The input to the activation function (e.g., output from a dense layer).
        - The ReLU function sets all negative values to 0.
        """
        # Apply the ReLU function: max(0, inputs)
        self.output = np.maximum(0, inputs)

# Create the first dense layer
# Input data has 2 features per sample (2D spiral data), and the layer has 5 neurons
layer1 = Layer_Dense(2, 5)

# Create the ReLU activation function for the output of the first layer
activation1 = Activation_ReLU()

# Perform a forward pass through the first dense layer
# Inputs: The spiral dataset (X)
layer1.forward(X)

# Perform a forward pass through the ReLU activation function
# Inputs: The output of the dense layer
activation1.forward(layer1.output)

# Print the output of the activation function (ReLU applied to the dense layer's output)
print(activation1.output)








# Notes
# Activation Function: A function used to determine the output of a node
# Every neuron in hidden layer and output layer will use a activation function
# A Step Input Function: Determineds the output by a fixed number, often one is activated, 2 is not
# Sigmoid Activation Function: A activation function with a more grangular output, has issue as the vanishing gradient
# Rectified Unit Activation Function: (x>0 ? x=x : x=0), Output can be grangular, fast, works well, and is most popular Activation Function
# nnfs: Package we will use too make sure all data is the same as the tutorial, and 