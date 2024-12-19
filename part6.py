
# Import necessary libraries
import sys  # For system-specific parameters and functions
import numpy as np  # For numerical computations
import matplotlib  # For data visualization (not used directly here)
from nnfs.datasets import spiral_data  # To generate a dataset for testing
import nnfs  # Helper library for neural networks
nnfs.init()
import math

print(f'Using A Single List:')
layer_outputs = [4.8, 1.21, 2.385]

# Get Eulers number
E = math.e

exp_values = []

# Get the exponential values
# exp_values = np.exp(layer_outputs) You can also do it like this
for output in layer_outputs:
    exp_values.append(E**output)

print(f'Exp Values = {exp_values}')

# Nomalize the values
# norm_values = exp_values /np.sum(exp_values) You can do this to normalize
norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

print(f'Normalized Values = {norm_values}')
print(f'Sum Norm Values is {sum(norm_values)}')

print(f'\nUsing A Batch:')

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026],]

# Fet the expenential values
exp_values = np.exp(layer_outputs)

# Normalize the data
norm_base = np.sum(exp_values, axis=1, keepdims=True)

norm_values = exp_values / norm_base 

print(norm_values)


print(f'\nPart 3:')
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

class Activation_Softmax:
    def forward(self, inputs):
        #Get exp values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Get probabilities
        probabilies = exp_values /np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilies


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

'''
Nots:
- Exponencial Function y=e^x: Solves negative issue by making no outputs negative but keeps track of negatives
- Normilization: The value divied by the sum of all the other values gives you a normalized value
- Softmax = Input -> Exponentiation -> Normalization -> Output
'''