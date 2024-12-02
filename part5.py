
import sys
import numpy as np
import matplotlib
from nnfs.datasets import spiral_data
import nnfs


nnfs.init()


np.random.seed(0)


X, y = spiral_data(100,3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Weights on n inputs * number of neurons we have
        self.weights = 0.10* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        # Rectified Unit Activation Function
        self.output = np.maximum(0, inputs)


# Make first Layer
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

# Move and activate first layer
layer1.forward(X)
activation1.forward(layer1.output)


print(activation1.output)








# Notes
# Activation Function: A function used to determine the output of a node
# Every neuron in hidden layer and output layer will use a activation function
# A Step Input Function: Determineds the output by a fixed number, often one is activated, 2 is not
# Sigmoid Activation Function: A activation finction with a more grangular output, has issue as the vanishing gradient
# Rectified Unit Activation Function: (x>0 ? x=x : x=0), Output can be grangular, fast, works well, and is most popular Activation Function
# nnfs: Package we will use too make sure all data is the same as the tutorial, and 