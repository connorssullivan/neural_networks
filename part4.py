import sys
import numpy as np
import matplotlib

np.random.seed(0)

X = [
    [1,2,3,2.5],
    [2.0, 5.0, -1.0,2.0],
    [-1.5, 2.7, 3.3, -0.8],
    ]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Weights on n inputs * number of neurons we have
        self.weights = 0.10* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2) # Input has to be same size as output from first layer

# Move first layer
layer1.forward(X)
print(layer1.output)

# Move second laayer
layer2.forward(layer1.output)
print(layer2)





'''
print("Python ", sys.version)
print('Numpy ', np.__version__)
print('Matplotlib: ', matplotlib.__version__)

# Every neuron has a unique connection to every previous neurons


# Input of batches
inputs = [
    [1,2,3,2.5],
    [2.0, 5.0, -1.0,2.0],
    [-1.5, 2.7, 3.3, -0.8],
    ]

# 1st layer of neurons
weights1 = [
    [0.2, 0.8, -0.5,1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27,0.17, 0.87]
]
biases1 = [2,3,0.5]

# Second Layer of Neurons
weights2 = [
    [0.5, 0.9, -0.14,.5],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27,0.17, 0.87]
]
biases2 = [-1, 2, -0.5]

# First layer outpust
layer1_output = np.dot(inputs, np.array(weights1).T) + biases1

# Second Layer output
layer2_output = np.dot(inputs, np.array(weights2).T) + biases2

print(layer2_output)
'''

#Notes: 
# We use gpus because they have more cores and designed for simple calculations like dot product
# Batches can be good because they help us generalize the data, by not training all data by once
# Transpose Swaps rows and columns
# X is the genric symbol for input
# Good to initalize weights between -1 and 1 