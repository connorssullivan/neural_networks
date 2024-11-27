import sys
import numpy as np
import matplotlib

print("Python ", sys.version)
print('Numpy ', np.__version__)
print('Matplotlib: ', matplotlib.__version__)

# Every neuron has a unique connection to every previous neurons

# Inputs could be an input layer, which is the values you are tracking
inputs = [1,2,3,2.5]

# Values of arcs on node, used to tweak the inputs
weights = [
    [0.2, 0.8, -0.5,1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27,0.17, 0.87]
]

# Value in node
biases = 2

#Get the dot product
output = np.dot(weights, inputs) + biases
print(output)