import sys
import numpy as np
import matplotlib

print("Python ", sys.version)
print('Numpy ', np.__version__)
print('Matplotlib: ', matplotlib.__version__)

# Every neuron has a unique connection to every previous neurons

# Inputs could be an input layer, which is the values you are tracking
input = [1,2,3,2.5]

# Values of arcs on node, used to tweak the inputs
weights1 = [0.2, 0.8, -0.5,1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27,0.17, 0.87]

# Value in node
bias1 = 2
bias2 = 3
bias3 = 0.5

# First step for neuron add input + wight + bias
output = [input[0]*weights1[0] + input[1]*weights1[1] + input[2]*weights1[2] + input[3]*weights1[3] + bias1,
          input[0]*weights2[0] + input[1]*weights2[1] + input[2]*weights2[2] + input[3]*weights2[3] + bias2,
          input[0]*weights3[0] + input[1]*weights3[1] + input[2]*weights3[2] + input[3]*weights3[3] + bias3,]
print(f'\n\nOutput: {output}')