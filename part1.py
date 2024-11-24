import sys
import numpy as np
import matplotlib

print("Python ", sys.version)
print('Numpy ', np.__version__)
print('Matplotlib: ', matplotlib.__version__)

# Every neuron has a unique connection to every previous neurons


input = [1.2,5.1,2.1]

weights = [3.1, 2.1, 8.7]

bias = 3

# First step for neuron add input + wight + bias
output = input[0]*weights[0] + input[1]*weights[1] + input[2]*weights[2] + bias
print(f'\n\nOutput: {output}')
