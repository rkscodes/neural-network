from time import time
import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_input, n_neurons):
        self.weight = np.random.randn(n_input, n_neurons) * 0.10
        self.biases = np.zeros((1, n_neurons)) * 0.10

    def forward(self, n_input):
        self.output = np.dot(n_input, self.weight) + self.biases


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)
