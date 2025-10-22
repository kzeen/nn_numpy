import numpy as np

# Activation function
def sigmoid(x):
    # f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Weight the inputs, add the bias, then pass to activation function
        weighted_sum = (self.weights @ inputs) + self.bias
        return sigmoid(weighted_sum)
    
if __name__ == "__main__":
    # Sample run with random values
    weights = np.array([0, 1])
    bias = 4

    neuron = Neuron(weights, bias)

    x = np.array([2, 3])
    y = neuron.feedforward(x)

    print(y)