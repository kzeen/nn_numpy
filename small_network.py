import numpy as np
from neuron import Neuron

class SmallNeuralNetwork:
    '''
    Neural network with:
    - 2 inputs
    - 1 hidden layer with 2 neurons (h1, h2)
    - output layer with 1 neuron (o1)

    Neurons have same weights and bias
    '''
    def __init__(self):
        # Random values
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
    
    def feedforward(self, x):
        h1_output = self.h1.feedforward(x)
        h2_output = self.h2.feedforward(x)

        o1_output = self.o1.feedforward(np.array([h1_output, h2_output]))

        return o1_output
    
if __name__ == "__main__":
    network = SmallNeuralNetwork()

    x = np.array([2, 3])

    print(network.feedforward(x))
    