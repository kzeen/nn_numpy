import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    # This is a nice form of e^(-x) / (1 + e^(-x))^2 
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    '''
    y_true and y_pred: NumPy arrays
    '''
    return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
    '''
    Neural network with:
    - 2 inputs
    - 1 hidden layer with 2 neurons
    - 1 output

    This neural network is extremely simplified and not optimal. Values used are random, instead of coming from a real dataset
    '''
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x (input) is a numpy array with 2 elements
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

        return o1
    
    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n number of samples in the dataset
        - all_y_trues is a numpy array with n elements, corresponding to the data
        '''
        n = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Doing a feedforward, values will be needed
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)

                y_pred = o1

                # Calculate partial derivatives
                # d_L_d_w1 represents "partial L / partial w1"

                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * sigmoid_deriv(sum_o1)
                d_ypred_d_w6 = h2 * sigmoid_deriv(sum_o1)
                d_ypred_d_b3 = sigmoid_deriv(sum_o1)

                d_ypred_d_h1 = self.w5 * sigmoid_deriv(sum_o1)
                d_ypred_d_h2 = self.w6 * sigmoid_deriv(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * sigmoid_deriv(sum_h1)
                d_h1_d_w2 = x[1] * sigmoid_deriv(sum_h1)
                d_h1_d_b1 = sigmoid_deriv(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * sigmoid_deriv(sum_h2)
                d_h2_d_w4 = x[1] * sigmoid_deriv(sum_h2)
                d_h2_d_b2 = sigmoid_deriv(sum_h2)

                # Update weights and biases

                # Neuron h1
                self.w1 -= n * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= n * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= n * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= n * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= n * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= n * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= n * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= n * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= n * d_L_d_ypred * d_ypred_d_b3

                # Get total loss at the end of each epoch
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)

                    print("Epoch %d loss: %.3f" % (epoch, loss))


if __name__ == "__main__":
    # Define dataset
    data = np.array([
        [-2, -1],
        [25, 6],
        [17, 4],
        [-15, -6],
    ])

    all_y_trues = np.array([
        1, 
        0,
        0,
        1,
    ])

    # Train neural network
    network = NeuralNetwork()
    network.train(data, all_y_trues)
