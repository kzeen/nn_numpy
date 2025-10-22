# Neural Network from Scratch in NumPy

This project implements a small fully-connected neural network **from scratch**, using only **NumPy**, no TensorFlow, no PyTorch. The goal is to clearly demonstrate how feedforward, backpropagation, and gradient descent work under the hood.

The code is intentionally kept (too) simple, readable, and educational. It is not meant for anything serious.

## Features

-   Single hidden-layer NN
-   Sigmoid activation function
-   MSE loss
-   Manual backpropagation with gradient descent
-   Neuron-by-neuron implementation for clarity
-   NumPy-only, no ML frameworks

## Network Architecture

```
Input layer: 2 neurons
Hidden layer: 2 neurons (h1, h2)
Output layer: 1 neuron (o1)
Activation: Sigmoid
Loss: MSE
```

Mathematically, each neuron computes:

```
output = sigmoid(w · x + b)
sigmoid(x) = 1 / (1 + e^(-x))
```

During training, gradients are computed manually and weights are updated via:

```
w := w - η * dL/dw
b := b - η * dL/db
```
