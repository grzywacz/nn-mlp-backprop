import numpy as np

from nn import FullyConnected


class SGD:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Initialize previous weight and bias updates for momentum
        for layer in self.model.layers:
            if isinstance(layer, FullyConnected):
                layer.prev_d_weights = np.zeros_like(layer.weights)
                layer.prev_d_biases = np.zeros_like(layer.biases)

    def step(self):
        for layer in self.model.layers:
            if isinstance(layer, FullyConnected):
                # Update weights and biases using momentum
                layer.prev_d_weights = self.momentum * layer.prev_d_weights + self.learning_rate * layer.d_weights
                layer.prev_d_biases = self.momentum * layer.prev_d_biases + self.learning_rate * layer.d_biases
                layer.weights -= layer.prev_d_weights
                layer.biases -= layer.prev_d_biases
