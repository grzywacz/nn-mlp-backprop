import numpy as np


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, d_output):
        return d_output * self.output * (1 - self.output)


class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, d_output):
        return d_output * (1 - self.output**2)


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, d_output):
        return d_output * self.output * (1 - self.output)


class FullyConnected:
    def __init__(self, input_neurons, output_neurons):
        # Xavier init
        self.weights = np.random.randn(input_neurons, output_neurons) * np.sqrt(2 / (input_neurons + output_neurons))
        self.biases = np.zeros((1, output_neurons))
        self.prev_d_weights = None
        self.prev_d_biases = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, d_output):
        d_input = np.dot(d_output, self.weights.T)
        self.d_weights = np.dot(self.input.T, d_output)
        self.d_biases = np.sum(d_output, axis=0, keepdims=True)
        return d_input


class CrossEntropyLoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(-np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

    def backward(self):
        return (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) + 1e-15)


class MLP:
    def __init__(self, input_neurons, hidden_layers, output_neurons, activation="tanh"):
        self.layers = []
        self.activation = activation

        if hidden_layers:
            # Create hidden layers
            for i, layer_size in enumerate(hidden_layers):
                if i == 0:
                    input_size = input_neurons
                else:
                    input_size = hidden_layers[i - 1]

                self.layers.append(FullyConnected(input_size, layer_size))
                if self.activation == "tanh":
                    self.layers.append(Tanh())
                elif self.activation == "sigmoid":
                    self.layers.append(Sigmoid())

            # Create output layer
            self.layers.append(FullyConnected(hidden_layers[-1], output_neurons))
        else:
            # Create output layer directly connecting input and output neurons
            self.layers.append(FullyConnected(input_neurons, output_neurons))

        # Add Softmax layer
        self.layers.append(Softmax())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_output):
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)
        return d_output
