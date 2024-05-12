import random

import numpy as np

from nn import MLP, CrossEntropyLoss
from optim import SGD


def one_hot_encoding(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def train_mlp(
    X,
    y,
    X_test,
    y_test,
    input_neurons,
    hidden_layers,
    output_neurons,
    activation="tanh",
    learning_rate=0.01,
    momentum=0.9,
    batch_size=32,
    epochs=100,
    seed=42,
):
    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)

    # One-hot encode the labels
    y_one_hot = one_hot_encoding(y, output_neurons)

    # Initialize the MLP model, the loss function, and the optimizer
    mlp = MLP(input_neurons, hidden_layers, output_neurons, activation)
    loss_function = CrossEntropyLoss()
    optimizer = SGD(mlp, learning_rate, momentum)

    # Collections to track training and test error
    error_train = []
    error_test = []

    for epoch in range(epochs):
        # Shuffle the dataset
        shuffle_idx = np.random.permutation(len(X))
        X_shuffled = X[shuffle_idx]
        y_shuffled = y_one_hot[shuffle_idx]

        # Initialize variables to compute average accuracy and loss
        avg_accuracy = 0
        avg_loss = 0
        num_batches = 0

        for batch_start in range(0, len(X), batch_size):
            num_batches += 1
            batch_end = batch_start + batch_size
            X_batch = X_shuffled[batch_start:batch_end]
            y_batch = y_shuffled[batch_start:batch_end]

            # Forward pass
            y_pred = mlp.forward(X_batch)

            # Compute the loss
            loss = loss_function.forward(y_pred, y_batch)
            avg_loss += loss

            # Compute the accuracy
            accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1)) / len(y_batch)
            avg_accuracy += accuracy

            # Compute the gradient of the loss w.r.t. the output of the last layer
            d_output = loss_function.backward()

            # Backward pass
            mlp.backward(d_output)

            # Update weights and biases with the optimizer
            optimizer.step()

        avg_accuracy /= num_batches
        avg_loss /= num_batches

        # Evaluate the accuracy on the test set
        y_pred = mlp.forward(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        test_accuracy = np.sum(y_pred_labels == y_test) / len(y_test)

        # Append train and test error
        error_train.append(1 - avg_accuracy)
        error_test.append(1 - test_accuracy)
        
        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}, Accuracy train: {avg_accuracy}, Accuracy test: {test_accuracy}")

    return mlp, error_train, error_test
