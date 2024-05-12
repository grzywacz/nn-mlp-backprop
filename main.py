import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from train import train_mlp


def main(args):
    # Load dataset
    if args.dataset == "wine":
        from sklearn.datasets import load_wine as load_dataset
    elif args.dataset == "breast_cancer":
        from sklearn.datasets import load_breast_cancer as load_dataset

    data = load_dataset()
    X = data.data #features
    y = data.target #labels

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Set the parameters for the MLP
    input_neurons = X_train.shape[1]
    hidden_layers = args.hidden_layers
    output_neurons = np.unique(y_train).size

    # Train the MLP
    mlp_model, error_train, error_test = train_mlp(
        X_train,
        y_train,
        X_test,
        y_test,
        input_neurons,
        hidden_layers,
        output_neurons,
        activation=args.activation,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # Plot train and test errors
    plt.plot(np.arange(1, args.epochs + 1), error_train, marker="o", label="Train")
    plt.plot(np.arange(1, args.epochs + 1), error_test, marker="o", label="Test")
    plt.ylabel("Error")
    plt.xlabel("Epoch")
    plt.grid()
    plt.legend()
    plt.savefig("plot.png", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP on the Wine dataset")
    parser.add_argument(
        "--hidden_layers",
        nargs="*",
        type=int,
        default=[],
        help="List of integers specifying the number of neurons in each hidden layer",
    )
    parser.add_argument(
        "--activation", choices=["tanh", "sigmoid"], default="tanh", help="Activation function for the hidden layers"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for weight updates")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for weight updates")
    parser.add_argument(
        "--dataset",
        choices=["wine", "breast_cancer"],
        default="wine",
        help="Dataset to use for training and evaluation",
    )

    args = parser.parse_args()
    main(args)
