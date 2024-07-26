import unittest
import sklearn
import sklearn.datasets as sk
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.deep_nn import forward_propagation, deep_nn_model

# Load crescent-shaped dataset
X, Y = sk.make_moons(n_samples=100, shuffle=True, noise=.1, random_state=None)

# Swap dimensions of array to fit function
X = X.T
Y = Y[np.newaxis, :]

# Train neural network model
layer_dims = [2, 10, 5, 1]
parameters, _ = deep_nn_model(X, Y, num_iterations=1000, layer_dims=layer_dims, learning_rate=.33, classification_method="binary")

def predict(X, parameters):
    A2, _ = forward_propagation(X.T, parameters)
    return (A2 > 0.5).astype(int)

def plot_decision_boundary(X, Y, parameters):
    # Create meshgrid
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Flatten the meshgrid to pass it to the model
    Z = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(Z, parameters)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=30, cmap=plt.cm.RdYlBu, edgecolor='k')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision Boundary and Data Points')
    plt.show()

plot_decision_boundary(X, Y, parameters)