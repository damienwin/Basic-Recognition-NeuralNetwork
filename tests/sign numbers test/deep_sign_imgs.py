import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import os
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from deep_nn import forward_propagation, deep_nn_model

X = np.load("datasets/signdigits/X.npy")
Y = np.load("datasets/signdigits/Y.npy")

# Split dataset into some training examples(15%) and shuffle data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

m_train = X_train.shape[0]
X_train_flat = X_train.reshape(m_train, -1).T

m_test = X_test.shape[0]
X_test_flat = X_test.reshape(m_test, -1).T

Y_train = Y_train.T
Y_test = Y_test.T

print(X_train_flat.shape)
print(Y_train.shape)
print(X_test_flat.shape)
print(Y_test.shape)

layer_dims = [4096, 256, 64, 10]
parameters, _ = deep_nn_model(X_train_flat, Y_train, num_iterations=4000, layer_dims=layer_dims, learning_rate=6.6, classification_method="multivariable")

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = np.argmax(AL, axis=0)  
    return predictions

def calculate_accuracy(predictions, labels):
    labels = np.argmax(labels, axis=0)
    return np.mean(predictions == labels)

with open('model_parameters.pkl', 'wb') as f:
    pickle.dump(parameters, f)

train_predictions = predict(X_train_flat[:, :100], parameters)
train_accuracy = calculate_accuracy(train_predictions, Y_train[:, :100])
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

test_predictions = predict(X_test_flat, parameters)
test_accuracy = calculate_accuracy(test_predictions, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")