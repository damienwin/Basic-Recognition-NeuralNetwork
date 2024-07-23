import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.deep_nn import forward_propagation, deep_nn_model

x_l = np.load('datasets/signdigits/X.npy')
# Load only ones and zeros signed from dataset  
x_array = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0)

# Create output array, first half of X classified as sign zero, second half as sign one
y_array = np.concatenate((np.zeros(205), np.ones(205)), axis=0).reshape(x_array.shape[0], 1)

# Split dataset into some training examples(15%) and shuffle data
X_train, X_test, Y_train, Y_test = train_test_split(x_array, y_array, test_size=0.15, random_state=42)

m_train = X_train.shape[0]
X_train_flat = X_train.reshape(m_train, X_train.shape[1] * X_train.shape[2]).T
Y_train = Y_train.T

m_test = X_test.shape[0]
X_test_flat = X_test.reshape(m_test, X_test.shape[1] * X_test.shape[2]).T
Y_test = Y_test.T

layer_dims = [4096, 20, 10, 5, 1]
parameters, _ = deep_nn_model(X_train_flat, Y_train, num_iterations=2500, layer_dims=layer_dims, learning_rate=.01)

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = (AL > 0.5).astype(int)  # Binary classification with a threshold of 0.5
    return predictions

def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels)

with open('model_parameters.pkl', 'wb') as f:
    pickle.dump(parameters, f)

train_predictions = predict(X_train_flat, parameters)
train_accuracy = calculate_accuracy(train_predictions, Y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

test_predictions = predict(X_test_flat, parameters)
test_accuracy = calculate_accuracy(test_predictions, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

