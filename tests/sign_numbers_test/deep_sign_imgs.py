import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import os
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from deep_nn import forward_propagation, deep_nn_model

preprocessed_data_dir = "datasets/signdigits/preprocessed"
X_train_flat_path = os.path.join(preprocessed_data_dir, "X_train_flat.npy")
Y_train_path = os.path.join(preprocessed_data_dir, "Y_train.npy")
X_test_flat_path = os.path.join(preprocessed_data_dir, "X_test_flat.npy")
Y_test_path = os.path.join(preprocessed_data_dir, "Y_test.npy")

# Check if preprocessed data exists
if all(os.path.exists(path) for path in [X_train_flat_path, Y_train_path, X_test_flat_path, Y_test_path]):
    X_train_flat = np.load(X_train_flat_path)
    Y_train = np.load(Y_train_path)
    X_test_flat = np.load(X_test_flat_path)
    Y_test = np.load(Y_test_path)
    print("Loaded preprocessed data.")
else:
    # Load original dataset
    X = np.load("datasets/signdigits/X.npy")
    Y = np.load("datasets/signdigits/Y.npy")

    # Split dataset into training and test sets (15% test size)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

    # Flatten the data
    m_train = X_train.shape[0]
    X_train_flat = X_train.reshape(m_train, -1).T

    m_test = X_test.shape[0]
    X_test_flat = X_test.reshape(m_test, -1).T

    Y_train = Y_train.T
    Y_test = Y_test.T

    # Save preprocessed data for future use
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)
    np.save(X_train_flat_path, X_train_flat)
    np.save(Y_train_path, Y_train)
    np.save(X_test_flat_path, X_test_flat)
    np.save(Y_test_path, Y_test)
    print("Preprocessed data saved.")

layer_dims = [4096, 512, 64, 10]
classification_method="multivariable"
num_iterations = 10000
learning_rate = 0.01
lambd = 0
keep_prob = 0.6

parameters, _ = deep_nn_model(X_train_flat, Y_train, 
                              num_iterations=num_iterations, 
                              layer_dims=layer_dims, 
                              learning_rate=learning_rate, 
                              classification_method=classification_method, 
                              lambd=lambd,
                              keep_prob=keep_prob)

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters, classification_type="multivariable")
    predictions = np.argmax(AL, axis=0)  
    return predictions

def calculate_accuracy(predictions, labels):
    labels = np.argmax(labels, axis=0)
    return np.mean(predictions == labels)

with open('model_parameters.pkl', 'wb') as f:
    pickle.dump(parameters, f)

train_predictions = predict(X_train_flat, parameters)
train_accuracy = calculate_accuracy(train_predictions, Y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

test_predictions = predict(X_test_flat, parameters)
test_accuracy = calculate_accuracy(test_predictions, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

def log_results(file_path, num_iterations, learning_rate, lambd, keep_prob, train_accuracy, test_accuracy):
    # Append the results to the file
    with open(file_path, 'a') as f:
        f.write(f"{num_iterations:<8} {learning_rate:<8} {lambd:<8} {keep_prob:<8} {train_accuracy * 100:<10.2f} {test_accuracy * 100:<10.2f}\n")
        print("Successfully logged")

log_file_path = "tests/sign_numbers_test/training_log.txt"
log_results(log_file_path, num_iterations, learning_rate, lambd, keep_prob, train_accuracy, test_accuracy)