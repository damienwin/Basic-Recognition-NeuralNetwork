import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.deep_nn import (
    initialize_layer_parameters,
    sigmoid,
    relu,
    relu_backward,
    sigmoid_backward,
    forward_linear,
    forward_activation,
    forward_propagation,
    cost_function,
    backward_linear,
    backward_activation,
    backward_propagation,
    update_parameters,
    deep_nn_model
)

class TestDeepNN(unittest.TestCase):

    def setUp(self):
        self.layer_dims = [5, 4, 3]
        self.X = np.random.randn(5, 10)
        self.Y = np.random.randint(0, 2, (3, 10))

    def test_initialize_layer_parameters(self):
        parameters = initialize_layer_parameters(self.layer_dims)
        self.assertEqual(parameters['W1'].shape, (4, 5))
        self.assertEqual(parameters['b1'].shape, (4, 1))
        self.assertEqual(parameters['W2'].shape, (3, 4))
        self.assertEqual(parameters['b2'].shape, (3, 1))

    def test_sigmoid(self):
        Z = np.array([[0, 2], [-1, -2]])
        A, _ = sigmoid(Z)
        expected_A = 1 / (1 + np.exp(-Z))
        np.testing.assert_array_almost_equal(A, expected_A)

    def test_relu(self):
        Z = np.array([[0, 2], [-1, -2]])
        A, _ = relu(Z)
        expected_A = np.maximum(0, Z)
        np.testing.assert_array_equal(A, expected_A)

    def test_relu_backward(self):
        # Define test inputs
        Z = np.array([[1, -2], [3, -4]])  # Sample input to ReLU
        dA = np.array([[0.1, 0.2], [0.3, 0.4]])  # Gradient coming from the next layer

        # Perform forward pass (ReLU)
        A, cache = relu(Z)

        # Compute backward pass
        dZ = relu_backward(dA, cache)

        # Expected output
        expected_dZ = np.array([[0.1, 0], [0.3, 0]])  # Only positive Z values should have gradients

        # Check if the output is as expected
        self.assertTrue(np.allclose(dZ, expected_dZ), f"Test failed: dZ = {dZ}, expected_dZ = {expected_dZ}")

    def test_sigmoid_backward(self):
        # Define test inputs
        Z = np.array([[0.5, -1.5], [1.0, -2.0]])  # Sample input to sigmoid
        dA = np.array([[0.1, 0.2], [0.3, 0.4]])  # Gradient coming from the next layer

        # Perform forward pass (sigmoid)
        A, cache = sigmoid(Z)

        # Compute backward pass
        dZ = sigmoid_backward(dA, cache)

        # Compute expected output
        s = 1 / (1 + np.exp(-Z))  # Sigmoid output
        expected_dZ = dA * s * (1 - s)  # Gradient of the sigmoid function

        # Check if the output is as expected
        self.assertTrue(np.allclose(dZ, expected_dZ), f"Test failed: dZ = {dZ}, expected_dZ = {expected_dZ}")

    def test_forward_linear(self):
        W = np.random.randn(4, 5)
        A = np.random.randn(5, 10)
        b = np.random.randn(4, 1)
        Z, _ = forward_linear(A, W, b)
        expected_Z = np.dot(W, A) + b
        np.testing.assert_array_almost_equal(Z, expected_Z)

    def test_forward_activation(self):
        W = np.random.randn(4, 5)
        A_prev = np.random.randn(5, 10)
        b = np.random.randn(4, 1)
        A, _ = forward_activation(A_prev, W, b, 'relu')
        Z, _ = forward_linear(A_prev, W, b)
        expected_A = np.maximum(0, Z)
        np.testing.assert_array_almost_equal(A, expected_A)

    def test_cost_function(self):
        AL = np.array([[0.8, 0.9, 0.4]])
        Y = np.array([[1, 0, 1]])
        cost = cost_function(AL, Y)
        expected_cost = -np.mean(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))
        self.assertAlmostEqual(cost, np.squeeze(expected_cost))

    def test_backward_linear(self):
        dZ = np.random.randn(4, 10)
        A_prev = np.random.randn(5, 10)
        W = np.random.randn(4, 5)
        b = np.random.randn(4, 1)
        cache = (A_prev, W, b)
        dA_prev, dW, db = backward_linear(dZ, cache)
        m = A_prev.shape[1]
        expected_dW = 1/m * np.dot(dZ, A_prev.T)
        expected_db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        expected_dA_prev = np.dot(W.T, dZ)
        np.testing.assert_array_almost_equal(dW, expected_dW)
        np.testing.assert_array_almost_equal(db, expected_db)
        np.testing.assert_array_almost_equal(dA_prev, expected_dA_prev)

    def test_backward_activation_relu(self):
        # Define test inputs for ReLU
        Z = np.array([[1, -2], [3, -4]])  # Sample input to ReLU
        dA = np.array([[0.1, 0.2], [0.3, 0.4]])  # Gradient coming from the next layer
        A_prev = np.array([[0.1, 0.2], [0.3, 0.4]])  # Input to the linear layer
        W = np.array([[0.5, 0.6], [0.7, 0.8]])  # Weights
        b = np.array([[0.1], [0.2]])  # Biases

        # Perform forward pass (ReLU)
        A, cache = relu(Z)

        # Compute backward pass using the activation function
        linear_cache = (W, A_prev, b)
        activation_cache = Z
        dW, dA_prev, db = backward_activation(dA, (linear_cache, activation_cache), "relu")

        # Compute expected outputs
        dZ = relu_backward(dA, activation_cache)
        expected_dA_prev, expected_dW, expected_db = backward_linear(dZ, linear_cache)

        # Check if the outputs are as expected
        self.assertTrue(np.allclose(dW, expected_dW), f"Test failed: dW = {dW}, expected_dW = {expected_dW}")
        self.assertTrue(np.allclose(dA_prev, expected_dA_prev), f"Test failed: dA_prev = {dA_prev}, expected_dA_prev = {expected_dA_prev}")
        self.assertTrue(np.allclose(db, expected_db), f"Test failed: db = {db}, expected_db = {expected_db}")

    def test_update_parameters(self):
        parameters = initialize_layer_parameters(self.layer_dims)
        grads = {
            'dW1': np.random.randn(4, 5),
            'db1': np.random.randn(4, 1),
            'dW2': np.random.randn(3, 4),
            'db2': np.random.randn(3, 1),
        }
        learning_rate = 0.01
        updated_parameters = update_parameters(parameters, grads, learning_rate)
        self.assertEqual(parameters['W1'].shape, updated_parameters['W1'].shape)
        self.assertEqual(parameters['b1'].shape, updated_parameters['b1'].shape)

if __name__ == '__main__':
    unittest.main()
