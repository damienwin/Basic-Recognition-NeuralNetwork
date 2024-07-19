import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.NeuralNetwork import forward_propagation, nn_model

class TestNN(unittest.TestCase):
    def test_xor(self):
        # Define the XOR dataset
        X = np.array([[0, 0, 1, 1],
                      [0, 1, 0, 1]])
        Y = np.array([[0, 1, 1, 0]])

        # Define layer sizes
        n_x = X.shape[0]
        n_h = 4
        n_y = Y.shape[0]

        # Train the neural network
        parameters = nn_model(X, Y, n_x, n_h, n_y, learning_rate=.1)

        # Make predictions on the XOR dataset
        A2 = forward_propagation(X, parameters)[0]
        predictions = (A2 > 0.5).astype(int)

        # Assert that predictions match the true labels
        np.testing.assert_array_equal(predictions, Y)

if __name__ == "__main__":
    unittest.main()