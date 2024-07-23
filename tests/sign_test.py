import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.deep_nn import forward_propagation

# Load the model parameters
with open('model_parameters.pkl', 'rb') as f:
    parameters = pickle.load(f)

# Load the test dataset
x_l = np.load('datasets/signdigits/X.npy')
x_array = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0)
y_array = np.concatenate((np.zeros(205), np.ones(205)), axis=0).reshape(x_array.shape[0], 1)
_, X_test, _, Y_test = train_test_split(x_array, y_array, test_size=0.15, random_state=42)
m_test = X_test.shape[0]
X_test_flat = X_test.reshape(m_test, X_test.shape[1] * X_test.shape[2]).T
Y_test = Y_test.T

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = (AL > 0.5).astype(int)
    return predictions

# Initialize the index for displaying random image 
from random import randrange
current_index = randrange(0, m_test)

img_array = X_test[current_index]
img = img_array.reshape(X_test.shape[1], X_test.shape[2])

plt.imshow(img)
plt.title(f"This image is a {'one' if predict(X_test_flat[:, current_index].reshape(-1, 1), parameters) == 1 else 'zero'} sign")
plt.show()

