import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import sys
import os
from random import randrange
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from deep_nn import forward_propagation

with open('model_parameters.pkl', 'rb') as f:
    parameters = pickle.load(f)

current_index = randrange(0, 300)

X = np.load("datasets/signdigits/X.npy")
Y = np.load("datasets/signdigits/Y.npy")

_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)


img_array = X_test[current_index]
img = img_array.reshape(X.shape[1], X.shape[2])

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters, "multivariable")
    predictions = np.argmax(AL, axis=0)  
    return predictions

test_example = X_test[current_index].reshape(-1, 1)

# Predict the label for the selected test example
predicted_label = predict(test_example, parameters)

# Visualize the image and prediction
plt.imshow(img, cmap='gray')
plt.title(f"This image is a {predicted_label[0]}")
plt.show()



