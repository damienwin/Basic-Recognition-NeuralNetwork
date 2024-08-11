import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.deep_nn import deep_nn_model, forward_propagation

# Parameters
n_samples = 2000  # Number of samples
n_features = 12288    # Number of features (x and y)
n_clusters = 10    # Number of clusters

# Generate synthetic dataset with 4 clustered groups
X, Y_unflat = make_blobs(n_samples=n_samples, 
                   n_features=n_features, 
                   centers=n_clusters, 
                   cluster_std=3, 
                   random_state=100)

def to_categorical(x, num_classes):
    x = np.array(x, dtype="int64")
    input_shape = x.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    x = x.reshape(-1)
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

X = X.T
Y = to_categorical(Y_unflat, num_classes=n_clusters).T


# Define the model
layer_dims = [12288, 64, 20, 10]
parameters, _ = deep_nn_model(X, Y, num_iterations=1000, layer_dims=layer_dims, learning_rate=1, classification_method="multivariable", lambd = 0, keep_prob = 0)

def predict(X, parameters):
    
    AL, _ = forward_propagation(X, parameters)
    predictions = np.argmax(AL, axis=0)
    return predictions

def plot_decision_boundaries(X, Y, parameters, num_classes):
    # Reduce dimensions to 2 for visualization
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.T)  # Transpose X back to its original shape before PCA
    
    # Define the grid for the decision boundary plot
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Flatten the grid to pass into the model
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Use PCA to transform grid points
    grid_points_reduced = pca.inverse_transform(grid_points)
    
    # Get model predictions for the grid points
    predictions = predict(grid_points_reduced.T, parameters)
    predictions = predictions.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, predictions, cmap=plt.get_cmap('viridis', num_classes), alpha=0.8)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y_unflat.flatten(), cmap=plt.get_cmap('viridis', num_classes), edgecolor='k', marker='o')
    plt.colorbar(ticks=range(num_classes))
    plt.title('Decision Boundaries')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Plot decision boundaries
plot_decision_boundaries(X, Y, parameters, num_classes=n_clusters)
