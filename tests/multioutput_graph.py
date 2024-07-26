import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.deep_nn import deep_nn_model, forward_propagation

# Parameters
n_samples = 1000  # Number of samples
n_features = 20    # Number of features (x and y)
n_clusters = 4    # Number of clusters

# Generate synthetic dataset with 4 clustered groups
X, Y = make_blobs(n_samples=n_samples, 
                   n_features=n_features, 
                   centers=n_clusters, 
                   cluster_std=5, 
                   random_state=100)

X = X.T
Y = Y[np.newaxis, :]
print(X.shape)
print(Y.shape)

# Define the model
layer_dims = [20, 10, 5, 4]
parameters, _ = deep_nn_model(X, Y, num_iterations=1000, layer_dims=layer_dims, learning_rate=10, classification_method="multivariable")

def predict(X, parameters):
    """
    Predict the class labels for input data X.
    """
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
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y.flatten(), cmap=plt.get_cmap('viridis', num_classes), edgecolor='k', marker='o')
    plt.colorbar(ticks=range(num_classes))
    plt.title('Decision Boundaries')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Plot decision boundaries
plot_decision_boundaries(X, Y, parameters, num_classes=n_clusters)