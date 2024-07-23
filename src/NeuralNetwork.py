import numpy as np

def initialize_params(n_x, n_h, n_y):
    # initialize weights and biases randomly using a normal distribution
    W1 = np.random.randn(n_h, n_x) * 0.1
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.1
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

    return parameters

def sigmoid(x):
    # sigmoid activation function
    sigmoid_activation = 1 / (1 + np.exp(-x))

    return sigmoid_activation

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    assert(A2.shape == (1, X.shape[1]))

    return A2, cache

def cost_function(A2, Y):
    # number of examples
    m = Y.shape[1] 

    logprob = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = -1/m * np.sum(logprob) 

    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply((W2.T * dZ2), 1 - np.power(A1, 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, 
             "dW2": dW2, 
             "db1": db1, 
             "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters

def nn_model(X, Y, n_h, learning_rate):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    num_iterations = 10000
    parameters = initialize_params(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = cost_function(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))


    return parameters