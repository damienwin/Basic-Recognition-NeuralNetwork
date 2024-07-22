import numpy as np

def initialize_layer_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    # Initialize parameters for each layer
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def sigmoid_backward(dA, activation_cache):
    dZ = dA * (1 - np.power(activation_cache, 2))

    return dZ

def relu_backward(dA, activation_cache):
    dZ = np.array(dA, copy=True)
    dZ[activation_cache <= 0] = 0

    return dZ

def forward_linear(W, A, b):
    Z = np.dot(W, A) + b
    cache = (W, A, b)

    return Z, cache

def forward_activation(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = forward_linear(W, A_prev, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = forward_linear(W, A_prev, b)
        A, activation_cache = relu(Z)
    else:
        raise ValueError("Invalid activation function!")

    cache = (linear_cache, activation_cache)

    return A, cache

def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the network

    # Forward propagation for each layer
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A, cache = forward_activation(A_prev, W, b, "relu")
        caches.append(cache)

    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    #AL is final output layer
    AL, cache = forward_activation(A, W, b, "sigmoid")
    caches.append(cache)

    return AL, caches

def cost_function(AL, Y):
    cost = -np.mean(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))
    cost = np.squeeze(cost)

    return cost

def backward_linear(dZ, cache):
    W, A_prev, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dW, dA_prev, db

def backward_activation(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)
    else:
        raise ValueError("Invalid activation function!")


    return dW, dA_prev, db

# GRADED FUNCTION: L_model_backward

def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Back propagation of output layer
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Gradient descent of output layer
    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = backward_activation(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # Store gradient descent for remaining layers
    for l in reversed(range(L-1)):
        dA_prev_temp, dW_temp, db_temp = backward_activation(grads["dA" + str(l+1)], caches[l], "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network

    # Update each parameter by layer  
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
        
    return parameters

