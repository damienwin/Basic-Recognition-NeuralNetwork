import numpy as np

def initialize_layer_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    # Initialize parameters for each layer
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def initialize_adam_params(parameters):
    L = len(parameters) // 2 # Number of layers
    v = {} # momentum
    s = {} # RMS
    
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v, s

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)  # To prevent overflow
    exp_Z = np.exp(Z_shifted)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def softmax_backward(dAL, cache):
    Z = cache
    m = Z.shape[1]
    dZ = dAL / m

    return dZ

def forward_linear(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

def forward_activation(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activation_cache = softmax(Z)
    else:
        raise ValueError("Invalid activation function!")

    cache = (linear_cache, activation_cache)

    return A, cache

def forward_propagation(X, parameters, classification_type):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the network
    activation = "sigmoid" if classification_type == "binary" else "softmax"

    # Forward propagation for each layer
    for l in range(1, L):
        A_prev = A
        A, cache = forward_activation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    #AL is final output layer
    AL, cache = forward_activation(A, parameters['W' + str(L)], parameters['b' + str(L)], activation)
    caches.append(cache)

    return AL, caches

def cost_function(AL, Y):
    if classification_type == "binary":
        cost = -np.mean(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))
    elif classification_type == "multivariable":
        epsilon = 1e-8  # Small constant to prevent log(0)
        AL = np.clip(AL, epsilon, 1 - epsilon)  # Clip AL to avoid log(0) and division by zero
        cost = -np.sum(Y * np.log(AL)) / Y.shape[1]  
    else:
        raise ValueError("Invalid classification type!")
    
    return cost

def cost_function_with_regularization(AL, Y, parameters, lambd):
    L = len(parameters) // 2  # number of layers in the neural network
    L2_regularization_cost = 0
    m = Y.shape[1] 
    
    for l in range(1, L + 1):
        W = parameters['W' + str(l)]
        L2_regularization_cost += np.sum(np.square(W))
    
    L2_regularization_cost = 1/m * lambd/2 * L2_regularization_cost
    cross_entropy_cost = cost_function(AL, Y)
    cost = cross_entropy_cost + L2_regularization_cost

    return cost

def backward_linear(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backward_activation(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)
    else:
        raise ValueError("Invalid activation function!")


    return dW, dA_prev, db

def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    activation = "sigmoid" if classification_type == "binary" else "softmax"
    
    if activation == "sigmoid":
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    if activation == "softmax":
        dAL = AL - Y

    # Gradient descent of output layer
    current_cache = caches[L - 1]
    dW_temp, dA_prev_temp, db_temp = backward_activation(dAL, current_cache, activation)
    grads["dW" + str(L)] = dW_temp
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["db" + str(L)] = db_temp

    # Store gradient descent for remaining layers
    for l in reversed(range(L-1)):
        dW_temp, dA_prev_temp, db_temp = backward_activation(grads["dA" + str(l+1)], caches[l], "relu")
        grads["dW" + str(l + 1)] = dW_temp
        grads["dA" + str(l)] = dA_prev_temp  
        grads["db" + str(l + 1)] = db_temp

    return grads

def backward_propagation_with_regularization(AL, Y, caches, lambd):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    activation = "sigmoid" if classification_type == "binary" else "softmax"
    
    if activation == "sigmoid":
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    if activation == "softmax":
        dAL = AL - Y

    # Gradient descent of output layer
    current_cache = caches[L - 1]
    dW_temp, dA_prev_temp, db_temp = backward_activation(dAL, current_cache, activation)
    W_temp = current_cache[0][1] # Take value of WL
    dW_temp += lambd/m * W_temp

    grads["dW" + str(L)] = dW_temp
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["db" + str(L)] = db_temp

    # Store gradient descent for remaining layers
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dW_temp, dA_prev_temp, db_temp = backward_activation(grads["dA" + str(l+1)], caches[l], "relu")

        W_temp = current_cache[0][1] # Take value of Wl
        dW_temp += lambd/m * W_temp

        grads["dW" + str(l + 1)] = dW_temp
        grads["dA" + str(l)] = dA_prev_temp  
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters_with_adam(parameters, grads, v, s, learning_rate):
    L = len(parameters) // 2 
    v_corrected = {}
    s_corrected = {}

    # General parameter settings
    t=2
    beta1=0.9
    beta2=0.999
    epsilon=1e-8

    for l in range(1, L + 1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))
        
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.square(grads["dW" + str(l)])
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.square(grads["db" + str(l)])

        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t))

        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v_corrected["dW" + str(l)] / np.sqrt(s_corrected["dW" + str(l)] + epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v_corrected["db" + str(l)] / np.sqrt(s_corrected["db" + str(l)] + epsilon)

    return parameters, v, s

def deep_nn_model(X, Y, num_iterations, layer_dims, learning_rate, classification_method, lambd):
    costs = []
    parameters = initialize_layer_parameters(layer_dims)
    v, s = initialize_adam_params(parameters)
    global classification_type
    classification_type = classification_method

    for i in range(0, num_iterations):
        AL, caches = forward_propagation(X, parameters, classification_type)
 
        if lambd == 0:
            cost = cost_function(AL, Y)
            grads = backward_propagation(AL, Y, caches)
        else:
            cost = cost_function_with_regularization(AL, Y, parameters, lambd)
            grads = backward_propagation_with_regularization(AL, Y, caches, lambd)
    
        parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, learning_rate)

        if i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs