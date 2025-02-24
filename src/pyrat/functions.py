import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

ACTIVATION_FUNCTIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "linear": (linear, linear_derivative),
    "softmax": (softmax, softmax_derivative)
}

def xavier_uniform(shape):
    fan_in, fan_out = shape
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

def xavier_normal(shape):
    fan_in, fan_out = shape
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0, std, size=shape)

def he_uniform(shape):
    fan_in, _ = shape
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)

def he_normal(shape):
    fan_in, _ = shape
    std = np.sqrt(2 / fan_in)
    return np.random.normal(0, std, size=shape)

WEIGHT_INIT_FUNCTIONS = {
    "xavier": xavier_normal,
    "he": he_normal,
    "xavier_uniform": xavier_uniform,
    "he_uniform": he_uniform
}