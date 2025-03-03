import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


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

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy_loss(y_pred, y_true):
    eps = 1e-15
    predictions = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(predictions), axis=1))

def cross_entropy_loss_derivative(y_pred, y_true):
    return y_pred - y_true

LOSS_FUNCTIONS = {
    "mse": [(mse_loss, mse_loss_derivative), "MSE"],
    "cross-entropy": [(cross_entropy_loss, cross_entropy_loss_derivative), "Cross-Entropy"]
}

def accuracy_scoring(model, X_val, y_val):
    predictions = model.predict(X_val)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_val, axis=1)
    return np.mean(predicted_classes == true_classes)

def cross_entropy_scoring(model, X_val, y_val):
    predictions = model.predict(X_val)
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)
    loss = -np.mean(np.sum(y_val * np.log(predictions), axis=1))
    return loss

SCORING_FUNCTIONS = {
    "accuracy": {"fn": accuracy_scoring, "compare": lambda current, best: current > best, "best_init": -float('inf')},
    "cross-entropy": {"fn": cross_entropy_scoring, "compare": lambda current, best: current < best, "best_init": float('inf')},
}