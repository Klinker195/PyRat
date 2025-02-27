from abc import ABC, abstractmethod
from . import functions as rat_func
from . import optimizers as rat_opt
import numpy as np

class Layer(ABC):
    @abstractmethod
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, grad_output):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
    
class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation_fn='relu', weight_init_fn='xavier'):
        super().__init__(input_size, output_size)
        
        self.weights = self.__init_weights(weight_init_fn)
        self.bias = np.zeros((1, output_size))
        
        self.__init_activation_fn(activation_fn)
        
    def __init_weights(self, weight_init_fn):
        if weight_init_fn not in rat_func.WEIGHT_INIT_FUNCTIONS:
            raise ValueError(f"'{weight_init_fn}' is not a valid weight init function.")
        else:
            self.weight_init_repr = weight_init_fn
            return rat_func.WEIGHT_INIT_FUNCTIONS[weight_init_fn]((self.input_size, self.output_size))
        
    def __init_activation_fn(self, activation_fn):
        if activation_fn not in rat_func.ACTIVATION_FUNCTIONS:
            raise ValueError(f"'{activation_fn}' is not a valid activation function.")
        else:
            self.activation_fn_repr = activation_fn
            self.activation, self.activation_derivative = rat_func.ACTIVATION_FUNCTIONS[activation_fn]

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.weights) + self.bias
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, grad_output, optimizer):
        grad_activation = grad_output * self.activation_derivative(self.z)
        
        grad_w = np.dot(self.x.T, grad_activation)
        grad_b = np.sum(grad_activation, axis=0, keepdims=True)

        self.weights, self.bias = optimizer.update(self.weights, self.bias, grad_w, grad_b)
        
        return np.dot(grad_activation, self.weights.T)
    
    def __repr__(self):
        return f"DenseLayer(input_size={self.input_size}, output_size={self.output_size}, activation_fn='{self.activation_fn_repr}')"
