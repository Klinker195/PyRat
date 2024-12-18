from abc import ABC, abstractmethod

class Layer(ABC):
    
    @abstractmethod
    def __init__(self, input_size, output_size):
        pass
    
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def backward(self):
        pass
    
class DenseLayer(Layer):
    
    def __init__(self, input_size, output_size):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass