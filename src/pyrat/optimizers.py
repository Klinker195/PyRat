from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self, weights, bias, grad_w, grad_b):
        pass


class RPROP(Optimizer):
    def __init__(self, eta_plus=1.2, eta_minus=0.5, delta_min=1e-6, delta_max=50):
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.initialized = False

    def update(self, weights, bias, grad_w, grad_b):
        if not self.initialized:
            self.delta_w = np.full(grad_w.shape, 0.01)
            self.delta_b = np.full(grad_b.shape, 0.01)
            self.prev_grad_w = np.zeros_like(grad_w)
            self.prev_grad_b = np.zeros_like(grad_b)
            self.initialized = True

        grad_sign_w = self.prev_grad_w * grad_w
        grad_sign_b = self.prev_grad_b * grad_b
        
        self.delta_w[grad_sign_w > 0] *= self.eta_plus
        self.delta_w[grad_sign_w < 0] *= self.eta_minus
        self.delta_w = np.clip(self.delta_w, self.delta_min, self.delta_max)
        
        self.delta_b[grad_sign_b > 0] *= self.eta_plus
        self.delta_b[grad_sign_b < 0] *= self.eta_minus
        self.delta_b = np.clip(self.delta_b, self.delta_min, self.delta_max)
        
        weights[grad_sign_w > 0] -= self.delta_w[grad_sign_w > 0] * np.sign(grad_w[grad_sign_w > 0])
        weights[grad_sign_w < 0] -= self.delta_w[grad_sign_w < 0] * np.sign(self.prev_grad_w[grad_sign_w < 0])
        
        bias[grad_sign_b > 0] -= self.delta_b[grad_sign_b > 0] * np.sign(grad_b[grad_sign_b > 0])
        bias[grad_sign_b < 0] -= self.delta_b[grad_sign_b < 0] * np.sign(self.prev_grad_b[grad_sign_b < 0])
        
        grad_w[grad_sign_w < 0] = 0
        grad_b[grad_sign_b < 0] = 0
        
        self.prev_grad_w = grad_w.copy()
        self.prev_grad_b = grad_b.copy()

        return weights, bias


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None

        self.t = 0
        self.initialized = False

    def update(self, weights, bias, grad_w, grad_b):
        if not self.initialized:
            self.m_w = np.zeros_like(grad_w)
            self.v_w = np.zeros_like(grad_w)
            self.m_b = np.zeros_like(grad_b)
            self.v_b = np.zeros_like(grad_b)
            self.initialized = True

        self.t += 1

        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w ** 2)

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)

        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return weights, bias
    
# TODO: Try to remove default value for learning rate using __init__ signature to discriminate optimizer

OPTIMIZERS = {
    "rprop": (RPROP, None),
    "adam": (Adam, 0.001)
}
