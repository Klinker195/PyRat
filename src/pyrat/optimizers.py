from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Type

class Optimizer(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(
        self, 
        weights: np.ndarray, 
        bias: np.ndarray, 
        grad_w: np.ndarray, 
        grad_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class RPROP(Optimizer):
    """
    The RPROP (Resilient Backpropagation) optimizer.
    
    RPROP adapts step sizes based on the sign of the gradient. 
    If the sign of the gradient remains the same across steps, 
    the step size is increased by 'eta_plus'. If the sign changes, 
    the step size is decreased by 'eta_minus'. 
    
    Attributes
    ----------
    eta_plus : float
        Factor by which the step size is increased when consecutive 
        gradients have the same sign.
    eta_minus : float
        Factor by which the step size is decreased when consecutive 
        gradients have opposite signs.
    delta_min : float
        Minimum permissible step size.
    delta_max : float
        Maximum permissible step size.
    state : Dict
        A dictionary to store the step sizes ('delta_w', 'delta_b') and 
        previous gradients ('prev_grad_w', 'prev_grad_b') for each 
        (weights, bias) pair, identified by their 'id'.
    """

    def __init__(
        self, 
        eta_plus: float = 1.2, 
        eta_minus: float = 0.5, 
        delta_min: float = 1e-6, 
        delta_max: float = 50
    ) -> None:
        """
        Initializes the RPROP optimizer.

        Parameters
        ----------
        eta_plus : float, optional
            Factor by which the step size is multiplied when 
            gradient signs are the same, by default 1.2
        eta_minus : float, optional
            Factor by which the step size is multiplied when 
            gradient signs are opposite, by default 0.5
        delta_min : float, optional
            Minimum permissible step size, by default 1e-6
        delta_max : float, optional
            Maximum permissible step size, by default 50
        """
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.delta_min = delta_min
        self.delta_max = delta_max

        # Store optimizer-specific states for each (weights, bias) pair.
        self.state: Dict = {}

    def update(
        self, 
        weights: np.ndarray, 
        bias: np.ndarray, 
        grad_w: np.ndarray, 
        grad_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the weights and bias using RPROP.

        RPROP modifies step sizes based on the sign of 
        the gradient compared to the previous iteration:
        
        - If the sign of grad(current) * grad(previous) > 0,
          the step size is multiplied by 'eta_plus'.
        - If the sign of grad(current) * grad(previous) < 0,
          the step size is multiplied by 'eta_minus'.
        - The step size is then clipped to the [delta_min, delta_max] range.
        
        Additionally, if the sign has changed, we set the 
        current gradient to zero to avoid immediate reversal.

        Parameters
        ----------
        weights : np.ndarray
            Current model weights.
        bias : np.ndarray
            Current model biases.
        grad_w : np.ndarray
            Gradient of the loss with respect to weights.
        grad_b : np.ndarray
            Gradient of the loss with respect to bias.

        Returns
        -------
        (np.ndarray, np.ndarray)
            The updated weights and bias.
        """
        # Create a unique key for this set of parameters
        key = (id(weights), id(bias))

        # If first time, initialize the step size (delta) and previous gradient storage
        if key not in self.state:
            self.state[key] = {}
            # Start with a small default step size for each element
            self.state[key]["delta_w"] = np.full_like(grad_w, 0.01)
            self.state[key]["delta_b"] = np.full_like(grad_b, 0.01)
            # Store previous gradients
            self.state[key]["prev_grad_w"] = np.zeros_like(grad_w)
            self.state[key]["prev_grad_b"] = np.zeros_like(grad_b)

        delta_w = self.state[key]["delta_w"]
        delta_b = self.state[key]["delta_b"]
        prev_grad_w = self.state[key]["prev_grad_w"]
        prev_grad_b = self.state[key]["prev_grad_b"]

        # Evaluate sign changes
        grad_sign_w = prev_grad_w * grad_w  # Elementwise product
        grad_sign_b = prev_grad_b * grad_b

        # Masks to identify positive/negative sign changes
        mask_plus_w = (grad_sign_w > 0)
        mask_minus_w = (grad_sign_w < 0)
        mask_plus_b = (grad_sign_b > 0)
        mask_minus_b = (grad_sign_b < 0)

        # Update step sizes (delta) for weights
        # For positive sign: multiply delta by eta_plus, but also ensure 
        # we don't exceed delta_max when scaling up.
        safe_limit_plus_w = self.delta_max / max(self.eta_plus, 1e-16)
        delta_w[mask_plus_w] = np.minimum(delta_w[mask_plus_w], safe_limit_plus_w)
        delta_w[mask_plus_w] *= self.eta_plus

        # For negative sign: multiply delta by eta_minus, again respecting delta_max.
        safe_limit_minus_w = self.delta_max / max(self.eta_minus, 1e-16)
        delta_w[mask_minus_w] = np.minimum(delta_w[mask_minus_w], safe_limit_minus_w)
        delta_w[mask_minus_w] *= self.eta_minus

        # Clip delta values between delta_min and delta_max
        delta_w = np.clip(delta_w, self.delta_min, self.delta_max)

        # Update step sizes (delta) for bias
        safe_limit_plus_b = self.delta_max / max(self.eta_plus, 1e-16)
        delta_b[mask_plus_b] = np.minimum(delta_b[mask_plus_b], safe_limit_plus_b)
        delta_b[mask_plus_b] *= self.eta_plus

        safe_limit_minus_b = self.delta_max / max(self.eta_minus, 1e-16)
        delta_b[mask_minus_b] = np.minimum(delta_b[mask_minus_b], safe_limit_minus_b)
        delta_b[mask_minus_b] *= self.eta_minus

        delta_b = np.clip(delta_b, self.delta_min, self.delta_max)

        # Update weights/bias with newly computed deltas
        # For positive sign: use the sign of current gradient
        weights[mask_plus_w] -= delta_w[mask_plus_w] * np.sign(grad_w[mask_plus_w])
        # For negative sign: use the sign of the previous gradient (to avoid immediate reversal)
        weights[mask_minus_w] -= delta_w[mask_minus_w] * np.sign(prev_grad_w[mask_minus_w])

        bias[mask_plus_b] -= delta_b[mask_plus_b] * np.sign(grad_b[mask_plus_b])
        bias[mask_minus_b] -= delta_b[mask_minus_b] * np.sign(prev_grad_b[mask_minus_b])

        # If sign changed (mask_minus), reset current gradient to 0 
        # to avoid "backtracking" issues in the next iteration
        grad_w[mask_minus_w] = 0
        grad_b[mask_minus_b] = 0

        # Store current gradients for the next iteration
        self.state[key]["prev_grad_w"] = grad_w.copy()
        self.state[key]["prev_grad_b"] = grad_b.copy()

        return weights, bias


class Adam(Optimizer):
    """
    Adam optimizer implementation.
    
    Adam (Adaptive Moment Estimation) maintains running averages 
    of both the gradients ('m') and the squared gradients ('v'). 
    It also applies bias correction to these running averages 
    to account for startup transients.
    
    Attributes
    ----------
    learning_rate : float
        The base learning rate for the optimizer.
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Small constant added for numerical stability.
    state : Dict
        Stores the first ('m_w', 'm_b') and second ('v_w', 'v_b') 
        moment estimates for each (weights, bias) pair.
    global_step : int
        Global step counter (useful if needed externally). 
        Also used to track how many times 'update' is called.
    """

    def __init__(
        self, 
        learning_rate: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        epsilon: float = 1e-8
    ) -> None:
        """
        Initializes the Adam optimizer.

        Parameters
        ----------
        learning_rate : float, optional
            The initial learning rate, by default 0.001
        beta1 : float, optional
            Exponential decay rate for the first moment estimates, by default 0.9
        beta2 : float, optional
            Exponential decay rate for the second moment estimates, by default 0.999
        epsilon : float, optional
            Small constant for numerical stability, by default 1e-8
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Dictionary to store moment estimates for each (weights, bias)
        self.state: Dict = {}
        self.global_step = 0  # Can be used for scheduling if needed

    def update(
        self, 
        weights: np.ndarray, 
        bias: np.ndarray, 
        grad_w: np.ndarray, 
        grad_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the given weights and bias using the Adam algorithm.

        Parameters
        ----------
        weights : np.ndarray
            Current model weights.
        bias : np.ndarray
            Current model biases.
        grad_w : np.ndarray
            Gradient of the loss with respect to weights.
        grad_b : np.ndarray
            Gradient of the loss with respect to bias.

        Returns
        -------
        (np.ndarray, np.ndarray)
            The updated weights and bias.
        """
        self.global_step += 1
        key = (id(weights), id(bias))

        # If first time using these parameters, initialize their state
        if key not in self.state:
            self.state[key] = {
                "m_w": np.zeros_like(grad_w),
                "v_w": np.zeros_like(grad_w),
                "m_b": np.zeros_like(grad_b),
                "v_b": np.zeros_like(grad_b),
                "t_local": 0  # Local timestep for these particular parameters
            }

        self.state[key]["t_local"] += 1
        t_local = self.state[key]["t_local"]

        m_w = self.state[key]["m_w"]
        v_w = self.state[key]["v_w"]
        m_b = self.state[key]["m_b"]
        v_b = self.state[key]["v_b"]

        # First moment estimates (m)
        m_w[:] = self.beta1 * m_w + (1 - self.beta1) * grad_w
        m_b[:] = self.beta1 * m_b + (1 - self.beta1) * grad_b
        
        # Second moment estimates (v)
        v_w[:] = self.beta2 * v_w + (1 - self.beta2) * (grad_w ** 2)
        v_b[:] = self.beta2 * v_b + (1 - self.beta2) * (grad_b ** 2)

        # Bias correction
        m_w_hat = m_w / (1 - self.beta1 ** t_local)
        v_w_hat = v_w / (1 - self.beta2 ** t_local)
        m_b_hat = m_b / (1 - self.beta1 ** t_local)
        v_b_hat = v_b / (1 - self.beta2 ** t_local)

        # Update parameters
        weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return weights, bias


# A dictionary of available optimizers for convenient instantiation by name.
OPTIMIZERS: Dict[str, Type[Optimizer]] = {
    "rprop": RPROP,
    "adam": Adam
}
