from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Type

class Optimizer(ABC):
    """
    Abstract base class for optimizers. 
    Every optimizer must implement an __init__ and an update method.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize any internal variables or state for the optimizer."""
        pass

    @abstractmethod
    def update(
        self, 
        weights: np.ndarray, 
        bias: np.ndarray, 
        grad_w: np.ndarray, 
        grad_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply an update step to the weights and biases based on their gradients.
        """
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
        # These hyperparameters control how aggressively the step sizes grow or shrink
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.delta_min = delta_min
        self.delta_max = delta_max

        # Dictionary holding per-parameter data structures:
        # step sizes (delta_w, delta_b) and the previous gradients (prev_grad_w, prev_grad_b).
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
        the current gradient compared to the previous iteration:
        
        - If grad(current) * grad(previous) > 0:
          the step size is multiplied by 'eta_plus'.
        - If grad(current) * grad(previous) < 0:
          the step size is multiplied by 'eta_minus'.
        - The step size is then clipped to the [delta_min, delta_max] range.
        
        Additionally, if the sign has changed, the current gradient is 
        effectively set to zero to avoid an immediate direction reversal.

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
        # Create a unique key for this set of weights/biases 
        # so that each parameter group has independent state.
        key = (id(weights), id(bias))

        # If we've never seen these particular parameters before,
        # initialize their step sizes and previous gradients.
        if key not in self.state:
            self.state[key] = {}
            # Start with a small default step size for every element
            self.state[key]["delta_w"] = np.full_like(grad_w, 0.01)
            self.state[key]["delta_b"] = np.full_like(grad_b, 0.01)
            # Store previous gradients (initially zero)
            self.state[key]["prev_grad_w"] = np.zeros_like(grad_w)
            self.state[key]["prev_grad_b"] = np.zeros_like(grad_b)

        # Retrieve stored step sizes and previous gradients
        delta_w = self.state[key]["delta_w"]
        delta_b = self.state[key]["delta_b"]
        prev_grad_w = self.state[key]["prev_grad_w"]
        prev_grad_b = self.state[key]["prev_grad_b"]

        # The elementwise product of prev_grad and current grad indicates sign consistency
        grad_sign_w = prev_grad_w * grad_w
        grad_sign_b = prev_grad_b * grad_b

        # Create boolean masks to identify where the gradient sign is consistent (+) or flipped (−)
        mask_plus_w = (grad_sign_w > 0)
        mask_minus_w = (grad_sign_w < 0)
        mask_plus_b = (grad_sign_b > 0)
        mask_minus_b = (grad_sign_b < 0)

        # Update step sizes (for W)
        # If sign is consistent (positive product), multiply the step size by eta_plus
        # but ensure we do not exceed delta_max when scaling up.
        safe_limit_plus_w = self.delta_max / max(self.eta_plus, 1e-16)
        delta_w[mask_plus_w] = np.minimum(delta_w[mask_plus_w], safe_limit_plus_w)
        delta_w[mask_plus_w] *= self.eta_plus

        # If sign is flipped (negative product), multiply the step size by eta_minus
        # again ensuring we don't exceed the delta_max after scaling.
        safe_limit_minus_w = self.delta_max / max(self.eta_minus, 1e-16)
        delta_w[mask_minus_w] = np.minimum(delta_w[mask_minus_w], safe_limit_minus_w)
        delta_w[mask_minus_w] *= self.eta_minus

        # Clip all deltas between delta_min and delta_max to avoid instability
        delta_w = np.clip(delta_w, self.delta_min, self.delta_max)

        # Update step sizes (for b)
        safe_limit_plus_b = self.delta_max / max(self.eta_plus, 1e-16)
        delta_b[mask_plus_b] = np.minimum(delta_b[mask_plus_b], safe_limit_plus_b)
        delta_b[mask_plus_b] *= self.eta_plus

        safe_limit_minus_b = self.delta_max / max(self.eta_minus, 1e-16)
        delta_b[mask_minus_b] = np.minimum(delta_b[mask_minus_b], safe_limit_minus_b)
        delta_b[mask_minus_b] *= self.eta_minus

        # As above, clip to the allowed range
        delta_b = np.clip(delta_b, self.delta_min, self.delta_max)

        # Apply weight and bias updates
        # For the elements with consistent sign, update using the current grad's sign
        weights[mask_plus_w] -= delta_w[mask_plus_w] * np.sign(grad_w[mask_plus_w])
        # For the elements with flipped sign, rely on the sign of the *previous* gradient
        weights[mask_minus_w] -= delta_w[mask_minus_w] * np.sign(prev_grad_w[mask_minus_w])

        bias[mask_plus_b] -= delta_b[mask_plus_b] * np.sign(grad_b[mask_plus_b])
        bias[mask_minus_b] -= delta_b[mask_minus_b] * np.sign(prev_grad_b[mask_minus_b])

        # Reset the current gradient to zero where the sign flipped,
        # to prevent a future immediate jump back in the opposite direction.
        grad_w[mask_minus_w] = 0
        grad_b[mask_minus_b] = 0

        # Finally, store these current gradients for the next iteration's sign check
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
        # Fundamental hyperparameters for the Adam algorithm
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Dictionary to store the moments for each parameter set
        self.state: Dict = {}
        # Global step can help with scheduling or diagnosing convergence
        self.global_step = 0

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
        # Increment global step on every call, so we know how many updates have been done
        self.global_step += 1
        # Create a unique key for this specific set of parameters
        key = (id(weights), id(bias))

        # If these parameters haven't been seen yet, initialize their state
        if key not in self.state:
            self.state[key] = {
                "m_w": np.zeros_like(grad_w),  # First moment for weights
                "v_w": np.zeros_like(grad_w),  # Second moment for weights
                "m_b": np.zeros_like(grad_b),  # First moment for bias
                "v_b": np.zeros_like(grad_b),  # Second moment for bias
                "t_local": 0                   # Local timestep for these parameters
            }

        # Increase the local timestep for these weights and biases
        self.state[key]["t_local"] += 1
        t_local = self.state[key]["t_local"]

        # Retrieve the moment estimates from the state dictionary
        m_w = self.state[key]["m_w"]
        v_w = self.state[key]["v_w"]
        m_b = self.state[key]["m_b"]
        v_b = self.state[key]["v_b"]

        # Update the first moments
        m_w[:] = self.beta1 * m_w + (1 - self.beta1) * grad_w
        m_b[:] = self.beta1 * m_b + (1 - self.beta1) * grad_b
        
        # Update the second moments
        v_w[:] = self.beta2 * v_w + (1 - self.beta2) * (grad_w ** 2)
        v_b[:] = self.beta2 * v_b + (1 - self.beta2) * (grad_b ** 2)

        # Bias correction
        m_w_hat = m_w / (1 - self.beta1 ** t_local)  # Correct first moment
        v_w_hat = v_w / (1 - self.beta2 ** t_local)  # Correct second moment
        m_b_hat = m_b / (1 - self.beta1 ** t_local)  # Correct first moment
        v_b_hat = v_b / (1 - self.beta2 ** t_local)  # Correct second moment

        # Apply the parameter updates
        # Compute the adaptive learning rate using the corrected moments
        weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return weights, bias


# A dictionary of available optimizers for convenient instantiation by name.
OPTIMIZERS: Dict[str, Type[Optimizer]] = {
    "rprop": RPROP,
    "adam": Adam
}
