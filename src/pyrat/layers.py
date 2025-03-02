from abc import ABC, abstractmethod
import numpy as np

from .functions import WEIGHT_INIT_FUNCTIONS, ACTIVATION_FUNCTIONS
from .optimizers import Optimizer


class Layer(ABC):
    """
    Abstract base class defining the interface for a generic neural network layer.
    Each layer must implement forward and backward passes.
    """

    @abstractmethod
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initialize a layer with specified input and output sizes.

        Parameters
        ----------
        input_size : int
            The dimensionality of the input to this layer.
        output_size : int
            The dimensionality of the output from this layer.
        """
        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass through this layer.

        Parameters
        ----------
        x : np.ndarray
            The input data for the forward pass.

        Returns
        -------
        np.ndarray
            The output of the layer.
        """
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray):
        """
        Compute the backward pass, returning the gradient needed by previous layers.

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the input of this layer.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class DenseLayer(Layer):
    """
    Fully connected (dense) layer.

    This layer computes the transformation: z = xW + b,
    followed by an activation function a = f(z).

    Parameters
    ----------
    input_size : int
        The dimensionality of the input to the layer.
    output_size : int
        The dimensionality of the output from the layer.
    activation_fn : str, optional
        The name of the activation function to use. Must be a valid key in
        `ACTIVATION_FUNCTIONS`. Default is 'relu'.
    weight_init_fn : str, optional
        The name of the function used to initialize the weights. Must be a valid key in
        `WEIGHT_INIT_FUNCTIONS`. Default is 'xavier'.

    Notes
    -----
    - Weights are stored in `self.weights` and biases in `self.bias`.
    - Activation function and its derivative are set up by `_init_activation_fn`.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation_fn: str = 'relu',
                 weight_init_fn: str = 'xavier') -> None:
        super().__init__(input_size, output_size)

        # Initialize weights and biases
        self.weights = self._init_weights(weight_init_fn)
        # Bias shape: (1, output_size) so that it can be broadcast during forward pass
        self.bias = np.zeros((1, output_size))

        # Set the activation function and its derivative
        self._init_activation_fn(activation_fn)

    def _init_weights(self, weight_init_fn: str) -> np.ndarray:
        """
        Initialize the layer's weights using a specified weight initialization strategy.

        Parameters
        ----------
        weight_init_fn : str
            The name of the weight initialization function (must be a valid key in
            `WEIGHT_INIT_FUNCTIONS`).

        Returns
        -------
        np.ndarray
            The initialized weight matrix of shape (input_size, output_size).

        Raises
        ------
        ValueError
            If `weight_init_fn` is not a valid key in `WEIGHT_INIT_FUNCTIONS`.
        """
        if weight_init_fn not in WEIGHT_INIT_FUNCTIONS:
            raise ValueError(f"'{weight_init_fn}' is not a valid weight initialization function.")
        # Keep track of the initialization method used for __repr__
        self.weight_init_repr = weight_init_fn

        # The function in WEIGHT_INIT_FUNCTIONS takes a shape (tuple) and returns an ndarray
        return WEIGHT_INIT_FUNCTIONS[weight_init_fn]((self.input_size, self.output_size))

    def _init_activation_fn(self, activation_fn: str) -> None:
        """
        Set up the activation function and its derivative.

        Parameters
        ----------
        activation_fn : str
            The name of the activation function (must be a valid key in `ACTIVATION_FUNCTIONS`).

        Raises
        ------
        ValueError
            If `activation_fn` is not a valid key in `ACTIVATION_FUNCTIONS`.
        """
        if activation_fn not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"'{activation_fn}' is not a valid activation function.")
        # Keep track of the activation method used for __repr__
        self.activation_fn_repr = activation_fn

        # Each activation entry is expected to be a tuple: (activation_function, derivative_function)
        self.activation, self.activation_derivative = ACTIVATION_FUNCTIONS[activation_fn]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass for a batch of inputs.

        Parameters
        ----------
        x : np.ndarray
            Input data of shape.

        Returns
        -------
        np.ndarray
            Activation of z trough activation function.

        Notes
        -----
        The final activation in `self.a` (for potential debugging or analysis).
        """
        # Store the input to use it later in backpropagation
        self.x = x

        # Compute the linear transformation
        self.z = np.dot(x, self.weights) + self.bias

        # Apply the activation function
        self.a = self.activation(self.z)
        return self.a

    def backward(self, grad_output: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        """
        Perform the backward pass (backpropagation) to compute gradients.

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the layer output.
        optimizer : Optimizer
            The optimizer instance that updates weights and biases.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the layer input.
        """
        # Compute gradient of activation function
        grad_activation = grad_output * self.activation_derivative(self.z)

        # Compute gradients for weights and biases
        grad_w = np.dot(self.x.T, grad_activation)
        grad_b = np.sum(grad_activation, axis=0, keepdims=True)

        # Update weights and biases using the optimizer
        self.weights, self.bias = optimizer.update(self.weights, self.bias, grad_w, grad_b)

        # Return gradient to pass back to previous layer
        return np.dot(grad_activation, self.weights.T)

    def __repr__(self) -> str:
        return (f"DenseLayer(input_size={self.input_size}, "
                f"output_size={self.output_size}, "
                f"activation_fn='{self.activation_fn_repr}', "
                f"weight_init='{self.weight_init_repr}')")
