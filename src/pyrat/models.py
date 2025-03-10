import sys
import time
import inspect
import random
from typing import List, Optional, Dict, Tuple, Any

import numpy as np

from .optimizers import OPTIMIZERS
from .functions import LOSS_FUNCTIONS
from .layers import Layer

class Model:
    """
    A neural network model class that encapsulates layers, loss function, and optimizer configuration.

    Parameters
    ----------
    layers : list of Layer, optional
        A list of Layer objects to be added to the model.
    loss_fn : str, default="cross-entropy"
        The name of the loss function to be used (must be a valid key in 'LOSS_FUNCTIONS').
    optimizer : str, default="rprop"
        The name of the optimizer to be used (must be a valid key in 'OPTIMIZERS').
    opt_params : dict of {str: float}, optional
        A dictionary of hyperparameters for the specified optimizer. Only parameters
        that match the optimizer's constructor signature will be used.
    """

    def __init__(self,
                 layers: Optional[List[Layer]] = None,
                 loss_fn: str = "cross-entropy",
                 optimizer: str = "rprop",
                 opt_params: Optional[Dict[str, float]] = None
                 ) -> None:
        """
        Constructor for the Model class. Initializes layers, loss function, and optimizer.
        """
        # Initialize and store model layers
        self._init_layers(layers)
        
        # Initialize and store the chosen loss function
        self._init_loss_fn(loss_fn)
        
        # Initialize and store the chosen optimizer (with hyperparameters)
        self._init_optimizer(optimizer, opt_params)

    def _init_layers(self, layers: Optional[List[Layer]]) -> None:
        """
        Initialize layers for the model.

        Parameters
        ----------
        layers : list of Layer or None
            List of layers to add. If None, no layers are added at initialization.
        """
        self.layers: List[Layer] = []
        if layers:
            for layer in layers:
                self.add(layer)

    def _init_loss_fn(self, loss_fn: str) -> None:
        """
        Configure the chosen loss function and its derivative.

        Parameters
        ----------
        loss_fn : str
            Name of the loss function to use. Must be a valid key in 'LOSS_FUNCTIONS'.

        Raises
        ------
        ValueError
            If the specified 'loss_fn' does not exist in 'LOSS_FUNCTIONS'.
        """
        # Confirm that the requested loss function exists in the lookup table
        if loss_fn not in LOSS_FUNCTIONS:
            raise ValueError(f"'{loss_fn}' is not a valid score function.")
        
        # LOSS_FUNCTIONS[loss_fn] returns a tuple: ((loss_func, loss_derivative), name)
        # Store both the function and its derivative for later usage
        (self.loss_function, self.loss_derivative), self.loss_function_name = LOSS_FUNCTIONS[loss_fn]

    def _init_optimizer(self, optimizer: str, opt_params: Optional[Dict[str, float]] = None) -> None:
        """
        Set the optimizer based on its name and provided parameters.

        Parameters
        ----------
        optimizer : str
            Name of the optimizer (must be a valid key in 'OPTIMIZERS').
        opt_params : dict of {str: float} or None
            Dictionary of hyperparameters for the optimizer. Only valid parameters
            in the constructor signature of the optimizer will be used.

        Raises
        ------
        ValueError
            If the specified 'optimizer' does not exist in 'OPTIMIZERS'.
        """
        # Look up the optimizer class by name
        opt_class = OPTIMIZERS.get(optimizer)
        if opt_class is None:
            raise ValueError(f"'{optimizer}' is not a valid optimizer.")
        
        opt_params = opt_params or {}
        
        # Extract only the parameters that match the optimizer's constructor signature
        valid_params = inspect.signature(opt_class.__init__).parameters
        valid_kwargs = {k: v for k, v in opt_params.items() if k in valid_params}
        
        # Instantiate the optimizer with the valid parameters
        self.optimizer = opt_class(**valid_kwargs)

    def add(self, layer: Layer) -> None:
        """
        Add a new Layer to the Model.

        Parameters
        ----------
        layer : Layer
            A layer instance to be appended to the model.

        Raises
        ------
        TypeError
            If the object to be added is not an instance of Layer.
        """
        # Ensure that the layer is indeed a Layer object
        if not isinstance(layer, Layer):
            raise TypeError("Can't add a non-Layer object to the model.")
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass on a batch of inputs through all layers.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            The output after passing through each layer.
        """
        # Sequentially pass data through each layer's forward method
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Perform backpropagation on the gradient of the loss.

        Parameters
        ----------
        loss_grad : np.ndarray
            The gradient of the loss function with respect to the output
            of the final layer.

        Notes
        -----
        This method calls each layer's backward method in reverse order
        (i.e., starting from the last layer).
        """
        # Iterate backwards through layers, passing gradients from the final layer
        # all the way back to the first layer
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, self.optimizer)

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the loss between predictions and true labels.

        Parameters
        ----------
        y_pred : np.ndarray
            Model predictions.
        y_true : np.ndarray
            Ground truth labels.

        Returns
        -------
        float
            The computed loss value.
        """
        # Use the currently stored loss function to compute a scalar loss
        return self.loss_function(y_pred, y_true)

    def compute_loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the loss with respect to predictions.

        Parameters
        ----------
        y_pred : np.ndarray
            Model predictions.
        y_true : np.ndarray
            True labels.

        Returns
        -------
        np.ndarray
            The gradient of the loss with respect to the model predictions.
        """
        # Return the derivative of the loss function for backprop
        return self.loss_derivative(y_pred, y_true)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 25,
            batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            shuffle: bool = True,
            verbose: int = 1,
            patience: Optional[int] = None,
            min_delta: float = 1e-5,
            random_state: Optional[int] = None
            ) -> Dict[str, Any]:
        """
        Train the model on the given dataset for a specified number of epochs.

        Parameters
        ----------
        X : np.ndarray
            Training input data.
        y : np.ndarray
            Training labels.
        epochs : int, optional
            Number of training epochs. Default is 25.
        batch_size : int, optional
            Number of samples in each mini-batch. Default is 32.
        validation_data : tuple of (np.ndarray, np.ndarray]), optional
            A tuple (X_val, y_val) for validation. If provided, validation loss
            (and accuracy, if applicable) is computed after each epoch.
        shuffle : bool, optional
            Whether to shuffle the training data at the beginning of each epoch. Default is True.
        verbose : int, optional
            Verbosity level. 0 for no output, 1 for progress bar output,
            and 2 for per-epoch output. Default is 1.
        patience : int, optional
            Number of epochs to wait for an improvement before stopping early.
            Default is the number of epochs (no early stopping).
        min_delta : float, optional
            Minimum improvement in loss to reset the patience counter. Default is 1e-5.
        random_state : int, optional
            Seed for the random number generator for reproducibility. Default is None.

        Returns
        -------
        dict
            A dictionary containing:
            - "loss": List of training loss values per epoch.
            - "val_loss": List of validation loss values per epoch (empty if no validation data).
            - "accuracy": List of training accuracy values per epoch.
            - "val_accuracy": List of validation accuracy values per epoch (empty if no validation data).
        """
        num_samples = X.shape[0]
        best_loss = float("inf")
        wait = 0

        loss_history: List[float] = []
        val_loss_history: List[float] = []
        accuracy_history: List[float] = []
        val_accuracy_history: List[float] = []

        n_batches = (num_samples + batch_size - 1) // batch_size
        patience = patience if patience is not None else epochs

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        for epoch in range(epochs):
            if shuffle:
                perm = np.random.permutation(num_samples)
                X_epoch = X[perm]
                y_epoch = y[perm]
            else:
                X_epoch, y_epoch = X, y

            epoch_loss_sum = 0.0
            total_samples = 0
            correct_preds = 0

            start_time = time.time()

            if verbose == 2:
                print(f"Epoch {epoch+1}/{epochs}")

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                X_batch = X_epoch[start_idx:end_idx]
                y_batch = y_epoch[start_idx:end_idx]

                y_pred = self.forward(X_batch)
                batch_loss = self.compute_loss(y_pred, y_batch)
                epoch_loss_sum += batch_loss * (end_idx - start_idx)
                total_samples += (end_idx - start_idx)

                pred_classes = np.argmax(y_pred, axis=1)
                true_classes = np.argmax(y_batch, axis=1)
                correct_preds += np.sum(pred_classes == true_classes)

                loss_grad = self.compute_loss_derivative(y_pred, y_batch)
                self.backward(loss_grad)

                if verbose == 1:
                    done = batch_idx + 1
                    percent = int(30 * done / n_batches)
                    bar = f"[{'=' * percent}{'.' * (30 - percent)}]"
                    sys.stdout.write(
                        f"\rEpoch {epoch+1}/{epochs} {done}/{n_batches} {bar} - loss: {batch_loss:.4f}"
                    )
                    sys.stdout.flush()

            elapsed_us = (time.time() - start_time) * 1e6
            avg_loss = epoch_loss_sum / total_samples
            train_accuracy = correct_preds / total_samples

            loss_history.append(avg_loss)
            accuracy_history.append(train_accuracy)

            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val_pred, y_val)
                val_loss_history.append(val_loss)
                val_acc = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
                val_accuracy_history.append(val_acc)
            else:
                val_loss = None

            if verbose == 1:
                sys.stdout.write("\n")
                if validation_data is not None:
                    print(f"Epoch {epoch+1}/{epochs} - {int(elapsed_us)}us - "
                        f"loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f} - "
                        f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - {int(elapsed_us)}us - "
                        f"loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f}")
            elif verbose == 2:
                if validation_data is not None:
                    print(f"{int(elapsed_us)}us - loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f} - "
                        f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                else:
                    print(f"{int(elapsed_us)}us - loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f}")

            current_loss = val_loss if val_loss is not None else avg_loss

            if current_loss + min_delta < best_loss:
                best_loss = current_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1} due to "
                            f"no improvement > {min_delta} for {patience} epochs.")
                    break

        return {
            "loss": loss_history,
            "val_loss": val_loss_history,
            "accuracy": accuracy_history,
            "val_accuracy": val_accuracy_history
        }


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Produce predictions for given input data by performing a forward pass.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            The model outputs or predictions.
        """
        # Forward pass through the model to get predictions
        return self.forward(X)
