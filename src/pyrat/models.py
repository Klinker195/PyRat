import sys
import time
import inspect
import numpy as np
from . import optimizers as rat_opt
from . import functions as rat_func
from .layers import Layer

class Model:
    def __init__(self, layers=None, loss_fn="cross-entropy", optimizer="rprop", opt_params=None):
        self.__init_layers(layers)
        self.__init_loss_fn(loss_fn)
        self.__init_optimizer(optimizer, opt_params)

    def __init_layers(self, layers):
        self.layers = []
        if layers is not None:
            for layer in layers: self.add(layer)

    def __init_loss_fn(self, loss_fn):
        if loss_fn not in rat_func.LOSS_FUNCTIONS:
            raise ValueError(f"'{loss_fn}' is not a valid loss function.")
        else:
            self.loss_function, self.loss_derivative = rat_func.LOSS_FUNCTIONS[loss_fn][0]
            self.loss_function_name = rat_func.LOSS_FUNCTIONS[loss_fn][1]

    # TODO: Fix optimizer default values for learning rate

    def __init_optimizer(self, optimizer, opt_params=None):
        if optimizer not in rat_opt.OPTIMIZERS:
            raise ValueError(f"'{optimizer}' is not a valid optimizer.")
        else:
            opt_class, default_lr = rat_opt.OPTIMIZERS[optimizer]

            if opt_params is None: opt_params = {}

            sig = inspect.signature(opt_class.__init__)
            valid_params = sig.parameters

            valid_kwargs = {}
            for k, v in opt_params.items():
                if k in valid_params:
                    valid_kwargs[k] = v
                # else:
                #     raise TypeError(f"Optimizer '{optimizer}' does not accept argument '{k}'.")
                
            if 'learning_rate' in valid_params and 'learning_rate' not in valid_kwargs:
                valid_kwargs['learning_rate'] = default_lr

            self.optimizer = opt_class(**valid_kwargs)

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError("Cannot add a non-layer object to model.")
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, self.optimizer)

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)

    def compute_loss_derivative(self, y_pred, y_true):
        return self.loss_derivative(y_pred, y_true)

    def fit(self, X, y, epochs=25, batch_size=32, validation_data=None, shuffle=True, verbose=1,
            patience=None, min_delta=1e-5):
        num_samples = X.shape[0]
        best_loss = float("inf")
        wait = 0

        loss_history = []
        val_loss_history = []
        
        n_batches = (num_samples + batch_size - 1) // batch_size
        patience = patience if patience is not None else epochs

        for epoch in range(epochs):
            if shuffle:
                perm = np.random.permutation(num_samples)
                X = X[perm]
                y = y[perm]

            epoch_loss_sum = 0.0
            total = 0
            correct = 0
            start_time = time.time()

            if verbose == 2:
                print(f"Epoch {epoch+1}/{epochs}")
            
            for batch_i in range(n_batches):
                start_idx = batch_i * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                y_pred = self.forward(X_batch)
                batch_loss = self.compute_loss(y_pred, y_batch)
                epoch_loss_sum += batch_loss * len(X_batch)
                total += len(X_batch)

                pred_classes = np.argmax(y_pred, axis=1)
                true_classes = np.argmax(y_batch, axis=1)
                correct += np.sum(pred_classes == true_classes)

                loss_grad = self.compute_loss_derivative(y_pred, y_batch)
                self.backward(loss_grad)

                if verbose == 1:
                    done = batch_i + 1
                    percent = int(30 * done / n_batches)
                    bar = "[" + "=" * percent + "." * (30 - percent) + "]"
                    sys.stdout.write(
                        f"\rEpoch {epoch+1}/{epochs} "
                        f"{done}/{n_batches} {bar} - loss: {batch_loss:.4f}"
                    )
                    sys.stdout.flush()

            elapsed_us = (time.time() - start_time) * 1e6
            avg_loss = epoch_loss_sum / total
            loss_history.append(avg_loss)
            train_accuracy = correct / total

            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val_pred, y_val)
                val_loss_history.append(val_loss)
                val_accuracy = np.mean(
                    np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1)
                )
            else:
                val_loss = None

            if verbose == 1:
                sys.stdout.write("\n")
                if validation_data is not None:
                    print(
                        f"Epoch {epoch+1}/{epochs} - {int(elapsed_us)}us "
                        f"- loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f} "
                        f"- val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch+1}/{epochs} - {int(elapsed_us)}us "
                        f"- loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f}"
                    )
            elif verbose == 2:
                if validation_data is not None:
                    print(
                        f"{int(elapsed_us)}us - loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f} "
                        f"- val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f}"
                    )
                else:
                    print(
                        f"{int(elapsed_us)}us - loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f}"
                    )

            current_loss = val_loss if val_loss is not None else avg_loss
            if current_loss + min_delta < best_loss:
                best_loss = current_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose != 0:
                        print(
                            f"\nEarly stopping at epoch {epoch+1} due to no improvement > {min_delta} for {patience} epochs."
                        )
                    break

        return {"loss": loss_history, "val_loss": val_loss_history}

    def predict(self, X):
        return self.forward(X)
