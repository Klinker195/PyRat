import time
import numpy as np
from .layers import Layer

LOSS_FUNCTIONS = {
    "mse": [(
        lambda y_pred, y_true: np.mean((y_pred - y_true) ** 2),
        lambda y_pred, y_true: 2 * (y_pred - y_true) / y_true.shape[0]
    ), "MSE"],
    "cross-entropy": [(
        lambda y_pred, y_true: -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1)),
        lambda y_pred, y_true: y_pred - y_true
    ), "Cross-Entropy"]
}

class Model:
    def __init__(self, layers=None, loss_fn="cross-entropy", optimizer="rprop", metrics=None):
        self.layers = layers if layers is not None else []

        if loss_fn not in LOSS_FUNCTIONS:
            raise ValueError(f"'{loss_fn}' is not a valid loss function.")
        
        self.loss_function, self.loss_derivative = LOSS_FUNCTIONS[loss_fn][0]
        self.loss_function_name = LOSS_FUNCTIONS[loss_fn][1]
        self.optimizer = optimizer
        self.metrics = metrics if metrics is not None else []

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError("Cannot add a non-layer object to model.")
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)

    def compute_loss_derivative(self, y_pred, y_true):
        return self.loss_derivative(y_pred, y_true)

    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def fit(self, X_train, y_train, epochs=25, batch_size=32, validation_data=None, verbose=True,
            patience=None, min_delta=1e-4):
        num_samples = X_train.shape[0]
        best_loss = float("inf")
        wait = 0
        loss_history = []
        val_loss_history = []
        
        patience = patience if patience is not None else epochs
        
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            epoch_loss_sum = 0.0
            total = 0
            correct = 0
            
            start_time = time.time()
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                
                y_pred = self.forward(X_batch)
                batch_loss = self.compute_loss(y_pred, y_batch)
                epoch_loss_sum += batch_loss * len(X_batch)
                total += len(X_batch)
                
                pred_classes = np.argmax(y_pred, axis=1)
                true_classes = np.argmax(y_batch, axis=1)
                correct += np.sum(pred_classes == true_classes)
                
                loss_grad = self.compute_loss_derivative(y_pred, y_batch)
                self.backward(loss_grad)
            
            elapsed = time.time() - start_time
            avg_loss = epoch_loss_sum / total
            loss_history.append(avg_loss)
            train_accuracy = correct / total
            
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val_pred, y_val)
                val_loss_history.append(val_loss)
            else:
                val_loss = None
            
            bar_length = 30
            progress_bar = "[" + "=" * bar_length + "]"
            if validation_data is not None:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"{total}/{num_samples} {progress_bar} - {int(elapsed)}s - loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"{total}/{num_samples} {progress_bar} - {int(elapsed)}s - loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f}")
            
            current_loss = val_loss if val_loss is not None else avg_loss
            if current_loss + min_delta < best_loss:
                best_loss = current_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1} due to no improvement greater than {min_delta} for {patience} consecutive epochs.")
                    break
                    
        if validation_data is not None:
            return {"loss": loss_history, "val_loss": val_loss_history}
        else:
            return loss_history

    def predict(self, X):
        return self.forward(X)
