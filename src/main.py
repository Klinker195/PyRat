import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pyrat.datasets as rat_datasets

from pyrat.models import Model
from pyrat.layers import DenseLayer
from pyrat.tuning import grid_search_cv

from sklearn.metrics import confusion_matrix

# pyrat - Rapid Artificial Training

if __name__ == "__main__":
    mnist_train_set = rat_datasets.MNIST(train=True)
    mnist_val_set = rat_datasets.MNIST(train=False)

    X_train = mnist_train_set.images
    y_train = mnist_train_set.labels

    X_train = X_train.reshape(X_train.shape[0], -1) # da (60000, 28, 28, 1) a (60000, 784)

    X_val = mnist_val_set.images
    y_val = mnist_val_set.labels
    
    X_val = X_val.reshape(X_val.shape[0], -1) # da (10000, 28, 28, 1) a (10000, 784)

    print("Train set shape:")
    print(str(X_train.shape))
    print(str(y_train.shape))
    print("Test set shape:")
    print(str(X_val.shape))
    print(str(y_val.shape))
    print()

    # Testing
    
    param_grid = {
    "loss_fn": ["cross-entropy"],
    "optimizer": ["rprop"],
    "opt_params": [
        {"eta_plus": 1.2, "eta_minus": 0.5, "delta_min": 1e-6, "delta_max": 50},
        {"eta_plus": 1.5, "eta_minus": 0.4, "delta_min": 1e-6, "delta_max": 100},
        {"eta_plus": 1.1, "eta_minus": 0.7, "delta_min": 1e-7, "delta_max": 25},
        {"eta_plus": 1.3, "eta_minus": 0.3, "delta_min": 1e-6, "delta_max": 75}
        ],
    "layers_config": [
        [DenseLayer(input_size=784, output_size=10, activation_fn='softmax')]
        ],
    "epochs": [100],
    "batch_size": [60000],
    }
    
    results = grid_search_cv(model_class=Model, param_grid=param_grid, X=X_train, y=y_train, cv=3, shuffle=True, random_state=42, verbose=1)
    
    print("Best Score:", results["best_score"])
    print("Best Params:", results["best_params"])

    print("Detailed results:")
    for combo, mean_score, std_score in results["results"]:
        print(combo, "->", f"{mean_score:.3f} ± {std_score:.3f}")
    
    # model = Model(loss_fn='cross-entropy', optimizer='rprop', opt_params=None) # default: opt='rprop' loss_fn='cross-entropy'
    # model.add(DenseLayer(input_size=784, output_size=10, activation_fn='softmax'))
    # loss_history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), patience=5, shuffle=True)
    
    model = results["best_model"]
    loss_history = results["best_loss_history"]
    
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss evolution during training')
    plt.legend()
    plt.show()
    
    y_pred = model.predict(X_val) # shape: (10000, 10)

    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_val, axis=1)

    accuracy = np.mean(pred_classes == true_classes)
    print()
    print("Test Accuracy:", accuracy)
    print()
    print("Predictions:")
    print(pred_classes[:10])
    print("Ref values:")
    print(true_classes[:10])

    indices = np.random.choice(X_val.shape[0], 15, replace=False)
    sample_images = X_val[indices]
    sample_true = true_classes[indices]
    sample_pred = pred_classes[indices]

    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(indices):
        plt.subplot(3, 5, i+1)
        plt.imshow(sample_images[i].reshape(28,28), cmap='gray')
        plt.title(f"Pred: {sample_pred[i]}\nTrue: {sample_true[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()