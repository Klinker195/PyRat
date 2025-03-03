import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pyrat.datasets as rat_datasets
from pyrat.models import Model
from pyrat.layers import DenseLayer
from pyrat.tuning import grid_search_cv
from pyrat.tuning import random_search_cv

from sklearn.metrics import confusion_matrix

# pyrat - Rapid Artificial Training

# TODO: Add comments and docstrings

def model_test(X_train, y_train, validation_data=None):
    model = Model(loss_fn='cross-entropy', optimizer='rprop')

    # model.add(DenseLayer(input_size=784, output_size=10, activation_fn='softmax'))
    
    model.add(DenseLayer(input_size=784, output_size=16, activation_fn='sigmoid'))
    model.add(DenseLayer(input_size=16, output_size=10, activation_fn='softmax'))

    loss_history = model.fit(X_train, y_train, epochs=100, batch_size=60000, validation_data=validation_data, shuffle=True, patience=20)

    return model, loss_history

def grid_search_test(X_train, y_train, validation_data=None):
    param_grid = {
        "loss_fn": ["cross-entropy"],
        "optimizer": ["adam", "rprop"],
        "opt_params": [
            {"eta_plus": 1.2, "eta_minus": 0.5, "delta_min": 1e-6, "delta_max": 50},
            {"eta_plus": 1.5, "eta_minus": 0.4, "delta_min": 1e-6, "delta_max": 100},
            {"eta_plus": 1.1, "eta_minus": 0.7, "delta_min": 1e-7, "delta_max": 25},
            {"eta_plus": 1.3, "eta_minus": 0.3, "delta_min": 1e-6, "delta_max": 75},
            {"learning_rate": 0.001},
            {"learning_rate": 0.0001}
        ],
        "layers_config": [
            [DenseLayer(input_size=784, output_size=16, activation_fn='sigmoid'), DenseLayer(input_size=16, output_size=10, activation_fn='softmax')],
            [DenseLayer(input_size=784, output_size=10, activation_fn='softmax')]
        ],
        "epochs": [50],
        "batch_size": [32, 60000],
        "shuffle": [False]
    } 

    results = grid_search_cv(model_class=Model, param_grid=param_grid, X=X_train, y=y_train, validation_data=validation_data, cv=3, scoring="cross-entropy", shuffle=False, verbose=1, n_jobs=6)

    print(f"Best Score: {results["best_score"]:.4f}\n")
    print(f"\nBest Params: {results["best_params"]}\n")

    sorted_results = sorted(results["results"], key=lambda x: x[1], reverse=True)

    # print("Detailed results:")
    # for combo, mean_score, std_score in sorted_results:
    #     print(combo, f"\navg_score: {mean_score:.4f} - std_score: {std_score:.4f}\n")

    best_model = results["best_model"]
    best_loss_history = results["best_loss_history"]

    return best_model, best_loss_history

def random_search_test(X_train, y_train, validation_data=None):
    param_grid = {
        "loss_fn": ["cross-entropy"],
        "optimizer": ["rprop"],
        "opt_params": [
            {"eta_plus": 1.2, "eta_minus": 0.5, "delta_min": 1e-6, "delta_max": 50},
            {"eta_plus": 1.5, "eta_minus": 0.4, "delta_min": 1e-6, "delta_max": 100},
            {"eta_plus": 1.1, "eta_minus": 0.7, "delta_min": 1e-7, "delta_max": 25},
            {"eta_plus": 1.3, "eta_minus": 0.3, "delta_min": 1e-6, "delta_max": 75},
            {"eta_plus": 1.0, "eta_minus": 0.6, "delta_min": 1e-8, "delta_max": 10},
            {"eta_plus": 1.4, "eta_minus": 0.5, "delta_min": 1e-6, "delta_max": 80}
        ],
        "layers_config": [
            [
                DenseLayer(input_size=784, output_size=16, activation_fn='sigmoid'),
                DenseLayer(input_size=16, output_size=10, activation_fn='softmax')
            ],
            [
                DenseLayer(input_size=784, output_size=10, activation_fn='softmax')
            ],
            [
                DenseLayer(input_size=784, output_size=32, activation_fn='relu'),
                DenseLayer(input_size=32, output_size=16, activation_fn='relu'),
                DenseLayer(input_size=16, output_size=10, activation_fn='softmax')
            ]
        ],
        "epochs": [50, 100],
        "batch_size": [32, 128, 60000],
        "shuffle": [False, True]
    }

    
    # Set the number of random configurations to evaluate
    n_iter = 10 
    
    results = random_search_cv(
        model_class=Model,
        param_grid=param_grid,
        n_iter=n_iter,
        X=X_train,
        y=y_train,
        validation_data=validation_data,
        cv=3,
        scoring="cross-entropy",
        shuffle=False,
        verbose=1,
        n_jobs=6
    )
    
    print(f"Best Score: {results['best_score']:.4f}\n")
    print(f"\nBest Params: {results['best_params']}\n")
    
    sorted_results = sorted(results["results"], key=lambda x: x[1], reverse=True)
    
    # Uncomment the lines below to print detailed results:
    # for combo, mean_score, std_score in sorted_results:
    #     print(combo, f"\navg_score: {mean_score:.4f} - std_score: {std_score:.4f}\n")
    
    best_model = results["best_model"]
    best_loss_history = results["best_loss_history"]
    
    return best_model, best_loss_history



if __name__ == "__main__":
    mnist_train_set = rat_datasets.MNIST(train=True)
    mnist_val_set = rat_datasets.MNIST(train=False)

    X_train = mnist_train_set.images
    y_train = mnist_train_set.labels
    X_train = X_train.reshape(X_train.shape[0], -1)

    X_val = mnist_val_set.images
    y_val = mnist_val_set.labels
    X_val = X_val.reshape(X_val.shape[0], -1)

    print("Train set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_val.shape, y_val.shape)
    print()

    model, loss_history = random_search_test(X_train, y_train, validation_data=(X_val, y_val))
    #model, loss_history = grid_search_test(X_train, y_train, validation_data=(X_val, y_val))
    #model, loss_history = model_test(X_train, y_train, validation_data=(X_val, y_val))
    
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history['loss'], label='Training Loss')
    if 'val_loss' in loss_history:
        plt.plot(loss_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss evolution during training')
    plt.legend()
    plt.show()

    y_pred = model.predict(X_val)
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_val, axis=1)

    accuracy = np.mean(pred_classes == true_classes)
    print("\nTest Accuracy:", accuracy, "\n")

    print("Predictions:", pred_classes[:10])
    print("Ref values:", true_classes[:10])

    indices = np.random.choice(X_val.shape[0], 15, replace=False)
    sample_images = X_val[indices]
    sample_true = true_classes[indices]
    sample_pred = pred_classes[indices]

    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(indices):
        plt.subplot(3, 5, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {sample_pred[i]}\nTrue: {sample_true[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
