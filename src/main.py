import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from pyrat.datasets import MNIST, train_val_split
from pyrat.models import Model
from pyrat.layers import DenseLayer
from pyrat.tuning import grid_search_cv, random_search_cv

if __name__ == "__main__":
    mnist_train = MNIST(train=True)
    mnist_test = MNIST(train=False)

    X_full = mnist_train.images
    y_full = mnist_train.labels
    X_full = X_full.reshape(X_full.shape[0], -1)

    X_test = mnist_test.images
    y_test = mnist_test.labels
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train, X_val, y_train, y_val = train_val_split(X_full, y_full, val_size=0.2, random_state=42)

    param_grid = {
        "loss_fn": ["cross-entropy"],
        "optimizer": ["rprop"],
        "eta_plus": [1.1, 1.2, 1.5],
        "eta_minus": [0.3, 0.4, 0.7],
        "delta_min": [1e-6, 1e-7],
        "delta_max": [50, 75],
        "layers_config": [
            [
                DenseLayer(input_size=784, output_size=16, activation_fn='sigmoid'),
                DenseLayer(input_size=16, output_size=10, activation_fn='softmax')
            ]
        ],
        "epochs": [100],
        "batch_size": [60000],
        "shuffle": [True]
    }

    use_random_search = True

    if use_random_search:
        results = random_search_cv(
            model_class=Model,
            param_grid=param_grid,
            n_iter=6,
            X=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            cv=4,
            scoring="cross-entropy",
            verbose=1,
            n_jobs=6
        )
    else:
        results = grid_search_cv(
            model_class=Model,
            param_grid=param_grid,
            X=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            cv=4,
            scoring="cross-entropy",
            verbose=1,
            n_jobs=6
        )

    print(f"Best Score: {results['best_score']:.4f}")
    print(f"Best Params: {results['best_params']}")

    all_results = results["results"]
    sorted_results = sorted(all_results, key=lambda x: x["mean_score"])
    print("\nDetailed results:\n")
    for res in sorted_results:
        ps = res["params"]
        ms = res["mean_score"]
        ss = res["std_score"]
        tt = res["train_time"]
        print(f"{ps} | mean_score: {ms:.4f}, std_score: {ss:.4f}, train_time: {tt:.2f}s\n")

    best_model = results["best_model"]
    best_loss_history = results["best_loss_history"]
    best_val_loss_history = results["best_val_loss_history"]
    best_accuracy_history = results["best_accuracy_history"]
    best_val_accuracy_history = results["best_val_accuracy_history"]

    plt.figure()
    plt.plot(best_loss_history, label='Training Loss')
    if best_val_loss_history:
        plt.plot(best_val_loss_history, label='Validation Loss')
    plt.title('Loss evolution')
    plt.legend()
    plt.show()

    if best_accuracy_history:
        plt.figure()
        plt.plot(best_accuracy_history, label='Training Accuracy')
        if best_val_accuracy_history:
            plt.plot(best_val_accuracy_history, label='Validation Accuracy')
        plt.title('Accuracy evolution')
        plt.legend()
        plt.show()

    y_pred_test = best_model.predict(X_test)
    pred_classes_test = np.argmax(y_pred_test, axis=1)
    true_classes_test = np.argmax(y_test, axis=1)
    test_accuracy = np.mean(pred_classes_test == true_classes_test)
    print("Test Accuracy:", test_accuracy)

    cm = confusion_matrix(true_classes_test, pred_classes_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    sample_indices = np.random.choice(X_test.shape[0], 15, replace=False)
    sample_images = X_test[sample_indices]
    sample_true = true_classes_test[sample_indices]
    sample_pred = pred_classes_test[sample_indices]

    plt.figure(figsize=(12, 4))
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {sample_pred[i]}\nTrue: {sample_true[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
