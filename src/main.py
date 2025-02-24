import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import pyrat.datasets as rat_datasets
import pyrat.layers as rat_layers
import pyrat.models as rat_models

# pyrat - Rapid Artificial Training

if __name__ == "__main__":
    mnist_train_set = rat_datasets.MNIST(train=True)
    mnist_test_set = rat_datasets.MNIST(train=False)

    X_train = mnist_train_set.images
    y_train = mnist_train_set.labels

    X_train = X_train.reshape(X_train.shape[0], -1) # da (60000, 28, 28, 1) a (60000, 784)

    X_test = mnist_test_set.images
    y_test = mnist_test_set.labels
    
    X_test = X_test.reshape(X_test.shape[0], -1) # da (60000, 28, 28, 1) a (60000, 784)

    print("Train set shape:")
    print(str(X_train.shape))
    print(str(y_train.shape))
    print("Test set shape:")
    print(str(X_test.shape))
    print(str(y_test.shape))
    print()

    # Testing
    
    model = rat_models.Model()
    model.add(rat_layers.DenseLayer(input_size=784, output_size=10, optimizer='rprop', activation_fn='softmax'))
    loss_history = model.fit(X_train, y_train, epochs=100, batch_size=60, validation_data=(X_test, y_test), patience=5)
    
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss evolution during training')
    plt.legend()
    plt.show()
    
    y_pred = model.predict(X_test) # shape: (10000, 10)

    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    accuracy = np.mean(pred_classes == true_classes)
    print()
    print("Test Accuracy:", accuracy)
    print()
    print("Predictions:")
    print(pred_classes[:10])
    print("Ref values:")
    print(true_classes[:10])

    indices = np.random.choice(X_test.shape[0], 15, replace=False)
    sample_images = X_test[indices]
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